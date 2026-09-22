"""Notebook-local VAE prototype, separate from the production model registry.

Nothing trains on import. Standard and beta variants share architecture; beta changes
only the KL weight. Decoder likelihood is N(reconstruction, I), in scaled input units.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from _exploration import ROOT  # sets up the existing repo modules
from models.autoencoder import ConvAutoencoder, LatentLOF
from evaluation.common import classification_metrics, smooth_by_trial
from training.common import pick_device


class ConvVAE(nn.Module):
    def __init__(self, channels=16, window=200):
        super().__init__()
        reference=ConvAutoencoder(channels,window)
        self.trunk=reference.encoder[:8]  # 16×200 -> 24×8, same first two AE blocks
        self.mu_head=nn.Conv1d(24,4,19,padding=9)
        self.logvar_head=nn.Conv1d(24,4,19,padding=9)
        self.decoder=reference.decoder

    def encode(self,x):
        h=self.trunk(x)
        return self.mu_head(h),self.logvar_head(h).clamp(-8,8)

    def forward(self,x,sample=None):
        mu,logvar=self.encode(x)
        sample=self.training if sample is None else sample
        z=mu+torch.exp(.5*logvar)*torch.randn_like(mu) if sample else mu
        return self.decoder(z),mu,logvar


def loss_terms(x,reconstruction,mu,logvar):
    """Per-window sums (NOT mean reconstruction mixed with summed KL)."""
    nll=.5*((x-reconstruction).square()+math.log(2*math.pi)).flatten(1).sum(1)
    kl=.5*(mu.square()+logvar.exp()-1-logvar).flatten(1).sum(1)
    return nll,kl


@dataclass
class VAEConfig:
    beta: float=1.0
    lr: float=1e-3
    batch_size: int=256
    max_epochs: int=50
    patience: int=5
    seed: int=0
    val_fraction: float=.2
    held_out: str | None=None  # None: original protocol; or SA / ST / TR
    device: str | None=None
    max_train_windows: int | None=None  # optional SMOKE cap; not a full benchmark
    lof_fit_samples: int=50_000
    mc_samples: int=8


def data_for(cfg):
    if cfg.held_out:
        from splits import build
        return build('autoencoder',cfg.held_out,batch_size=cfg.batch_size)
    from dataset import AutoencoderData
    return AutoencoderData(batch_size=cfg.batch_size)


def _epoch(model,loader,device,beta,opt=None):
    model.train(opt is not None)
    totals=np.zeros(3);count=0
    for x in loader:
        x=x.to(device)
        with torch.set_grad_enabled(opt is not None):
            rec,mu,lv=model(x,sample=True)
            nll,kl=loss_terms(x,rec,mu,lv)
            loss=(nll+beta*kl).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite VAE objective')
            if opt is not None:
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        totals+=np.array([loss.item(),nll.mean().item(),kl.mean().item()])*len(x)
        count+=len(x)
    return dict(zip(('loss','nll','kl'),(totals/max(count,1)).tolist()))


@torch.inference_mode()
def encode_means(model,loader,device):
    model.eval()
    return np.concatenate([model.encode(x.to(device))[0].flatten(1).cpu().numpy() for x in loader])


@torch.inference_mode()
def score_split(model,loader,device,mc_samples):
    model.eval(); means=[]; nlls=[]; kls=[]
    for x in loader:
        x=x.to(device);mu,lv=model.encode(x)
        values=[]
        for _ in range(mc_samples):
            z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
            nll,kl=loss_terms(x,model.decoder(z),mu,lv);values.append(nll)
        means.append(mu.flatten(1).cpu().numpy())
        nlls.append(torch.stack(values).mean(0).cpu().numpy());kls.append(kl.cpu().numpy())
    nll=np.concatenate(nlls);kl=np.concatenate(kls)
    return np.concatenate(means),{'reconstruction_nll':nll,'kl':kl,'negative_elbo':nll+kl}


def train_experiment(out,cfg):
    """Explicit opt-in training/evaluation; refuses to overwrite existing runs."""
    if cfg.beta<0 or cfg.mc_samples<1 or cfg.max_epochs<1 or cfg.patience<1:
        raise ValueError('Invalid beta, Monte Carlo sample count, epoch count or patience')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    dat=data_for(cfg);device=pick_device(cfg.device)
    torch.manual_seed(cfg.seed);rng=np.random.default_rng(cfg.seed)
    tr,va,val_subjects=dat.train_val_indices(cfg.val_fraction,cfg.seed)
    if cfg.max_train_windows is not None:
        tr=np.sort(rng.choice(tr,min(len(tr),cfg.max_train_windows),replace=False))
        va=np.sort(rng.choice(va,min(len(va),cfg.max_train_windows),replace=False))
    model=ConvVAE(dat.n_channels,dat.window).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    best=float('inf');best_state=None;best_epoch=-1;history=[]
    for epoch in range(cfg.max_epochs):
        train=_epoch(model,dat.loader('train',indices=tr,shuffle=True,seed=cfg.seed+epoch),device,cfg.beta,opt)
        # A fixed Monte Carlo stream avoids changing the validation noise every epoch.
        torch.manual_seed(cfg.seed+10_000)
        val=_epoch(model,dat.loader('train',indices=va),device,cfg.beta)
        row={'epoch':epoch,**{f'train_{k}':v for k,v in train.items()},
             **{f'val_{k}':v for k,v in val.items()}};history.append(row)
        print(f"epoch {epoch+1}: train={train['loss']:.2f} val={val['loss']:.2f} KL={val['kl']:.2f}")
        if val['loss']<best:
            best=val['loss'];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch>=cfg.patience:break
    model.load_state_dict(best_state);model.eval()
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'model':best_state,'config':asdict(cfg)},out/'final.pt')
    # Fit LOF on ID train only, just as the AE baseline. Never on test latents.
    fit_idx=np.sort(rng.choice(len(dat.train),min(len(dat.train),cfg.lof_fit_samples),replace=False))
    fit_mu=encode_means(model,dat.loader('train',indices=fit_idx),device)
    lof=LatentLOF(fit_samples=None,seed=cfg.seed).fit(fit_mu)
    thresholds={};evaluations={}
    for split in ('train','val','test'):
        torch.manual_seed(cfg.seed+20_000)
        sp=dat.splits[split];mu,raw=score_split(model,dat.loader(sp),device,cfg.mc_samples)
        raw['latent_lof']=lof.score(mu)
        meta=sp.meta.copy().reset_index(drop=True);meta['kind']='ID'
        if split!='train':
            ood=dat.ood[split]
            om,os=score_split(model,dat.loader(ood),device,cfg.mc_samples)
            os['latent_lof']=lof.score(om);mu=np.concatenate([mu,om])
            raw={k:np.r_[v,os[k]] for k,v in raw.items()}
            other=ood.meta.copy();other['kind']='OOD';meta=pd.concat([meta,other],ignore_index=True)
        smooth={k:smooth_by_trial(v,meta,10,'median') for k,v in raw.items()}
        if split=='train':
            thresholds={k:float(np.percentile(v,99.5)) for k,v in smooth.items()}
            continue
        lab=(meta.kind=='OOD').to_numpy().astype(int)
        scores={k:classification_metrics(lab,v,thresholds[k])|{'threshold':thresholds[k]}
                for k,v in smooth.items()}
        rows=[]
        for mode,g in meta.groupby('mode',sort=False):
            idx=g.index.to_numpy();s=smooth['latent_lof'][idx]
            rows.append({'mode':mode,'kind':g.kind.iloc[0],'windows':len(idx),
                         'median_psi':float(np.median(s)),
                         'flagged_%':float(100*np.mean(s>thresholds['latent_lof']))})
        report={'model':'notebook_conv_vae','split':split,'scores':scores,'per_mode':rows,
                'primary_score':'latent_lof','held_out':cfg.held_out,
                'smoke_run':cfg.max_train_windows is not None}
        (out/f'eval_{split}.json').write_text(json.dumps(report,indent=2))
        np.savez_compressed(out/f'latents_{split}.npz',Z=mu,labels=lab,
                            **{f'smooth_{k}':v for k,v in smooth.items()})
        meta.to_parquet(out/f'latents_{split}.parquet',index=False)
        evaluations[split]=report
    report={'config':asdict(cfg),'history':history,'best_epoch':best_epoch,
            'val_subjects':list(val_subjects),'thresholds':thresholds,
            'smoke_run':cfg.max_train_windows is not None,'latent_dim':32}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    np.save(out/'train_reference_mu.npy',fit_mu)
    return model,dat,report,evaluations


def plot_representation(out):
    """Fit the display projection on training latents, then project held-out test data."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    out=Path(out)
    if not (out/'latents_test.npz').exists():
        print('No trained VAE representation yet. Enable training above, then rerun.');return
    reference=np.load(out/'train_reference_mu.npy')
    z=np.load(out/'latents_test.npz');meta=pd.read_parquet(out/'latents_test.parquet')
    pca=PCA(n_components=2).fit(reference)
    points=pca.transform(z['Z']);rng=np.random.default_rng(0)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for mode,g in meta.groupby('mode',sort=False):
        idx=g.index.to_numpy();idx=rng.choice(idx,min(700,len(idx)),replace=False)
        axes[0].scatter(*points[idx].T,s=8,alpha=.5,label=mode)
    axes[0].set(xlabel='Training-fit PC1',ylabel='Training-fit PC2',title='Posterior means by task')
    axes[0].legend(markerscale=2,ncol=2)
    for kind in (0,1):
        values=z['smooth_latent_lof'][z['labels']==kind]
        axes[1].hist(values,bins=60,density=True,alpha=.5,label='OOD' if kind else 'ID')
    threshold=json.loads((out/'report.json').read_text())['thresholds']['latent_lof']
    axes[1].axvline(threshold,color='black',ls='--',label='Train threshold')
    axes[1].set(xlabel='Latent LOF score',ylabel='Density',title='Test score distributions')
    axes[1].legend();fig.tight_layout();plt.show()
    # Representative traces are actual recorded inputs, not synthetic placeholders.
    saved=json.loads((out/'report.json').read_text())
    cfg=VAEConfig(**saved['config'])
    try:
        dat=data_for(cfg)
    except FileNotFoundError:
        print('Input arrays unavailable; reconstruction traces skipped.');return
    model=ConvVAE(dat.n_channels,dat.window).eval()
    model.load_state_dict(torch.load(out/'final.pt',map_location='cpu',weights_only=True)['model'])
    channel=dat.channels.index('enc_angle_l')
    fig,axes=plt.subplots(1,2,figsize=(12,3.5))
    for ax,sp,kind in zip(axes,(dat.test,dat.ood['test']),('ID','OOD')):
        x=torch.from_numpy(sp.take([0]))
        with torch.inference_mode():rec,_,_=model(x,sample=False)
        times=(np.arange(dat.window)-dat.window+1)/200
        ax.plot(times,x[0,channel],label='Recorded input')
        ax.plot(times,rec[0,channel],ls='--',label='Decode posterior mean')
        row=sp.meta.iloc[0]
        ax.set(title=f'{kind}: {row["subject"]}, {row["mode"]} (first window)',
               xlabel='Time relative to window end (s)',ylabel='Standardized left encoder angle')
        ax.legend()
    fig.tight_layout();plt.show()
