"""Small synthetic wiring tests, NOT trained-model benchmarks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import _vae_experiment as vae
from _exploration import run_commands

torch.set_num_threads(2)


class ToySplit:
    def __init__(self,n,subject,mode,seed):
        self.x=np.random.default_rng(seed).normal(size=(n,16,200)).astype('float32')
        self.meta=pd.DataFrame({'subject':[subject]*n,'trial':['trial1']*n,
                                'mode':[mode]*n,'end_idx':199+10*np.arange(n)})

    def __len__(self):return len(self.x)
    def take(self,idx):return self.x[idx].copy()


class ToyData:
    n_channels=16
    window=200
    channels=['enc_angle_l']+[f'channel_{i}' for i in range(15)]

    def __init__(self):
        self.splits={s:ToySplit(n,s,'LG',i) for i,(s,n) in
                     enumerate([('train',32),('val',8),('test',8)])}
        self.train,self.val,self.test=(self.splits[s] for s in ('train','val','test'))
        self.ood={s:ToySplit(8,s,'SA',10+i) for i,s in enumerate(('val','test'))}

    def train_val_indices(self,*args):return np.arange(24),np.arange(24,32),['internal-validation']

    def loader(self,split,indices=None,**kwargs):
        sp=self.splits[split] if isinstance(split,str) else split
        idx=np.arange(len(sp)) if indices is None else indices
        for i in range(0,len(idx),8):yield torch.from_numpy(sp.take(idx[i:i+8]))


class ExplorationTests(unittest.TestCase):
    def test_shapes_gradients_and_eval_determinism(self):
        torch.manual_seed(0);m=vae.ConvVAE();x=torch.randn(3,16,200)
        rec,mu,lv=m(x);self.assertEqual(tuple(mu.shape),(3,4,8))
        self.assertEqual(rec.shape,x.shape)
        nll,kl=vae.loss_terms(x,rec,mu,lv)
        self.assertTrue(torch.all(kl>=0));(nll+4*kl).mean().backward()
        self.assertTrue(torch.isfinite(m.mu_head.weight.grad).all())
        self.assertGreater(m.logvar_head.weight.grad.abs().sum(),0)
        m.eval()
        with torch.inference_mode():a=m(x)[0];b=m(x)[0]
        torch.testing.assert_close(a,b)

    def test_kl_zero_at_prior(self):
        x=torch.zeros(2,16,200);mu=torch.zeros(2,4,8)
        _,kl=vae.loss_terms(x,x,mu,torch.zeros_like(mu))
        torch.testing.assert_close(kl,torch.zeros(2))

    def test_opt_in_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';out.mkdir();(out/'existing').touch()
            with patch('_exploration.subprocess.run') as call:
                run_commands([['python','never-run.py']],out,enabled=False)
                call.assert_not_called()
                with self.assertRaises(FileExistsError):
                    run_commands([['python','never-run.py']],out,enabled=True)
                call.assert_not_called()
            with self.assertRaises(FileExistsError):vae.train_experiment(out,vae.VAEConfig())

    def test_tiny_training_scoring_and_plot_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(vae,'data_for',return_value=ToyData()):
            out=Path(tmp)/'toy'
            cfg=vae.VAEConfig(max_epochs=2,patience=1,mc_samples=2,
                              lof_fit_samples=32,device='cpu',batch_size=8)
            model,dat,report,ev=vae.train_experiment(out,cfg)
            self.assertEqual(set(ev),{'val','test'})
            self.assertEqual(set(ev['test']['scores']),{'latent_lof','reconstruction_nll','kl','negative_elbo'})
            for values in ev['test']['scores'].values():
                self.assertTrue(np.isfinite(values['threshold']))
                self.assertEqual(sum(values['counts'].values()),16)
            self.assertEqual(np.load(out/'latents_test.npz')['Z'].shape,(16,32))
            self.assertTrue((out/'final.pt').exists())
            vae.plot_representation(out);plt.close('all')


if __name__=='__main__':unittest.main()
