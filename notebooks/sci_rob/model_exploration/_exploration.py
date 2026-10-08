"""Read-only presentation helpers shared by the model exploration notebooks."""
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display


def find_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / 'ml/sci_rob/vanilla/registry.py').is_file():
            return candidate
    raise FileNotFoundError('Open this notebook inside the repository checkout.')


ROOT = find_root(Path(__file__).parent)
for folder in ('ml/common', 'ml/sci_rob/vanilla', 'ml/sci_rob/prob_ensemble', 'ml/sci_rob/LOTO', 'ml/sci_rob/VAE', 'ml/sci_rob/awareness'):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.append(path)

plt.rcParams.update({'figure.figsize': (10, 4), 'figure.dpi': 110,
                     'font.size': 11, 'axes.titlesize': 13, 'legend.fontsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

RUNS = {
    'ensemble': 'ml/sci_rob/vanilla/runs/ensemble_gp',
    'autoencoder': 'ml/sci_rob/vanilla/runs/autoencoder',
    'gan': 'ml/sci_rob/vanilla/runs/gan',
    'synthetic': 'ml/sci_rob/vanilla/runs/synthetic',
    'forecast-angle': 'ml/sci_rob/vanilla/runs/forecast_angle',
    'forecast-all': 'ml/sci_rob/vanilla/runs/forecast_all',
    'prob_ensemble': 'ml/sci_rob/prob_ensemble/runs/prob_forecast',
}
LOTO = {'ensemble': 'vanilla/Ensemble_GP', 'autoencoder': 'vanilla/AE',
        'gan': 'vanilla/GAN', 'synthetic': 'vanilla/Ensemble_synthetic',
        'forecast-angle': 'vanilla/Ensemble_forecast', 'prob_ensemble': 'prob_ensemble/run'}


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def inventory(run):
    run = Path(run)
    display(Markdown(f'**Artifact directory:** `{run.relative_to(ROOT)}`'))
    display(pd.DataFrame([{'artifact': n, 'available': (run/n).exists()}
                          for n in ('report.json', 'final.pt', 'eval_test.json', 'figures')]))
    if not (run/'report.json').exists():
        print('No completed training report here. Architecture/commands below are not results.')


def diagram(rows, title):
    """Rows are independent flows, not implied connections between separate branches."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.9*len(rows)), squeeze=False)
    for ax, boxes in zip(axes.flat, rows):
        ax.set_xlim(0, len(boxes)); ax.set_ylim(0, 1); ax.axis('off')
        for i, label in enumerate(boxes):
            box = FancyBboxPatch((i+.035, .18), .84, .64,
                                 boxstyle='round,pad=0.018', facecolor='#edf4f8',
                                 edgecolor='#487b96', linewidth=1.2)
            ax.add_patch(box)
            ax.text(i+.455, .5, label, ha='center', va='center', fontsize=10)
            if i < len(boxes)-1:
                ax.annotate('', (i+1.025, .5), (i+.90, .5),
                            arrowprops={'arrowstyle': '->', 'color': '#354b5e'})
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(); plt.show()


def config_and_history(run):
    report = read_json(Path(run)/'report.json')
    if not report:
        print('No saved training history.'); return
    display(pd.Series(report.get('config', {}), name='saved value').to_frame())
    summary = {k:v for k,v in report.items() if k not in ('config','history','stage2_history')
               and not isinstance(v, (dict, list))}
    if report.get('val_subjects'):
        summary['internal early-stopping participants'] = ', '.join(report['val_subjects'])
    if 'stage1' in report:
        summary['LOSO selected epochs'] = report['stage1'].get('n_epochs')
    display(pd.Series(summary, name='saved result').to_frame())
    history = pd.DataFrame(report.get('history', report.get('stage2_history', [])))
    if history.empty:
        return
    fields = [c for c in ('train_loss','val_loss','train_nll','loss_g','loss_d') if c in history]
    if fields:
        fig, ax = plt.subplots()
        for col in fields:
            ax.plot(history['epoch']+1, history[col], label=col)
        ax.set(xlabel='Epoch (1-based)', ylabel='Recorded objective', title='Saved training history')
        ax.legend(); fig.tight_layout(); plt.show()
    display(history.tail(5))
    folds = read_json(Path(run)/'loso_summary.json') or report.get('stage1', {})
    if folds:
        display(Markdown('**LOSO summary (epoch selection, not LOTO):**'))
        display(folds)


def results_table(run):
    result = read_json(Path(run)/'eval_test.json')
    if not result:
        print('No saved test evaluation; no performance is inferred.'); return
    scores = result.get('scores', {'primary':result.get('metrics', {})})
    rows = []
    for name, m in scores.items():
        rows.append({'score':name, 'AUROC':m.get('auroc'),
                     'recall (%)':m.get('recall'), 'specificity (%)':m.get('specificity'),
                     'J (−1 to 1)':m.get('j_statistic',float('nan'))/100,
                     'threshold':m.get('threshold',result.get('threshold'))})
    display(pd.DataFrame(rows).set_index('score').round(4))
    if result.get('calibration'):
        display(pd.Series(result['calibration'],name='saved calibration diagnostic').to_frame())
    pm = result.get('per_mode', [])
    if isinstance(pm,dict): pm = pm.get('epistemic', [])
    if pm: display(pd.DataFrame(pm).round(4))


def gallery(run, entries):
    for filename, explanation in entries:
        path = Path(run)/'figures'/filename
        display(Markdown(f'**{filename}** — {explanation}'))
        if path.exists():
            display(Image(filename=str(path), width=950))
        else:
            print('Not available:',path.relative_to(ROOT))


def loto_results(key):
    leaf = LOTO.get(key)
    if leaf is None:
        print('No LOTO run mapping exists for this variant.'); return
    rows = []
    for task in ('SA','ST','TR'):
        r=read_json(ROOT/'ml/sci_rob/LOTO'/task/leaf/'eval_test.json')
        if not r: continue
        for score,m in r.get('scores',{'primary':r.get('metrics',{})}).items():
            c=m.get('counts',{})
            rows.append({'held out':task,'score':score,'AUROC':m.get('auroc'),
                         'recall (%)':m.get('recall'),
                         'ID false alarms (%)':100-m.get('specificity',100),
                         'J':m.get('j_statistic',0)/100,
                         'OOD windows':c.get('tp',0)+c.get('fn',0)})
    if not rows:
        print('No completed LOTO evaluations found.'); return
    df=pd.DataFrame(rows); display(df.round(4))
    primary=df[df.score.isin(['primary','epistemic'])]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,col in zip(axes,['AUROC','recall (%)']):
        bars=ax.bar(primary['held out'],primary[col],color='#367f9c')
        ax.set(xlabel='Task excluded from training',ylabel=col,
               ylim=(0,1 if col=='AUROC' else 100))
        if col=='AUROC':ax.axhline(.5,color='gray',ls='--',label='chance');ax.legend()
        ax.bar_label(bars,fmt='%.3f' if col=='AUROC' else '%.1f',padding=3)
    fig.suptitle('LOTO: ranking and the training-calibrated operating point')
    fig.tight_layout();plt.show()


def source_links(paths):
    display(Markdown('Implementation sources: '+', '.join(
        f'[{Path(p).name}]({os.path.relpath(ROOT / p, Path(__file__).parent)})' for p in paths)))


def training_commands(key, out):
    out=str(out)
    if key!='prob_ensemble':
        return [[sys.executable,str(ROOT/'ml/sci_rob/vanilla/main.py'),'all','--model',key,'--out',out]]
    train=str(ROOT/'ml/sci_rob/prob_ensemble/train.py')
    commands=[[sys.executable,train,'fold','--index',str(i),'--out',out] for i in range(12)]
    commands += [[sys.executable,train,stage,'--out',out] for stage in ('collect','final')]
    commands += [[sys.executable,str(ROOT/'ml/sci_rob/prob_ensemble'/f'{s}.py'),'--out',out,'--split','test']
                 for s in ('evaluate','plots')]
    return commands


def run_commands(commands, out, enabled=False):
    for cmd in commands:print(shlex.join(cmd))
    if not enabled:
        print('\nPreview only. Set RUN_TRAINING=True explicitly to run.');return
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a populated run directory: {out}')
    for cmd in commands:
        subprocess.run(cmd,cwd=ROOT,check=True)


def model_shapes(model, example):
    """Forward pass with a fresh model: architecture check, never a performance result."""
    import torch
    model.eval()
    with torch.inference_mode():output=model(example)
    tensors=output if isinstance(output,tuple) else (output,)
    print(f'{type(model).__name__}: {sum(p.numel() for p in model.parameters()):,} parameters')
    print('Input:',tuple(example.shape),'  Outputs:',[tuple(x.shape) for x in tensors])
    inspected=model.members[0] if hasattr(model,'members') else model
    if inspected is not model:print('Layer table shows one representative branch; weights are not shared.')
    display(pd.DataFrame([{'module':name,'class':type(m).__name__,
                           'direct parameters':sum(p.numel() for p in m.parameters(recurse=False))}
                          for name,m in inspected.named_modules() if len(list(m.children()))==0]))
    return output
