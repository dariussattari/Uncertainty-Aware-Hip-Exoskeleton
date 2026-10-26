"""The training-class ablation's task list: every Nature autoencoder and VAE trained so far, retrained on other classes.

    python ml/nature/awareness/train_class_tasks.py > ml/nature/awareness/slurm/train_class_tasks.txt
    python ml/nature/awareness/train_class_tasks.py --arch tcn > ml/nature/awareness/slurm/tcn_tasks.txt   # a new VAE, every class

With ``--arch``, a VAE architecture not trained yet gets its whole grid instead: every input x ``train_vae.CORE_CELLS`` x
seeds, on ID-safe (out-root ``runs/vae``, as the other VAEs) and on each class in ``ABLATION``.

Reads every finished ID-safe run under ``notebooks/nature/model_exploration/runs/{ae,vae}`` (the configurations actually
trained) and prints one trainer command per (training class in ``ABLATION``, run): the same model, input, cell and seed,
with ``--train-class`` and an ``--out-root`` under ``runs_train_class/<folder>/<ae|vae>``. Every command rebuilds its
run's folder name exactly (checked here), so a retrained run pairs with its ID-safe twin by name.
``slurm/train_class.sbatch`` runs line i as array task i and adds ``--data-dir``. Counts go to stderr.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common'))                 # ml/common: dataset-agnostic models
sys.path.insert(0, str(HERE))
import train_ae as ta  # noqa: E402
import train_vae as tv  # noqa: E402

RUNS = HERE.parents[2] / 'notebooks' / 'nature' / 'model_exploration' / 'runs'
OUT_ROOT = 'ml/nature/awareness/runs_train_class'                   # on the cluster, relative to the repo
ABLATION = {'ID-safe+OOD-safe': 'id_ood_safe', 'OOD-safe': 'ood_safe'}   # training class -> results folder


def get_ae_task(config: dict) -> tuple[str, str]:
    """(train_ae.py arguments, run name) that reproduce an autoencoder run."""
    inputs, latent = config.get('inputs', 'raw_1s'), config.get('latent', '4x8')
    args = f"train_ae.py --variant {config['variant']} --seed {config['seed']} --inputs {inputs}"
    if config['variant'] == 'sparse':
        args += f" --sparsity-weight {config['sparsity_weight']:g}"
    if latent != '4x8':
        args += f' --latent {latent}'
    name = ta.get_run_name(ta.AeConfig(variant = config['variant'], seed = config['seed'], inputs = inputs,
                                       sparsity_weight = config['sparsity_weight'], latent = latent))
    return args, name


def get_vae_task(config: dict) -> tuple[str, str]:
    """(train_vae.py arguments, run name) that reproduce a VAE run."""
    arch, inputs = config.get('arch', 'conv'), config.get('inputs', 'raw_1s')
    args = (f"train_vae.py --arch {arch} --inputs {inputs} --beta {config['beta']:g} "
            f"--decoder-variance {config['decoder_variance']} --seed {config['seed']}")
    name = tv.get_run_name(tv.VaeConfig(beta = config['beta'], decoder_variance = config['decoder_variance'], seed = config['seed'],
                                        inputs = inputs, arch = arch))
    return args, name


def get_tasks(runs: Path = RUNS) -> list[str]:
    """One command per (training class, finished ID-safe run), training classes first, then AE before VAE, then run name."""
    makers = {'ae': get_ae_task, 'vae': get_vae_task}
    finished = {family: [p.parent for p in sorted((runs / family).glob('*/summary.json')) if not p.parent.name.endswith('_smoke')]
                for family in makers}
    tasks = []
    for train_class, folder in ABLATION.items():
        for family, make in makers.items():
            for run in finished[family]:
                config = json.loads((run / 'config.json').read_text())
                args, name = make(config)
                if name != run.name or config.get('train_class', 'ID-safe') != 'ID-safe':
                    raise ValueError(f'{run}: the task rebuilds {name!r} trained on {config.get("train_class")!r}; '
                                     'expected the same name and ID-safe')
                tasks.append(f'{args} --train-class {train_class} --out-root {OUT_ROOT}/{folder}/{family}')
    return tasks


def get_grid_tasks(arch: str) -> list[str]:
    """Every input x CORE_CELLS x seed of one VAE architecture, on ID-safe and on each class in ``ABLATION``."""
    roots = {'ID-safe': 'ml/nature/awareness/runs/vae', **{c: f'{OUT_ROOT}/{folder}/vae' for c, folder in ABLATION.items()}}
    return [f'train_vae.py --arch {arch} --inputs {inputs} --beta {beta:g} --decoder-variance {variance} --seed {seed} '
            f'--train-class {train_class} --out-root {root}'
            for train_class, root in roots.items() for inputs in ta.INPUTS for beta, variance in tv.CORE_CELLS for seed in tv.SEEDS]


if __name__ == '__main__':
    lines = get_grid_tasks(sys.argv[sys.argv.index('--arch') + 1]) if '--arch' in sys.argv else get_tasks()
    print('\n'.join(lines))
    print(f'{len(lines)} tasks: {", ".join(f"{c} {sum(f"--train-class {c} " in t for t in lines)}" for c in ("ID-safe", *ABLATION))}; '
          f'AE {sum(t.startswith("train_ae") for t in lines)}, VAE {sum(t.startswith("train_vae") for t in lines)}', file = sys.stderr)
