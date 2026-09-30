"""Execute safe notebook defaults in fresh kernels; save their rendered outputs."""
import argparse
import ast
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from nbformat.sign import NotebookNotary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--notebook',default='*.ipynb',help='Glob within this folder')
    parser.add_argument('--timeout',type=int,default=180,help='seconds per cell (UMAP cells over trained runs need more)')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    # The generic Python kernel resolves python via PATH; use this interpreter's environment.
    env=dict(os.environ);env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    for path in sorted(here.glob(args.notebook)):
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)
                if 'RUN_TRAINING =' in cell.source:
                    assert 'RUN_TRAINING = False' in cell.source
        if args.execute:
            print('Executing',path.name,flush=True)
            client=NotebookClient(nb,timeout=args.timeout,kernel_name='python3',
                                  resources={'metadata':{'path':str(here)}})
            client.execute(env=env)
            errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
            assert not errors
            nbformat.write(nb,path)  # generated execution outputs, not source changes
            # Sign the executed notebook so JupyterLab treats it as trusted: otherwise it hides
            # JavaScript-backed outputs (the interactive Plotly figures) until 'Trust Notebook'.
            NotebookNotary().sign(nbformat.read(path,as_version=4))
            images=sum('image/png' in o.get('data',{}) for c in nb.cells
                       if c.cell_type=='code' for o in c.outputs)
            print(f'  PASS: {len(nb.cells)} cells; {images} embedded figures',flush=True)
        else:print('Valid:',path.name)


if __name__=='__main__':main()
