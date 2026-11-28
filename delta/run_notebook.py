"""Execute a notebook headless without ever damaging the original.

    python delta/run_notebook.py NOTEBOOK.ipynb KERNEL

Runs into delta/logs/<name>-<jobid>.ipynb and stops at the first failing cell (so a broken setup cell
does not start hours of later work). Only if every cell succeeds is the executed copy written back over
the notebook, so its outputs can be committed. Exit code 0 = success, 1 = a cell failed.
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

path, kernel = sys.argv[1], sys.argv[2]
name = os.path.splitext(os.path.basename(path))[0]
job = os.environ.get('SLURM_JOB_ID', 'local')
out = os.path.join('delta', 'logs', f'{name}-{job}.ipynb')
os.makedirs(os.path.dirname(out), exist_ok=True)

nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, kernel_name=kernel, timeout=None, allow_errors=False,
                        resources={'metadata': {'path': os.path.dirname(os.path.abspath(path))}})
t0, failed = time.time(), None
try:
    client.execute()
except CellExecutionError as e:
    failed = e
except Exception as e:  # kernel died, etc.
    failed = e
finally:
    nbformat.write(nb, out)

mins = (time.time() - t0) / 60
if failed is None:
    nbformat.write(nb, path)
    print(f'OK: {path} finished in {mins:.0f} min; outputs saved into the notebook (copy: {out})')
    sys.exit(0)
print(f'FAILED: {path} after {mins:.0f} min. The original notebook is unchanged.')
print(f'Partial run with the error: {out}')
print(str(failed)[-3000:])
sys.exit(1)
