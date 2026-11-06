#!/usr/bin/env python3
"""Two-rank matched native phase resources, measured only after correctness."""
import sys,os,time,json,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from diffspecfem import Solver,objective
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--case');parser.add_argument('--evidence',default='evidence/mpi2');args=parser.parse_args()
ev=root/args.evidence;s=Solver(root,ranks=2,case=args.case)
if not json.loads((ev/'validation.json').read_text())['passed']:raise SystemExit('correctness required first')
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
rows=[]
def timed(name,exe,mode=None):
 e=env.copy();e.pop('DIFFSPECFEM_MODE',None)
 if mode:e['DIFFSPECFEM_MODE']=mode
 prefix=ev/name;log=ev/(name+'.log');t=time.perf_counter()
 argv=['nice','-n','10','mpirun','--bind-to','core','-np','2',str(root/'scripts/mpi_rank_time.sh'),str(prefix),str(exe)]
 with log.open('w') as f:subprocess.run(argv,cwd=s.work,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
 row={'name':name,'wall_seconds':time.perf_counter()-t,'launched_argv':argv,'mode':mode,'rank_max_rss_kib':[]}
 for rank in range(2):
  f=ev/f'{name}.rank{rank}.time.txt'
  for line in f.read_text().splitlines():
   if 'Maximum resident set size' in line:row['rank_max_rss_kib'].append(int(line.split(':')[-1]))
 rows.append(row)
s.prepare(np.zeros(4));timed('perf_native_forward',root/'native-baseline-mpi-xspecfem2D')
timed('perf_extension_nohistory',root/'vendor/specfem2d-mpi/bin/xspecfem2D')
timed('perf_extension_record',root/'vendor/specfem2d-mpi/bin/xspecfem2D','record')
d=s.traces();target=np.load(ev/'target.npy');leaf=torch.tensor(d,dtype=torch.float64,requires_grad=True)
t=time.perf_counter();v=objective(leaf,torch.tensor(target,dtype=torch.float64),'processed');cot,=torch.autograd.grad(v,leaf)
rows.append({'name':'data_objective_vjp','wall_seconds':time.perf_counter()-t,'torch_dtype':'float64'})
cot.detach().numpy().T.copy().tofile(s.work/'DATA/ds_cotangent.bin')
timed('perf_native_adjoint',root/'vendor/specfem2d-mpi/bin/xspecfem2D','adjoint')
start=time.perf_counter();result=s.value_gradient(np.zeros(4),target,'processed')
rows.append({'name':'complete_processed_gradient','wall_seconds':time.perf_counter()-start,'scope':'mesh + native MPI full-history forward + data torch objective/VJP + source file + native MPI adjoint + kernel read/material VJP; excludes build/imports'})
history=[{'file':p.name,'bytes':p.stat().st_size} for p in sorted((s.work/'OUTPUT_FILES').glob('*history.bin'))]
report={'case':args.case or 'examples/sh_two_layer','observations':rows,'history':history,'history_bytes_sum':sum(p['bytes'] for p in history),'ranks':2,'threads_per_rank':1,'gpu':'not used','rss_semantics':'independent per-rank process RSS peaks; do not add peaks and call it simultaneous RSS','contention':'shared host, no exclusive reservation; startup included; no speedup claim'}
(ev/'performance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
