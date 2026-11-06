#!/usr/bin/env python3
"""Small matched CPU resource observations, after numerical acceptance.

No benchmark or checkpoint-capacity claim is inferred from this one small case.
"""
import sys,os,time,json,subprocess,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from diffspecfem import Solver,objective
root=Path(__file__).resolve().parents[1];ev=root/'evidence';s=Solver(root)
if not json.loads((ev/'validation.json').read_text())['passed']:raise SystemExit('correctness required first')
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
rows=[]
def timed(name,cmd,cwd,mode=None):
 e=env.copy();e.pop('DIFFSPECFEM_MODE',None)
 if mode:e['DIFFSPECFEM_MODE']=mode
 out=ev/(name+'.time.txt');log=ev/(name+'.log');t=time.perf_counter()
 with log.open('w') as f:
  subprocess.run(['/usr/bin/time','-v','-o',str(out),'nice','-n','10',*map(str,cmd)],cwd=cwd,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
 row={'name':name,'wall_seconds':time.perf_counter()-t,'resource_file':out.name,'mode':mode}
 for line in out.read_text().splitlines():
  if 'Maximum resident set size' in line:row['max_rss_kib']=int(line.split(':')[-1])
 rows.append(row)
s.prepare(np.zeros(4))
timed('perf_native_forward',[root/'native-baseline-xspecfem2D'],s.work)
timed('perf_extension_nohistory',[root/'vendor/specfem2d/bin/xspecfem2D'],s.work)
timed('perf_extension_record',[root/'vendor/specfem2d/bin/xspecfem2D'],s.work,'record')
d=s.traces();target=np.load(ev/'target.npy');leaf=torch.tensor(d,dtype=torch.float64,requires_grad=True)
t=time.perf_counter();v=objective(leaf,torch.tensor(target,dtype=torch.float64),'processed');cot,=torch.autograd.grad(v,leaf)
rows.append({'name':'data_objective_vjp','wall_seconds':time.perf_counter()-t,'torch_dtype':'float64'})
cot.detach().numpy().T.copy().tofile(s.work/'DATA/ds_cotangent.bin')
timed('perf_native_adjoint',[root/'vendor/specfem2d/bin/xspecfem2D'],s.work,'adjoint')
start=time.perf_counter();result=s.value_gradient(np.zeros(4),target,'processed')
rows.append({'name':'complete_processed_gradient','wall_seconds':time.perf_counter()-start,'scope':'mesh + native full-history forward + data torch objective/VJP + source file + native adjoint + raw kernel read/material VJP; excludes initial build and imports'})
report={'observations':rows,'history_bytes':(s.work/'OUTPUT_FILES/ds_history.bin').stat().st_size,'ranks':1,'threads':1,'gpu':'not used','contention':'shared host, no exclusive reservation; measurements are observations','no_capacity_or_causal_skill_effort_claim':True}
(ev/'performance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
