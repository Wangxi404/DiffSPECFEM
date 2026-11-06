#!/usr/bin/env python3
from pathlib import Path
import sys,json,shutil,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from diffspecfem import Solver,objective
from diffspecfem.numerics import check_adjoint,taylor_test
root=Path(__file__).resolve().parents[1];ev=root/'evidence/mpi2';ev.mkdir(exist_ok=True)
s=Solver(root,ranks=2);rng=np.random.default_rng(901);m=np.zeros(4)
tol=json.loads((root/'evidence/acceptance.json').read_text())
report={'ranks':2,'accepted_thresholds':tol,'gates':{},'passed':False,'status':'running'};t0=time.perf_counter()
(ev/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
def save():
 (ev/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
def archived(name):
 dst=ev/name
 if dst.exists():shutil.rmtree(dst)
 shutil.copytree(s.work,dst)
# Target is physical, separately run, heterogeneous perturbation of both families.
s.prepare(m)
s._run([root/'native-baseline-mpi-xspecfem2D'],label='native_baseline_mpi2')
native_reference=s.traces();np.save(ev/'native_reference.npy',native_reference);archived('native_baseline')
target=s.forward(np.array([.025,-.02,.035,-.03]));np.save(ev/'target.npy',target)
for kind in ['l2','processed']:
 result=s.value_gradient(m,target,kind);archived('gradient_'+kind)
 np.savez(ev/f'gradient_{kind}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
 report['gates'][kind]={'value':result['value'],'gradient':result['gradient'].tolist(),'seconds':result['seconds'],
 'data_grad_fn':result['data_grad_fn'],'objective_grad_fn':result['objective_grad_fn']}
 # Save matched original baseline comparison.
 if kind=='l2':
  ref=native_reference
  report['gates']['native_forward']={'passed':bool(np.array_equal(ref,result['data'])),'absolute_linf':float(np.max(np.abs(ref-result['data']))),'native_norm':float(np.linalg.norm(ref))}
  xs=[];ats=[];ys=[];axs=[];sampled=np.zeros_like(result['data']);histbytes=0
  for rank in range(2):
   prefix=f'ds_rank{rank:06d}_'
   nodes=np.loadtxt(s.work/'OUTPUT_FILES'/f'{prefix}sampling_native_nodes.txt',ndmin=2)
   recs=np.loadtxt(s.work/'OUTPUT_FILES'/f'{prefix}sampling_native_receivers.txt',ndmin=2)
   xs.append(nodes[:,1]/np.sqrt(nodes[:,3]));ats.append(nodes[:,2]/np.sqrt(nodes[:,3]))
   ys.append(recs[:,1]);axs.append(recs[:,2])
   history=s.work/'OUTPUT_FILES'/f'{prefix}history.bin';histbytes+=history.stat().st_size
   with history.open('rb') as f:
    hdr=np.fromfile(f,dtype='<i4',count=3);saved=np.fromfile(f,dtype='<f8').reshape(int(hdr[1]),2,int(hdr[0]))
   rows=np.loadtxt(s.work/'OUTPUT_FILES'/f'{prefix}sampling.txt',ndmin=2);C=np.zeros((4,int(hdr[0])))
   for r,k,w in rows:C[int(r)-1,int(k)-1]+=w
   sampled+=C@saved[:,0,:].T
  report['gates']['sampling_native']=check_adjoint(lambda x:np.concatenate(axs),lambda y:np.concatenate(ats),np.concatenate(xs),np.concatenate(ys),rtol=1e-12,atol=1e-12)
  report['gates']['sampling_native']['coordinates']='partitioned nodal Euclidean product weighted by inverse interface multiplicity; unique receiver samples'
  difference=sampled-result['data']
  report['gates']['sampling_native_history']={'passed':bool(np.allclose(sampled,result['data'],rtol=1e-12,atol=1e-14)),'absolute_linf':float(np.max(np.abs(difference))),'relative_l2':float(np.linalg.norm(difference)/np.linalg.norm(result['data'])),'history_bytes_sum':histbytes}
 # Data VJP is checked independently of native propagation.
 p=rng.normal(size=result['data'].shape);p=p/np.linalg.norm(p)*max(np.linalg.norm(result['data']),.001)
 evald=lambda d: objective(torch.tensor(d,dtype=torch.float64),torch.tensor(target,dtype=torch.float64),kind).item()
 report['gates'][kind]['data_vjp']=taylor_test(evald,result['data'],result['cotangent'],p,steps=[.01,.003,.001,.0003,.0001,.00003],atol=1e-10,rtol=1e-6)
 save()
 # Family-separated and combined tests retain full epsilon sweep and raw objectives.
 for name,p in [('rho',np.array([.7,-.4,0,0])),('vs',np.array([0,0,.6,-.8])),('mixed',np.array([.5,-.3,.7,-.4]))]:
  cache={tuple(m):result['value']}
  def evalm(x):
   key=tuple(x)
   if key not in cache:
    d=s.forward(x);cache[key]=objective(torch.tensor(d,dtype=torch.float64),torch.tensor(target,dtype=torch.float64),kind).item()
   return cache[key]
  report['gates'][kind]['taylor_'+name]=taylor_test(evalm,m,result['gradient'],p,steps=tol['model_taylor']['steps'],atol=1e-9,rtol=1e-5)
  save()
 # Native propagation/model dot product, JVP supplied only by check-only two native perturbations.
 p=np.array([.5,-.3,.7,-.4]);h=1e-5
 jvp=(s.forward(m+h*p)-s.forward(m-h*p))/(2*h)
 report['gates'][kind]['model_dot']=check_adjoint(lambda x:jvp,lambda y:result['gradient'],p,result['cotangent'],rtol=1e-5,atol=1e-9)
 save()
report['total_seconds']=time.perf_counter()-t0
checks=[]
for key,x in report['gates'].items():
 if key in ['l2','processed']:checks += [v['passed'] for v in x.values() if isinstance(v,dict) and 'passed' in v]
 else:checks.append(x['passed'])
report['passed']=all(checks);report['status']='passed' if report['passed'] else 'failed';save();print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
