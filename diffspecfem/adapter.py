"""File-boundary PyTorch data VJP -> exact discrete native SH adjoint.

Production gradient calls 1 native forward and 1 native adjoint, no model FD.
Model coordinates: [logrho(bottom), logrho(top), logvs(bottom), logvs(top)]
relative to the provided nonuniform two-layer reference. VP held fixed.
"""
from pathlib import Path
import json, os, re, shutil, subprocess, time
import numpy as np
import torch

BASE_RHO=np.array([2500.,2800.]); BASE_VS=np.array([1600.,1900.])
ROOT=Path(__file__).resolve().parents[1]
class Solver:
    def __init__(self,root=ROOT,ranks=1,case=None):
        if ranks not in (1,2):raise ValueError('Only validated ranks=1 or ranks=2 supported')
        self.root=Path(root); self.ranks=ranks; self.work=self.root/('run' if ranks==1 else f'run_mpi{ranks}'); self.vendor=self.root/('vendor/specfem2d' if ranks==1 else 'vendor/specfem2d-mpi')
        self.case=self.root/'examples/sh_two_layer' if case is None else self.root/Path(case)
        if case is not None:self.work=self.root/f'run_mpi{ranks}_{self.case.name}'
        self.calls=[]
    def _run(self,command,mode=None,label='native'):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env.pop('DIFFSPECFEM_MODE',None)
        if mode:env['DIFFSPECFEM_MODE']=mode
        log=self.work/f'{label}.log';t=time.perf_counter()
        with log.open('w') as f:
            launch=['nice','-n','10']+(['mpirun','--bind-to','core','-np',str(self.ranks)] if self.ranks>1 and 'specfem2D' in str(command[0]) else [])
            p=subprocess.run([*launch,*map(str,command)],cwd=self.work,env=env,stdout=f,stderr=subprocess.STDOUT)
        row=dict(command=list(map(str,command)),launched_argv=[*launch,*map(str,command)],ranks=self.ranks,mode=mode,seconds=time.perf_counter()-t,returncode=p.returncode,log=str(log))
        self.calls.append(row)
        with (self.root/'evidence/native_calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if p.returncode or 'Error detected' in log.read_text() or 'STOP ' in log.read_text():
            raise RuntimeError(log.read_text()[-4000:])
    def prepare(self,m):
        m=np.asarray(m,dtype=float)
        if m.shape!=(4,):raise ValueError('Only four log material coordinates are supported')
        if self.work.exists():shutil.rmtree(self.work)
        shutil.copytree(self.case,self.work)
        (self.work/'OUTPUT_FILES').mkdir()
        p=self.work/'DATA/Par_file';s=p.read_text()
        s=re.sub(r'(?m)^(NPROC\s*=)[^\n]*',rf'\g<1> {self.ranks}',s)
        rho=BASE_RHO*np.exp(m[:2]);vs=BASE_VS*np.exp(m[2:])
        for i,(r,v,vp) in enumerate(zip(rho,vs,[3000.,3500.]),1):
            s=re.sub(rf'(?m)^{i} 1 .*$',f'{i} 1 {r:.17g} {vp:.17g} {v:.17g} 0 0 9999. 9999. 0 0 0 0 0 0',s)
        p.write_text(s)
        self._run([self.vendor/'bin/xmeshfem2D'],label='mesh')
    def traces(self):
        files=sorted((self.work/'OUTPUT_FILES').glob('*.semd'))
        if len(files)!=4:raise RuntimeError(f'Missing native traces: {files}')
        d=np.stack([np.loadtxt(p)[:,1] for p in files])
        if d.shape!=(4,800) or not np.isfinite(d).all():raise RuntimeError('invalid native data')
        return d
    def forward(self,m,record=False):
        self.prepare(m)
        self._run([self.vendor/'bin/xspecfem2D'],mode='record' if record else None,label='forward')
        return self.traces()
    def adjoint(self,cotangent):
        cot=np.asarray(cotangent,dtype='<f8')
        if cot.shape!=(4,800):raise ValueError('cotangent shape mismatch')
        cot.T.copy().tofile(self.work/'DATA/ds_cotangent.bin')
        self._run([self.vendor/'bin/xspecfem2D'],mode='adjoint',label='adjoint')
        paths=sorted((self.work/'OUTPUT_FILES').glob('ds_*raw_kernels.txt'))
        raw=np.concatenate([np.loadtxt(p,ndmin=2) for p in paths])
        # Raw rows: e,i,j,k,x,z,rho,mu,dJ/drho_GLL,dJ/dmu_GLL.
        gradient=np.zeros(4)
        for region in range(2):
            selected=raw[:,10]==region+1
            rr=raw[selected]
            gradient[region]=np.sum(rr[:,6]*rr[:,8]+rr[:,7]*rr[:,9])
            gradient[region+2]=np.sum(2*rr[:,7]*rr[:,9])
        return gradient,raw
    def value_gradient(self,m,target,kind='l2'):
        t=time.perf_counter();d=self.forward(m,record=True)
        leaf=torch.tensor(d,dtype=torch.float64,requires_grad=True)
        value=objective(leaf,torch.tensor(target,dtype=torch.float64),kind)
        cot,=torch.autograd.grad(value,leaf)
        g,raw=self.adjoint(cot.detach().numpy())
        return dict(value=value.item(),gradient=g,data=d,cotangent=cot.detach().numpy(),raw=raw,seconds=time.perf_counter()-t,
                    data_grad_fn=str(leaf.grad_fn),objective_grad_fn=type(value.grad_fn).__name__)

def objective(d,target,kind):
    if kind=='l2':return .5*torch.sum((d-target)**2)
    if kind=='processed':
        # Fixed FIR, nonlinearity, receiver mixing and time-dependent weights.
        filt=torch.tensor([.2,-.1,.6,.4,-.1],dtype=d.dtype,device=d.device).view(1,1,5)
        def process(x):
            y=torch.nn.functional.conv1d(x[:,None,:],filt,padding=2)[:,0,:]
            y=torch.tanh(y/0.02)
            return y+0.17*torch.roll(y,1,0)
        weight=torch.linspace(.3,1.2,d.shape[-1],dtype=d.dtype,device=d.device)
        return .5*torch.sum(weight*(process(d)-process(target))**2)
    raise ValueError('unsupported objective')
