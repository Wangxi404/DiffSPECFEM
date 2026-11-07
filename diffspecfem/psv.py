"""Dense active-solid Vp/Vs/rho for native coupled acoustic / P-SV.

SPDX-License-Identifier: GPL-3.0-or-later

Water is a fixed native acoustic material, with Vs=0. Solid tomography has
increasing native x,z axes and uses the official bilinear interpolation plus
constant border extension. Physical model coordinates are (Vp,Vs,rho).
Native propagation and its exact discrete transpose run outside Torch AD.
"""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import torch

from .adapter import ROOT
from .tomography import BilinearMap, write_tomography


def psv_objective(data, target, kind='l2'):
    """Data-only differentiable objectives on [receiver,Cartesian component,time]."""
    if kind == 'l2':
        return .5*torch.sum((data-target)**2)
    if kind == 'processed':
        filt=torch.tensor([.2,-.1,.6,.4,-.1],dtype=data.dtype,device=data.device).view(1,1,5)
        def process(x):
            flat=x.reshape(-1,1,x.shape[-1])
            y=torch.nn.functional.conv1d(flat,filt,padding=2).reshape_as(x)
            y=torch.tanh(y/.02)
            return y+.17*torch.roll(y,1,0)+.11*torch.roll(y,1,1)
        weight=torch.linspace(.3,1.2,data.shape[-1],dtype=data.dtype,device=data.device)
        return .5*torch.sum(weight*(process(data)-process(target))**2)
    raise ValueError('Unsupported P-SV objective')


class PSVSolver:
    """One native forward + data AD VJP + one native coupled adjoint.

    No temporal AD, parameter finite differences, or checkpoint replay in the
    production path. The six native histories per time step are streamed to
    disk and read one frame at a time in reverse, with O(nglob) live states.
    All receivers are solid Cartesian displacement receivers; water material
    and its source terms are fixed, so unsupported active-water derivatives
    cannot be requested through these solid-grid coordinates.
    """
    def __init__(self,x,z,case,*,root=ROOT,ranks=2,work='run_psv',evidence='evidence/psv',par_file='Par_file'):
        if ranks not in (1,2):
            raise ValueError('Only one or two MPI ranks are supported')
        self.root=Path(root).resolve();self.case=(self.root/Path(case)).resolve()
        self.work=(self.root/Path(work)).resolve();self.evidence=self.root/Path(evidence)
        self.evidence.mkdir(parents=True,exist_ok=True)
        self.x,self.z=np.asarray(x,float),np.asarray(z,float)
        self.shape=(3,len(self.z),len(self.x));self.ranks=ranks
        self.vendor=self.root/'vendor/specfem2d-psv';self.calls=[]
        self.nstep=None;self.stations=[]
        self.par_file=par_file

    def _run(self,command,mode=None,label='native'):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env.pop('DIFFSPECFEM_MODE',None);env.pop('DIFFSPECFEM_PSV_MODE',None)
        if mode:env['DIFFSPECFEM_PSV_MODE']=mode
        launch=['nice','-n','10']
        if self.ranks>1 and 'specfem2D' in str(command[0]):
            launch+=['mpirun','--bind-to','core','-np',str(self.ranks)]
        argv=[*launch,*map(str,command)]
        log=self.work/f'{label}.log';start=time.perf_counter()
        with log.open('w') as f:
            p=subprocess.run(['/usr/bin/time','-v','-o',str(self.work/f'{label}.resources.txt'),*argv],cwd=self.work,env=env,stdout=f,stderr=subprocess.STDOUT)
        row=dict(launched_argv=argv,ranks=self.ranks,mode=mode,seconds=time.perf_counter()-start,
                 returncode=p.returncode,log=str(log),resource_log=str(self.work/f'{label}.resources.txt'))
        self.calls.append(row)
        with (self.evidence/'native_calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        content=log.read_text()
        if p.returncode or 'Error detected' in content or 'STOP ' in content:
            raise RuntimeError(content[-8000:])

    def prepare(self,model):
        model=np.asarray(model,float)
        if model.shape!=self.shape or not np.isfinite(model).all() or np.min(model)<=0:
            raise ValueError(f'Expected finite positive solid [Vp,Vs,rho] with shape {self.shape}; fixed water is outside these coordinates')
        if np.any(model[0]**2<=2*model[1]**2):
            raise ValueError('This isotropic case requires positive solid lambda (Vp²>2Vs²)')
        if self.work.exists():
            if self.work in (self.root,self.case) or not self.work.name.startswith('run_'):
                raise ValueError('Unsafe native run directory')
            shutil.rmtree(self.work)
        shutil.copytree(self.case,self.work);(self.work/'OUTPUT_FILES').mkdir(exist_ok=True)
        if self.par_file!='Par_file':shutil.copyfile(self.work/'DATA'/self.par_file,self.work/'DATA/Par_file')
        p=self.work/'DATA/Par_file';text=p.read_text()
        text=re.sub(r'(?m)^(NPROC\s*=)[^\n]*',rf'\g<1> {self.ranks}',text)
        self.nstep=int(re.search(r'(?m)^NSTEP\s*=\s*(\d+)',text)[1]);p.write_text(text)
        rows=[line.split() for line in (self.work/'DATA/STATIONS').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        self.stations=[(r[0],r[1]) for r in rows]
        if len(set(self.stations))!=len(self.stations):raise ValueError('Station/network identifiers must be unique')
        write_tomography(self.work/'DATA/tomo_file.xyz',self.x,self.z,*model)
        self._run([self.vendor/'bin/xmeshfem2D'],label='mesh')

    def traces(self):
        data=[];common=None
        for station,network in self.stations:
            components=[]
            for component in ['BXX','BXZ']:
                path=self.work/'OUTPUT_FILES'/f'{network}.{station}.{component}.semd'
                trace=np.loadtxt(path,ndmin=2)
                if trace.shape!=(self.nstep,2) or not np.isfinite(trace).all():raise RuntimeError(f'Invalid native trace {path}')
                if common is None:common=trace[:,0]
                elif not np.array_equal(common,trace[:,0]):raise RuntimeError('Native sample times differ')
                components.append(trace[:,1])
            data.append(components)
        self.sample_time=common
        return np.array(data)

    def forward(self,model,record=False,baseline=False):
        self.prepare(model)
        binary=self.root/'psv-baseline-xspecfem2D' if baseline else self.vendor/'bin/xspecfem2D'
        self._run([binary],mode='record' if record else None,label='native_baseline' if baseline else 'forward')
        return self.traces()

    def adjoint(self,cotangent):
        cotangent=np.asarray(cotangent,dtype='<f8')
        if cotangent.shape!=(len(self.stations),2,self.nstep):raise ValueError('Expected data cotangent [STATIONS,UX/UZ,NSTEP]')
        cotangent.transpose(2,1,0).copy().tofile(self.work/'DATA/psv_cotangent.bin')
        self._run([self.vendor/'bin/xspecfem2D'],mode='adjoint',label='adjoint')
        paths=sorted((self.work/'OUTPUT_FILES').glob('psv_*raw_kernels.txt'))
        raw=np.concatenate([np.loadtxt(path,ndmin=2) for path in paths])
        solid=raw[:,12]==1;rr=raw[solid]
        mapping=BilinearMap(self.x,self.z,rr[:,4],rr[:,5])
        rho,lam,mu=rr[:,6],rr[:,7],rr[:,8]
        vp=np.sqrt((lam+2*mu)/rho);vs=np.sqrt(mu/rho)
        gr,gl,gm=rr[:,9],rr[:,10],rr[:,11]
        gradient=np.stack([mapping.transpose(2*rho*vp*gl),
                           mapping.transpose(2*rho*vs*(gm-2*gl)),
                           mapping.transpose(gr+(vp**2-2*vs**2)*gl+vs**2*gm)])
        return gradient,raw

    def value_gradient(self,model,target,kind='l2'):
        start=time.perf_counter();data=self.forward(model,record=True)
        leaf=torch.tensor(data,dtype=torch.float64,requires_grad=True)
        value=psv_objective(leaf,torch.as_tensor(target,dtype=torch.float64),kind)
        cot,=torch.autograd.grad(value,leaf)
        gradient,raw=self.adjoint(cot.detach().numpy())
        return dict(value=value.item(),gradient=gradient,raw=raw,data=data,cotangent=cot.detach().numpy(),
                    sample_time=self.sample_time,seconds=time.perf_counter()-start,
                    data_grad_fn=str(leaf.grad_fn),objective_grad_fn=type(value.grad_fn).__name__,
                    production_route='one unchanged native coupled forward + data-only Torch AD VJP + exact native P-SV/acoustic transpose + static solid tomography transpose; temporal AD=0; parameter FD=0; replay=0')
