#!/usr/bin/env python3
"""Independent-shot native P-SV FWI, one global Vp update per iteration.

SPDX-License-Identifier: GPL-3.0-or-later
Only observations generation reads unknown target Vp. The optimizer's input
file physically contains no target Vp. Native kernels and history format are
unchanged; independent shots run serially with one temporary history at a time.
"""
from pathlib import Path
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diffspecfem.psv import PSVSolver
from scripts.experiment_gradient_detail_psv import digest,set_parameter
from scripts.experiment_vponly_psv import array_hash,materials


class CandidateRejected(ValueError):
    """A physically inadmissible candidate, not a provenance failure."""


def jsave(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def nsave(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    os.replace(temporary,path)


def read(path):return json.loads(Path(path).read_text())


def packet_check(packet,expected):
    manifest=read(Path(packet)/'manifest.json');rows=[]
    for row in manifest['files']:
        p=Path(packet)/'skill'/row['path']
        rows.append(dict(path=row['path'],bytes=p.stat().st_size,sha256=digest(p)))
    actual=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if actual!=expected or rows!=manifest['files'] or actual!=manifest['packet_sha256']:
        raise ValueError('Frozen packet contents do not match the independently pinned manifest')
    return dict(passed=True,version=manifest['version'],packet_sha256=actual,files=rows)


def parameters(path):
    text=Path(path).read_text()
    def number(key):
        value=re.search(r'(?m)^'+key+r'\s*=\s*([0-9.eEdD+-]+)',text)[1]
        return float(value.lower().replace('d','e'))
    return text,number


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name):raise ValueError('Unsafe namespace')
    destination=ROOT/'data/inversion'/args.name
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    unit=args.unit
    source=ROOT/'data/prepared'/('negative_lambda_20261107-vponly.npz' if unit else 'marmousi2_20261107-vponly.npz')
    with np.load(source,allow_pickle=False) as f:
        x,z=f['x_native'],f['z_native']
        if unit:
            ix=np.arange(x.size);iz=np.arange(z.size);origin_x=0.;bottom=2000.;water=400.
        else:
            ix=np.flatnonzero((x>=3000)&(x<=5500));iz=np.flatnonzero(3465-z<=2025)
            origin_x=float(x[ix[0]]+5000);bottom=2025.;water=450.
        xnew=x[ix]-x[ix[0]];znew=z[iz]-z[iz[0]]
        vp0=f['vp_initial'][np.ix_(iz,ix)];vpt=f['vp_true'][np.ix_(iz,ix)]
        vs=f['vs_true'][np.ix_(iz,ix)];rho=f['rho_true'][np.ix_(iz,ix)]
    # The training file deliberately has no target Vp key or target-model path.
    nsave(destination/'training_input.npz',vp_initial=vp0,vs_fixed=vs,rho_fixed=rho,x_native=xnew,z_native=znew,active_mask=np.ones_like(vp0,dtype=np.uint8))
    nsave(destination/'truth_for_observations_only.npz',vp=vpt,vs=vs,rho=rho,x_native=xnew,z_native=znew)
    width=float(xnew[-1]);shots=[350.,1000.,1650.] if unit else [250.,750.,1237.5,1725.,2225.]
    profiles=[dict(id='unit',f0=1.5,nx=8,ns=8,nw=2,dt=.001,nt=2400,epochs=2)] if unit else [
        dict(id='coarse',f0=2.,nx=56,ns=36,nw=10,dt=.0005,nt=5000,epochs=8),
        dict(id='fine',f0=4.,nx=112,ns=72,nw=20,dt=.00025,nt=10000,epochs=32)]
    parent=ROOT/'examples'/('psv_negative_lambda_20261107-vponly' if unit else 'psv_gradient_detail_water2')
    for stage in profiles:
        nx,ns,nw=stage['nx'],stage['ns'],stage['nw']
        stage['forecast_history_gb']=6*(4*nx+1)*(4*(ns+nw)+1)*stage['nt']*8*1.01/1.e9
        for sid,sx in enumerate(shots):
            case=destination/'cases'/stage['id']/f'shot_{sid:03d}'
            (case/'DATA').mkdir(parents=True)
            for filename in ['Par_file','SOURCE','STATIONS','interfaces_elastic_analytic.dat']:
                shutil.copyfile(parent/'DATA'/filename,case/'DATA'/filename)
            p=case/'DATA/Par_file';text=p.read_text()
            for key,value in [('xmax',width),('nx',nx),('NSTEP',stage['nt']),('DT',stage['dt']),('USER_T0',1.2/stage['f0']),('NSOURCES',1),('SAVE_MODEL','default')]:
                text=set_parameter(text,key,value)
            text,n1=re.subn(r'(?m)^1\s+\d+\s+1\s+\d+\s+2\s*$',f'1 {nx} 1 {ns} 2',text)
            text,n2=re.subn(r'(?m)^1\s+\d+\s+\d+\s+\d+\s+1\s*$',f'1 {nx} {ns+1} {ns+nw} 1',text)
            assert n1==n2==1
            p.write_text(text)
            interface=bottom-water
            (case/'DATA/interfaces_elastic_analytic.dat').write_text(f'3\n2\n0 0\n{width} 0\n2\n0 {interface}\n{width} {interface}\n2\n0 {bottom}\n{width} {bottom}\n{ns}\n{nw}\n')
            p=case/'DATA/SOURCE';text=p.read_text()
            for key,value in [('xs',sx),('zs',bottom-150),('time_function_type',3),('f0',stage['f0']),('factor','1e8')]:text=set_parameter(text,key,value)
            p.write_text(text)
            if not unit:
                (case/'DATA/STATIONS').write_text(''.join(f'R{i+1:03d} XX {rx:.17g} {bottom-540:.17g} 0 0\n' for i,rx in enumerate(np.linspace(50,width-50,161))))
        stage['case_hashes']={str(sid):{name:digest(destination/'cases'/stage['id']/f'shot_{sid:03d}'/'DATA'/name) for name in ['Par_file','SOURCE','STATIONS','interfaces_elastic_analytic.dat']} for sid in range(len(shots))}
    protocol=dict(name=args.name,unit=unit,stages=profiles,shot_ids=list(range(len(shots))),shot_x_m=shots,
        original_x_origin_m=origin_x,width_m=width,bottom_depth_m=bottom,water_depth_m=water,
        source_depth_m=150.,receiver_depth_m=None if unit else 540.,
        unit_actual_receiver_depths_m=[650.,900.,650.,550.] if unit else None,
        fixed_known_fields=['vs_fixed','rho_fixed'],
        active_domain='All solid tomography nodes only; no acoustic-water Vp coordinate',
        water_material=dict(vp=1500.,vs=0.,rho=1000. if unit else 2450.),
        objective='J=mean_shots[0.5*mean_data(((native_displacement-observed)/fixed_observed_RMS_s)^2)]',
        normalization='Exactly one data-size factor per shot and one global number-of-shots factor; observed RMS fixed before inversion.',
        optimizer=dict(method='projected L-BFGS',memory=5,max_vp_step_m_s=60.,vp_upper_m_s=6000.,bulk_margin=1.01,
            max_backtracks=5,armijo_c1=1.e-4,gradient_relative_stop=1.e-3,stagnation_relative_loss=1.e-4,stagnation_step_m_s=.5,stagnation_epochs=3),
        batch_size=2,max_history_gb=160.,boundaries='Reflecting/free native ROI boundaries; this separate boxed ROI is not the old full domain or a continuum marine-field claim.',
        parent_model_sha256=digest(source),training_input_sha256=digest(destination/'training_input.npz'),
        truth_file_for_observations_only_sha256=digest(destination/'truth_for_observations_only.npz'),
        initial_materials=materials(np.stack([vp0,vs,rho])))
    jsave(destination/'preparation_diagnostics.json',dict(initial=materials(np.stack([vp0,vs,rho])),
        target_generation_only=materials(np.stack([vpt,vs,rho])),optimizer_reads_this_file=False))
    jsave(destination/'protocol.json',protocol)
    print(json.dumps(dict(prepared=str(destination),profiles=profiles),indent=2))


def guide(path):
    text=Path(path).read_text()
    f=re.search(r'Maximum suggested \(Ricker\) source frequency =\s*([0-9.Ee+-]+)',text)
    d=re.search(r'Maximum suggested time step[^=]*=\s*([0-9.Ee+-]+)',text)
    if not f or not d:raise ValueError('Native grid guidance missing')
    return dict(max_f0_hz=float(f[1]),max_dt_s=float(d[1]))


class Engine:
    def __init__(self,prepared,runroot):
        self.prepared=Path(prepared);self.root=Path(runroot);self.root.mkdir(parents=True,exist_ok=True)
        self.protocol=read(self.prepared/'protocol.json');self.shots=self.protocol['shot_ids']
        protocol_copy=self.root/'protocol_copy.json'
        if protocol_copy.exists() and digest(protocol_copy)!=digest(self.prepared/'protocol.json'):
            raise ValueError('Run protocol changed; start a new namespace for a different objective or budget')
        if not protocol_copy.exists():shutil.copyfile(self.prepared/'protocol.json',protocol_copy)
        if digest(self.prepared/'training_input.npz')!=self.protocol['training_input_sha256']:raise ValueError('Training input changed')
        with np.load(self.prepared/'training_input.npz',allow_pickle=False) as f:
            assert set(f.files)=={'vp_initial','vs_fixed','rho_fixed','x_native','z_native','active_mask'}
            self.initial=f['vp_initial'].copy();self.vs=f['vs_fixed'].copy();self.rho=f['rho_fixed'].copy();self.x=f['x_native'].copy();self.z=f['z_native'].copy()
            self.active=f['active_mask'].astype(bool);assert self.active.all() and np.min(self.vs)>0
        nsave(self.root/'visual_map.npz',x_native=self.x,z_native=self.z,active_mask=self.active,
              original_x_origin_m=np.array(self.protocol['original_x_origin_m']),bottom_depth_m=np.array(self.protocol['bottom_depth_m']),water_depth_m=np.array(self.protocol['water_depth_m']))
        self.lower=np.sqrt(4./3.)*self.vs*self.protocol['optimizer']['bulk_margin']
        self.upper=self.protocol['optimizer']['vp_upper_m_s'];self.events=[]
        self.ledger=self.root/'ledger.jsonl'
        if self.ledger.exists():self.events=[json.loads(line) for line in self.ledger.read_text().splitlines() if line.strip()]
        self.calls_path=self.root/'calls';self.calls_path.mkdir(exist_ok=True)
        signature=self.runtime_signature()
        signature_path=self.root/'runtime_signature.json'
        if signature_path.exists() and read(signature_path)!=signature:raise ValueError('Runtime/API/native binary signature changed')
        if not signature_path.exists():jsave(signature_path,signature)

    def runtime_signature(self):
        paths=[Path(__file__).resolve(),ROOT/'diffspecfem/psv.py',ROOT/'diffspecfem/tomography.py',
            ROOT/'scripts/experiment_gradient_detail_psv.py',ROOT/'scripts/experiment_vponly_psv.py',
            ROOT/'native/psv_adjoint.f90',ROOT/'native/psv_iterate_time.patch',
            ROOT/'vendor/specfem2d-psv/bin/xspecfem2D',ROOT/'vendor/specfem2d-psv/bin/xmeshfem2D',ROOT/'psv-baseline-xspecfem2D']
        return {str(p.relative_to(ROOT)):digest(p) for p in paths}

    def bind_targets(self,stage):
        if self.runtime_signature()!=read(self.root/'runtime_signature.json'):raise ValueError('Runtime signature changed during run')
        directory=self.prepared/'observations'/stage['id'];manifest=read(directory/'manifest.json')
        signature=dict(stage_id=stage['id'],manifest_sha256=digest(directory/'manifest.json'),files={})
        for sid in self.shots:
            self.case(stage,sid)
            path=directory/f'shot_{sid:03d}.npz';entry=manifest[str(sid)]
            if digest(path)!=entry['sha256'] or entry['source_sha256']!=stage['case_hashes'][str(sid)]['SOURCE']:
                raise ValueError('Target/source provenance changed')
            signature['files'][str(sid)]=entry
        frozen=self.root/'target_signatures'/(stage['id']+'.json')
        if frozen.exists() and read(frozen)!=signature:raise ValueError('Frozen stage observations changed')
        if not frozen.exists():jsave(frozen,signature)
        return digest(frozen)

    def fingerprint(self,vp):return array_hash(np.stack([vp,self.vs,self.rho]))

    def event(self,event,**values):
        row=dict(event=event,**values)
        # Idempotent log continuation accompanies verified immutable caches.
        if row in self.events:return
        if event in ('iteration_start','iteration_end','evaluation_start','evaluation_complete','evaluation_abort','accepted_update'):
            for old in self.events:
                if old['event']==event and old['iteration_id']==row['iteration_id'] and old.get('evaluation_id')==row.get('evaluation_id'):
                    raise ValueError('An immutable ledger event changed on resumption')
        self.events.append(row)
        with self.ledger.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())

    def case(self,stage,sid):
        case=self.prepared/'cases'/stage['id']/f'shot_{sid:03d}'
        for name,expected in stage['case_hashes'][str(sid)].items():
            if digest(case/'DATA'/name)!=expected:raise ValueError('Immutable source/case changed')
        return case

    def native(self,vp,stage,sid,gradient=False,baseline=False,label='evaluation'):
        import torch
        torch.set_num_threads(1)
        if self.runtime_signature()!=read(self.root/'runtime_signature.json'):raise ValueError('Runtime changed before native call')
        model=np.stack([vp,self.vs,self.rho]);mh=self.fingerprint(vp)
        if not materials(model)['stable']:raise CandidateRejected('Nonpositive isotropic material energy')
        solver=PSVSolver(self.x,self.z,self.case(stage,sid),root=ROOT,ranks=2,
            work=self.root/'run_native',evidence=self.calls_path,allow_negative_lambda=True)
        solver.prepare(model)
        if digest(solver.work/'DATA/SOURCE')!=stage['case_hashes'][str(sid)]['SOURCE']:raise AssertionError('Actual native source changed')
        cache=self.root/'guides'/stage['id']/(mh+'.json')
        if not cache.exists():
            p=solver.work/'DATA/Par_file';original=p.read_text();p.write_text(set_parameter(original,'NSTEP',10))
            try:
                solver._run([solver.vendor/'bin/xmeshfem2D'],label='diagnostic_mesh')
                solver._run([ROOT/'psv-baseline-xspecfem2D'],label='diagnostic_ten_steps')
                first=next((solver.work/'OUTPUT_FILES').glob('*.semd'))
                if np.loadtxt(first,ndmin=2).shape!=(10,2):raise AssertionError('Diagnostic was not exactly ten native steps')
            finally:p.write_text(original)
            measured=guide(solver.work/'diagnostic_ten_steps.log')
            valid=measured['max_f0_hz']>=stage['f0'] and measured['max_dt_s']>=stage['dt']
            jsave(cache,dict(model_sha256=mh,stage_id=stage['id'],source_id=sid,diagnostic_native_steps=10,passed=valid,**measured))
            if not valid:raise CandidateRejected('Actual native frequency/DT guidance rejects this candidate')
            for path in (solver.work/'OUTPUT_FILES').glob('*.semd'):path.unlink()
            solver._run([solver.vendor/'bin/xmeshfem2D'],label='mesh_after_diagnostic')
        if not read(cache)['passed']:raise CandidateRejected('Cached native guidance rejected this exact model/stage')
        binary=ROOT/'psv-baseline-xspecfem2D' if baseline else solver.vendor/'bin/xspecfem2D'
        solver._run([binary],mode='record' if gradient else None,label='forward')
        data=solver.traces();actual=guide(solver.work/'forward.log')
        if actual['max_f0_hz']<stage['f0'] or actual['max_dt_s']<stage['dt']:raise CandidateRejected('Native guide failed')
        result=dict(data=data,sample_time=solver.sample_time.copy(),model_sha256=mh,vp_sha256=array_hash(vp),native_calls=solver.calls)
        if gradient:
            obs=self.observation(stage,sid);leaf=torch.tensor(data,dtype=torch.float64,requires_grad=True)
            value=.5*torch.mean(((leaf-torch.as_tensor(obs['observed']))/float(obs['scale']))**2)
            cot,=torch.autograd.grad(value,leaf);cot=cot.detach().numpy()
            expected=(data-obs['observed'])/(data.size*float(obs['scale'])**2)
            denominator=float(np.linalg.norm(expected));error=float(np.linalg.norm(cot-expected))
            assert error<1.e-12*denominator if denominator else error==0
            histories=list((solver.work/'OUTPUT_FILES').glob('psv_*history.bin'));size=sum(p.stat().st_size for p in histories)
            if size>self.protocol['max_history_gb']*1.e9:raise RuntimeError('History cap exceeded')
            g,raw=solver.adjoint(cot)
            if not np.isfinite(g).all() or not np.isfinite(raw).all():raise FloatingPointError('Non-finite native gradient')
            fluid=raw[:,12]==0;water=self.protocol['water_material']
            fluid_ok=bool(np.allclose(raw[fluid,6],water['rho'],rtol=1.e-13) and np.allclose(raw[fluid,8],0.,atol=0.) and np.allclose(raw[fluid,7]/raw[fluid,6],water['vp']**2,rtol=1.e-13) and np.count_nonzero(raw[fluid,9:12])==0)
            if not fluid_ok:raise AssertionError('Fixed native acoustic-water material/gradient proof failed')
            cl=cr=bl=br=length=0.;edges=0
            for rank in range(2):
                prefix=f'psv_rank{rank:06d}_';directory=solver.work/'OUTPUT_FILES'
                nodes=np.loadtxt(directory/(prefix+'sampling_native_nodes.txt'),ndmin=2)
                recs=np.loadtxt(directory/(prefix+'sampling_native_receivers.txt'),ndmin=2)
                cl+=float(np.sum(recs[:,1:3]*recs[:,3:5]));cr+=float(np.sum(nodes[:,1:3]*nodes[:,3:5]/nodes[:,5,None]))
                cp=np.loadtxt(directory/(prefix+'coupling_probe.txt'),ndmin=2)
                bl+=float(np.sum(cp[:,1]*cp[:,2]/cp[:,7]));br+=float(np.sum(np.sum(cp[:,3:5]*cp[:,5:7],axis=1)/cp[:,8]))
                geometry=np.loadtxt(directory/(prefix+'coupling_geometry.txt'));edges+=int(geometry[0]);length-=float(geometry[1])
            probes=dict(sampling_relative_error=abs(cl-cr)/max(abs(cl),abs(cr)),coupling_relative_error=abs(bl-br)/max(abs(bl),abs(br)),native_edges=edges,interface_length_m=length)
            if probes['sampling_relative_error']>1.e-12 or probes['coupling_relative_error']>1.e-12 or edges!=stage['nx'] or not np.isclose(length,self.protocol['width_m'],atol=1.e-8,rtol=1.e-12):raise AssertionError('Native sampling/coupling/interface gate failed')
            result.update(loss=float(value.detach()),gradient=g[0],cotangent=cot,history_bytes=size,
                fixed_water_proof=dict(passed=fluid_ok,raw_fluid_rows=int(fluid.sum()),material=water,fluid_gradient_zero=True),
                native_probes=probes,
                temporal_AD=0,replay_count=0,data_leaf_grad_fn=str(leaf.grad_fn),objective_grad_fn=type(value.grad_fn).__name__,native_calls=solver.calls)
            # Unit-only native joint-cotangent proof, before deleting this shot's
            # history. Production FWI uses exactly one adjoint per shot.
            if label=='unit_joint_probe':
                gj,_=solver.adjoint(cot/len(self.shots));result['joint_gradient']=gj[0]
                result['raw']=raw
            if sid==self.shots[len(self.shots)//2] and re.search(r'_e(000|008|016|024|031)_gradient$',label):
                fields=dict(model_sha256=np.array(mh),source_id=np.array(sid),stage_id=np.array(stage['id']))
                for rank in range(2):
                    prefix=f'psv_rank{rank:06d}_';directory=solver.work/'OUTPUT_FILES'
                    path=directory/(prefix+'displacement_snapshots.bin');header=np.fromfile(path,dtype='<i4',count=5)
                    nspec,nt,ni,nj,nd=map(int,header);count_values=nspec*ni*nj*nd
                    dtype=np.dtype([('step','<i4'),('u','<f8',(count_values,))]);count=(path.stat().st_size-20)//dtype.itemsize
                    mm=np.memmap(path,dtype=dtype,mode='r',offset=20,shape=(count,));idx=np.unique([count//4,count//2,count-1])
                    fields[f'rank{rank}_steps']=np.array(mm['step'][idx]);fields[f'rank{rank}_u']=np.array(mm['u'][idx]).reshape(len(idx),nspec,nj,ni,nd)
                    geometry=np.loadtxt(directory/(prefix+'gll_geometry.txt'))
                    fields[f'rank{rank}_x']=geometry[:,3].reshape(nspec,nj,ni);fields[f'rank{rank}_z']=geometry[:,4].reshape(nspec,nj,ni);fields[f'rank{rank}_fluid']=geometry[:,5].reshape(nspec,nj,ni).astype(np.uint8)
                nsave(self.root/'fields'/(label+f'_source{sid}.npz'),**fields)
            for p in histories:p.unlink()
        if self.fingerprint(vp)!=mh:raise AssertionError('Anchor changed during native evaluation')
        logdir=self.root/'native_logs'/label/stage['id']/f'shot_{sid:03d}'
        logdir.mkdir(parents=True,exist_ok=True)
        for path in solver.work.glob('*.log'):shutil.copyfile(path,logdir/path.name)
        for path in solver.work.glob('*.resources.txt'):shutil.copyfile(path,logdir/path.name)
        return result

    def observation(self,stage,sid):
        cache=self.prepared/'observations'/stage['id']/f'shot_{sid:03d}.npz'
        manifest=read(self.prepared/'observations'/stage['id']/'manifest.json')
        if digest(cache)!=manifest[str(sid)]['sha256']:raise ValueError('Fixed observations changed')
        with np.load(cache,allow_pickle=False) as f:return {key:f[key].copy() for key in f.files}

    def observe(self):
        # This separate method is the only engine operation reading unknown Vp.
        with np.load(self.prepared/'truth_for_observations_only.npz',allow_pickle=False) as f:
            truth=f['vp'].copy();assert np.array_equal(f['vs'],self.vs) and np.array_equal(f['rho'],self.rho)
        for stage in self.protocol['stages']:
            directory=self.prepared/'observations'/stage['id'];directory.mkdir(parents=True,exist_ok=True)
            if (directory/'manifest.json').exists():self.bind_targets(stage);continue
            manifest={}
            for sid in self.shots:
                path=directory/f'shot_{sid:03d}.npz'
                if not path.exists():
                    out=self.native(truth,stage,sid,label='observation')
                    scale=float(np.sqrt(np.mean(out['data']**2)));assert scale>0
                    nsave(path,observed=out['data'],sample_time=out['sample_time'],scale=np.array(scale))
                manifest[str(sid)]=dict(sha256=digest(path),source_sha256=stage['case_hashes'][str(sid)]['SOURCE'])
            jsave(directory/'manifest.json',manifest)
            self.bind_targets(stage)

    def batch(self,vp,stage,directory,iteration_id,evaluation_id,gradient=False,stop_after=None,label=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);mh=self.fingerprint(vp)
        target_signature=self.bind_targets(stage)
        runtime_signature=digest(self.root/'runtime_signature.json')
        self.event('evaluation_start',iteration_id=iteration_id,evaluation_id=evaluation_id,phase='gradient' if gradient else 'trial',model_sha256=mh,objective_id=stage['id'])
        total=0.;g=np.zeros_like(vp);done=0
        for start in range(0,len(self.shots),self.protocol['batch_size']):
            group=self.shots[start:start+self.protocol['batch_size']]
            for sid in group:
                path=directory/f'shot_{sid:03d}.npz';receipt=directory/f'shot_{sid:03d}.json'
                if receipt.exists():
                    meta=read(receipt)
                    if meta['model_sha256']!=mh or meta['stage_id']!=stage['id'] or meta['array_sha256']!=digest(path):raise ValueError('Stale or damaged shot cache')
                    self.observation(stage,sid);self.case(stage,sid)
                    if meta['target_signature_sha256']!=target_signature or meta['runtime_signature_sha256']!=runtime_signature or meta['source_sha256']!=stage['case_hashes'][str(sid)]['SOURCE'] or meta['gradient_requested']!=gradient:
                        raise ValueError('Shot cache objective/source/runtime signature mismatch')
                    expected_target=read(self.prepared/'observations'/stage['id']/'manifest.json')[str(sid)]['sha256']
                    if meta['target_sha256']!=expected_target or meta['vs_fixed_sha256']!=array_hash(self.vs) or meta['rho_fixed_sha256']!=array_hash(self.rho):raise ValueError('Cached target or fixed fields changed')
                    with np.load(path,allow_pickle=False) as f:result={key:f[key].copy() for key in f.files}
                else:
                    out=self.native(vp,stage,sid,gradient=gradient,label=label or evaluation_id)
                    obs=self.observation(stage,sid)
                    value=out['loss'] if gradient else float(.5*np.mean(((out['data']-obs['observed'])/float(obs['scale']))**2))
                    result=dict(loss=np.array(value),data=out['data'],sample_time=out['sample_time'])
                    if gradient:result.update(gradient=out['gradient'],cotangent=out['cotangent'])
                    if 'joint_gradient' in out:result.update(joint_gradient=out['joint_gradient'],raw=out['raw'])
                    nsave(path,**result)
                    meta=dict(shot_id=sid,stage_id=stage['id'],model_sha256=mh,vp_sha256=array_hash(vp),
                        target_signature_sha256=target_signature,runtime_signature_sha256=runtime_signature,gradient_requested=gradient,
                        source_sha256=stage['case_hashes'][str(sid)]['SOURCE'],target_sha256=read(self.prepared/'observations'/stage['id']/'manifest.json')[str(sid)]['sha256'],
                        array_sha256=digest(path),loss=float(value),native_calls=out['native_calls'],history_bytes=out.get('history_bytes',0),
                        fixed_water_proof=out.get('fixed_water_proof'),
                        native_probes=out.get('native_probes'),
                        vs_fixed_sha256=array_hash(self.vs),rho_fixed_sha256=array_hash(self.rho))
                    jsave(receipt,meta)
                total+=float(result['loss'])
                if gradient:g+=result['gradient']
                done+=1
                if stop_after is not None and done==stop_after:raise InterruptedError('Intentional unit partial-accumulation resume check')
            self.event('shot_batch',iteration_id=iteration_id,evaluation_id=evaluation_id,shot_ids=group,model_sha256=mh)
        self.event('evaluation_complete',iteration_id=iteration_id,evaluation_id=evaluation_id,model_sha256=mh,normalizer=len(self.shots))
        nsave(directory/'aggregate.npz',loss=np.array(total/len(self.shots)),loss_sum=np.array(total),gradient_sum=g,gradient=g/len(self.shots),vp=vp)
        return total/len(self.shots),g/len(self.shots)


def lbfgs_direction(g,s_history,y_history):
    q=g.copy();alphas=[];rhos=[]
    for s,y in zip(s_history[::-1],y_history[::-1]):
        rho=1/float(np.vdot(s,y));alpha=rho*float(np.vdot(s,q));q-=alpha*y;alphas.append(alpha);rhos.append(rho)
    gamma=float(np.vdot(s_history[-1],y_history[-1])/np.vdot(y_history[-1],y_history[-1])) if s_history else 1.
    out=gamma*q
    for s,y,alpha,rho in zip(s_history,y_history,alphas[::-1],rhos[::-1]):out+=s*(alpha-rho*float(np.vdot(y,out)))
    return -out


def save_state(engine,state,vp,prev_vp,prev_g,s_history,y_history,commit=True):
    version=int(state.get('version',-1))+1;path=engine.root/'states'/f'state_{version:05d}.npz'
    shape=vp.shape
    nsave(path,vp=vp,prev_vp=prev_vp,prev_g=prev_g,s=np.stack(s_history) if s_history else np.empty((0,*shape)),y=np.stack(y_history) if y_history else np.empty((0,*shape)))
    state.update(version=version,array_path=str(path.relative_to(engine.root)),array_sha256=digest(path),model_sha256=engine.fingerprint(vp))
    state['runtime_signature_sha256']=digest(engine.root/'runtime_signature.json')
    state['protocol_sha256']=digest(engine.prepared/'protocol.json')
    if commit:jsave(engine.root/'state.json',state)


def optimize(engine,limit=None):
    root=engine.root;cfg=engine.protocol['optimizer'];state_path=root/'state.json'
    if state_path.exists():
        state=read(state_path);path=root/state['array_path'];assert digest(path)==state['array_sha256']
        if state['runtime_signature_sha256']!=digest(root/'runtime_signature.json') or state['protocol_sha256']!=digest(engine.prepared/'protocol.json'):raise ValueError('Checkpoint runtime/objective protocol changed')
        with np.load(path,allow_pickle=False) as f:
            vp=f['vp'].copy();prev_vp=f['prev_vp'].copy();prev_g=f['prev_g'].copy();ss=list(f['s'].copy());ys=list(f['y'].copy())
    else:
        vp=engine.initial.copy();prev_vp=vp.copy();prev_g=np.zeros_like(vp);ss=[];ys=[]
        state=dict(stage_index=0,epoch_index=0,updates=0,status='running',initial_gradient_norm=None,recent=[])
        save_state(engine,state,vp,prev_vp,prev_g,ss,ys)
    while state['stage_index']<len(engine.protocol['stages']):
        stage=engine.protocol['stages'][state['stage_index']];eid=f"{stage['id']}_e{state['epoch_index']:03d}";directory=root/'epochs'/eid
        if (directory/'accepted.json').exists():
            # Recovery after durable accepted output but before global pointer.
            accepted=read(directory/'accepted.json');next_state=accepted['next_state']
            if state['model_sha256']==accepted['model_before_sha256']:
                post=root/next_state['array_path'];assert digest(post)==next_state['array_sha256']
                with np.load(post,allow_pickle=False) as f:vp=f['vp'].copy();prev_vp=f['prev_vp'].copy();prev_g=f['prev_g'].copy();ss=list(f['s'].copy());ys=list(f['y'].copy())
                engine.event('accepted_update',iteration_id=eid,gradient_evaluation_id=eid+'_gradient',candidate_evaluation_id=accepted['candidate_evaluation_id'],model_before_sha256=accepted['model_before_sha256'],model_after_sha256=accepted['model_after_sha256'])
                engine.event('iteration_end',iteration_id=eid,status='accepted')
                state=next_state;jsave(state_path,state);continue
        mh=engine.fingerprint(vp)
        engine.event('iteration_start',iteration_id=eid,shot_ids=engine.shots,model_sha256=mh,reduction='mean',objective_id=stage['id'])
        value,g=engine.batch(vp,stage,directory/'gradient',eid,eid+'_gradient',gradient=True)
        gnorm=float(np.linalg.norm(g))
        if state['initial_gradient_norm'] is None:state['initial_gradient_norm']=gnorm
        projected=np.where(((vp<=engine.lower)&(g>0))|((vp>=engine.upper)&(g<0)),0.,g)
        stationary=np.linalg.norm(projected)<=cfg['gradient_relative_stop']*max(state['initial_gradient_norm'],np.finfo(float).tiny)
        if stationary:
            engine.event('iteration_end',iteration_id=eid,status='converged')
            if state['stage_index']+1==len(engine.protocol['stages']):
                state['status']='data_gradient_stationarity';save_state(engine,state,vp,prev_vp,prev_g,ss,ys);return state
            state.update(stage_index=state['stage_index']+1,epoch_index=0,initial_gradient_norm=None,recent=[])
            save_state(engine,state,vp,vp,np.zeros_like(g),[],[]);ss=[];ys=[];continue
        if state['epoch_index'] and float(np.vdot(vp-prev_vp,g-prev_g))>1.e-10*np.linalg.norm(vp-prev_vp)*np.linalg.norm(g-prev_g):
            ss.append(vp-prev_vp);ys.append(g-prev_g);ss=ss[-cfg['memory']:];ys=ys[-cfg['memory']:]
        direction=lbfgs_direction(g,ss,ys)
        direction=np.where(((vp<=engine.lower)&(direction<0))|((vp>=engine.upper)&(direction>0)),0,direction)
        if float(np.vdot(g,direction))>=0:direction=-g;ss=[];ys=[]
        scale=float(np.max(abs(direction)))
        if scale==0:raise RuntimeError('Zero feasible optimizer direction')
        direction*=cfg['max_vp_step_m_s']/scale
        nsave(directory/'direction.npz',vp_anchor=vp,raw_gradient=g,direction=direction)
        accepted=None
        for trial in range(cfg['max_backtracks']):
            alpha=.5**trial;candidate=np.clip(vp+alpha*direction,engine.lower,engine.upper);delta=candidate-vp
            slope=float(np.vdot(g,delta));tid=eid+f'_trial{trial:02d}'
            if slope>=0:continue
            try:new_loss,_=engine.batch(candidate,stage,directory/f'trial_{trial:02d}',eid,tid)
            except CandidateRejected as exc:
                engine.event('evaluation_abort',iteration_id=eid,evaluation_id=tid,reason=str(exc));continue
            decision=dict(alpha=alpha,loss=new_loss,armijo_rhs=value+cfg['armijo_c1']*slope,step_linf_m_s=float(np.max(abs(delta))),passed=bool(new_loss<=value+cfg['armijo_c1']*slope))
            jsave(directory/f'trial_{trial:02d}'/'decision.json',decision)
            if decision['passed']:accepted=(candidate,new_loss,tid,decision);break
        if accepted is None:
            engine.event('iteration_end',iteration_id=eid,status='rejected');state['status']='line_search_stalled'
            if state['stage_index']+1==len(engine.protocol['stages']):jsave(state_path,state);return state
            state.update(stage_index=state['stage_index']+1,epoch_index=0,initial_gradient_norm=None,recent=[],status='running');ss=[];ys=[]
            save_state(engine,state,vp,vp,np.zeros_like(g),ss,ys);continue
        candidate,new_loss,tid,decision=accepted
        relative=(value-new_loss)/max(abs(value),np.finfo(float).tiny)
        state['recent']=(state['recent']+[dict(relative_loss_decrease=relative,step_linf_m_s=decision['step_linf_m_s'])])[-cfg['stagnation_epochs']:]
        stop=len(state['recent'])==cfg['stagnation_epochs'] and all(r['relative_loss_decrease']<cfg['stagnation_relative_loss'] and r['step_linf_m_s']<cfg['stagnation_step_m_s'] for r in state['recent'])
        stop=stop or gnorm/max(state['initial_gradient_norm'],np.finfo(float).tiny)<cfg['gradient_relative_stop']
        old_vp=vp.copy();vp=candidate;prev_vp=old_vp;prev_g=g;state['updates']+=1;state['epoch_index']+=1
        change_stage=stop or state['epoch_index']>=stage['epochs']
        if change_stage:
            state.update(stage_index=state['stage_index']+1,epoch_index=0,initial_gradient_norm=None,recent=[]);ss=[];ys=[]
        if state['stage_index']==len(engine.protocol['stages']):state['status']='data_stagnation' if stop else 'epoch_cap'
        # Immutable new checkpoint precedes the pointer and the accepted event.
        previous_pointer=read(state_path)
        save_state(engine,state,vp,prev_vp,prev_g,ss,ys,commit=False)
        jsave(directory/'accepted.json',dict(iteration_id=eid,candidate_evaluation_id=tid,model_before_sha256=mh,model_after_sha256=engine.fingerprint(vp),loss_before=value,loss_after=new_loss,decision=decision,next_state=state,previous_pointer=previous_pointer))
        engine.event('accepted_update',iteration_id=eid,gradient_evaluation_id=eid+'_gradient',candidate_evaluation_id=tid,model_before_sha256=mh,model_after_sha256=engine.fingerprint(vp))
        engine.event('iteration_end',iteration_id=eid,status='accepted')
        jsave(state_path,state)
        if limit and state['updates']>=limit:return state
    return state


def unit_check(engine):
    import torch
    engine.observe();stage=engine.protocol['stages'][0];vp=engine.initial.copy();iid='unit_sum_check';mh=engine.fingerprint(vp)
    engine.event('iteration_start',iteration_id=iid,shot_ids=engine.shots,model_sha256=mh,reduction='mean',objective_id=stage['id'])
    directory=engine.root/'unit_gradient'
    try:engine.batch(vp,stage,directory,iid,'unit_gradient',gradient=True,stop_after=1,label='unit_joint_probe')
    except InterruptedError:pass
    assert not (engine.root/'state.json').exists()
    engine=Engine(engine.prepared,engine.root)
    value,g=engine.batch(vp,stage,directory,iid,'unit_gradient',gradient=True,label='unit_joint_probe')
    results=[];observed=[];scales=[];joint=np.zeros_like(vp);baseline_bits=[]
    for sid in engine.shots:
        with np.load(directory/f'shot_{sid:03d}.npz',allow_pickle=False) as f:
            results.append(f['data'].copy());joint+=f['joint_gradient']
        obs=engine.observation(stage,sid);observed.append(obs['observed']);scales.append(float(obs['scale']))
        baseline=engine.native(vp,stage,sid,baseline=True,label='unit_original_baseline')['data']
        baseline_bits.append(bool(np.array_equal(baseline.view(np.uint64),results[-1].view(np.uint64))))
    data=np.stack(results);obs=np.stack(observed);scale=np.array(scales)[:,None,None,None]
    leaf=torch.tensor(data,dtype=torch.float64,requires_grad=True);loss=.5*torch.mean(((leaf-torch.tensor(obs))/torch.tensor(scale))**2)
    cot,=torch.autograd.grad(loss,leaf);cot=cot.detach().numpy()
    expected=(data-obs)/(data.size*scale**2)
    assert np.linalg.norm(cot-expected)/np.linalg.norm(expected)<1.e-12
    grad_error=float(np.linalg.norm(g-joint)/np.linalg.norm(g));assert grad_error<1.e-12
    xx,zz=np.meshgrid(engine.x,engine.z);p=vp*(.45+.2*np.cos(xx/max(engine.x[-1],1)*2*np.pi)*np.sin(zz/max(engine.z[-1],1)*np.pi))
    rhs=float(np.vdot(p,g));checks=[]
    for h in [.0003,.0001]:
        plus=[];minus=[]
        for sid in engine.shots:
            plus.append(engine.native(vp+h*p,stage,sid,label=f'unit_fd_plus_{h}')['data']);minus.append(engine.native(vp-h*p,stage,sid,label=f'unit_fd_minus_{h}')['data'])
        lhs=float(np.vdot((np.stack(plus)-np.stack(minus))/(2*h),cot));relative=abs(lhs-rhs)/max(abs(lhs),abs(rhs))
        checks.append(dict(h=h,lhs=lhs,rhs=rhs,relative_error=relative,passed=relative<1.e-5))
    engine.event('iteration_end',iteration_id=iid,status='converged')
    if not all(r['passed'] for r in checks):raise AssertionError(checks)
    # A separate ledger preserves the derivative-check iteration's model chain.
    engine.ledger=engine.root/'optimizer_ledger.jsonl';engine.events=[]
    state=optimize(engine,limit=1)
    assert all(baseline_bits)
    report=dict(passed=True,shot_ids=engine.shots,batch_sizes=[2,1],normalization=engine.protocol['normalization'],joint_AD_loss=float(loss.detach()),mean_separate_loss=value,joint_cotangent_relative_l2=float(np.linalg.norm(cot-expected)/np.linalg.norm(expected)),native_global_cot_vs_mean_independent_adjoints_relative_l2=grad_error,full_batch_model_dot=checks,resumed_partial_gradient=True,updates_before_all_shots=0,updates_after_all_shots=state['updates'],training_has_target_vp=False,original_native_baseline_bits_equal=baseline_bits)
    jsave(engine.root/'unit_validation.json',report);print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','observe','unit-check','invert'])
    parser.add_argument('--name',required=True);parser.add_argument('--unit',action='store_true');parser.add_argument('--limit-updates',type=int)
    parser.add_argument('--packet',type=Path);parser.add_argument('--packet-sha')
    args=parser.parse_args()
    if args.command=='prepare':return prepare(args)
    prepared=ROOT/'data/inversion'/args.name;runroot=ROOT/('run_'+args.name)
    runroot.mkdir(exist_ok=True)
    with (runroot/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=runroot/'driver_executed.py'
        if snapshot.exists() and digest(snapshot)!=digest(__file__):raise ValueError('Current source differs from this run snapshot; use a new namespace')
        if not snapshot.exists():shutil.copyfile(__file__,snapshot)
        if args.packet:jsave(runroot/'packet_pre.json',packet_check(args.packet,args.packet_sha))
        engine=Engine(prepared,runroot)
        if args.command=='observe':engine.observe()
        elif args.command=='unit-check':unit_check(engine)
        else:
            started=time.perf_counter();state=optimize(engine,args.limit_updates);print(json.dumps(dict(state=state,command_seconds=time.perf_counter()-started)),flush=True)
        if args.packet:jsave(runroot/'packet_post.json',packet_check(args.packet,args.packet_sha))


if __name__=='__main__':main()
