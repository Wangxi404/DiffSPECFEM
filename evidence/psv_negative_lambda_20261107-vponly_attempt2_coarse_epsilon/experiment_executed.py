#!/usr/bin/env python3
"""Fresh native Vp-only experiments with true Vs/rho held exactly fixed.

SPDX-License-Identifier: GPL-3.0-or-later
No native numerical kernel is changed. Negative lambda is an explicit stable-
isotropic material opt-in; positive density, shear and bulk moduli are required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_gradient_detail_psv import digest, write_json, temporal_spectrum, gradient_statistics, set_parameter


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest()


def materials(model):
    vp, vs, rho = model
    mu = rho*vs**2
    lam = rho*(vp**2-2*vs**2)
    k2 = lam+mu
    bulk = lam+2*mu/3
    poisson = lam/(2*(lam+mu))
    return dict(finite=bool(np.isfinite(model).all()), rho_min=float(rho.min()),
        vp_min=float(vp.min()), vp_max=float(vp.max()), vs_min=float(vs.min()),
        vs_max=float(vs.max()), vp_vs_min_ratio=float((vp/vs).min()),
        mu_min_pa=float(mu.min()), lambda_min_pa=float(lam.min()),
        negative_lambda_count=int(np.count_nonzero(lam<0)), kappa2D_min_pa=float(k2.min()),
        bulk3D_min_pa=float(bulk.min()), poisson_min=float(poisson.min()),
        stable=bool(np.isfinite(model).all() and np.min(model)>0 and np.min(mu)>0 and np.min(bulk)>0),
        field_sha256={name:array_hash(value) for name,value in zip(['vp','vs','rho'],model)})


def verify_packet(packet):
    packet = Path(packet)
    manifest = json.loads((packet/'manifest.json').read_text())
    assert manifest['version']=='0.2.0'
    assert manifest['packet_sha256']=='51b19130b19625fb074f76db5165e6517d662a61079723e0381d62bfb96f947a'
    rows = []
    for row in manifest['files']:
        p = packet/'skill'/row['path']
        observed = dict(path=row['path'], bytes=p.stat().st_size, sha256=digest(p))
        assert observed['bytes']==row['bytes'] and observed['sha256']==row['sha256'], p
        rows.append(observed)
    # Match the frozen packet's official tools/freeze.py digest: this is
    # the canonical file-manifest hash, not the tar archive's byte hash.
    observed_packet=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert observed_packet==manifest['packet_sha256']
    return dict(passed=True, version=manifest['version'], packet_sha256=manifest['packet_sha256'],
                manifest_sha256=digest(packet/'manifest.json'), archive_tar_sha256=digest(packet/'skill.tar'), files=rows,
                packet_path=str(packet), packet_modified=False)


def names(unit):
    return 'psv_negative_lambda_20261107-vponly' if unit else 'psv_marmousi2_20261107-vponly'


def prepare(unit=False):
    name = names(unit)
    case = ROOT/'examples'/name
    path = ROOT/'data/prepared'/('negative_lambda_20261107-vponly.npz' if unit else 'marmousi2_20261107-vponly.npz')
    if case.exists() or path.exists():
        raise FileExistsError('Use a fresh namespace; never overwrite a model or case')
    parent = ROOT/'examples'/('psv_coupled_unit' if unit else 'psv_gradient_detail_water2')
    source = parent/'DATA/unit_model.npz' if unit else ROOT/'data/prepared/marmousi2_full_psv_solid_domain.npz'
    with np.load(source,allow_pickle=False) as f:
        data={key:f[key].copy() for key in f.files}
    if unit:
        xx,zz=np.meshgrid(data['x_native']/2000.,data['z_native']/1600.)
        data['vp_true']=1.28*data['vs_true']
        data['vp_initial']=1.2*data['vs_true']*(1+.015*np.cos(2*np.pi*xx)*np.sin(np.pi*zz))
    for key in ['vs','rho']:
        data[key+'_initial']=data[key+'_true'].copy()
    provenance=dict(name=name,parent_model_sha256=digest(source),model_role='synthetic negative-lambda coupled regression' if unit else 'Marmousi-II solid-domain input with original sigma150m Vp initial; true Vs/rho fixed',
        active_coordinates=['vp'],fixed_coordinates=['vs_true','rho_true','native acoustic water'],
        no_clipping_or_floors=True,water_density_convention='Fixed native water rho1000 in synthetic unit; rho2450 in Marmousi case (controlled convention, not original Marmousi density)',
        negative_lambda_policy='explicit opt-in, positive rho/mu/bulk3D; locally negative Poisson allowed')
    data['metadata_json']=np.array(json.dumps(provenance))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**data)
    (case/'DATA').mkdir(parents=True)
    for filename in ['Par_file','SOURCE','STATIONS','interfaces_elastic_analytic.dat']:
        src=parent/'DATA'/('Par_file_tomography' if unit and filename=='Par_file' else filename)
        shutil.copyfile(src,case/'DATA'/filename)
    par=case/'DATA/Par_file'
    par.write_text(set_parameter(par.read_text(),'SAVE_MODEL','ascii'))
    truth=np.stack([data[key+'_true'] for key in ['vp','vs','rho']])
    initial=np.stack([data[key+'_initial'] for key in ['vp','vs','rho']])
    assert materials(initial)['stable'] and materials(truth)['stable']
    record=dict(**provenance,model_path=str(path.relative_to(ROOT)),model_sha256=digest(path),
                initial=materials(initial),target=materials(truth),status='prepared; native acceptance pending',
                native_case='unchanged parent numerics/acquisition, SAVE_MODEL=ascii adds diagnostic GLL material export only')
    write_json(case/'CASE_PROVENANCE.json',record)
    return record


def data_checks(data,target):
    """Independent data-only standard/processed VJPs, with no wave solver AD."""
    import torch
    scale=float(np.sqrt(np.mean(target*target)))
    assert scale>0
    direction=scale*(.6+.3*np.sin(np.arange(data.size).reshape(data.shape)*.017))
    weight=np.linspace(.3,1.2,data.shape[-1])
    linear=lambda x:.7*x+.2*np.roll(x,1,-1)+.1*np.roll(x,-1,-1)
    result={}
    for kind in ['l2','processed_data_only']:
        leaf=torch.tensor(data,dtype=torch.float64,requires_grad=True)
        obs=torch.tensor(target,dtype=torch.float64)
        if kind=='l2':
            objective=.5*torch.sum((leaf-obs)**2)
            expected=data-target
            value=lambda x:float(.5*np.sum((x-target)**2))
        else:
            transform=lambda x:torch.tanh((.7*x+.2*torch.roll(x,1,-1)+.1*torch.roll(x,-1,-1))/scale)
            q,qt=transform(leaf),transform(obs)
            objective=.5*torch.sum(torch.tensor(weight)*(q-qt)**2)
            qn,qn_obs=np.tanh(linear(data)/scale),np.tanh(linear(target)/scale)
            gq=weight*(qn-qn_obs)*(1-qn*qn)/scale
            expected=.7*gq+.2*np.roll(gq,-1,-1)+.1*np.roll(gq,1,-1)
            value=lambda x:float(.5*np.sum(weight*(np.tanh(linear(x)/scale)-qn_obs)**2))
        cot,=torch.autograd.grad(objective,leaf)
        cot=cot.detach().numpy()
        error=float(np.linalg.norm(cot-expected)/np.linalg.norm(expected))
        rhs=float(np.vdot(cot,direction))
        centered=[]
        for h in [.0003,.0001]:
            lhs=(value(data+h*direction)-value(data-h*direction))/(2*h)
            rel=abs(lhs-rhs)/max(abs(lhs),abs(rhs))
            centered.append(dict(epsilon=h,lhs=lhs,rhs=rhs,relative_error=rel))
        result[kind]=dict(value=float(objective.detach()),numpy_vjp_relative_l2=error,
            centered_data_directions=centered,passed=error<1.e-12 and all(r['relative_error']<1.e-6 for r in centered),
            leaf_grad_fn=str(leaf.grad_fn),objective_grad_fn=type(objective.grad_fn).__name__)
    return result


def run(args):
    import torch
    torch.set_num_threads(1)
    from diffspecfem.psv import PSVSolver
    from diffspecfem.numerics import check_adjoint
    from diffspecfem.tomography import BilinearMap
    unit=args.unit;name=names(unit)
    ev=ROOT/'evidence'/name;case=ROOT/'examples'/name
    if ev.exists():raise FileExistsError('Use a fresh evidence namespace; retain all failed attempts')
    ev.mkdir();shutil.copyfile(__file__,ev/'experiment_executed.py')
    write_json(ev/'packet_pre_native.json',verify_packet(args.packet))
    provenance=json.loads((case/'CASE_PROVENANCE.json').read_text())
    modelpath=ROOT/provenance['model_path']
    with np.load(modelpath,allow_pickle=False) as f:
        x,z=f['x_native'],f['z_native']
        truth=np.stack([f[key+'_true'] for key in ['vp','vs','rho']])
        initial=np.stack([f[key+'_initial'] for key in ['vp','vs','rho']])
    assert np.array_equal(initial[1:],truth[1:])
    solver=PSVSolver(x,z,case,root=ROOT,ranks=2,work='run_'+name,evidence=ev,allow_negative_lambda=True)
    # This public experiment has only Vp coordinates. The existing native API
    # still computes three physical material derivatives for private diagnosis.
    def forward(vp,**options):
        model=np.stack([vp,truth[1],truth[2]])
        assert np.array_equal(model[1:],truth[1:])
        report['model_calls'].append(materials(model))
        return solver.forward(model,**options)
    par=(case/'DATA/Par_file').read_text()
    steps=int(re.search(r'(?m)^NSTEP\s*=\s*(\d+)',par)[1]);dt=float(re.search(r'(?m)^DT\s*=\s*([0-9.eE+-]+)',par)[1])
    frequency=float(re.search(r'(?m)^f0\s*=\s*([0-9.eE+-]+)',(case/'DATA/SOURCE').read_text())[1])
    nx=8 if unit else 192;nz=10 if unit else 84
    estimate=6*(4*nx+1)*(4*nz+1)*steps*8*1.01/1.e9
    if estimate>args.max_history_gb or shutil.disk_usage(ROOT).free<2*estimate*1.e9:
        raise RuntimeError('History budget or free disk gate rejected this run')
    dependencies=['diffspecfem/psv.py','diffspecfem/tomography.py','native/psv_adjoint.f90','native/psv_iterate_time.patch','scripts/experiment_gradient_detail_psv.py']
    report=dict(status='running',passed=False,case=str(case.relative_to(ROOT)),active_coordinates=['physical solid Vp (m/s)'],
        fixed_coordinates='Vs=true, rho=true at every model call; water Vp/rho fixed and Vs=0',
        experiment='negative-lambda synthetic control' if unit else 'new matched Marmousi-II Vp-only experiment',
        initial=materials(initial),target=materials(truth),model_sha256=digest(modelpath),model_calls=[],
        input_provenance=provenance,forecast_history_gb=estimate,max_history_gb=args.max_history_gb,
        config=dict(nstep=steps,dt_s=dt,duration_s=steps*dt,frequency_hz=frequency,mpi_ranks=2,threads_per_rank=1),
        material_policy=dict(allow_negative_lambda=True,bulk3D_positive_required=True,old_default_unchanged=True,no_model_clipping=True),
        runtime_script_sha256=digest(ev/'experiment_executed.py'),dependency_sha256={p:digest(ROOT/p) for p in dependencies},
        binary_sha256={p:digest(ROOT/p) for p in ['vendor/specfem2d-psv/bin/xspecfem2D','psv-baseline-xspecfem2D']},
        acceptance_predeclared=dict(native_forward='bitwise',data_vjp_rtol=1.e-12,model_dot_rtol=1.e-5,model_dot_epsilons=[.001,.0003,.0001] if unit else [.0003,.0001],
            model_dot_direction='Vp only; dVs=dRho=0; source/receivers/water/true Vs/rho fixed',no_absolute_only_pass=True),
        route='one native recorded forward -> actual Torch data-only L2 VJP -> one exact native coupled adjoint -> physical material map transpose -> select Vp; temporal AD=0, production FD=0, replay=0',gates={})
    start=time.perf_counter()
    def save():
        report['elapsed_seconds']=time.perf_counter()-start
        write_json(ev/'validation.json',report)
    def archive(label):
        shutil.copytree(solver.work,ev/label,ignore=lambda d,n:[v for v in n if v.endswith('history.bin') or v.endswith('.vtk') or v.endswith('.jpg')])
    def guides(path):
        text=path.read_text()
        f=re.search(r'Maximum suggested \(Ricker\) source frequency =\s*([0-9.Ee+-]+)',text)
        t=re.search(r'Maximum suggested time step[^=]*=\s*([0-9.Ee+-]+)',text)
        return dict(max_f0_hz=float(f[1]) if f else None,max_dt_s=float(t[1]) if t else None)
    save()
    # A conservative default must still reject this same negative-lambda input.
    strict=PSVSolver(x,z,case,root=ROOT,ranks=2,work='run_unused_'+name,evidence=ev)
    try:strict.prepare(initial)
    except ValueError as exc:report['default_material_guard']=dict(rejected=True,reason=str(exc))
    else:raise AssertionError('Default material contract unexpectedly changed')
    control=forward(initial[0],baseline=True);initial_guide=guides(solver.work/'native_baseline.log');archive('native_control')
    target=forward(truth[0]);target_guide=guides(solver.work/'forward.log');archive('target')
    report['gates']['native_guidance']=dict(initial=initial_guide,target=target_guide,passed=all(g['max_f0_hz'] is not None and g['max_f0_hz']>=frequency and g['max_dt_s'] is not None and g['max_dt_s']>=dt for g in [initial_guide,target_guide]))
    save()
    if not report['gates']['native_guidance']['passed']:
        report['status']='failed_native_guidance';save();return 2
    if not unit:
        with np.load(ROOT/'evidence/psv_gradient_detail_water2/public/data_and_fixed_window.npz',allow_pickle=False) as old:
            same=np.array_equal(target,old['target'])
            report['gates']['fresh_target_matches_old_physics']=dict(passed=bool(same),new_trace_sha256=array_hash(target),old_trace_sha256=array_hash(old['target']),old_target_reused_as_input=False)
    data=forward(initial[0],record=True)
    times=solver.sample_time.copy()
    report['gates']['native_forward']=dict(passed=bool(np.array_equal(data,control)),absolute_linf=float(np.max(abs(data-control))),trace_sha256=array_hash(data))
    report['data_objective_checks']=data_checks(data,target)
    report['data_spectrum']=dict(predicted=temporal_spectrum(data,dt),target=temporal_spectrum(target,dt),residual=temporal_spectrum(data-target,dt))
    leaf=torch.tensor(data,dtype=torch.float64,requires_grad=True)
    objective=.5*torch.sum((leaf-torch.tensor(target,dtype=torch.float64))**2)
    cot,=torch.autograd.grad(objective,leaf);cot=cot.detach().numpy()
    report['data_leaf_grad_fn']=str(leaf.grad_fn);report['objective_grad_fn']=type(objective.grad_fn).__name__
    report['objective_value']=float(objective.detach())
    histories=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted((solver.work/'OUTPUT_FILES').glob('psv_*history.bin'))]
    report['history']=dict(files=histories,total_bytes=sum(p['bytes'] for p in histories),replay_count=0,live_state='O(nglob); six-field history streamed from disk',retention='temporary full history; hash retained and full forward/adjoint archive excludes it; deleted by prepare only after adjoint')
    if report['history']['total_bytes']>args.max_history_gb*1.e9:raise RuntimeError('Actual native history exceeds declared cap')
    begin=time.perf_counter();physical_gradient,raw=solver.adjoint(cot)
    report['adjoint_and_static_seconds']=time.perf_counter()-begin
    vp_gradient=physical_gradient[0].copy()
    np.savez_compressed(ev/'private_gradient.npz',gradient=physical_gradient,raw=raw,data=data,target=target,cotangent=cot,sample_time=times)
    np.save(ev/'direction_vp.npy',initial[0]*(.55+.25*np.sin(2*np.pi*np.meshgrid(x,z)[0]/x[-1])*np.cos(2*np.pi*np.meshgrid(x,z)[1]/z[-1])))
    archive('native_gradient')
    solid=raw[:,12]==1;rr=raw[solid]
    mapping=BilinearMap(x,z,rr[:,4],rr[:,5])
    rho,lam,mu=rr[:,6],rr[:,7],rr[:,8];vp=np.sqrt((lam+2*mu)/rho)
    gll_truth_rho=mapping.apply(truth[2]);gll_truth_vs=mapping.apply(truth[1])
    report['gates']['fixed_native_coefficients']=dict(passed=bool(np.allclose(rho,gll_truth_rho,rtol=1.e-14,atol=0) and np.allclose(mu,gll_truth_rho*gll_truth_vs**2,rtol=1.e-14,atol=0) and np.min(lam+2*mu/3)>0 and np.min(mu)>0),
        explanation='Fresh actual native rho/mu match interpolation of fixed true grid fields; Vp alters lambda only. Independent target/perturbed SAVE_MODEL exports retained privately.',
        rho_relative_linf=float(np.max(abs(rho-gll_truth_rho))/np.max(abs(rho))),mu_relative_linf=float(np.max(abs(mu-gll_truth_rho*gll_truth_vs**2))/np.max(abs(mu))),
        actual_gll=dict(rho_min=float(rho.min()),mu_min_pa=float(mu.min()),lambda_min_pa=float(lam.min()),bulk3D_min_pa=float((lam+2*mu/3).min()),kappa2D_min_pa=float((lam+mu).min())),water_raw_derivative_zero=bool(np.count_nonzero(raw[~solid,9:12])==0))
    if not unit:
        stats=gradient_statistics(np.stack([vp_gradient,np.zeros_like(vp_gradient),np.zeros_like(vp_gradient)]),x,z,truth,initial)
        report['vp_gradient_statistics']=stats['vp']
    direction=np.load(ev/'direction_vp.npy')
    slope=float(np.vdot(vp_gradient,direction));angle=abs(slope)/(np.linalg.norm(vp_gradient)*np.linalg.norm(direction))
    report['direction']=dict(active=['vp'],dvs_zero=True,drho_zero=True,vp_sha256=array_hash(direction),slope=slope,gradient_direction_abs_cosine=float(angle),informative=bool(angle>1.e-4))
    report['model_dot']=[]
    for h in report['acceptance_predeclared']['model_dot_epsilons']:
        plus=forward(initial[0]+h*direction);archive('check_plus_'+str(h))
        minus=forward(initial[0]-h*direction);archive('check_minus_'+str(h))
        jvp=(plus-minus)/(2*h)
        dot=check_adjoint(lambda p:jvp,lambda q:vp_gradient,direction,cot,rtol=1.e-5,atol=0.)
        dot.update(epsilon=h,informative=bool(angle>1.e-4),taylor_remainder_plus=abs(.5*np.sum((plus-target)**2)-report['objective_value']-h*slope),taylor_remainder_minus=abs(.5*np.sum((minus-target)**2)-report['objective_value']+h*slope))
        dot['passed']=bool(dot['relative_error']<1.e-5 and dot['informative'])
        report['model_dot'].append(dot);save()
    post=verify_packet(args.packet);write_json(ev/'packet_post_native.json',post)
    report['gates']['frozen_packet']=dict(passed=True,packet_sha256=post['packet_sha256'])
    report['native_calls']=solver.calls
    report['gates']['fixed_fields_all_model_calls']=dict(passed=all(call['field_sha256'][key]==report['target']['field_sha256'][key] for call in report['model_calls'] for key in ['vs','rho']))
    report['gates']['stable_all_model_calls']=dict(passed=all(call['stable'] for call in report['model_calls']))
    report['passed']=all(g['passed'] for g in report['gates'].values()) and all(v['passed'] for v in report['data_objective_checks'].values()) and all(v['passed'] for v in report['model_dot'])
    report['status']='passed' if report['passed'] else 'failed';save()
    if report['passed']:
        public=ev/'public';public.mkdir()
        np.savez_compressed(public/'vp_gradient.npz',gradient_vp=vp_gradient,x_native=x,z_native=z)
        rec=np.unique(np.linspace(0,data.shape[0]-1,min(5,data.shape[0])).round().astype(int));idx=np.unique(np.linspace(0,steps-1,min(241,steps)).round().astype(int))
        np.savez_compressed(public/'trace_samples.npz',receiver_index=rec,time_index=idx,sample_time=times[idx],data=data[rec][:,:,idx],target=target[rec][:,:,idx],cotangent=cot[rec][:,:,idx])
        report['public_samples']=dict(rule='5 evenly spaced receiver ids and241 evenly spaced native sample indices; model-free QA sample only, NOT full data or plotting a substitute waveform',full_data_shape=list(data.shape),full_data_sha256=array_hash(data),full_target_sha256=array_hash(target),full_cotangent_sha256=array_hash(cot))
        save()
        for filename in ['validation.json','experiment_executed.py','packet_pre_native.json','packet_post_native.json']:
            shutil.copyfile(ev/filename,public/filename)
        for filename in ['Par_file','SOURCE','STATIONS','interfaces_elastic_analytic.dat']:
            shutil.copyfile(case/'DATA'/filename,public/filename)
        write_json(public/'manifest.json',dict(active_coordinates=['vp'],fixed='true Vs and rho; fixed native acoustic water',no_gradient_postprocessing=True,files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(public.iterdir())]))
    print(json.dumps(dict(status=report['status'],evidence=str(ev),elapsed_seconds=report['elapsed_seconds'])),flush=True)
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--unit',action='store_true')
    parser.add_argument('--packet',type=Path,default=ROOT/'data/frozen-skill-v0.2.0')
    parser.add_argument('--max-history-gb',type=float,default=160.)
    args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(args.unit),indent=2));return 0
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
