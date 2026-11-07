#!/usr/bin/env python3
"""Read-only resolution diagnostics and a native water-source P-SV pilot.

SPDX-License-Identifier: GPL-3.0-or-later
This experiment changes the acquisition and the differentiable data objective.
It never sharpens, smooths, rescales, or substitutes a model for a gradient.
The original native core, established cases, and published figures are untouched.
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def spectral_fraction(field, dx=45., dz=45.):
    """Windowed spatial bandwidth, NOT a claim of resolved model features."""
    field = np.asarray(field, float)
    window = np.outer(np.hanning(field.shape[0]), np.hanning(field.shape[1]))
    mean = np.sum(window*field)/np.sum(window)
    power = np.abs(np.fft.fft2(window*(field-mean)))**2
    kz, kx = np.meshgrid(np.fft.fftfreq(field.shape[0], dz),
                         np.fft.fftfreq(field.shape[1], dx), indexing='ij')
    k = np.hypot(kx, kz)
    total = power.sum()
    return {f'wavelength_below_{length:g}m': float(power[k>1/length].sum()/total) if total else None
            for length in [250, 500, 1000, 2000]}


def gradient_statistics(gradient, x, z, truth=None, initial=None, focus=(4500., 2925.)):
    xx, zz = np.meshgrid(x, z)
    depth = 3465.-zz
    radius = np.hypot(xx-focus[0], zz-focus[1])
    out = {}
    for index, name in enumerate(['vp', 'vs', 'rho']):
        g = np.asarray(gradient[index], float)
        power = g*g
        total = power.sum()
        entry = dict(l2=float(np.linalg.norm(g)), linf=float(np.max(abs(g))),
                     finite=bool(np.isfinite(g).all()),
                     common_active_focus_native_m=list(focus),
                     energy_inside_common_solid_focus_radius={str(r): float(power[radius<r].sum()/total) if total else None
                                                  for r in [300, 600, 1000]},
                     energy_by_depth={}, windowed_spatial_spectrum=spectral_fraction(g))
        for lo, hi in [(495,1000), (1000,1500), (1500,2000), (2000,3000), (3000,3466)]:
            selected = (depth>=lo)&(depth<hi)
            entry['energy_by_depth'][f'{lo}_{hi}m'] = float(power[selected].sum()/total) if total else None
        deep_rows = (3465.-z>=1000)&(3465.-z<=3000)
        entry['deep_1000_3000m_windowed_spectrum'] = spectral_fraction(g[deep_rows])
        if truth is not None and initial is not None:
            error = truth[index]-initial[index]
            norm = np.linalg.norm(error)*np.linalg.norm(g)
            entry['negative_gradient_model_correction_cosine'] = float(np.vdot(-g,error)/norm) if norm else None
        out[name] = entry
    return out


def temporal_spectrum(data, dt):
    data = np.asarray(data, float)
    window = np.hanning(data.shape[-1])
    centered = data-data.mean(axis=-1, keepdims=True)
    power = np.sum(abs(np.fft.rfft(centered*window, axis=-1))**2,
                   axis=tuple(range(data.ndim-1)))
    freq = np.fft.rfftfreq(data.shape[-1], dt)
    total = power.sum()
    if total == 0:
        return dict(rms=0., f50_hz=None, f90_hz=None)
    cdf = np.cumsum(power)/total
    result = dict(rms=float(np.sqrt(np.mean(data*data))), linf=float(np.max(abs(data))))
    for quantile in [.5,.9,.95]:
        result[f'f{int(100*quantile)}_hz'] = float(freq[min(np.searchsorted(cdf,quantile),len(freq)-1)])
    result['energy_above_hz'] = {str(f):float(power[freq>f].sum()/total) for f in [.5,1.,1.5,2.,3.]}
    return result


def analyze(args):
    ev = ROOT/args.existing
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    model = np.load(ROOT/args.model, allow_pickle=False)
    x, z = model['x_native'], model['z_native']
    truth = np.stack([model[k+'_true'] for k in ['vp','vs','rho']])
    initial = np.stack([model[k+'_initial'] for k in ['vp','vs','rho']])
    target = np.load(ev/'target.npy')
    report = dict(status='read-only diagnostics', source_case=args.existing,
                  input_model_sha256=digest(ROOT/args.model), gradients={}, data={},
                  caution='Spatial Fourier power and source/depth concentration diagnose bandwidth/illumination; high wavenumber at a point source is not resolved geology.',
                  resource_forecasts={})
    stations = np.loadtxt(ev/'gradient_l2/DATA/STATIONS', usecols=[2,3])
    offset = abs(stations[:,0]-4500.)
    for kind in ['l2','processed']:
        result = np.load(ev/f'gradient_{kind}.npz')
        report['gradients'][kind] = gradient_statistics(result['gradient'],x,z,truth,initial)
        dt = float(np.median(np.diff(result['sample_time'])))
        residual = result['data']-target
        report['data'][kind] = dict(predicted=temporal_spectrum(result['data'],dt),
                                  residual=temporal_spectrum(residual,dt))
        total = float(np.sum(residual*residual))
        report['data'][kind]['residual_energy_by_horizontal_offset'] = {
            f'{lo}_{hi}m': float(np.sum(residual[(offset>=lo)&(offset<hi)]**2)/total)
            for lo,hi in [(0,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,10000)]}
    # Six full native arrays, plus conservative 1% MPI interface replicas.
    for label,nx,ns,nw,dt,duration in [
        ('existing_1Hz_full_6s',128,48,8,.00075,6.),
        ('proposed_2Hz_full_4s',192,72,12,.0005,4.),
        ('proposed_3Hz_full_4s',288,108,18,1/3000,4.),
        ('proposed_2Hz_2970m_ROI_4s',96,72,12,.0005,4.)]:
        nodes = (4*nx+1)*(4*(ns+nw)+1)
        steps = int(round(duration/dt))
        report['resource_forecasts'][label] = dict(elements=nx*(ns+nw), native_steps=steps,
            full_history_gb=6*nodes*steps*8*1.01/1.e9,
            not_a_native_resolution_pass='Both true/initial native f0/dt guides must be measured before any recorded run')
    write_json(out/'diagnostics.json', report)
    print(json.dumps(report, indent=2))


def raised_cosine(x):
    return .5-.5*np.cos(np.pi*np.clip(x,0,1))


def fixed_mute(stations, sample_time, source_x=4500., source_depth=150.,
               source_t0=1.2, vref=2500., delay=.7, fade=.35,
               offset_min=350., offset_full=700., offset_max=3500., offset_taper=500., end=4.25):
    """Fixed data-coordinate window, frozen for model perturbations.

    This reflection-oriented selection is not claimed to separate all direct,
    converted, or outer-boundary reflected waves. The reference speed is an
    explicitly chosen constant, not recomputed from the changing model.
    """
    offset = abs(stations[:,0]-source_x)
    receiver_depth = 3465.-stations[:,1]
    start = source_t0+np.hypot(offset,receiver_depth-source_depth)/vref+delay
    elapsed = sample_time+source_t0+float(np.median(np.diff(sample_time)))
    w = raised_cosine((elapsed[None,:]-start[:,None])/fade)
    w *= raised_cosine((end-elapsed[None,:])/fade)
    ow = raised_cosine((offset-offset_min)/(offset_full-offset_min))
    ow *= raised_cosine((offset_max+offset_taper-offset)/offset_taper)
    return w[:,None,:]*ow[:,None,None], dict(elapsed_s=elapsed.tolist(),
        mute_start_elapsed_s=start.tolist(), receiver_horizontal_offset_m=offset.tolist(),
        fixed_vref_m_s=vref, delay_s=delay, fade_s=fade, end_elapsed_s=end,
        offset_min_m=offset_min, offset_full_m=offset_full, offset_max_m=offset_max,
        offset_taper_m=offset_taper)


def set_parameter(text, key, value):
    pattern = rf'(?m)^({re.escape(key)}\s*=)[^\n]*'
    if not re.search(pattern,text):
        raise ValueError('Missing native parameter '+key)
    return re.sub(pattern, lambda match:match[1]+' '+str(value), text)


def make_water_case(name, factor=1.e8):
    case = ROOT/'examples'/name
    if case.exists():
        raise FileExistsError(case)
    shutil.copytree(ROOT/'examples/psv_marmousi2_full',case)
    p = case/'DATA/SOURCE'
    text = p.read_text()
    for key,value in [('zs','3315.'),('time_function_type','3'),('factor',f'{factor:.17g}')]:
        text = set_parameter(text,key,value)
    p.write_text(text)
    p=case/'CASE_PROVENANCE.json'
    metadata=json.loads(p.read_text())
    metadata.update(case=name,status='new water-source gradient-detail pilot, awaiting native checks',
                    source_native_m=[4500.,3315.],source_original_m=[9500.,150.],
                    source_type='fixed native acoustic source in water',
                    source_function='Gaussian potential chi (time_function_type3)',
                    source_factor=factor,validation='new experiment pilot.json; prior parent-case validation is not inherited')
    write_json(p,metadata)
    write_json(case/'EXPERIMENT.json',dict(case=name,physics='native coupled acoustic/P-SV, fixed water, active solid Vp/Vs/rho',
        parent_case='psv_marmousi2_full', changed=dict(source_depth_m=150.,source_native_z_m=3315.,
        source_function='Gaussian potential chi, native time_function_type3',source_factor=factor),
        unchanged='prepared true/initial material arrays, 128x56 mesh, 1Hz, dt=.00075s, 6s, 41 solid two-component displacement receivers, reflecting/free outer boundaries',
        source_units='Acoustic and elastic force source factors have different physical meaning; equal numeric factors do not assert equal source energy.'))
    return case


def pilot(args):
    import torch
    from diffspecfem.psv import PSVSolver
    from diffspecfem.numerics import check_adjoint

    ev = ROOT/args.output
    ev.mkdir(parents=True,exist_ok=True)
    if (ev/'pilot.json').exists():
        raise RuntimeError('Use a new namespace, preserving every prior attempt')
    case = make_water_case(args.case_name,args.factor)
    source = np.load(ROOT/args.model,allow_pickle=False)
    x,z = source['x_native'],source['z_native']
    initial = np.stack([source[k+'_initial'] for k in ['vp','vs','rho']])
    truth = np.stack([source[k+'_true'] for k in ['vp','vs','rho']])
    solver = PSVSolver(x,z,case,root=ROOT,ranks=2,
                      work='run_'+args.case_name,evidence=args.output)
    report = dict(status='running',passed=False,case=str(case.relative_to(ROOT)),
        model_sha256=digest(ROOT/args.model), source_factor=args.factor,
        source='native acoustic Gaussian chi source in fixed water at original depth150m',
        storage='one shared six-field native forward history on disk, two distinct data-AD cotangents/native adjoints; no replay, no temporal AD; temporary history removed only after archives and before check-only forward',
        accepted_before_run=dict(native_forward='bitwise',source_f0_hz=1.,dt_s=.00075,
            model_dot_rtol=1.e-5,model_dot_atol=1.e-16,epsilons=[.0003,.0001]),
        objectives={}, gates={})
    started=time.perf_counter()
    def save():
        report['elapsed_seconds']=time.perf_counter()-started
        write_json(ev/'pilot.json',report)
    save()
    def archive(label):
        # Large history stays only in the dedicated run directory. Keep real
        # native vector snapshots, coefficient gradients and logs for auditing.
        shutil.copytree(solver.work,ev/label,
            ignore=lambda d,n:[v for v in n if v.endswith('history.bin') or v.endswith('.vtk')])
    def guides(path):
        text=Path(path).read_text()
        f=re.search(r'Maximum suggested \(Ricker\) source frequency =\s*([0-9.Ee+-]+)',text)
        dt=re.search(r'Maximum suggested time step[^=]*=\s*([0-9.Ee+-]+)',text)
        return dict(max_f0_hz=float(f[1]) if f else None,max_dt_s=float(dt[1]) if dt else None)
    control=solver.forward(initial,baseline=True);archive('water_native_control')
    initial_guides=guides(solver.work/'native_baseline.log')
    target=solver.forward(truth);archive('water_target')
    true_guides=guides(solver.work/'forward.log')
    report['gates']['source_resolution']=dict(passed=all(g['max_f0_hz'] is not None and g['max_f0_hz']>=1. and g['max_dt_s'] is not None and g['max_dt_s']>=.00075 for g in [initial_guides,true_guides]),initial=initial_guides,true=true_guides)
    report['data_control']=temporal_spectrum(control,.00075)
    report['target_spectrum']=temporal_spectrum(target,.00075)
    save()
    if not report['gates']['source_resolution']['passed']:
        report['status']='failed_source_guidance';save();return 2
    # Exactly one recorded native forward is reused by the two adjoint calls.
    data=solver.forward(initial,record=True)
    report['gates']['native_forward']=dict(passed=bool(np.array_equal(data,control)),absolute_linf=float(np.max(abs(data-control))))
    times=solver.sample_time.copy()
    stations=np.loadtxt(solver.work/'DATA/STATIONS',usecols=[2,3])
    weight,window_config=fixed_mute(stations,times)
    np.savez_compressed(ev/'data_and_fixed_window.npz',data=data,target=target,sample_time=times,weight=weight)
    report['fixed_window']=window_config
    report['fixed_window']['note']='Chosen before inspecting the gradient; held exactly fixed for all model differences. May still contain physical outer-boundary reflections.'
    histories=[]
    for path in sorted((solver.work/'OUTPUT_FILES').glob('psv_*history.bin')):
        histories.append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path)))
    write_json(ev/'history_manifest.json',histories)
    grads={};cots={}
    for kind in ['l2','reflection_window_l2']:
        leaf=torch.tensor(data,dtype=torch.float64,requires_grad=True)
        residual=leaf-torch.tensor(target,dtype=torch.float64)
        w=torch.ones_like(leaf) if kind=='l2' else torch.as_tensor(weight,dtype=torch.float64)
        value=.5*torch.sum(w*residual**2)
        cot,=torch.autograd.grad(value,leaf)
        t=time.perf_counter();gradient,raw=solver.adjoint(cot.detach().numpy())
        grads[kind]=gradient;cots[kind]=cot.detach().numpy()
        phase_seconds=time.perf_counter()-t
        np.savez_compressed(ev/f'gradient_{kind}_private.npz',gradient=gradient,raw=raw,
            data=data,cotangent=cots[kind],sample_time=times)
        archive('gradient_'+kind)
        expected=(data-target)*(1. if kind=='l2' else weight)
        cot_error=float(np.linalg.norm(cots[kind]-expected)/max(np.linalg.norm(expected),np.finfo(float).tiny))
        report['objectives'][kind]=dict(value=value.item(),adjoint_and_static_vjp_seconds=phase_seconds,
            derivative_statistics=gradient_statistics(gradient,x,z,truth,initial),
            common_focus_note='Same active solid coordinate x4500/native_z2925 as the old solid source, for fair concentration comparisons; actual new source is in water.',
            data_vjp_check=dict(passed=cot_error<1.e-12,relative_error=cot_error),
            data_grad_fn=str(leaf.grad_fn),objective_grad_fn=type(value.grad_fn).__name__)
        save()
    # Check-only native differences, after both native adjoints finished.
    xx,zz=np.meshgrid(x,z);a=.55+.25*np.sin(2*np.pi*xx/x[-1])*np.cos(2*np.pi*zz/z[-1]);b=.45+.2*np.cos(3*np.pi*xx/x[-1]+zz/z[-1])
    direction=initial*np.stack([a,b,.5+.2*np.sin(2*np.pi*xx/x[-1])])
    for h in [.0003,.0001]:
        jvp=(solver.forward(initial+h*direction)-solver.forward(initial-h*direction))/(2*h)
        for kind in grads:
            dot=check_adjoint(lambda p:jvp,lambda q:grads[kind],direction,cots[kind],rtol=1.e-5,atol=1.e-16)
            dot['epsilon']=h
            report['objectives'][kind].setdefault('model_dot',[]).append(dot)
        save()
    checks=[g['passed'] for g in report['gates'].values()]
    checks += [v['data_vjp_check']['passed'] for v in report['objectives'].values()]
    checks += [v['passed'] for objective in report['objectives'].values() for v in objective['model_dot']]
    report['passed']=all(checks);report['status']='passed' if report['passed'] else 'failed';save()
    if report['passed']:
        public=ev/'public';public.mkdir()
        shutil.copyfile(ev/'pilot.json',public/'pilot.json')
        shutil.copyfile(ev/'data_and_fixed_window.npz',public/'data_and_fixed_window.npz')
        for kind in grads:
            np.savez_compressed(public/f'gradient_{kind}.npz',gradient=grads[kind],data=data,
                                cotangent=cots[kind],sample_time=times)
        for name in ['SOURCE','STATIONS','Par_file','interfaces_elastic_analytic.dat']:
            shutil.copyfile(case/'DATA'/name,public/name)
        manifest=dict(scope='Actual native gradients/data/cotangents and fixed objective weights only; no model material arrays or large wavefields',
            native_case=str(case.relative_to(ROOT)),input_model_sha256=report['model_sha256'],
            gradient_coordinates='physical solid [Vp,Vs,rho], increasing native(z,x), shape(3,67,200); fixed water',
            history_policy=report['storage'],no_gradient_postprocessing=True,
            outputs=[dict(path=str(p.relative_to(public)),bytes=p.stat().st_size,sha256=digest(p))
                     for p in sorted(public.iterdir()) if p.is_file()])
        write_json(public/'manifest.json',manifest)
    print(json.dumps(dict(status=report['status'],evidence=str(ev),seconds=report['elapsed_seconds']),indent=2))
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('analyze');p.add_argument('--existing',default='evidence/psv_marmousi128')
    p.add_argument('--model',default='data/prepared/marmousi2_full_psv_solid_domain.npz')
    p.add_argument('--output',default='evidence/psv_gradient_detail_diagnostics')
    p=commands.add_parser('pilot');p.add_argument('--model',default='data/prepared/marmousi2_full_psv_solid_domain.npz')
    p.add_argument('--output',default='evidence/psv_gradient_detail_water1')
    p.add_argument('--case-name',default='psv_gradient_detail_water1');p.add_argument('--factor',type=float,default=1.e8)
    args=parser.parse_args()
    return analyze(args) if args.command=='analyze' else pilot(args)


if __name__=='__main__':
    raise SystemExit(main())
