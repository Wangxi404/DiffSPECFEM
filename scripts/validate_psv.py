#!/usr/bin/env python3
"""Verify the exact native coupled P-SV adapter on a dense solid model.

SPDX-License-Identifier: GPL-3.0-or-later
Check-only native differences never supply a production gradient.
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
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffspecfem.psv import PSVSolver,psv_objective
from diffspecfem.tomography import BilinearMap
from diffspecfem.numerics import check_adjoint,taylor_test


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--case',required=True)
    p.add_argument('--evidence',required=True);p.add_argument('--ranks',type=int,default=2)
    p.add_argument('--par-file',default='Par_file')
    p.add_argument('--steps',type=float,nargs='+',default=[.02,.006,.002,.0006,.0002,.00006])
    p.add_argument('--directions',nargs='+',default=['vp','vs','rho','mixed','localized'])
    p.add_argument('--mode',choices=['full','illustration','smoke'],default='full')
    args=p.parse_args();root=Path(__file__).resolve().parents[1];ev=root/args.evidence
    ev.mkdir(parents=True,exist_ok=True)
    if (ev/'validation.json').exists():raise RuntimeError('Use a new namespace; retain earlier attempts')
    model=np.load(args.model,allow_pickle=False);x,z=model['x_native'],model['z_native']
    initial=np.stack([model[k+'_initial'] for k in ['vp','vs','rho']]);truth=np.stack([model[k+'_true'] for k in ['vp','vs','rho']])
    s=PSVSolver(x,z,args.case,root=root,ranks=args.ranks,work='run_'+Path(args.evidence).name,evidence=args.evidence,par_file=args.par_file)
    report=dict(status='running',passed=False,case=args.case,mode=args.mode,model_sha256=sha(args.model),
        model_shape=list(initial.shape),ranks=args.ranks,active_coordinates='physical solid [Vp(m/s),Vs(m/s),rho(kg/m^3)] on increasing native(z,x)',
        fixed='acoustic water coefficients, source, receiver geometry, mesh, dt, reflecting/free boundaries',
        history_policy='six native post-step arrays on disk; O(nglob) live reverse state, no replay',
        accepted_thresholds=dict(declared_before_run=True,native_forward='bitwise',interpolation_rtol=2.e-13,
            sampling_rtol=1.e-12,coupling_rtol=1.e-12,data_vjp_rtol=1.e-6,model_rtol=1.e-5,model_atol=1.e-9,
            steps=args.steps,min_taylor_order=1.8,max_taylor_order=2.2),gates={})
    started=time.perf_counter()
    def save():
        report['elapsed_seconds']=time.perf_counter()-started
        (ev/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()

    def archive(name,history=False):
        dest=ev/name
        shutil.copytree(s.work,dest,ignore=lambda d,n:[v for v in n if v.endswith('history.bin') or v.endswith('.vtk')])
        if not history:return
        arrays={};sources=[]
        for rank in range(args.ranks):
            prefix=f'psv_rank{rank:06d}_';path=s.work/'OUTPUT_FILES'/f'{prefix}history.bin'
            header=np.fromfile(path,dtype='<i4',count=3);ng,nt,nr=map(int,header)
            mm=np.memmap(path,dtype='<f8',mode='r',offset=12,shape=(nt,6*ng))
            idx=np.unique(np.r_[np.arange(0,nt,max(1,nt//48)),nt-1]);selected=np.array(mm[idx])
            states=np.concatenate([selected[:,:2*ng].reshape(len(idx),ng,2).transpose(0,2,1),
                selected[:,2*ng:4*ng].reshape(len(idx),ng,2).transpose(0,2,1),selected[:,4*ng:].reshape(len(idx),2,ng)],axis=1)
            arrays[f'rank{rank}_header']=header;arrays[f'rank{rank}_states']=states;arrays['step_index_zero_based']=idx
            sources.append(dict(file=str(path),bytes=path.stat().st_size,sha256=sha(path),header=header.tolist()))
        arrays['provenance_json']=np.array(json.dumps(dict(sources=sources,fields=['ux','uz','ax','az','chi','chi_ddot'],
            selection='48 approximately uniformly spaced actual native post-step frames + last; no spatial subsampling or synthetic wavefield')))
        np.savez_compressed(dest/'native_frames.npz',**arrays)
        (dest/'history_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')

    def diagnostics(path):
        text=Path(path).read_text()
        fm=re.search(r'Maximum suggested \(Ricker\) source frequency =\s*([0-9.Ee+-]+)',text)
        dm=re.search(r'Maximum suggested time step[^=]*=\s*([0-9.Ee+-]+)',text)
        return dict(max_suggested_f0_hz=float(fm[1]) if fm else None,max_suggested_dt_s=float(dm[1]) if dm else None)
    native=s.forward(initial,baseline=True);np.save(ev/'native_reference.npy',native);archive('native_baseline')
    report['initial_diagnostics']=diagnostics(s.work/'native_baseline.log')
    target=s.forward(truth);np.save(ev/'target.npy',target);archive('target_native')
    report['true_diagnostics']=diagnostics(s.work/'forward.log')
    source=(s.work/'DATA/SOURCE').read_text();f0=float(re.search(r'(?m)^f0\s*=\s*([0-9.Ee+-]+)',source)[1])
    par=(s.work/'DATA/Par_file').read_text();dt=float(re.search(r'(?m)^DT\s*=\s*([0-9.EedD+-]+)',par)[1].lower().replace('d','e'))
    report['gates']['source_resolution']=dict(passed=all(d['max_suggested_f0_hz'] is not None and f0<=d['max_suggested_f0_hz'] and d['max_suggested_dt_s'] is not None and dt<=d['max_suggested_dt_s'] for d in (report['initial_diagnostics'],report['true_diagnostics'])),source_f0_hz=f0,dt_s=dt)
    save()
    if not report['gates']['source_resolution']['passed']:
        report['status']='failed_source_resolution';save();return 2
    results={}
    for kind in ['l2','processed']:
        result=s.value_gradient(initial,target,kind);results[kind]=result
        np.savez_compressed(ev/f'gradient_{kind}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        archive('gradient_'+kind,history=True)
        report['gates'][kind]=dict(value=result['value'],seconds=result['seconds'],gradient_norms=[float(np.linalg.norm(g)) for g in result['gradient']],
            data_grad_fn=result['data_grad_fn'],objective_grad_fn=result['objective_grad_fn'],production_route=result['production_route'])
        if kind=='l2':
            report['gates']['native_forward']=dict(passed=bool(np.array_equal(native,result['data'])),absolute_linf=float(np.max(abs(native-result['data']))))
            raw=result['raw'];rr=raw[raw[:,12]==1];mapping=BilinearMap(x,z,rr[:,4],rr[:,5])
            rp,vp,vs=mapping.apply(initial[2]),mapping.apply(initial[0]),mapping.apply(initial[1])
            comparisons=[np.max(abs(rr[:,6]-rp))/np.max(abs(rp)),np.max(abs(rr[:,7]-rp*(vp**2-2*vs**2)))/np.max(abs(rr[:,7])),np.max(abs(rr[:,8]-rp*vs**2))/np.max(abs(rr[:,8]))]
            report['gates']['native_material_map']=dict(passed=bool(max(comparisons)<2.e-13),relative_linf_rho_lambda_mu=list(map(float,comparisons)))
            probe=np.sin(np.arange(initial[0].size).reshape(initial[0].shape)*.013)+.2;v=np.cos(np.arange(len(rr))*.017)+.3
            report['gates']['interpolation_transpose']=check_adjoint(mapping.apply,mapping.transpose,probe,v,atol=1.e-10,rtol=1.e-12)
            xs=[];ats=[];ys=[];axs=[];coupling_lhs=0.;coupling_rhs=0.;edges=0;length=0.;histbytes=0;sample_error=0.
            frames=np.load(ev/'gradient_l2/native_frames.npz');idx=frames['step_index_zero_based'];sampled=np.zeros((len(s.stations),2,len(idx)))
            for rank in range(args.ranks):
                prefix=f'psv_rank{rank:06d}_';out=s.work/'OUTPUT_FILES'
                nodes=np.loadtxt(out/f'{prefix}sampling_native_nodes.txt',ndmin=2);recs=np.loadtxt(out/f'{prefix}sampling_native_receivers.txt',ndmin=2)
                xs.append((nodes[:,1:3]/np.sqrt(nodes[:,5,None])).ravel());ats.append((nodes[:,3:5]/np.sqrt(nodes[:,5,None])).ravel())
                ys.append(recs[:,1:3].ravel());axs.append(recs[:,3:5].ravel())
                coupling=np.loadtxt(out/f'{prefix}coupling_probe.txt',ndmin=2)
                coupling_lhs+=float(np.sum(coupling[:,1]*coupling[:,2]/coupling[:,7]))
                coupling_rhs+=float(np.sum(np.sum(coupling[:,3:5]*coupling[:,5:7],axis=1)/coupling[:,8]))
                geom=np.loadtxt(out/f'{prefix}coupling_geometry.txt');edges+=int(geom[0]);length-=float(geom[1])
                rows=np.loadtxt(out/f'{prefix}sampling.txt',ndmin=2);states=frames[f'rank{rank}_states']
                for r,k,w in rows:sampled[int(r)-1]+=w*states[:,:2,int(k)-1].T
                histbytes+=(out/f'{prefix}history.bin').stat().st_size
            report['gates']['sampling_native']=check_adjoint(lambda q:np.concatenate(axs),lambda q:np.concatenate(ats),np.concatenate(xs),np.concatenate(ys),atol=1.e-12,rtol=1.e-12)
            report['gates']['coupling_transpose']=dict(passed=abs(coupling_lhs-coupling_rhs)<=1.e-12+1.e-12*max(abs(coupling_lhs),abs(coupling_rhs)),lhs=coupling_lhs,rhs=coupling_rhs,relative_error=abs(coupling_lhs-coupling_rhs)/max(abs(coupling_lhs),abs(coupling_rhs)))
            report['gates']['coupling_geometry']=dict(passed=bool(np.isclose(length,x[-1]-x[0],atol=1.e-8,rtol=1.e-12)),native_edges_sum=edges,flat_interface_length_m=length,expected_length_m=float(x[-1]-x[0]))
            report['gates']['sampling_history']=dict(passed=bool(np.allclose(sampled,result['data'][:,:,idx],rtol=1.e-12,atol=1.e-14)),absolute_linf=float(np.max(abs(sampled-result['data'][:,:,idx]))),history_bytes_sum=histbytes,actual_steps_checked=idx.tolist())
        rng=np.random.default_rng(7013);dp=rng.normal(size=result['data'].shape);dp*=max(np.linalg.norm(result['data']),.001)/np.linalg.norm(dp)
        evald=lambda d:psv_objective(torch.as_tensor(d,dtype=torch.float64),torch.as_tensor(target,dtype=torch.float64),kind).item()
        report['gates'][kind]['data_vjp']=taylor_test(evald,result['data'],result['cotangent'],dp,steps=[.01,.003,.001,.0003,.0001,.00003],atol=1.e-10,rtol=1.e-6)
        save()
    if args.mode!='smoke':
        xx,zz=np.meshgrid(x,z);xn=(xx-x[0])/(x[-1]-x[0]);zn=(zz-z[0])/(z[-1]-z[0])
        a=.55+.25*np.sin(2*np.pi*xn)*np.cos(2*np.pi*zn)+.2*np.cos(5*np.pi*xn+2*np.pi*zn)
        b=.45+.3*np.cos(3*np.pi*xn)*np.sin(2*np.pi*zn)+.15*np.sin(4*np.pi*xn+zn)
        c=.5+.2*np.cos(2*np.pi*xn+zn)
        local=np.exp(-((xn-.53)/.15)**2-((zn-.78)/.18)**2);zero=np.zeros_like(a)
        directions=dict(vp=np.stack([initial[0]*a,zero,zero]),vs=np.stack([zero,initial[1]*b,zero]),rho=np.stack([zero,zero,initial[2]*c]),mixed=initial*np.stack([a,b,c]),localized=initial*np.stack([local,.7*local,.3*local]))
        cache={sha256bytes(initial):results['l2']['data']}
        def data_at(m):
            key=sha256bytes(m)
            if key not in cache:cache[key]=s.forward(m)
            return cache[key]
        for name in args.directions:
            direction=directions[name];np.save(ev/f'direction_{name}.npy',direction)
            for kind,result in results.items():
                if args.mode=='full':
                    def evalm(m):return psv_objective(torch.as_tensor(data_at(m),dtype=torch.float64),torch.as_tensor(target,dtype=torch.float64),kind).item()
                    report['gates'][kind]['taylor_'+name]=taylor_test(evalm,initial,result['gradient'],direction,steps=args.steps,atol=1.e-9,rtol=1.e-5)
                else:
                    rows=[]
                    for h in [.0003,.0001]:
                        jvp=(data_at(initial+h*direction)-data_at(initial-h*direction))/(2*h)
                        check=check_adjoint(lambda p:jvp,lambda q:result['gradient'],direction,result['cotangent'],rtol=1.e-5,atol=1.e-9);check['epsilon']=h;rows.append(check)
                    report['gates'][kind]['directional_dot_'+name]=rows
                save()
        direction=directions['mixed'];h=.00006;jvp=(data_at(initial+h*direction)-data_at(initial-h*direction))/(2*h)
        for kind,result in results.items():report['gates'][kind]['model_dot']=check_adjoint(lambda p:jvp,lambda q:result['gradient'],direction,result['cotangent'],rtol=1.e-5,atol=1.e-9)
    def passes(v):
        if isinstance(v,dict):return ([v['passed']] if 'passed' in v else [])+[p for k,x in v.items() if k!='passed' for p in passes(x)]
        if isinstance(v,list):return [p for x in v for p in passes(x)]
        return []
    report['passed']=bool(all(passes(report['gates'])));report['status']='passed' if report['passed'] else 'failed';report['native_call_count']=len(s.calls);save()
    print(json.dumps(dict(status=report['status'],evidence=str(ev),seconds=report['elapsed_seconds']),indent=2));return 0 if report['passed'] else 1


def sha256bytes(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        import traceback
        if '--evidence' in sys.argv:
            ev=Path(__file__).resolve().parents[1]/sys.argv[sys.argv.index('--evidence')+1]
            path=ev/'validation.json'
            if path.exists():
                report=json.loads(path.read_text());report.update(status='failed_exception',passed=False,error=str(error),traceback=traceback.format_exc())
                path.write_text(json.dumps(report,indent=2)+'\n')
                run=Path(__file__).resolve().parents[1]/('run_'+ev.name)
                if run.exists() and not (ev/'failed_native_run').exists():
                    shutil.copytree(run,ev/'failed_native_run',ignore=lambda d,n:[v for v in n if v.endswith('history.bin') or v.endswith('.vtk')])
        raise
