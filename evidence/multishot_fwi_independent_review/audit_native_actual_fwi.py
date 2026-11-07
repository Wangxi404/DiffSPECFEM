"""Read completed native FWI unit + immutable ROI epoch; never launch solver."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--case',action='append',help='Prepared/run namespace; may be repeated')
parser.add_argument('--expected-driver-sha256',default='28eed616dacde81f48eb1d470b927f7969166fe1a74468be88e6dd3ac01716c6')
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args()
ROOT=options.repo_root.resolve()
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='native-actual-fwi-independent-'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-300))
def bits(a,b):return bool(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a.view(np.uint64),b.view(np.uint64)))
def fingerprint(vp,vs,rho):return ah(np.stack([vp,vs,rho]))
def map_info(x,z,qx,qz):
    i=np.clip(np.searchsorted(x,qx,side='right')-1,0,len(x)-2)
    j=np.clip(np.searchsorted(z,qz,side='right')-1,0,len(z)-2)
    tx=np.clip((qx-x[i])/(x[i+1]-x[i]),0,1)
    tz=np.clip((qz-z[j])/(z[j+1]-z[j]),0,1)
    return [(j,i,(1-tx)*(1-tz)),(j,i+1,tx*(1-tz)),(j+1,i,(1-tx)*tz),(j+1,i+1,tx*tz)]
def interp(value,info):return sum(value[j,i]*w for j,i,w in info)
def transpose(values,info,shape):
    output=np.zeros(shape)
    for j,i,w in info:np.add.at(output,(j,i),w*values)
    return output

def base_audit(name):
    run=ROOT/('run_'+name);prepared=ROOT/'data/inversion'/name
    protocol=read(prepared/'protocol.json');training=arrays(prepared/'training_input.npz')
    runtime=read(run/'runtime_signature.json')
    runtime_ok={p:sha(ROOT/p)==v for p,v in runtime.items()}
    assert all(runtime_ok.values())
    assert sha(prepared/'training_input.npz')==protocol['training_input_sha256']
    assert sha(run/'protocol_copy.json')==sha(prepared/'protocol.json')
    assert sha(run/'driver_executed.py')==runtime['scripts/invert_vponly_psv.py']==options.expected_driver_sha256
    assert set(training)=={'vp_initial','vs_fixed','rho_fixed','x_native','z_native','active_mask'}
    truth=arrays(prepared/'truth_for_observations_only.npz')
    assert sha(prepared/'truth_for_observations_only.npz')==protocol['truth_file_for_observations_only_sha256']
    assert bits(training['vs_fixed'],truth['vs']) and bits(training['rho_fixed'],truth['rho'])
    stage=protocol['stages'][0]
    frozen=read(run/'target_signatures'/f"{stage['id']}.json")
    manifest=read(prepared/'observations'/stage['id']/'manifest.json')
    assert frozen['manifest_sha256']==sha(prepared/'observations'/stage['id']/'manifest.json')
    for sid in protocol['shot_ids']:
        for file,digest in stage['case_hashes'][str(sid)].items():assert sha(prepared/'cases'/stage['id']/f'shot_{sid:03d}'/'DATA'/file)==digest
        assert sha(prepared/'observations'/stage['id']/f'shot_{sid:03d}.npz')==manifest[str(sid)]['sha256']
        assert manifest[str(sid)]['source_sha256']==stage['case_hashes'][str(sid)]['SOURCE']
    packet={}
    for p in ('packet_pre.json','packet_post.json'):
        if (run/p).exists():packet[p]=read(run/p)['packet_sha256']
    assert not packet or len(set(packet.values()))==1
    return run,prepared,protocol,training,stage,dict(
        runtime_all_hashes_match=runtime_ok,driver_snapshot_sha256=sha(run/'driver_executed.py'),
        training_input_sha256=sha(prepared/'training_input.npz'),protocol_sha256=sha(prepared/'protocol.json'),
        target_manifest_sha256=sha(prepared/'observations'/stage['id']/'manifest.json'),
        no_unknown_true_Vp_in_training=True,known_Vs_rho_uint64_equal_observation_priors=True,
        native_field_composition_source='Every native call stacks [trialVp, fixedVs, fixedRho] before PSVSolver.prepare; source/runtime hashes checked.',
        packet_pre_post=packet)

def audit_group(run,prepared,protocol,training,stage,directory,vp,raw=False):
    n=len(protocol['shot_ids']);gs=[];losses=[];cot_errors=[];scale_errors=[];rows=[];joint=[];static=[]
    mh=fingerprint(vp,training['vs_fixed'],training['rho_fixed'])
    for sid in protocol['shot_ids']:
        path=directory/f'shot_{sid:03d}.npz';meta=read(directory/f'shot_{sid:03d}.json');a=arrays(path)
        obs=arrays(prepared/'observations'/stage['id']/f'shot_{sid:03d}.npz');scale=float(obs['scale'])
        assert meta['array_sha256']==sha(path) and meta['model_sha256']==mh and meta['vp_sha256']==ah(vp)
        assert meta['target_sha256']==sha(prepared/'observations'/stage['id']/f'shot_{sid:03d}.npz')
        assert meta['source_sha256']==stage['case_hashes'][str(sid)]['SOURCE']
        assert meta['vs_fixed_sha256']==ah(training['vs_fixed']) and meta['rho_fixed_sha256']==ah(training['rho_fixed'])
        assert meta['runtime_signature_sha256']==sha(run/'runtime_signature.json')
        assert meta['target_signature_sha256']==sha(run/'target_signatures'/f"{stage['id']}.json")
        assert bits(a['sample_time'],obs['sample_time'])
        measured=float(np.sqrt(np.mean(obs['observed']**2)));scale_errors.append(abs(scale-measured)/scale)
        residual=a['data']-obs['observed'];value=.5*float(np.mean((residual/scale)**2));losses.append(value)
        assert abs(value-float(a['loss']))<=1e-13*max(abs(value),1e-300)
        if 'gradient' in a:
            cot=residual/(a['data'].size*scale**2)
            error=rel(a['cotangent'],cot);assert error<1e-12;cot_errors.append(error);gs.append(a['gradient'])
            if 'joint_gradient' in a:joint.append(a['joint_gradient'])
        probes=meta.get('native_probes');water=meta.get('fixed_water_proof')
        if probes:
            assert probes['sampling_relative_error']<1e-12 and probes['coupling_relative_error']<1e-12
            assert probes['native_edges']==stage['nx'] and abs(probes['interface_length_m']-protocol['width_m'])<1e-8
        if water:assert water['passed'] and water['raw_fluid_rows']>0 and water['fluid_gradient_zero']
        if raw:
            r=a['raw'];solid=r[:,12]==1;fluid=~solid;rr=r[solid];rho,lam,mu=rr[:,6],rr[:,7],rr[:,8]
            actualVp=np.sqrt((lam+2*mu)/rho);actualVs=np.sqrt(mu/rho)
            info=map_info(training['x_native'],training['z_native'],rr[:,4],rr[:,5])
            independent=transpose(2*rho*actualVp*rr[:,10],info,vp.shape)
            e=rel(independent,a['gradient']);assert e<1e-13
            coeff={key:rel(actual,interp(grid,info)) for key,actual,grid in
                [('rho',rho,training['rho_fixed']),('vs',actualVs,training['vs_fixed']),('vp',actualVp,vp)]}
            assert max(coeff.values())<1e-13
            watermat=protocol['water_material'];assert fluid.sum()>0
            assert np.allclose(r[fluid,6],watermat['rho'],rtol=1e-13) and np.count_nonzero(r[fluid,8])==0
            assert np.allclose(r[fluid,7]/r[fluid,6],watermat['vp']**2,rtol=1e-13)
            assert np.count_nonzero(r[fluid,9:12])==0
            static.append(dict(shot_id=sid,raw_rows=len(r),solid_rows=int(solid.sum()),fluid_rows=int(fluid.sum()),
                raw_to_Vp_relative_l2=e,actual_GLL_vs_input_interpolation=coeff,negative_lambda_count=int((lam<0).sum()),
                positive_bulk3d_min_pa=float(np.min(lam+2*mu/3)),fixed_water_raw_gradient_exact_zero=True))
        rows.append(dict(shot_id=sid,model_sha256=meta['model_sha256'],source_sha256=meta['source_sha256'],
            target_sha256=meta['target_sha256'],cache_sha256=sha(path),data_shape=list(a['data'].shape),
            native_samples_per_shot=int(a['data'].size),loss_recomputed=value,native_probes=probes,
            history_bytes=meta.get('history_bytes')))
    aggregate=arrays(directory/'aggregate.npz');expected_loss=float(sum(losses)/n)
    assert bits(aggregate['vp'],vp)
    assert abs(float(aggregate['loss'])-expected_loss)<1e-14*max(abs(expected_loss),1e-300)
    gradient=np.sum(gs,axis=0)/n if gs else None
    gradient_error=rel(aggregate['gradient'],gradient) if gs else None
    if gs:assert gradient_error<1e-14
    joint_error=rel(np.sum(joint,axis=0),gradient) if joint else None
    if joint:assert joint_error<1e-12
    return dict(shot_count=n,one_global_normalizer=n,recomputed_mean_loss=expected_loss,
        max_per_shot_fixed_RMS_relative_error=max(scale_errors),
        max_closed_form_per_shot_cot_relative_l2=max(cot_errors) if cot_errors else None,
        independent_sum_then_mean_gradient_relative_l2=gradient_error,
        independent_native_joint_cot_adjoints_vs_mean_gradient_relative_l2=joint_error,
        dense_gradient_shape=list(aggregate['gradient'].shape),dense_gradient_sha256=ah(aggregate['gradient']),
        rows=rows,raw_static_map=static),aggregate,gradient

def accepted_audit(run,prepared,p,t,s):
    directory=run/'epochs'/f"{s['id']}_e000";accepted=read(directory/'accepted.json');state=accepted['next_state']
    assert state['updates']==1 and sha(run/state['array_path'])==state['array_sha256']
    post=arrays(run/state['array_path']);before=t['vp_initial'];after=post['vp']
    assert accepted['model_before_sha256']==fingerprint(before,t['vs_fixed'],t['rho_fixed'])
    assert accepted['model_after_sha256']==state['model_sha256']==fingerprint(after,t['vs_fixed'],t['rho_fixed'])
    gr,ga,g=audit_group(run,prepared,p,t,s,directory/'gradient',before)
    tid=accepted['candidate_evaluation_id'];trial=int(tid.rsplit('trial',1)[1]);trialdir=directory/f'trial_{trial:02d}'
    tr,ta,_=audit_group(run,prepared,p,t,s,trialdir,after)
    decision=read(trialdir/'decision.json');direction=arrays(directory/'direction.npz')['direction']
    expected=np.clip(before+decision['alpha']*direction,np.sqrt(4/3)*t['vs_fixed']*p['optimizer']['bulk_margin'],p['optimizer']['vp_upper_m_s'])
    assert bits(expected,after)
    slope=float(np.vdot(g,after-before));rhs=float(ga['loss'])+p['optimizer']['armijo_c1']*slope
    assert slope<0 and float(ta['loss'])<=rhs and abs(rhs-decision['armijo_rhs'])<1e-15
    step=float(np.max(np.abs(after-before)));assert step<=p['optimizer']['max_vp_step_m_s']*(1+1e-12)
    ledgerpath=run/('optimizer_ledger.jsonl' if p['unit'] else 'ledger.jsonl')
    ledger=[json.loads(line) for line in ledgerpath.read_text().splitlines() if line.strip()]
    events=[r for r in ledger if r['iteration_id']==accepted['iteration_id']]
    updates=[r for r in events if r['event']=='accepted_update'];assert len(updates)==1
    index=next(i for i,r in enumerate(events) if r['event']=='accepted_update')
    grad=[r for r in events if r['event']=='evaluation_start' and r['phase']=='gradient'];assert len(grad)==1
    gid=grad[0]['evaluation_id'];seen=[]
    for i,r in enumerate(events):
        if r['event']=='shot_batch' and r['evaluation_id']==gid:
            assert i<index and r['model_sha256']==accepted['model_before_sha256'];seen+=r['shot_ids']
    assert sorted(seen)==p['shot_ids'] and len(seen)==len(p['shot_ids'])
    assert any(r['event']=='evaluation_complete' and r['evaluation_id']==tid and i<index for i,r in enumerate(events))
    return dict(passed=True,scope='Immutable first accepted all-shot iteration only; not completion of declared schedule',
        gradient=gr,candidate=tr,model_before_sha256=accepted['model_before_sha256'],model_after_sha256=accepted['model_after_sha256'],
        checkpoint_sha256=state['array_sha256'],gradient_dot_actual_delta_vp=slope,Armijo_rhs_recomputed=rhs,
        actual_max_delta_vp_m_s=step,relative_loss_decrease=(float(ga['loss'])-float(ta['loss']))/float(ga['loss']),
        only_one_accepted_update=True,all_shots_same_model_before_update=True,actual_candidate_equals_clipped_declared_direction_uint64=True,
        ledger_path=str(ledgerpath),ledger_sha256=sha(ledgerpath),first_iteration_ledger=events),g

reports={}
for name in (options.case or ('psv_fwi_unit_20261108_final','psv_fwi_roi_20261108')):
    run,prepared,p,t,s,provenance=base_audit(name)
    accepted,g=accepted_audit(run,prepared,p,t,s)
    report=dict(passed=True,case=name,provenance=provenance,first_accepted_iteration=accepted,
        scope='Read-only NumPy audit, no solver/GPU/temporal AD/FD rerun; model arrays and raw native coefficient columns remain remote.')
    if p['unit']:
        validation=read(run/'unit_validation.json')
        unit,aggregate,ug=audit_group(run,prepared,p,t,s,run/'unit_gradient',t['vp_initial'],raw=True)
        assert rel(g,ug)<1e-13
        xx,zz=np.meshgrid(t['x_native'],t['z_native'])
        direction=t['vp_initial']*(.45+.2*np.cos(xx/max(t['x_native'][-1],1)*2*np.pi)*np.sin(zz/max(t['z_native'][-1],1)*np.pi))
        rhs=float(np.vdot(direction,ug));fd=[]
        for row in validation['full_batch_model_dot']:
            assert abs(rhs-row['rhs'])<1e-13*max(abs(rhs),1e-300)
            relative=abs(row['lhs']-rhs)/max(abs(row['lhs']),abs(rhs))
            assert relative<1e-5
            fd.append(dict(h=row['h'],LHS_reported_not_raw_recomputed=row['lhs'],RHS_independently_recomputed=rhs,relative_error=relative))
        order=math.log(fd[0]['relative_error']/fd[1]['relative_error'])/math.log(fd[0]['h']/fd[1]['h'])
        fd_logs=sorted(run.glob('native_logs/unit_fd_*/unit/shot_*/forward.log'))
        baseline_logs=sorted(run.glob('native_logs/unit_original_baseline/unit/shot_*/forward.log'))
        assert len(fd_logs)==12 and len(baseline_logs)==3
        report.update(unit_gradient=unit,unit_vs_optimizer_anchor_gradient_relative_l2=rel(g,ug),
            joint_loss_vs_independent_numpy_mean_relative_error=abs(validation['joint_AD_loss']-unit['recomputed_mean_loss'])/unit['recomputed_mean_loss'],
            model_dot=dict(passed=True,direction='Executed source analytic physical Vp direction reconstructed from fixed input axes and initial Vp',checks=fd,observed_error_order=order,actual_forward_log_count=len(fd_logs),
                evidence_limit='Original ±FD trace arrays were not archived by the unit runner. RHS/order and actual per-source forward logs are independently checked; LHS values remain executed-source report evidence and are explicitly not claimed as independently recomputed raw traces.'),
            baseline=dict(executed_report_uint64_equal=validation['original_native_baseline_bits_equal'],actual_original_native_forward_log_count=len(baseline_logs),
                evidence_limit='Original baseline arrays are not archived in this unit namespace. Source uses actual uint64 comparisons; independent array re-comparison is unavailable.'),
            derivative_qualification_ledger=[json.loads(line) for line in (run/'ledger.jsonl').read_text().splitlines() if line.strip()],
            source_runtime_unit_validation_sha256=sha(run/'unit_validation.json'))
    reports[name]=report
(OUT/'independent_actual_native_fwi_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps({'directory':str(OUT),'reports':reports},indent=2))
