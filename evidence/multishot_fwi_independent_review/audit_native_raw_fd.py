"""Independently recompute a frozen five-shot native FD qualification.

Reads existing arrays only. No native solver, adjoint, optimizer or GPU runs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--fd-run',type=Path,default=Path('run_psv_fwi_roi_fd_20261108'))
parser.add_argument('--anchor-run',type=Path,default=Path('run_psv_fwi_roi_20261108'))
parser.add_argument('--prepared',type=Path,default=Path('data/inversion/psv_fwi_roi_20261108'))
parser.add_argument('--epoch',default='coarse_e000')
parser.add_argument('--expected-driver-sha256',default='28eed616dacde81f48eb1d470b927f7969166fe1a74468be88e6dd3ac01716c6')
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args()
ROOT=options.repo_root.resolve()
def resolve(path):return path.resolve() if path.is_absolute() else (ROOT/path).resolve()
FD=resolve(options.fd_run);ANCHOR=resolve(options.anchor_run);PREP=resolve(options.prepared)
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='native-raw-fd-independent-'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {key:f[key].copy() for key in f.files}
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint64),b.view(np.uint64))
def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-300))

verification=read(FD/'verification.json');protocol=read(PREP/'protocol.json')
training=arrays(PREP/'training_input.npz');anchor=arrays(FD/'anchor_and_direction.npz')
aggregate=arrays(ANCHOR/'epochs'/options.epoch/'gradient/aggregate.npz')
assert verification['driver_sha256']==options.expected_driver_sha256
assert sha(ROOT/'scripts/invert_vponly_psv.py')==options.expected_driver_sha256
assert sha(ROOT/'scripts/check_native_fullshot_fd.py')==verification['checker_sha256']
assert verification['tolerance_relative']==1e-5 and verification['absolute_tolerance']==0
assert sha(PREP/'training_input.npz')==protocol['training_input_sha256']
assert 'vp_true' not in training and 'vp' not in training
assert bits(anchor['vp'],aggregate['vp']) and bits(anchor['gradient'],aggregate['gradient'])
assert bits(anchor['vp'],training['vp_initial'])
assert bits(anchor['x_native'],training['x_native']) and bits(anchor['z_native'],training['z_native'])
assert ah(anchor['direction'])==verification['direction_sha256']
assert bits(anchor['direction'],anchor['gradient']/np.max(np.abs(anchor['gradient'])))
fingerprint=lambda vp:ah(np.stack([vp,training['vs_fixed'],training['rho_fixed']]))
assert fingerprint(anchor['vp'])==verification['model_sha256']
assert ah(anchor['vp'])==verification['vp_sha256']
rhs=float(np.vdot(anchor['gradient'],anchor['direction']))
cosine=abs(rhs)/(np.linalg.norm(anchor['gradient'])*np.linalg.norm(anchor['direction']))
assert np.isfinite(rhs) and abs(rhs)>0 and cosine>1e-4
manifest=read(FD/'raw_manifest.json');manifest_bytes=0
for name,item in manifest.items():
    path=FD/name
    assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    manifest_bytes+=item['bytes']
runtime=read(FD/'runtime_signature.json')
assert all(sha(ROOT/name)==expected for name,expected in runtime.items())
assert sha(FD/'protocol_copy.json')==sha(PREP/'protocol.json')
stage=protocol['stages'][0];shot_ids=protocol['shot_ids'];nshots=len(shot_ids)
assert shot_ids==verification['shot_ids'] and nshots==5
observations={};sources={};anchors=[];gradient_sum=np.zeros_like(anchor['gradient'])
for sid in shot_ids:
    for file,expected in stage['case_hashes'][str(sid)].items():
        assert sha(PREP/'cases'/stage['id']/f'shot_{sid:03d}'/'DATA'/file)==expected
    observation_path=PREP/'observations'/stage['id']/f'shot_{sid:03d}.npz'
    obs=arrays(observation_path);observations[sid]=obs
    expected_target=verification['anchor_target_source_hashes'][str(sid)]
    assert sha(observation_path)==expected_target['target_sha256']
    assert expected_target['source_sha256']==stage['case_hashes'][str(sid)]['SOURCE']
    original_path=ANCHOR/'epochs'/options.epoch/'gradient'/f'shot_{sid:03d}.npz'
    original=arrays(original_path);saved=arrays(FD/f'anchor_shot_{sid:03d}.npz')
    assert sha(original_path)==expected_target['anchor_array_sha256']
    assert all(bits(saved[key],original[key]) for key in ('data','cotangent','sample_time'))
    assert bits(saved['sample_time'],obs['sample_time'])
    N=saved['data'].size;scale=float(obs['scale'])
    expected_cot=(saved['data']-obs['observed'])/(N*scale**2)
    cot_error=relative(expected_cot,saved['cotangent']);assert cot_error<1e-12
    assert abs(scale-np.sqrt(np.mean(obs['observed']**2)))/scale<1e-14
    gradient_sum+=original['gradient']
    sources[sid]={'data':saved['data'],'cotangent_independent':expected_cot,'sample_time':saved['sample_time']}
    anchors.append({'shot_id':sid,'shape':list(saved['data'].shape),'full_per_shot_N':N,'cotangent_relative_l2':cot_error,
        'original_anchor_cache_sha256':sha(original_path),'target_sha256':sha(observation_path),
        'source_sha256':expected_target['source_sha256'],'cached_anchor_data_and_cot_uint64_equal_original':True})
assert bits(gradient_sum/nshots,anchor['gradient'])

checks=[];raw_files=[]
for h in (.1,.05):
    loss={'plus':[],'minus':[]};lhs_sum=0.;terms=[]
    models={sign:anchor['vp']+multiplier*h*anchor['direction'] for sign,multiplier in [('plus',1),('minus',-1)]}
    for model in models.values():assert np.all(model**2>4/3*training['vs_fixed']**2)
    for sid in shot_ids:
        data={}
        for sign in ('plus','minus'):
            directory=FD/f'h{h}_{sign}';path=directory/f'shot_{sid:03d}.npz'
            meta=read(directory/f'shot_{sid:03d}.json');a=arrays(path)
            assert sha(path)==meta['array_sha256']
            assert meta['model_sha256']==fingerprint(models[sign]) and meta['vp_sha256']==ah(models[sign])
            assert meta['target_sha256']==anchors[sid]['target_sha256'] and meta['source_sha256']==anchors[sid]['source_sha256']
            assert bits(a['sample_time'],sources[sid]['sample_time'])
            assert a['data'].shape==sources[sid]['data'].shape and np.isfinite(a['data']).all()
            value=.5*float(np.mean(((a['data']-observations[sid]['observed'])/float(observations[sid]['scale']))**2))
            assert abs(value-meta['objective_value'])<1e-13*max(abs(value),1e-300)
            loss[sign].append(value);data[sign]=a['data']
            raw_files.append({'path':str(path.relative_to(FD)),'sha256':sha(path),'model_sha256':meta['model_sha256']})
        jvp=(data['plus']-data['minus'])/(2*h)
        term=float(np.sum(jvp*sources[sid]['cotangent_independent']))
        terms.append(term);lhs_sum+=term
    lhs=lhs_sum/nshots
    plus=float(sum(loss['plus'])/nshots);minus=float(sum(loss['minus'])/nshots)
    scalar=(plus-minus)/(2*h)
    error=abs(lhs-rhs)/max(abs(lhs),abs(rhs));scalar_error=abs(scalar-rhs)/max(abs(scalar),abs(rhs))
    assert error<1e-5 and scalar_error<1e-5
    recorded=next(row for row in verification['checks'] if row['h']==h)
    assert abs(lhs-recorded['lhs_JVP'])<1e-12*abs(rhs)
    assert abs(scalar-recorded['centered_scalar_objective_derivative'])<1e-11*abs(rhs)
    checks.append({'h_m_s':h,'rhs_independently_recomputed':rhs,'raw_full_shot_JVP_cot_lhs':lhs,
        'JVP_relative_error':error,'raw_full_mean_scalar_FD_lhs':scalar,'scalar_relative_error':scalar_error,
        'plus_recomputed_global_mean_loss':plus,'minus_recomputed_global_mean_loss':minus,
        'actual_scalar_loss_difference':plus-minus,'per_shot_JVP_cot_terms_before_global_mean':terms,
        'actual_model_plus_sha256':fingerprint(models['plus']),'actual_model_minus_sha256':fingerprint(models['minus']),
        'maximum_actual_plus_Vp_change_m_s':float(np.max(np.abs(models['plus']-anchor['vp']))),
        'maximum_actual_minus_Vp_change_m_s':float(np.max(np.abs(models['minus']-anchor['vp']))),'passed':True})
calls=[json.loads(line) for line in (FD/'calls/native_calls.jsonl').read_text().splitlines() if line.strip()]
assert not any(row['mode'] in ('adjoint','record') for row in calls)
full_forwards=[row for row in calls if Path(row['log']).name=='forward.log']
assert len(full_forwards)==20 and all(row['ranks']==2 and row['returncode']==0 for row in full_forwards)
assert not (FD/'state.json').exists() and not (FD/'ledger.jsonl').exists()
report={'passed':True,'scope':'Independent read-only recomputation from all twenty actual native ±FD receiver arrays, original five-shot gradient/cotangents and fixed target data. First coarse anchor only; no solver/adjoint/GPU or optimizer run by reviewer.',
    'numerical_driver_sha256':options.expected_driver_sha256,'executed_checker_sha256':verification['checker_sha256'],
    'verification_json_sha256':sha(FD/'verification.json'),'raw_manifest_sha256':sha(FD/'raw_manifest.json'),
    'manifest_files_verified':len(manifest),'manifest_bytes_verified':manifest_bytes,
    'direction_sha256':ah(anchor['direction']),'anchor_vp_sha256':ah(anchor['vp']),'global_mean_gradient_sha256':ah(anchor['gradient']),
    'direction_gradient_abs_cosine':float(cosine),'original_gradient_recomputed':False,
    'original_gradient_sum_then_mean_uint64_equal':True,'unknown_true_Vp_read_by_independent_auditor':False,
    'shot_count':nshots,'full_global_sample_N':sum(row['full_per_shot_N'] for row in anchors),
    'anchors':anchors,'checks':checks,'JVP_error_order':math.log(checks[0]['JVP_relative_error']/checks[1]['JVP_relative_error'])/math.log(2),
    'scalar_error_order':math.log(checks[0]['scalar_relative_error']/checks[1]['scalar_relative_error'])/math.log(2),
    'relative_gate_unchanged':1e-5,'absolute_gate_not_used':True,'twenty_raw_FD_files':raw_files,
    'recorded_native_full_forwards':len(full_forwards),'recorded_native_MPI_ranks':2,
    'no_adjoint_record_or_optimizer_in_check_namespace':True,'output_directory':str(OUT),
    'limits':['This closes the raw finite-difference LHS evidence gap for the first coarse ROI anchor only. Original unit missing-trace limits remain unchanged.','It does not certify later models, the fine-grid stage, the full bounded inversion schedule or geological recovery.']}
(OUT/'independent_native_raw_fd_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
