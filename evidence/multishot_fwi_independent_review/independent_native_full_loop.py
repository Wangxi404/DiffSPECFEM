"""Actual Native optimizer/cache control with analytic tiny CPU observations.

The native wave solver is never invoked. This tests the surrounding production
loop; actual native derivative/unit qualification is a separate evidence item.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--skill-root',type=Path,help='differentiate-seismic-solver skill directory')
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args()
REPO=options.repo_root.resolve()
SKILL=(options.skill_root or REPO.parent/'SeismicDiffSkill/skills/differentiate-seismic-solver').resolve()
SOURCE=REPO/'scripts/invert_vponly_psv.py'
SPEC=importlib.util.spec_from_file_location('actual_native_loop',SOURCE)
RUN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RUN)
SOURCE_SHA=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='native_real_loop_cpu_'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)
PREP=OUT/'prepared';PREP.mkdir()
vp0=np.full((3,4),2500.);vs=np.full_like(vp0,1200.);rho=np.full_like(vp0,2250.)
RUN.nsave(PREP/'training_input.npz',vp_initial=vp0,vs_fixed=vs,rho_fixed=rho,x_native=np.arange(4.),z_native=np.arange(3.),active_mask=np.ones_like(vp0,dtype=np.uint8))
stages=[]
for i in range(2):
    stage={'id':f's{i}','epochs':2,'factor':1+.3*i,'case_hashes':{}}
    manifest={}
    for sid in range(3):
        source=PREP/'cases'/stage['id']/f'shot_{sid:03d}/DATA/SOURCE'
        source.parent.mkdir(parents=True);source.write_text(f'independent-analytic-source-{sid}\n')
        stage['case_hashes'][str(sid)]={'SOURCE':RUN.digest(source)}
        x=np.arange(vp0.size).reshape(vp0.shape)
        weight=(.2+(x+sid+1)**1.3)/(15.+sid)*stage['factor']
        operator=np.sqrt(weight)
        target=2500.+30*np.sin(x*.4)+10*sid
        obs=PREP/'observations'/stage['id']/f'shot_{sid:03d}.npz'
        RUN.nsave(obs,observed=operator*target,sample_time=np.arange(4.),scale=np.array(100.))
        manifest[str(sid)]={'sha256':RUN.digest(obs),'source_sha256':RUN.digest(source)}
    RUN.jsave(PREP/'observations'/stage['id']/'manifest.json',manifest)
    stages.append(stage)
protocol={'shot_ids':[0,1,2],'stages':stages,'training_input_sha256':RUN.digest(PREP/'training_input.npz'),
    'original_x_origin_m':0.,'bottom_depth_m':3.,'water_depth_m':0.,
    'batch_size':2,'max_history_gb':1.,'optimizer':{'bulk_margin':1.01,'vp_upper_m_s':6000.,'memory':5,
        'max_vp_step_m_s':60.,'max_backtracks':8,'armijo_c1':1e-4,'gradient_relative_stop':1e-6,
        'stagnation_relative_loss':1e-7,'stagnation_step_m_s':.01,'stagnation_epochs':3}}
RUN.jsave(PREP/'protocol.json',protocol)
(OUT/'mock_binary').write_bytes(b'fixture-only-no-execution')

class InjectedPause(RuntimeError):pass

class AnalyticEngine(RUN.Engine):
    pause_partial=False
    pause_commit=False
    native_calls=0
    def runtime_signature(self):
        # Actual binary hashing method is source-reviewed; no binaries are
        # present in this model-free fixture. Bind a local inert file instead.
        return {'driver':SOURCE_SHA,'inert_runtime_fixture':RUN.digest(OUT/'mock_binary')}
    def native(self,vp,stage,sid,gradient=False,baseline=False,label='evaluation'):
        AnalyticEngine.native_calls+=1
        x=np.arange(vp.size).reshape(vp.shape)
        weight=(.2+(x+sid+1)**1.3)/(15.+sid)*stage['factor']
        operator=np.sqrt(weight);data=operator*vp
        observed=self.observation(stage,sid);scale=float(observed['scale'])
        residual=data-observed['observed'];cot=residual/(scale*scale*vp.size)
        result={'data':data,'sample_time':np.arange(4.),'native_calls':[]}
        if gradient:result.update(loss=float(.5*np.mean((residual/scale)**2)),gradient=operator*cot,
            cotangent=cot,history_bytes=0)
        return result
    def event(self,event,**values):
        if event=='shot_batch' and values.get('evaluation_id','').endswith('_gradient') and AnalyticEngine.pause_partial:
            AnalyticEngine.pause_partial=False
            raise InjectedPause('Two exact-model shot caches complete before partial-batch ledger append')
        if event=='accepted_update' and AnalyticEngine.pause_commit:
            identifier=values['iteration_id']
            assert (self.root/'epochs'/identifier/'accepted.json').exists()
            AnalyticEngine.pause_commit=False
            raise InjectedPause('Durable accepted artifact written before global pointer/event')
        return super().event(event,**values)

first=AnalyticEngine(PREP,OUT/'uninterrupted')
first_state=RUN.optimize(first)
AnalyticEngine.pause_partial=True
AnalyticEngine.pause_commit=True
for pause in range(2):
    try:RUN.optimize(AnalyticEngine(PREP,OUT/'resumed'))
    except InjectedPause:pass
    else:raise AssertionError('Required partial/cache or durable-commit branch not reached')
    assert json.loads((OUT/'resumed/state.json').read_text())['updates']==0
last=AnalyticEngine(PREP,OUT/'resumed');last_state=RUN.optimize(last)

spec=importlib.util.spec_from_file_location('checker',SKILL/'scripts/check_iterations.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
details={}
for name in ('uninterrupted','resumed'):
    root=OUT/name;state=RUN.read(root/'state.json')
    events=[json.loads(line) for line in (root/'ledger.jsonl').read_text().splitlines()]
    audit=checker.audit_events(events)
    assert audit['passed'] and audit['accepted_updates']==4
    assert state['updates']==4 and state['status']=='epoch_cap'
    slopes=[];stage_first_history={}
    for accepted_file in sorted((root/'epochs').glob('*/accepted.json')):
        accepted=RUN.read(accepted_file);directory=accepted_file.parent
        with np.load(directory/'gradient/aggregate.npz') as a:
            g=a['gradient'];anchor=a['vp'];value=float(a['loss'])
        trial_id=accepted['candidate_evaluation_id'];trial=int(trial_id[-2:])
        with np.load(directory/f'trial_{trial:02d}/aggregate.npz') as a:candidate=a['vp'];new_loss=float(a['loss'])
        slope=float(np.vdot(g,candidate-anchor));slopes.append(slope)
        assert new_loss<=value+protocol['optimizer']['armijo_c1']*slope
        assert np.max(np.abs(candidate-anchor))<=60*(1+1e-10)
        stage_id=directory.name.split('_')[0]
        with np.load(root/accepted['next_state']['array_path']) as a:history=len(a['s'])
        stage_first_history.setdefault(stage_id,history)
        batches=[row['shot_ids'] for row in events if row['iteration_id']==directory.name and row['event']=='shot_batch' and row['evaluation_id'].endswith('_gradient')]
        assert batches==[[0,1],[2]]
        shot_gradients=[];shot_losses=[]
        for sid in (0,1,2):
            with np.load(directory/f'gradient/shot_{sid:03d}.npz') as a:
                shot_gradients.append(a['gradient']);shot_losses.append(float(a['loss']))
        assert np.array_equal(np.mean(shot_gradients,axis=0),g)
        assert np.mean(shot_losses)==value
    assert stage_first_history=={'s0':0,'s1':0}
    details[name]={'ledger':audit,'status':state['status'],'stage_first_history_counts':stage_first_history,
        'accepted_actual_delta_slopes':slopes}
with np.load(OUT/'uninterrupted'/first_state['array_path']) as a,np.load(OUT/'resumed'/last_state['array_path']) as b:
    exact=bool(np.array_equal(a['vp'].view(np.uint64),b['vp'].view(np.uint64)))
assert exact
class ZeroEngine(AnalyticEngine):
    def native(self,vp,stage,sid,gradient=False,baseline=False,label='evaluation'):
        obs=self.observation(stage,sid)
        result={'data':obs['observed'],'sample_time':np.arange(4.),'native_calls':[]}
        if gradient:result.update(loss=0.,gradient=np.zeros_like(vp),cotangent=np.zeros_like(obs['observed']),history_bytes=0)
        return result
zero_engine=ZeroEngine(PREP,OUT/'zero_gradient');zero_state=RUN.optimize(zero_engine)
zero_audit=checker.audit_events(zero_engine.events)
assert zero_state['status']=='data_gradient_stationarity' and zero_state['updates']==0
assert zero_audit['iteration_statuses']['converged']==2
class AtLowerEngine(AnalyticEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.lower=self.initial.copy()
    def native(self,vp,stage,sid,gradient=False,baseline=False,label='evaluation'):
        result=super().native(vp,stage,sid,gradient,baseline,label)
        if gradient:result.update(loss=1.,gradient=np.ones_like(vp)*1e-8)
        return result
projected_engine=AtLowerEngine(PREP,OUT/'projected_stationary');projected_state=RUN.optimize(projected_engine)
projected_audit=checker.audit_events(projected_engine.events)
assert projected_state['status']=='data_gradient_stationarity' and projected_state['updates']==0
assert projected_audit['iteration_statuses']['converged']==2

# Actual final binding/cache methods must reject changed targets/cases/runtime.
stage=stages[0]
def must_reject(operation):
    try:operation()
    except ValueError as error:return {'rejected':True,'message':str(error)}
    raise AssertionError('Changed immutable provenance was accepted')
obs=PREP/'observations/s0/shot_000.npz';manifest=PREP/'observations/s0/manifest.json'
saved_obs,saved_manifest=obs.read_bytes(),manifest.read_bytes()
RUN.nsave(obs,observed=np.ones_like(vp0)*3,sample_time=np.arange(4.),scale=np.array(100.))
m=RUN.read(manifest);m['0']['sha256']=RUN.digest(obs);RUN.jsave(manifest,m)
target_rejection=must_reject(lambda:last.batch(vp0,stage,OUT/'resumed/epochs/s0_e000/gradient','s0_e000','s0_e000_gradient',gradient=True))
obs.write_bytes(saved_obs);manifest.write_bytes(saved_manifest)
source=PREP/'cases/s0/shot_000/DATA/SOURCE';saved_source=source.read_bytes();source.write_text('CHANGED SOURCE\n')
source_rejection=must_reject(lambda:last.bind_targets(stage));source.write_bytes(saved_source)
(OUT/'mock_binary').write_bytes(b'CHANGED RUNTIME')
runtime_rejection=must_reject(lambda:AnalyticEngine(PREP,OUT/'resumed'))

report={'passed':True,'scope':'Actual optimize/batch/Armijo/L-BFGS/stage/cache/checkpoint/ledger control flow with an analytic tiny CPU observation engine. No native wave solver, MPI or GPU process launched; derivative physics remains separately qualified.',
    'source_sha256':SOURCE_SHA,'directory':str(OUT),'uninterrupted_vs_two_interruption_resume_final_Vp_uint64_equal':exact,
    'checks':details,'immutable_target_rejection':target_rejection,'immutable_source_rejection':source_rejection,
    'immutable_runtime_rejection':runtime_rejection,'actual_analytic_forward_calls':AnalyticEngine.native_calls,
    'zero_gradient_no_update':{'status':zero_state['status'],'ledger':zero_audit},
    'projected_stationarity_no_update':{'status':projected_state['status'],'ledger':projected_audit},
    'limitation':'Native forward/gradient and binary-file availability are mocked solely for surrounding control flow. This does not substitute for actual native unit and directional checks.'}
(OUT/'independent_native_full_loop.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
