#!/usr/bin/env python3
"""Exercise the actual v1/v2 controller on a model-free small-step quadratic.

SPDX-License-Identifier: GPL-3.0-or-later
This analytic fixture tests optimizer control only, not native propagation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def load(filename):
    path=ROOT/'scripts'/filename
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


def probe(module,root):
    prepared=root/'prepared';prepared.mkdir(parents=True)
    protocol=dict(optimizer=dict(memory=5,max_vp_step_m_s=60.,max_backtracks=5,
        armijo_c1=1.e-4,gradient_relative_stop=1.e-3,stagnation_epochs=3,
        stagnation_relative_loss=1.e-4,stagnation_step_m_s=.5),
        stages=[dict(id='quadratic',epochs=2)])
    module.jsave(prepared/'protocol.json',protocol)

    class Quadratic:
        def __init__(self):
            self.root=root;self.prepared=prepared;self.protocol=protocol
            self.shots=[0,1,2];self.lower=np.array([[2000.]]);self.upper=6000.
            self.initial=np.array([[2500.]])
            self.events=[];self.evaluations=[]
        def fingerprint(self,vp):return hashlib.sha256(np.ascontiguousarray(vp).tobytes()).hexdigest()
        def event(self,event,**values):self.events.append(dict(event=event,**values))
        def batch(self,vp,stage,directory,iteration_id,evaluation_id,gradient=False):
            # Actual 3-shot mean: weights differ, so summing/counting matters.
            residual=vp-2500.125;weights=[.5,1.,1.5]
            losses=[float(.5*w*np.vdot(residual,residual)) for w in weights]
            gradients=[w*residual for w in weights]
            value=sum(losses)/3;g=sum(gradients)/3
            self.evaluations.append(dict(model_sha256=self.fingerprint(vp),shot_ids=self.shots,
                gradient=gradient,loss=value,all_shots_same_model=True))
            return value,g

    engine=Quadratic();module.jsave(root/'runtime_signature.json',{'fixture':'analytic-control-only'})
    # A valid curvature pair is reconstructed by optimize from two preceding
    # accepted anchors: s=.125, y=.125. The exact quasi-Newton step is +.125.
    state=dict(stage_index=0,epoch_index=1,updates=1,status='running',
        initial_gradient_norm=1.,recent=[])
    module.save_state(engine,state,engine.initial,np.array([[2499.875]]),np.array([[-.25]]),[],[])
    result=module.optimize(engine)
    accepted=root/'epochs/quadratic_e001/accepted.json'
    record=module.read(accepted) if accepted.exists() else None
    return dict(state=result,accepted=record is not None,
        step_m_s=None if record is None else record['decision']['step_linf_m_s'],
        loss_after=None if record is None else record['loss_after'],evaluations=engine.evaluations)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();old,old_path=load('invert_vponly_psv.py');new,new_path=load('invert_vponly_psv_v2.py')
    with tempfile.TemporaryDirectory(prefix='psv-v2-optimizer-') as tmp:
        old_result=probe(old,Path(tmp)/'v1');new_result=probe(new,Path(tmp)/'v2')
    assert old_result['state']['status']=='line_search_stalled' and not old_result['accepted']
    assert new_result['accepted'] and new_result['step_m_s']==.125 and new_result['loss_after']==0.
    first,first_meta=new.scale_direction(np.array([[.125]]),False,60.)
    small,small_meta=new.scale_direction(np.array([[.125]]),True,60.)
    large,large_meta=new.scale_direction(np.array([[120.]]),True,60.)
    assert first.item()==60. and small.item()==.125 and large.item()==60.
    report=dict(passed=True,scope='Analytic three-shot controller fixture; no native solver or target model used',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [old_path,new_path,Path(__file__)]},
        natural_quasi_newton_step_m_s=.125,v1=old_result,v2=new_result,
        policies=dict(no_history=first_meta,small_history=small_meta,large_history=large_meta))
    if args.output:new.jsave(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
