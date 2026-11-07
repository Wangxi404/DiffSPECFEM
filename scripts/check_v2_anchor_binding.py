#!/usr/bin/env python3
"""Bind a freshly recomputed v2 initial gradient to retained v1 raw FD proof.

SPDX-License-Identifier: GPL-3.0-or-later
This reader performs no native simulation and never imports the optimizer.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def function_ast(path,class_name,function_name):
    tree=ast.parse(Path(path).read_text())
    body=tree.body if class_name is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==function_name)
    return ast.dump(node,include_attributes=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['old-run','new-run','old-prepared','new-prepared','fd-proof','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();old=args.old_run;new=args.new_run;epoch='coarse_e000'
    old_source=old/'driver_executed.py';new_source=new/'driver_executed.py'
    fd=read(args.fd_proof);assert fd['passed'] and fd['driver_sha256']==digest(old_source)
    old_runtime=read(old/'runtime_signature.json');new_runtime=read(new/'runtime_signature.json')
    assert {k:v for k,v in old_runtime.items() if k!='scripts/invert_vponly_psv.py'}=={k:v for k,v in new_runtime.items() if k!='scripts/invert_vponly_psv_v2.py'}
    functions={}
    for cls,name in [('Engine','native'),('Engine','batch'),('Engine','observation'),('Engine','bind_targets'),('Engine','case'),(None,'lbfgs_direction')]:
        key=(cls+'.' if cls else '')+name
        equal=function_ast(old_source,cls,name)==function_ast(new_source,cls,name)
        assert equal,key;functions[key]=dict(equal=True,ast_sha256=hashlib.sha256(function_ast(new_source,cls,name).encode()).hexdigest())
    for filename in ['training_input.npz','truth_for_observations_only.npz']:
        assert digest(args.old_prepared/filename)==digest(args.new_prepared/filename)
    with np.load(args.new_prepared/'training_input.npz',allow_pickle=False) as f:
        assert set(f.files)=={'vp_initial','vs_fixed','rho_fixed','x_native','z_native','active_mask'}
    shots=read(args.new_prepared/'protocol.json')['shot_ids'];checks=[]
    for sid in shots:
        stem=f'shot_{sid:03d}';oldp=old/'epochs'/epoch/'gradient';newp=new/'epochs'/epoch/'gradient'
        a=read(oldp/(stem+'.json'));b=read(newp/(stem+'.json'))
        reference=fd['anchor_target_source_hashes'][str(sid)]
        assert reference['anchor_array_sha256']==digest(oldp/(stem+'.npz'))
        assert reference['source_sha256']==b['source_sha256'] and reference['target_sha256']==b['target_sha256']
        assert fd['model_sha256']==b['model_sha256']
        for key in ['model_sha256','vp_sha256','source_sha256','target_sha256','vs_fixed_sha256','rho_fixed_sha256','stage_id']:
            assert a[key]==b[key],(sid,key)
        assert b['gradient_requested'] is True
        assert digest(args.new_prepared/'observations/coarse'/(stem+'.npz'))==b['target_sha256']
        assert digest(args.new_prepared/'cases/coarse'/stem/'DATA/SOURCE')==b['source_sha256']
        assert b['array_sha256']==digest(newp/(stem+'.npz'))
        modes=[c['mode'] for c in b['native_calls']]
        assert modes.count('record')==1 and modes.count('adjoint')==1
        assert a['runtime_signature_sha256']!=b['runtime_signature_sha256']
        with np.load(oldp/(stem+'.npz'),allow_pickle=False) as x,np.load(newp/(stem+'.npz'),allow_pickle=False) as y:
            keys=['data','cotangent','gradient','loss','sample_time']
            for key in keys:
                assert x[key].dtype==y[key].dtype and x[key].shape==y[key].shape
                assert np.array_equal(x[key].view(np.uint64),y[key].view(np.uint64)),(sid,key)
        checks.append(dict(shot_id=sid,anchor_sha256=b['model_sha256'],source_sha256=b['source_sha256'],target_sha256=b['target_sha256'],fresh_native_record=1,fresh_native_adjoint=1,exact_uint64_fields=keys))
    aggregate={}
    with np.load(old/'epochs'/epoch/'gradient/aggregate.npz',allow_pickle=False) as a,np.load(new/'epochs'/epoch/'gradient/aggregate.npz',allow_pickle=False) as b:
        for key in ['vp','gradient_sum','gradient','loss_sum','loss']:
            assert np.array_equal(a[key].view(np.uint64),b[key].view(np.uint64)),key
            aggregate[key]=hashlib.sha256(b[key].tobytes()).hexdigest()
    report=dict(passed=True,scope='Fresh initial coarse anchor only; unchanged native/data/gradient functions and exact physical inputs bind the retained two-step raw-native FD evidence. Does not certify all later iterates or replace their shot/physical gates.',
        old_driver_sha256=digest(old_source),new_driver_sha256=digest(new_source),
        reader_sha256=digest(__file__),unchanged_functions=functions,shots=checks,
        exact_aggregate_field_sha256=aggregate,retained_fd_proof_sha256=digest(args.fd_proof),
        retained_fd_checks=fd['checks'],noncontroller_runtime_hashes_equal=True,
        training_input_contains_unknown_target_vp=False,no_gradient_cache_copied=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
