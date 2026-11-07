#!/usr/bin/env python3
"""Check-only independent-shot FD replay, preserving every raw receiver array.

SPDX-License-Identifier: GPL-3.0-or-later
No optimizer, adjoint recomputation, target-model read, or production-gradient
replacement. Run only when the production MPI workers are stopped or absent.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.invert_vponly_psv import Engine,jsave,nsave
from scripts.experiment_gradient_detail_psv import digest
from scripts.experiment_vponly_psv import array_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--anchor',type=Path,required=True,help='Completed immutable gradient evaluation directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--unit-direction',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve every earlier check attempt')
    args.output.mkdir(parents=True)
    started=time.perf_counter();engine=Engine(args.prepared,args.output)
    stage=engine.protocol['stages'][0]
    with np.load(args.anchor/'aggregate.npz',allow_pickle=False) as a:
        vp=a['vp'].copy();gradient=a['gradient'].copy()
    if args.unit_direction:
        xx,zz=np.meshgrid(engine.x,engine.z)
        direction=vp*(.45+.2*np.cos(xx/max(engine.x[-1],1)*2*np.pi)*np.sin(zz/max(engine.z[-1],1)*np.pi))
        steps=[.0003,.0001]
    else:
        direction=gradient/np.max(np.abs(gradient));steps=[.1,.05]
    rhs=float(np.vdot(gradient,direction));cosine=abs(rhs)/(np.linalg.norm(gradient)*np.linalg.norm(direction))
    if not np.isfinite(rhs) or cosine<1.e-4:raise ValueError('Uninformative FD direction')
    nsave(args.output/'anchor_and_direction.npz',vp=vp,gradient=gradient,direction=direction,x_native=engine.x,z_native=engine.z)
    report=dict(role='Fresh check-only native replay after the production gradient; no adjoint or update',
        passed=False,stage=stage['id'],shot_ids=engine.shots,normalization=engine.protocol['normalization'],
        model_sha256=engine.fingerprint(vp),vp_sha256=array_hash(vp),direction_sha256=array_hash(direction),
        direction_max_abs=float(np.max(abs(direction))),direction_gradient_abs_cosine=float(cosine),rhs=rhs,
        tolerance_relative=1.e-5,absolute_tolerance=0.,checks=[],anchor_target_source_hashes={},
        driver_sha256=digest(ROOT/'scripts/invert_vponly_psv.py'),checker_sha256=digest(__file__),
        production_gradient_recomputed=False,unknown_true_vp_read=False)
    sources={}
    for sid in engine.shots:
        meta=json.loads((args.anchor/f'shot_{sid:03d}.json').read_text())
        if meta['model_sha256']!=engine.fingerprint(vp):raise ValueError('Anchor shot was evaluated at another model')
        with np.load(args.anchor/f'shot_{sid:03d}.npz',allow_pickle=False) as data:
            sources[sid]=dict(data=data['data'].copy(),cotangent=data['cotangent'].copy(),sample_time=data['sample_time'].copy())
        obs=engine.observation(stage,sid)
        expected=(sources[sid]['data']-obs['observed'])/(sources[sid]['data'].size*float(obs['scale'])**2)
        if np.linalg.norm(expected-sources[sid]['cotangent'])>1.e-12*np.linalg.norm(expected):raise AssertionError('Anchor AD cotangent mismatch')
        report['anchor_target_source_hashes'][str(sid)]=dict(anchor_array_sha256=digest(args.anchor/f'shot_{sid:03d}.npz'),target_sha256=meta['target_sha256'],source_sha256=meta['source_sha256'])
        nsave(args.output/f'anchor_shot_{sid:03d}.npz',**sources[sid])
    for h in steps:
        results={};losses={}
        for sign,multiplier in [('plus',1),('minus',-1)]:
            model=vp+multiplier*h*direction;directory=args.output/(f'h{h}_{sign}');directory.mkdir()
            results[sign]=[];losses[sign]=[]
            for sid in engine.shots:
                out=engine.native(model,stage,sid,label=f'FD_h{h}_{sign}')
                observation=engine.observation(stage,sid)
                if not np.array_equal(out['sample_time'],sources[sid]['sample_time']):raise AssertionError('FD sample times changed')
                results[sign].append(out['data'])
                losses[sign].append(float(.5*np.mean(((out['data']-observation['observed'])/float(observation['scale']))**2)))
                path=directory/f'shot_{sid:03d}.npz';nsave(path,data=out['data'],sample_time=out['sample_time'])
                jsave(directory/f'shot_{sid:03d}.json',dict(shot_id=sid,model_sha256=out['model_sha256'],vp_sha256=out['vp_sha256'],
                    array_sha256=digest(path),target_sha256=report['anchor_target_source_hashes'][str(sid)]['target_sha256'],
                    source_sha256=report['anchor_target_source_hashes'][str(sid)]['source_sha256'],native_calls=out['native_calls'],objective_value=losses[sign][-1]))
        lhs=float(sum(np.vdot((a-b)/(2*h),sources[sid]['cotangent']) for sid,a,b in zip(engine.shots,results['plus'],results['minus']))/len(engine.shots))
        scalar=(float(np.mean(losses['plus']))-float(np.mean(losses['minus'])))/(2*h)
        error=abs(lhs-rhs)/max(abs(lhs),abs(rhs));scalar_error=abs(scalar-rhs)/max(abs(scalar),abs(rhs))
        report['checks'].append(dict(h=h,maximum_vp_perturbation_m_s=h*float(np.max(abs(direction))),lhs_JVP=lhs,rhs=rhs,relative_error=error,
            centered_scalar_objective_derivative=scalar,scalar_relative_error=scalar_error,passed=bool(error<1.e-5 and scalar_error<1.e-5),
            plus_full_shot_mean=float(np.mean(losses['plus'])),minus_full_shot_mean=float(np.mean(losses['minus']))))
        report['elapsed_seconds']=time.perf_counter()-started;jsave(args.output/'verification.json',report)
    report['passed']=all(row['passed'] for row in report['checks']);report['elapsed_seconds']=time.perf_counter()-started
    jsave(args.output/'verification.json',report)
    jsave(args.output/'raw_manifest.json',{str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(args.output.rglob('*')) if p.is_file() and p.suffix in ('.npz','.json') and p.name!='raw_manifest.json'})
    print(json.dumps(report),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
