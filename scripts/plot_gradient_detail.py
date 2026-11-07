#!/usr/bin/env python3
"""Compare actual P-SV gradient-detail experiments without altering derivatives.

SPDX-License-Identifier: GPL-3.0-or-later
Source/mesh changes are separate physical experiments. Display saturation is
reported; no sharpening, smoothing or inserted model texture is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/diffspecfem-gradient-detail-mpl')
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig,axes,bars,output,name):
    fig.canvas.draw()
    for image,ax,label,ticks in bars:
        box=ax.get_position();height=fig.get_figheight()*72
        cax=fig.add_axes([box.x0,box.y0-36/height,box.width,5/height])
        cb=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=ticks,
            format=mpl.ticker.FuncFormatter(lambda value,position:f'{value:.2g}'))
        cb.ax.tick_params(labelsize=7,pad=2);cb.set_label(label,fontsize=7,labelpad=3)
    fig.canvas.draw()
    labels=np.array(list('abcdefghijklmnopqrstuvwxyz')[:axes.size]).reshape(axes.shape)
    require_matplotlib_panel_alignment(fig,axes=axes.ravel().tolist(),panel_ids=labels.ravel().tolist(),
        row_groups=labels.tolist(),column_groups=labels.T.tolist(),
        json_out=output/'qa'/f'{name}.alignment.json',overlay_svg=output/'qa'/f'{name}.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(output/f'{name}.png',dpi=300)
    fig.savefig(output/f'{name}.pdf',dpi=300)
    fig.savefig(output/f'{name}.svg',dpi=300)
    plt.close(fig)


def domain(ax):
    ax.set(xlim=(5,13.955),ylim=(3.465,0),xlabel='Original x (km)',ylabel='Depth (km)')
    ax.set_xticks([5,8,11,13.955],['5','8','11','13.955']);ax.set_yticks([0,.45,1.5,3])
    ax.set_aspect('equal',adjustable='box');ax.axhline(.45,color='#555555',ls='--',lw=.5)


def source_frequency(directory):
    if not (directory/'SOURCE').exists():
        report=json.loads((directory/'pilot.json').read_text())
        return float(report['accepted_before_run']['source_f0_hz'])
    text=(directory/'SOURCE').read_text()
    return float(re.search(r'(?m)^f0\s*=\s*([0-9.Ee+-]+)',text)[1])


def read_case(directory,kind='l2'):
    report=json.loads((directory/'pilot.json').read_text())
    if not report.get('passed'):
        raise ValueError('Only render checked new experiments as validated results')
    compact=directory/'grid_gradients.npz'
    data=np.load(compact if compact.exists() else directory/f'gradient_{kind}.npz',allow_pickle=False)
    return data['gradient_'+kind] if compact.exists() else data['gradient'],report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'evidence/psv_marmousi128')
    parser.add_argument('--water1',type=Path,required=True)
    parser.add_argument('--best',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=ROOT/'data/prepared/marmousi2_full_psv.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'figures/gradient_detail')
    parser.add_argument('--percentile',type=float,default=99.)
    args=parser.parse_args()
    if not 95<=args.percentile<=100:
        raise ValueError('Declare a conservative display percentile')
    output=args.output;output.mkdir(parents=True,exist_ok=True);(output/'qa').mkdir(exist_ok=True)
    old_validation=json.loads((args.baseline/'validation.json').read_text())
    if not old_validation['passed']:
        raise ValueError('Unverified reference')
    old=np.load(args.baseline/'gradient_l2.npz')['gradient']
    water,water_report=read_case(args.water1)
    best,best_report=read_case(args.best)
    window,unused=read_case(args.water1,'reflection_window_l2')
    if any(g.shape!=(3,67,200) or not np.isfinite(g).all() for g in [old,water,best,window]):
        raise ValueError('All comparisons must use the same physical solid input coordinates')
    model=np.load(args.model,allow_pickle=False)
    x=np.asarray(model['x_original_m'])/1000;depth=np.asarray(model['depth_original_m'])/1000
    gz=depth[11:];scale_region=np.broadcast_to(gz[:,None]>=1.,(67,200))
    contract=dict(claim='Moving the fixed source into water and increasing bandwidth within native mesh guidance changes real subsurface objective sensitivity; raw gradients are retained.',
        backend='python',archetype='image plate + quant',
        input_identity='Same full Marmousi-II prepared fields and sigma150m initial model; source/mesh/frequency changes explicitly separated.',
        integrity='No derivative filtering, illumination correction, depth gain or inserted model texture; all samples retained with declared display saturation.',
        main_panels='true/initial inputs and old/best actual derivatives',
        ablation_panels='solid source, water source, water source plus fixed data window; no claim that window always improves depth sensitivity')
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    records={};fbest=source_frequency(args.best)

    def gradient_map(fig,ax,g,param,letter,text,bars,record_key):
        values=np.flipud(g[param])
        limit=float(np.percentile(abs(values[scale_region]),args.percentile))
        if not np.isfinite(limit) or limit<=0:
            raise ValueError('Empty derivative scale')
        image=ax.pcolormesh(x,gz,values,shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
        ax.axhspan(0,.45,facecolor='#e6e9ed',hatch='///',edgecolor='#c3c9ce',lw=0)
        domain(ax);ax.set_title(f'{letter}  {text}',loc='left',pad=9)
        bars.append((image,ax,'Raw dJ/dm (objective / [m/s])',[-limit,0,limit]))
        records[record_key]=dict(absolute_limit=limit,percentile=args.percentile,
            scale_region='solid input samples at depth>=1000m; display limit only, no samples removed',
            saturated_fraction_all_solid=float(np.mean(abs(values)>limit)),derivative_modified=False)

    fig,axes=plt.subplots(2,4,figsize=(13.7,6.6));bars=[]
    for row,(name,index) in enumerate([('vp',0),('vs',1)]):
        display={'vp':'Vp','vs':'Vs'}[name]
        truth,initial=(np.flipud(model[name+'_'+state])/1000 for state in ['true','initial'])
        lo,hi=float(min(truth.min(),initial.min())),float(max(truth.max(),initial.max()))
        for col,array in enumerate([truth,initial]):
            image=axes[row,col].pcolormesh(x,depth,array,shading='nearest',cmap='viridis',vmin=lo,vmax=hi,rasterized=True)
            domain(axes[row,col]);axes[row,col].set_title(f'{"abcdefgh"[row*4+col]}  {display}: {["True","Initial"][col]}',loc='left',pad=9)
            bars.append((image,axes[row,col],display+' (km/s)',[lo,(lo+hi)/2,hi]))
        gradient_map(fig,axes[row,2],old,index,'abcdefgh'[row*4+2],f'{display}: solid source, 1 Hz',bars,'old_'+name)
        gradient_map(fig,axes[row,3],best,index,'abcdefgh'[row*4+3],f'{display}: water source, {fbest:g} Hz',bars,'best_'+name)
    fig.suptitle('Actual Marmousi-II gradients: source, frequency and mesh experiments\n'
                 'Linear limits from depth ≥ 1 km; all derivative samples retained; gray water is fixed',y=.99,fontsize=10)
    fig.subplots_adjust(left=.055,right=.975,top=.88,bottom=.17,wspace=.4,hspace=.64)
    save(fig,axes,bars,output,'models_and_gradient_detail')

    fig,axes=plt.subplots(2,3,figsize=(11.6,6.5));bars=[]
    for row,(name,index) in enumerate([('vp',0),('vs',1)]):
        display={'vp':'Vp','vs':'Vs'}[name]
        for col,(g,text,key) in enumerate([(old,'Solid source / L2','solid'),(water,'Water source / L2','water'),(window,'Water source / fixed window','window')]):
            gradient_map(fig,axes[row,col],g,index,'abcdef'[row*3+col],f'{display}: {text}',bars,key+'_'+name)
    fig.suptitle('1 Hz acquisition/objective ablation at the same model and mesh\n'
                 'The fixed late/offset window is retained even when it reduces deep sensitivity',y=.99,fontsize=10)
    fig.subplots_adjust(left=.065,right=.975,top=.88,bottom=.17,wspace=.39,hspace=.65)
    save(fig,axes,bars,output,'water_source_and_window_ablation')
    manifest=dict(status='rendered; final independent PDF/visual QA required',gradient_postprocessing='none',
        comparison='Physical source units and energy differ; no gradient-amplitude effect size or inversion-recovery claim.',
        best_frequency_hz=fbest,display_scales=records,
        plotter_sha256=sha(__file__),local_model_sha256=sha(args.model),
        inputs=[dict(path=str(p),sha256=sha(p)) for p in [args.baseline/'gradient_l2.npz',
            args.water1/'grid_gradients.npz' if (args.water1/'grid_gradients.npz').exists() else args.water1/'gradient_l2.npz',
            args.best/'grid_gradients.npz' if (args.best/'grid_gradients.npz').exists() else args.best/'gradient_l2.npz',args.best/'pilot.json']],
        source_case_reports={'water_1Hz':water_report['case'],'best':best_report['case']},
        density_note='True and initial density are constant; density sensitivity is not a density reconstruction and is retained in the numerical bundles.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
