#!/usr/bin/env python3
"""Plot fresh Vp-only native derivatives with a shared old/new raw scale.

SPDX-License-Identifier: GPL-3.0-or-later
Model correction is an input reference, never a replacement for a gradient.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.plot_gradient_detail import plt, mpl, domain, sha
from scripts.figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def save(fig,axes,bars,output,name):
    fig.canvas.draw()
    for image,ax,label,ticks in bars:
        box=ax.get_position();height=fig.get_figheight()*72
        cax=fig.add_axes([box.x0,box.y0-36/height,box.width,5/height])
        cb=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=ticks,
            format=mpl.ticker.FuncFormatter(lambda value,position:f'{value:.2g}'))
        cb.ax.tick_params(labelsize=7,pad=2);cb.set_label(label,fontsize=7,labelpad=3)
    fig.canvas.draw()
    labels=np.array(list('abcdef')).reshape(axes.shape)
    require_matplotlib_panel_alignment(fig,axes=axes.ravel().tolist(),panel_ids=labels.ravel().tolist(),
        row_groups=labels.tolist(),column_groups=labels.T.tolist(),
        json_out=output/'qa'/f'{name}.alignment.json',overlay_svg=output/'qa'/f'{name}.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(output/f'{name}.png',dpi=300)
    fig.savefig(output/f'{name}.pdf',dpi=300)
    fig.savefig(output/f'{name}.svg',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=ROOT/'evidence/psv_marmousi2_20261107-vponly')
    parser.add_argument('--reference',type=Path,default=ROOT/'evidence/psv_gradient_detail_water2')
    parser.add_argument('--model',type=Path,default=ROOT/'data/prepared/marmousi2_20261107-vponly.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'figures/vponly')
    parser.add_argument('--percentile',type=float,default=99.)
    args=parser.parse_args()
    if not 95<=args.percentile<=100:
        raise ValueError('Display percentile must be declared conservatively')
    validation=json.loads((args.evidence/'validation.json').read_text())
    old_report=json.loads((args.reference/'pilot.json').read_text())
    if not validation['passed'] or not old_report['passed']:
        raise ValueError('Both actual numerical experiments must pass before plotting')
    if not validation['gates']['fresh_target_matches_old_physics']['passed']:
        raise ValueError('Shared target/source/objective comparison was not verified')
    if validation['input_provenance']['parent_model_sha256']!=old_report['model_sha256']:
        raise ValueError('The old/new experiment did not start from the same prepared parent model')
    if validation['model_sha256']!=sha(args.model):
        raise ValueError('This is not the actual Vp-only input')
    manifest=json.loads((args.evidence/'manifest.json').read_text())
    for row in manifest['files']:
        path=args.evidence/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Public numerical evidence hash changed: '+str(path))
    with np.load(args.model,allow_pickle=False) as source:
        x,z=source['x_native'].copy(),source['z_native'].copy()
        truth=source['vp_true'].copy();initial=source['vp_initial'].copy();fixed_vs=source['vs_true'].copy()
        if not np.array_equal(source['vs_initial'],fixed_vs) or not np.array_equal(source['rho_initial'],source['rho_true']):
            raise ValueError('Only Vp is allowed to differ from truth')
    with np.load(args.evidence/'vp_gradient.npz',allow_pickle=False) as result:
        gradient=result['gradient_vp'].copy()
        if not np.array_equal(result['x_native'],x) or not np.array_equal(result['z_native'],z):
            raise ValueError('Vp gradient coordinates differ from the actual input')
    with np.load(args.reference/'grid_gradients.npz',allow_pickle=False) as result:
        reference=result['gradient_l2'][0].copy()
        if not np.array_equal(result['x_native'],x) or not np.array_equal(result['z_native'],z):
            raise ValueError('Old/new physical Vp coordinates differ')
    if any(a.shape!=(67,200) or not np.isfinite(a).all() for a in [gradient,reference,truth,initial,fixed_vs]):
        raise ValueError('Expected all original67x200 finite solid-grid samples')
    depth=3465.-z;deep=depth>=1000.
    limit=float(np.percentile(np.concatenate([abs(reference[deep]).ravel(),abs(gradient[deep]).ravel()]),args.percentile))
    if limit<=0:raise ValueError('All-zero derivative display')
    x_original=(x+5000.)/1000
    vp_limits=(min(1.5,truth.min()/1000,initial.min()/1000),max(truth.max(),initial.max())/1000)
    vs_limits=(0.,fixed_vs.max()/1000)
    correction=(truth-initial)/1000
    correction_limit=float(np.max(abs(correction)))
    output=args.output;output.mkdir(parents=True,exist_ok=True);(output/'qa').mkdir(exist_ok=True)
    contract=dict(question='How does fixing true Vs/rho change the fresh Vp derivative at the same blurred Vp input?',
        claim='A controlled known-Vs/rho Vp-only sensitivity experiment; interpretation follows the actual raw result, not an assumed improvement.',
        backend='Python matplotlib',active_coordinates=['physical solid Vp'],
        models='true Vp, actual sigma150m Vp start, fixed true Vs; native fixed acoustic water and constant rho2450 convention',
        gradient_integrity='No smoothing, gain, sharpening, illumination correction or model texture. Both raw gradients share one linear scale.',
        correction_panel='True-minus-initial Vp is an explicitly labeled input correction, NOT a gradient or reconstruction.',
        exports='Report-sized PNG300dpi and editable PDF/SVG; min5pt glyphs, strict1.5pt alignment, final-PDF QA required')
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(13.2,6.9));bars=[]
    # Official tomography is one-sided from the first solid row at495m to the
    # conforming450m interface. Show that native model extension explicitly.
    model_depth=np.r_[450.,depth[::-1]]/1000
    def model_panel(ax,field,water,limits,cmap):
        values=np.vstack([field[-1],field[::-1]])
        image=ax.pcolormesh(x_original,model_depth,values,cmap=cmap,vmin=limits[0],vmax=limits[1],shading='nearest',rasterized=True)
        color=mpl.colormaps[cmap](mpl.colors.Normalize(*limits)(water))
        ax.axhspan(0,.45,color=color,lw=0)
        return image
    for col,(field,water,limits,cmap,label) in enumerate([
        (truth/1000,1.5,vp_limits,'viridis','Vp (km/s)'),
        (initial/1000,1.5,vp_limits,'viridis','Vp (km/s)'),
        (fixed_vs/1000,0.,vs_limits,'viridis','Known Vs (km/s)')]):
        image=model_panel(axes[0,col],field,water,limits,cmap)
        bars.append((image,axes[0,col],label,[limits[0],sum(limits)/2,limits[1]]))
    for col,field in enumerate([reference,gradient]):
        axes[1,col].set_facecolor('#e6e9ed')
        image=axes[1,col].pcolormesh(x_original,depth/1000,field,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest',rasterized=True)
        bars.append((image,axes[1,col],'dJ/dVp (objective / [m/s]); shared scale',[-limit,0,limit]))
    image=model_panel(axes[1,2],correction,0.,(-correction_limit,correction_limit),'RdBu_r')
    bars.append((image,axes[1,2],'True − initial Vp (km/s); NOT a gradient',[-correction_limit,0,correction_limit]))
    titles=['a  True Vp','b  Initial Vp: σ =150m','c  Fixed true Vs (known prior)',
            'd  Previous joint-start dJ/dVp','e  New Vp-only dJ/dVp','f  Input correction: ΔVp']
    for ax,title in zip(axes.ravel(),titles):
        domain(ax);ax.set_title(title,loc='left',pad=9)
    fig.suptitle('Marmousi-II: Vp-only coordinates with known true Vs and density\n'
        'Fresh native forward + data AD + native adjoint · 2Hz · same source/target/objective · fixed ρ =2450kg/m³',y=.985,fontsize=11)
    fig.subplots_adjust(left=.055,right=.975,top=.87,bottom=.15,wspace=.27,hspace=.72)
    save(fig,axes,bars,output,'vponly_model_and_gradient')
    display=dict(rule='One linear symmetric scale from concatenated |old gVp| and |new gVp| at input-grid depth>=1000m; all67x200 nodes drawn, no derivative values modified',
        percentile=args.percentile,absolute_limit=limit,
        saturated_fraction_all_nodes=dict(previous_joint=float(np.mean(abs(reference)>limit)),new_vponly=float(np.mean(abs(gradient)>limit))),
        independent_rescaling=False,gradient_postprocessing='none',
        gradient_grid='original67x200 solid parameter coordinates depth495..3465m; first-row VJP includes native one-sided extension to450m, not duplicated as extra gradient samples')
    report=dict(status='rendered; final independent PDF/visual QA required',figure='vponly_model_and_gradient',
        plotter_sha256=sha(__file__),model_sha256=sha(args.model),new_validation_sha256=sha(args.evidence/'validation.json'),
        new_vp_gradient_sha256=sha(args.evidence/'vp_gradient.npz'),old_gradient_bundle_sha256=sha(args.reference/'grid_gradients.npz'),
        active_coordinates=['vp'],fixed_true_vs_and_rho=True,fresh_target_bitwise_equal=True,
        display=display,input_correction_is_not_gradient=True,known_vs_is_not_inverted=True,
        source_units_and_objective='Matched old/new native acoustic Gaussian chi source at150m, same numeric factor1e8, same .5sum displacement-residual squared',
        limitation='Single-gradient controlled experiment with a known true Vs prior; no blind inversion, recovered model or thin-layer recovery claim.',
        comparison_statistics=dict(previous_joint=old_report['objectives']['l2']['derivative_statistics']['vp'],new_vponly=validation['vp_gradient_statistics']))
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
