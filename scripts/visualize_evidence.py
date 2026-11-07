#!/usr/bin/env python3
"""Render the validated native SH evidence. No propagation or fabricated fields.

Input frames are extracted by extract_native_frames.py from checksum-matched
native histories. All source arrays, transformations and scales are manifested.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/diffspecfem-matplotlib')
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import PIL

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--qa-scripts', type=Path, default=Path(os.environ.get(
    'NATURE_FIGURE_QA', Path.home()/'.codex/skills/nature-figure/scripts')))
parser.add_argument('--skip-animation', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(args.qa_scripts))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = ROOT/'figures'
QA = OUT/'qa'
OUT.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)
EV = ROOT/'evidence/mpi2_32'
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.3,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
BLUE, ORANGE, GREEN = '#2166ac', '#d97721', '#26806c'
sources = set()


def load_np(path):
    sources.add(Path(path))
    return np.load(path)


def load_json(path):
    sources.add(Path(path))
    return json.loads(Path(path).read_text())


def label(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', pad=9)


def save(fig, name, axes):
    fig.canvas.draw()
    shape = np.shape(axes)
    panel_ids = np.array(list('abcdefghijklmnopqrstuvwxyz')[:int(np.prod(shape))]).reshape(shape)
    require_matplotlib_panel_alignment(
        fig, axes=list(np.ravel(axes)),
        panel_ids=panel_ids.ravel().tolist(),
        row_groups=[row.tolist() for row in panel_ids],
        column_groups=[col.tolist() for col in panel_ids.T],
        json_out=QA/f'{name}.alignment.json',
        overlay_svg=QA/f'{name}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    # Fixed canvas, no tight-bbox crop that changes the physical contract.
    fig.savefig(OUT/f'{name}.png', dpi=300)
    fig.savefig(OUT/f'{name}.pdf')
    svg = OUT/f'{name}.svg'
    fig.savefig(svg)
    # Normalize generated XML whitespace only; all SVG geometry/text is intact.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def map_axes(ax):
    ax.set(xlim=(0, 2), ylim=(-2, 0), xlabel='x (km)', ylabel='z (km)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-2, -1, 0])


def colorbar(fig, im, ax, title, *, shrink=1):
    cb = fig.colorbar(im, ax=ax, location='bottom', fraction=.06,
                      pad=.23, shrink=shrink, aspect=35)
    cb.set_label(title, fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=8, pad=2)
    return cb


start = load_np(EV/'gradient_l2.npz')
processed = load_np(EV/'gradient_processed.npz')
target = load_np(EV/'target.npy')
validation = load_json(EV/'validation.json')
coarse_validation = load_json(ROOT/'evidence/mpi2/validation.json')
frames = load_np(ROOT/'evidence/visuals/native_frames_mpi2_32.npz')
frame_provenance = json.loads(str(frames['provenance_json']))
indices = frames['step_index_zero_based']
time_file = EV/'gradient_l2/OUTPUT_FILES/XX.R01.BXY.semd'
sources.add(time_file)
t = np.loadtxt(time_file)[:, 0]
assert np.allclose(t, np.arange(800)*.001-.22, atol=1e-14)
assert np.array_equal(start['data'], load_np(EV/'native_reference.npy'))
assert np.array_equal(start['data'][0], start['data'][2])
assert np.array_equal(start['cotangent'], start['data']-target)

# Actual 32x32 rectilinear element boundaries; material ids from raw native rows.
raw = processed['raw']
assert raw.shape == (32*32*25, 11)
edges = np.linspace(0, 2, 33)
zedges = np.linspace(-2, 0, 33)
rho_ref = np.empty((32, 32))
vs_ref = np.empty((32, 32))
material = np.empty((32, 32), dtype=int)
element_maps = np.zeros((4, 32, 32))
seen_elements = set()
for rank in range(2):
    path = EV/f'gradient_processed/OUTPUT_FILES/ds_rank{rank:06d}_raw_kernels.txt'
    sources.add(path)
    rr = np.loadtxt(path)
    for element_id in np.unique(rr[:, 0]).astype(int):
        rows = rr[rr[:, 0] == element_id]
        assert rows.shape == (25, 11)
        x0, z0 = rows[:, 4:6].min(axis=0)/1000
        ix, iz = int(round(x0/.0625)), int(round((z0+2)/.0625))
        assert (iz, ix) not in seen_elements
        seen_elements.add((iz, ix))
        assert np.ptp(rows[:, 6]) == 0 and np.ptp(rows[:, 7]) == 0
        rho_ref[iz, ix] = rows[0, 6]
        vs_ref[iz, ix] = np.sqrt(rows[0, 7]/rows[0, 6])
        material[iz, ix] = int(rows[0, 10])
        element_maps[:, iz, ix] = [rows[:, 8].sum(), rows[:, 9].sum(),
            (rows[:, 6]*rows[:, 8]+rows[:, 7]*rows[:, 9]).sum(),
            (2*rows[:, 7]*rows[:, 9]).sum()]
assert len(seen_elements) == 1024
rho_true = rho_ref*np.exp(np.array([.025, -.02])[material-1])
vs_true = vs_ref*np.exp(np.array([.035, -.03])[material-1])
regional = np.array([element_maps[2][material==r].sum() for r in [1, 2]]+
                    [element_maps[3][material==r].sum() for r in [1, 2]])
assert np.allclose(regional, processed['gradient'], rtol=1e-13, atol=1e-15)

# A physical node appears on each MPI interface owner. Merge only after exact
# equality of all extracted u and a states, and multiplicity verification.
coordinates, states, multiplicities = [], [], []
rank_nodes = []
for rank in range(2):
    path = EV/f'gradient_l2/OUTPUT_FILES/ds_rank{rank:06d}_sampling_native_nodes.txt'
    sources.add(path)
    nodes = np.loadtxt(path)
    assert np.array_equal(nodes[:, 0], np.arange(1, len(nodes)+1))
    assert frames[f'rank{rank}_states'].shape == (len(indices), 2, len(nodes))
    coordinates.append(nodes[:, 4:6])
    states.append(frames[f'rank{rank}_states'])
    multiplicities.append(nodes[:, 3].astype(int))
    rank_nodes.append(nodes)
coords = np.concatenate(coordinates)
allstates = np.concatenate(states, axis=2)
mult = np.concatenate(multiplicities)
unique, first, inverse, counts = np.unique(coords, axis=0, return_index=True,
                                           return_inverse=True, return_counts=True)
assert np.array_equal(counts[inverse], mult)
assert np.array_equal(allstates, allstates[:, :, first][:, :, inverse])
# Rectilinear axes differ by <= 7e-13 m under native element arithmetic.
# Canonicalize display coordinates only; no nodes are merged or values changed.
display_coordinates_m = np.round(unique, 12)
assert len(np.unique(display_coordinates_m, axis=0)) == len(unique)
coordinate_display_rounding_m = float(np.max(np.abs(display_coordinates_m-unique)))
assert coordinate_display_rounding_m < 1e-12
xy = display_coordinates_m/1000
xx, zz = np.unique(xy[:, 0]), np.unique(xy[:, 1])
assert len(xx) == len(zz) == 129 and len(unique) == 16641
ix, iz = np.searchsorted(xx, xy[:, 0]), np.searchsorted(zz, xy[:, 1])
field = np.empty((len(indices), 2, len(zz), len(xx)))
field[:, :, iz, ix] = allstates[:, :, first]
# Native receiver scatter applied to retained states must reproduce their traces.
sampled = np.zeros((4, len(indices)))
for rank, nodes in enumerate(rank_nodes):
    path = EV/f'gradient_l2/OUTPUT_FILES/ds_rank{rank:06d}_sampling.txt'
    sources.add(path)
    for r, k, weight in np.loadtxt(path):
        sampled[int(r)-1] += weight*frames[f'rank{rank}_states'][:, 0, int(k)-1]
assert np.allclose(sampled, start['data'][:, indices], rtol=1e-12, atol=1e-14)

source = np.array([.650, -.650])
receivers = np.array([[.850, -.450], [1.150, -1.150], [.850, -.450], [1.450, -.550]])
for f in ['DATA/Par_file', 'DATA/SOURCE', 'DATA/STATIONS', 'DATA/interfaces.dat']:
    path = ROOT/'examples/sh_two_layer_32'/f
    if path.exists(): sources.add(path)
sources.add(ROOT/'native/discrete_adjoint.f90')
sources.add(ROOT/'diffspecfem/adapter.py')
sources.add(ROOT/'scripts/validate_mpi.py')
sources.add(Path(__file__).resolve())
sources.add(ROOT/'scripts/extract_native_frames.py')
sources.add(EV/'gradient_l2/history_manifest.json')

# Plate 1: exact parameter contrast, no invented Gaussian-blurred model.
fig, axs = plt.subplots(2, 3, figsize=(10.6, 8.0))
fig.subplots_adjust(left=.075, right=.97, top=.90, bottom=.10, wspace=.38, hspace=.55)
fig.suptitle('The validated SH contrast changes two material regions', fontsize=14, y=.98)
for row, (ref, truth, symbol, unit) in enumerate([
    (rho_ref, rho_true, 'ρ', 'kg/m³'), (vs_ref, vs_true, 'Vs', 'm/s')]):
    lo, hi = min(ref.min(), truth.min()), max(ref.max(), truth.max())
    delta = truth-ref
    for col, (values, title) in enumerate(zip([ref, truth, delta],
              ['Reference / starting', 'Perturbed target', 'Target − starting'])):
        ax = axs[row, col]
        lim = np.max(np.abs(delta))
        im = ax.pcolormesh(edges, zedges, values, shading='flat', rasterized=True,
                          cmap='RdBu_r' if col==2 else 'viridis',
                          vmin=-lim if col==2 else lo, vmax=lim if col==2 else hi)
        map_axes(ax)
        label(ax, 'abcdef'[row*3+col], f'{title} {symbol}')
        colorbar(fig, im, ax, f'{"Δ" if col==2 else ""}{symbol} ({unit})')
fig.text(.5, .025, 'Exact regional parameter map · fixed 2 km × 2 km geometry · 32 × 32 elements · no smoothing',
         ha='center', fontsize=9)
save(fig, '01_models', axs)

# Plate 2: original native nodal values, no spatial downsampling/interpolation.
wave_lim = float(np.max(np.abs(field[:, 0]))*1000)
fig, axs = plt.subplots(2, 2, figsize=(8.2, 8.5))
fig.subplots_adjust(left=.10, right=.96, top=.90, bottom=.20, wspace=.32, hspace=.35)
fig.suptitle('Native post-step SH displacement', fontsize=14, y=.98)
for ax, letter, index in zip(axs.ravel(), 'abcd', [199, 399, 599, 799]):
    j = int(np.flatnonzero(indices==index)[0])
    im = ax.pcolormesh(xx, zz, field[j, 0]*1000, shading='nearest',
                      cmap='RdBu_r', vmin=-wave_lim, vmax=wave_lim, rasterized=True)
    ax.axhline(-1, color='#777777', ls=':', lw=.6)
    ax.plot(*source, marker='*', ms=8, color='black', linestyle='none')
    ax.plot(receivers[:, 0], receivers[:, 1], 'o', ms=4, mfc='none', mec='black', mew=.8)
    map_axes(ax)
    label(ax, letter, f'n = {index+1}; t = {t[index]:.3f} s')
cb = fig.colorbar(im, ax=axs.ravel().tolist(), location='bottom', fraction=.035, pad=.11, aspect=45)
cb.set_label('SH displacement u (mm); one scale for every snapshot and animation frame')
fig.legend(handles=[Line2D([], [], marker='*', color='black', lw=0, label='Source'),
    Line2D([], [], marker='o', mfc='none', color='black', lw=0, label='Receivers (R01 = R03)'),
    Line2D([], [], ls=':', color='#777777', label='Material interface')],
    loc='lower center', bbox_to_anchor=(.5, .065), ncol=3)
fig.text(.5, .027, 't is the upstream trace label: (n − 1) × 0.001 − 0.22 s; elapsed native time is n × 0.001 s',
         ha='center', fontsize=8)
save(fig, '02_native_wavefield', axs)

# Plate 3: all four receivers, 800 samples, real AD data cotangent.
fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.6))
fig.subplots_adjust(left=.075, right=.97, top=.89, bottom=.12, wspace=.40, hspace=.65)
fig.suptitle('Native data, mismatch and the data-objective VJP', fontsize=14, y=.98)
residual = start['data']-target
amplitude = max(np.max(np.abs(start['data'])), np.max(np.abs(target)))*1000
reslim = np.max(np.abs(residual))*1000
items = [(target*1000, amplitude, 'Target traces', 'Displacement (mm)'),
         (start['data']*1000, amplitude, 'Starting traces', 'Displacement (mm)'),
         (residual*1000, reslim, 'Starting − target', 'Residual (mm)'),
         (start['cotangent']*1000, reslim, 'L2 data cotangent', '∂J / ∂d (10⁻³ m)'),
         (processed['cotangent'], np.max(np.abs(processed['cotangent'])),
          'Processed data cotangent', '∂J / ∂d (m⁻¹)')]
for ax, letter, (values, lim, title, unit) in zip(axs.ravel(), 'abcde', items):
    im = ax.imshow(values, extent=(t[0]-.0005, t[-1]+.0005, 4.5, .5),
                   origin='upper', aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                   interpolation='nearest', rasterized=True)
    ax.set(xlim=(t[0], t[-1]), xlabel='Upstream time label (s)', ylabel='Receiver')
    ax.set_yticks([1,2,3,4], ['R01','R02','R03','R04'])
    label(ax, letter, title)
    colorbar(fig, im, ax, unit)
ax = axs[1, 2]
ax.plot(t, target[3]*1000, color=ORANGE, label='Target')
ax.plot(t, start['data'][3]*1000, color=BLUE, ls='--', label='Starting')
ax.set(xlim=(t[0], t[-1]), ylim=(-amplitude, amplitude), xlabel='Upstream time label (s)',
       ylabel='R04 displacement (mm)')
label(ax, 'f', 'Waveform contrast at R04')
ax.legend(loc='upper left')
# Match the data rectangle after reserving the same colorbar-sized space.
placeholder = fig.colorbar(im, ax=ax, location='bottom', fraction=.06, pad=.23, aspect=35)
placeholder.ax.set_visible(False)
fig.text(.5, .025, 'Four stations × 800 samples · R01 and R03 coincide · cotangents use post-step sample order, with no dt factor or reversal',
         ha='center', fontsize=8.5)
save(fig, '03_data_and_adjoint_source', axs)

# Plate 4: exact element sums of discrete coefficient derivatives.
fig, axs = plt.subplots(2, 2, figsize=(8.8, 8.4))
fig.subplots_adjust(left=.095, right=.96, top=.90, bottom=.12, wspace=.36, hspace=.58)
fig.suptitle('Native spatial sensitivities for the processed objective', fontsize=14, y=.98)
titles = ['Density coefficient derivative', 'Shear-modulus coefficient derivative',
          'Relative log-density derivative', 'Relative log-Vs derivative']
units = ['Σ GLL ∂J/∂ρ  (m³/kg)', 'Σ GLL ∂J/∂μ  (Pa⁻¹)',
         'Σ GLL (ρ ∂J/∂ρ + μ ∂J/∂μ)  (dimensionless)',
         'Σ GLL 2μ ∂J/∂μ  (dimensionless)']
for ax, letter, values, title, unit in zip(axs.ravel(), 'abcd', element_maps, titles, units):
    lim = np.max(np.abs(values))
    im = ax.pcolormesh(edges, zedges, values, shading='flat', cmap='RdBu_r',
                      vmin=-lim, vmax=lim, rasterized=True)
    map_axes(ax)
    label(ax, letter, title)
    colorbar(fig, im, ax, unit)
fig.text(.5, .037, 'Each cell sums its 25 raw GLL contributions, including quadrature and Jacobian. J is dimensionless here.',
         ha='center', fontsize=8.5)
fig.text(.5, .016, 'No smoothing, clipping, illumination correction or conversion to continuous kernel density.',
         ha='center', fontsize=8.5)
save(fig, '04_spatial_gradients', axs)

# Plate 5: every tested epsilon, including those outside the central tolerance.
fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.5))
fig.subplots_adjust(left=.10, right=.96, top=.83, bottom=.13, wspace=.33, hspace=.50)
fig.suptitle('The native model gradient predicts independent perturbations', fontsize=14, y=.98)
family_colors = [BLUE, ORANGE, GREEN]
taylor_rows = []
for row, kind in enumerate(['l2', 'processed']):
    for family, color in zip(['rho', 'vs', 'mixed'], family_colors):
        test = validation['gates'][kind]['taylor_'+family]
        steps = test['steps']
        eps = np.array([s['step'] for s in steps])
        rem = np.array([s['first_order_residual'] for s in steps])
        err = np.array([s['central_directional_error'] for s in steps])
        display_family = {'rho':'Density', 'vs':'Vs', 'mixed':'Mixed'}[family]
        axs[row, 0].loglog(eps, rem, 'o-', ms=3.5, color=color, label=display_family)
        axs[row, 1].loglog(eps, err, 'o-', ms=3.5, color=color, label=display_family)
        threshold = 1e-9+1e-5*abs(test['directional_derivative'])
        axs[row, 1].axhline(threshold, color=color, ls=':', lw=.9)
        for s in steps:
            taylor_rows.append({'objective':kind, 'family':family, **s})
    reference = validation['gates'][kind]['taylor_mixed']['steps']
    eps = np.array([s['step'] for s in reference])
    ref_y = np.array([s['first_order_residual'] for s in reference])
    axs[row, 0].loglog(eps, .35*ref_y[-1]*(eps/eps[-1])**2, '--', color='#888888',
                       label='ε² guide', lw=1)
    for col in range(2):
        ax = axs[row, col]
        ax.set_xlabel('Perturbation ε (dimensionless log coordinates)')
        ax.grid(alpha=.18, lw=.5)
        label(ax, 'abcd'[row*2+col], f'{"L2" if row==0 else "Processed"}: '+
              ('first-order remainder' if col==0 else 'centered derivative error'))
        ax.set_ylabel('Absolute objective remainder' if col==0 else 'Absolute directional error')
handles, legend_labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5,.94), ncol=4)
fig.text(.5, .032, 'All 8 declared epsilons per family · dotted lines: 10⁻⁹ + 10⁻⁵ |g · p| acceptance · no dropped points',
         ha='center', fontsize=8.5)
save(fig, '05_taylor_evidence', axs)

if not args.skip_animation:
    fig, ax = plt.subplots(figsize=(6.2, 6.1))
    fig.subplots_adjust(left=.14, right=.94, bottom=.21, top=.90)
    im = ax.pcolormesh(xx, zz, field[0,0]*1000, shading='nearest', cmap='RdBu_r',
                      vmin=-wave_lim, vmax=wave_lim, rasterized=True)
    ax.axhline(-1, color='#777777', ls=':', lw=.7)
    ax.plot(*source, marker='*', ms=10, color='black', linestyle='none')
    ax.plot(receivers[:,0], receivers[:,1], 'o', ms=5, mfc='none', mec='black')
    map_axes(ax)
    title = ax.set_title('', fontsize=11, pad=10)
    cb = fig.colorbar(im, ax=ax, location='bottom', fraction=.045, pad=.16, aspect=40)
    cb.set_label('Native SH displacement u (mm); fixed amplitude scale')
    fig.text(.5, .035, '44 extracted native states; no spatial subsampling or physics rerun', ha='center', fontsize=9)
    def animate(j):
        im.set_array(field[j, 0]*1000)
        title.set_text(f'Native SH · n = {indices[j]+1} · upstream t = {t[indices[j]]:.3f} s')
        return im, title
    animation = FuncAnimation(fig, animate, frames=len(indices), blit=False)
    animation.save(OUT/'native_sh_wavefield.gif', writer=PillowWriter(fps=10), dpi=300)
    plt.close(fig)

# Numerical source data and manifest make every visible transformation reviewable.
np.savez_compressed(ROOT/'evidence/visuals/figure_source_data.npz',
    x_element_edges_km=edges, z_element_edges_km=zedges, material=material,
    rho_start_kg_m3=rho_ref, rho_target_kg_m3=rho_true,
    vs_start_m_s=vs_ref, vs_target_m_s=vs_true,
    element_sum_gradients=element_maps, upstream_time_s=t,
    target=target, starting=start['data'], residual=residual,
    cotangent_l2=start['cotangent'], cotangent_processed=processed['cotangent'],
    unique_x_km=xx, unique_z_km=zz, native_state_indices=indices)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    'backend':'Python / matplotlib; no generated or synthetic scientific data',
    'runtime':{'python':sys.version.split()[0], 'numpy':np.__version__,
               'matplotlib':mpl.__version__, 'Pillow':PIL.__version__},
    'export_contract':'300 dpi PNG and editable PDF/SVG; fixed report canvases, no cropping',
    'animation':{'display_fps':10, 'native_frames':len(indices), 'pixels':[1860,1830],
                 'playback_policy':'illustrative, actual native timestamps in titles'},
    'case':'examples/sh_two_layer_32', 'ranks':2,
    'data_shape':[4,800], 'unique_physical_nodes':len(unique),
    'local_node_count_sum':len(coords), 'duplicated_interface_nodes':int((counts==2).sum()),
    'shared_node_merge':'exact x,z match; all extracted displacement/acceleration states bitwise equal; multiplicities checked',
    'display_axis_canonicalization':'round physical coordinates to 12 decimal places in meters only to remove element-arithmetic axis roundoff; unique node count unchanged',
    'max_display_coordinate_rounding_m':coordinate_display_rounding_m,
    'history_trace_reconstruction_linf':float(np.max(np.abs(sampled-start['data'][:,indices]))),
    'frame_extraction':frame_provenance,
    'source_xy_km':source.tolist(), 'receiver_xy_km':receivers.tolist(),
    'source_and_receiver_policy':'all 4 receivers retained, including duplicate; full 2 km square retained',
    'model_policy':'exact native two-region map; reference/start is heterogeneous; target log perturbation [.025,-.02,.035,-.03]; no Gaussian smoothing',
    'wavefield_policy':'nearest-cell display of all 16641 physical GLL nodes; no interpolation to a regular pixel grid; 44 declared native times',
    'wavefield_common_limit_mm':wave_lim,
    'gradient_policy':'sum each element\u2019s 25 discrete GLL coefficient derivatives; includes quadrature/Jacobian, not continuous density; no smoothing/clipping/scaling',
    'processed_regional_gradient_from_maps':regional.tolist(),
    'processed_regional_gradient_saved':processed['gradient'].tolist(),
    'objective_units':{'l2':'m²', 'processed':'dimensionless'},
    'ad_cotangent_policy':'dJ/dd at same native post-step sample; no dt multiplication, sign change, time reversal or source-force conversion',
    'source_files':[{'file':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size, 'sha256':digest(p)}
                    for p in sorted(sources)],
    'outputs':[str(p.relative_to(ROOT)) for p in sorted(OUT.glob('*')) if p.is_file()],
    'limits':'Fixed discrete SH test, four regional parameters, two MPI ranks; grid/source guidance is not continuum convergence or full-field inversion.',
}
(ROOT/'evidence/visuals/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'evidence/visuals/taylor_source_data.json').write_text(json.dumps(taylor_rows,indent=2)+'\n')
print(json.dumps({'plates':5, 'unique_nodes':len(unique), 'frames':len(indices),
                  'native_sample_reconstruction_linf':manifest['history_trace_reconstruction_linf'],
                  'regional_gradient_closure_linf':float(np.max(np.abs(regional-processed['gradient'])))},indent=2))
