#!/usr/bin/env python3
"""Plot passed native P-SV evidence exported by package_psv_evidence.py.

SPDX-License-Identifier: GPL-3.0-or-later
No simulation, input material array or approximate gradient is constructed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/diffspecfem-psv-mpl')
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'lines.linewidth': 1.1,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False,
    'savefig.facecolor': 'white',
})
BLUE, ORANGE = '#2166ac', '#d97721'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(fig, name, axes, output):
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    letters = np.array(list('abcdefghijklmnopqrstuvwxyz')[:axes.size]).reshape(axes.shape)
    fig.canvas.draw()
    # Anchor colorbars to final aspect-correct plot boxes, below x-axis text.
    # Automatic space stealing can overlap labels on shallow seismic sections.
    for image, ax, text in getattr(fig, '_psv_colorbars', []):
        box = ax.get_position()
        points = fig.get_figheight() * 72
        cax = fig.add_axes([box.x0, box.y0 - 36 / points, box.width, 5 / points])
        maximum = max(abs(image.norm.vmin), abs(image.norm.vmax))
        cb = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[-maximum, 0, maximum],
                         format=mpl.ticker.FuncFormatter(lambda value, position: f'{value:.2g}'))
        cb.set_label(text, fontsize=7, labelpad=3)
        cb.ax.tick_params(labelsize=7, pad=2)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=axes.ravel().tolist(), panel_ids=letters.ravel().tolist(),
        row_groups=[group for group in letters.tolist() if len(group) > 1],
        column_groups=[group for group in letters.T.tolist() if len(group) > 1],
        json_out=output / 'qa' / f'{name}.alignment.json',
        overlay_svg=output / 'qa' / f'{name}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(output / f'{name}.png', dpi=300)
    fig.savefig(output / f'{name}.pdf', dpi=300)
    fig.savefig(output / f'{name}.svg', dpi=300)
    plt.close(fig)


def title(ax, letter, text):
    ax.set_title(f'{letter}  {text}', loc='left', pad=8)


def domain(ax):
    ax.set(xlim=(5, 13.955), ylim=(3.465, 0), xlabel='Original x (km)', ylabel='Depth (km)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([5, 8, 11, 13.955], ['5', '8', '11', '13.955'])
    ax.set_yticks([0, .45, 1.5, 3])
    ax.axhline(.45, color='#555555', lw=.5, ls='--')


def bar(fig, image, ax, label):
    if not hasattr(fig, '_psv_colorbars'):
        fig._psv_colorbars = []
    fig._psv_colorbars.append((image, ax, label))


def triangulation(geometry):
    if len(geometry) % 25:
        raise ValueError('Expected complete element-local GLL geometry')
    # Explicit element-local connectivity preserves both sides of the interface.
    # It never Delaunay-triangulates or averages fluid and solid values.
    element_count = len(geometry) // 25
    expected = np.tile(np.column_stack([np.tile(np.arange(1, 6), 5),
                                       np.repeat(np.arange(1, 6), 5)]), (element_count, 1))
    if not np.array_equal(geometry[:, 1:3], expected):
        raise ValueError('Unexpected native i/j serialization')
    triangles = []
    for base in range(0, len(geometry), 25):
        for j in range(4):
            for i in range(4):
                a = base + j * 5 + i
                triangles.extend([[a, a + 1, a + 5], [a + 1, a + 6, a + 5]])
    return mtri.Triangulation(5 + geometry[:, 3] / 1000,
                             (3465 - geometry[:, 4]) / 1000, np.asarray(triangles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True,
                        help='Passed fine-mesh public evidence directory')
    parser.add_argument('--correctness', type=Path, required=True,
                        help='Full multistep validation JSON on the declared correctness mesh')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/marmousi2')
    parser.add_argument('--skip-animation', action='store_true')
    args = parser.parse_args()
    ev, output = args.evidence, args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'qa').mkdir(exist_ok=True)
    report = json.loads((ev / 'validation.json').read_text())
    correctness = json.loads(args.correctness.read_text())
    if not report.get('passed') or not correctness.get('passed'):
        raise ValueError('Do not render unvalidated results as accepted evidence')
    inputs = [ev / 'validation.json', args.correctness, ev / 'manifest.json']
    target = np.load(ev / 'target.npy')
    results = {kind: np.load(ev / f'gradient_{kind}.npz', allow_pickle=False) for kind in ['l2', 'processed']}
    first = results['l2']
    native = np.load(ev / 'native_reference.npy')
    if not np.array_equal(first['data'], native):
        raise ValueError('Original native and extended forward outputs differ')
    if not np.array_equal(first['cotangent'], first['data'] - target):
        raise ValueError('L2 cotangent is not the actual residual')
    time = first['sample_time']
    source_files = sorted(ev.glob('wavefield_rank*.npz'))
    if len(source_files) != report['ranks']:
        raise ValueError('Native vector bundle is incomplete; wait for every rank')
    waves, geometries, triangles, steps = [], [], [], None
    for path in source_files:
        bundle = np.load(path, allow_pickle=False)
        schedule = bundle['step_one_based']
        if steps is not None and not np.array_equal(schedule, steps):
            raise ValueError('Rank snapshot schedules differ')
        steps = schedule
        geometries.append(bundle['geometry'])
        waves.append(bundle['displacement'])
        triangles.append(triangulation(geometries[-1]))
        inputs.append(path)
    if not waves:
        raise ValueError('Missing actual native P-SV displacement snapshots')
    if not all(wave.shape[:2] == (len(steps), 2) for wave in waves):
        raise ValueError('Unexpected vector shape')
    max_wave = max(float(np.max(abs(wave))) for wave in waves)
    if max_wave <= 0:
        raise ValueError('All-zero native wavefield')
    # A fixed full-record maximum: no frame-wise normalization or saturation.
    wave_norm = mpl.colors.SymLogNorm(linthresh=max_wave * .001,
                                      vmin=-max_wave, vmax=max_wave, base=10)
    selected = np.unique(np.linspace(0, len(steps) - 1, 5, dtype=int)[1:])
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.4))
    for component in range(2):
        for column, frame in enumerate(selected):
            ax = axes[component, column]
            for tri, wave in zip(triangles, waves):
                image = ax.tripcolor(tri, wave[frame, component], cmap='RdBu_r',
                                    norm=wave_norm, shading='gouraud', rasterized=True)
            domain(ax)
            title(ax, 'abcdefgh'[component * 4 + column],
                  f'{["UX", "UZ"][component]}, t = {time[steps[frame] - 1]:.2f} s')
            bar(fig, image, ax, 'Displacement (m); fixed signed-log scale')
    fig.suptitle('Native P–SV displacement at the smoothed initial model', y=.99, fontsize=10)
    fig.subplots_adjust(left=.065, right=.97, top=.9, bottom=.19, wspace=.36, hspace=.75)
    save(fig, '02_psv_wavefields', axes, output)

    if not args.skip_animation:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.9))
        artists = [[], []]
        for component, ax in enumerate(axes):
            for tri, wave in zip(triangles, waves):
                artists[component].append(ax.tripcolor(tri, wave[0, component], cmap='RdBu_r',
                                                      norm=wave_norm, shading='gouraud', rasterized=True))
            domain(ax)
            title(ax, 'ab'[component], ['UX — native horizontal displacement', 'UZ — native vertical displacement'][component])
        fig.subplots_adjust(left=.09, right=.80, top=.91, bottom=.09, hspace=.38)
        cax = fig.add_axes([.88, .2, .018, .6])
        cb = fig.colorbar(artists[0][0], cax=cax,
                          ticks=[-max_wave, 0, max_wave],
                          format=mpl.ticker.FuncFormatter(lambda value, position: f'{value:.2g}'))
        cb.set_label('Displacement (m); fixed signed-log scale', fontsize=8)
        clock = fig.suptitle('', fontsize=10, y=.98)
        def update(frame):
            for component in range(2):
                for artist, wave in zip(artists[component], waves):
                    artist.set_array(wave[frame, component])
            clock.set_text(f'Smoothed initial model: native P–SV, source-relative time {time[steps[frame] - 1]:.2f} s')
            return [clock] + artists[0] + artists[1]
        update(len(steps) - 1)
        # Save a static authoritative layout companion with identical axes.
        save(fig, 'native_psv_animation_layout', axes[:, None], output)
        # save() closes the window, but Agg canvas remains usable by animation.
        animation = FuncAnimation(fig, update, frames=len(steps), interval=100, blit=False)
        # GIF is a screen preview; static submission companions are 300 dpi.
        animation_preview_resolution = 110
        animation.save(output / 'native_psv_wavefield.gif', writer=PillowWriter(fps=10), dpi=animation_preview_resolution)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.6))
    gradient_scales = {}
    for row, kind in enumerate(['l2', 'processed']):
        gradient = results[kind]['gradient']
        if gradient.shape != (3, 67, 200):
            raise ValueError('Expected the full active 67x200 solid input grid')
        for column, name in enumerate(['Vp', 'Vs', 'Density']):
            ax = axes[row, column]
            values = np.flipud(gradient[column])
            maximum = float(np.max(abs(values)))
            if maximum <= 0:
                raise ValueError('All-zero advertised parameter gradient')
            norm = mpl.colors.SymLogNorm(linthresh=maximum * .001, vmin=-maximum, vmax=maximum, base=10)
            image = ax.pcolormesh(np.linspace(5, 13.955, 200), np.linspace(.495, 3.465, 67),
                                  values, shading='nearest', cmap='RdBu_r', norm=norm, rasterized=True)
            ax.axhspan(0, .45, facecolor='#eef3f6', hatch='///', edgecolor='#bac6cf', lw=0)
            domain(ax)
            title(ax, 'abcdef'[row * 3 + column], f'{kind.upper() if kind == "l2" else "Processed"}: {name}')
            unit = 'J / (kg m⁻³)' if name == 'Density' else 'J / (m s⁻¹)'
            bar(fig, image, ax, f'Discrete derivative ({unit})')
            gradient_scales[f'{kind}_{name}'] = dict(absolute_max=maximum, linthresh=maximum * .001,
                                                    clipping='none', model='smoothed initial; observations from true')
    fig.suptitle('Solid input-grid gradients at the smoothed initial model\nSigned-log display; J denotes the objective; each panel retains raw units', y=.99, fontsize=10)
    fig.subplots_adjust(left=.065, right=.97, top=.89, bottom=.16, wspace=.36, hspace=.65)
    save(fig, '04_psv_gradients', axes, output)

    stations = (ev / 'gradient_l2/DATA/STATIONS').read_text().splitlines()
    positions = np.array([float(line.split()[2]) for line in stations if line.strip()]) / 1000 + 5
    fig, axes = plt.subplots(2, 4, figsize=(12.3, 6.2))
    display_scales = {}
    for component in range(2):
        matrices = [target[:, component], first['data'][:, component],
                    first['data'][:, component] - target[:, component],
                    results['processed']['cotangent'][:, component]]
        common = max(float(np.max(abs(matrix))) for matrix in matrices[:2])
        for column, matrix in enumerate(matrices):
            limit = common if column < 2 else float(np.max(abs(matrix)))
            if limit <= 0:
                raise ValueError('Uninformative data panel')
            image = axes[component, column].pcolormesh(time, positions, matrix,
                shading='nearest', cmap='RdBu_r', vmin=-limit, vmax=limit, rasterized=True)
            axes[component, column].set(xlim=(time[0], time[-1]), ylim=(positions[0], positions[-1]))
            title(axes[component, column], 'abcdefgh'[component * 4 + column],
                  f'{["UX", "UZ"][component]}: {["Target", "Initial", "Residual", "Processed AD source"][column]}')
            axes[component, column].set(xlabel='Source-relative time (s)', ylabel='Receiver x (km)')
            unit = 'dJ / dData (m⁻¹)' if column == 3 else 'Displacement (m)'
            bar(fig, image, axes[component, column], unit)
            display_scales[f'{component}_{column}'] = limit
    fig.subplots_adjust(left=.065, right=.97, top=.92, bottom=.16, wspace=.46, hspace=.75)
    save(fig, '03_psv_data_adjoint_source', axes, output)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.7))
    colors = ['#2166ac', '#e08214', '#26806c', '#7b3294', '#666666']
    for row, kind in enumerate(['l2', 'processed']):
        gates = correctness['gates'][kind]
        for color, direction in zip(colors, ['vp', 'vs', 'rho', 'mixed', 'localized']):
            check = gates['taylor_' + direction]
            if not check['passed'] or not check['informative']:
                raise ValueError('Unpassed Taylor family')
            points = check['steps']
            hs = np.array([item['step'] for item in points])
            # Parameter directions have different scales. Normalize only the
            # plotted remainders by their own nonzero directional slope.
            slope = abs(check['directional_derivative'])
            remainder = np.array([item['first_order_residual'] for item in points]) / slope
            error = np.array([item['central_directional_error'] for item in points]) / slope
            if np.any(remainder <= 0) or np.any(error <= 0):
                raise ValueError('Log plot requires positive raw residuals')
            axes[row, 0].loglog(hs, remainder, marker='o', ms=3.4, color=color, label=direction.upper() if direction in ['vp', 'vs'] else direction.capitalize())
            axes[row, 1].loglog(hs, error, marker='o', ms=3.4, color=color)
        for column in range(2):
            ax = axes[row, column]
            title(ax, 'abcd'[row * 2 + column], f'{kind.upper() if kind == "l2" else "Processed"}: {["First-order remainder", "Centered derivative error"][column]}')
            ax.set(xlabel='Relative model step h', ylabel=['R₁ / |g · p|', 'Derivative error / |g · p|'][column])
            ax.grid(which='major', alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    settings = correctness['gates']['source_resolution']
    geometry_record = ROOT / correctness['case'] / 'CASE_PROVENANCE.json'
    inputs.append(geometry_record)
    mesh = json.loads(geometry_record.read_text())['mesh_elements']
    mesh_label = f"{mesh['nx']} × {mesh['nz_solid'] + mesh['nz_water']} elements"
    fig.suptitle(f"Full six-step Taylor sweeps: {mesh_label}, f0 = {settings['source_f0_hz']:g} Hz, DT = {settings['dt_s'] * 1000:g} ms", y=.995, fontsize=10)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .955), ncol=5)
    fig.subplots_adjust(left=.1, right=.98, top=.83, bottom=.1, wspace=.35, hspace=.5)
    save(fig, '05_psv_verification', axes, output)
    for name in ['target.npy', 'native_reference.npy', 'gradient_l2.npz', 'gradient_processed.npz']:
        inputs.append(ev / name)
    provenance = dict(physics='native coupled acoustic/elastic P-SV', fine_case=report['case'],
        correctness_case=correctness['case'], native_timestamps='sample_time[step_one_based - 1]',
        wavefield=dict(frames=len(steps), element_local_gll_occurrences=sum(len(g) for g in geometries),
                       component_absolute_max_m=max_wave, linthresh_m=max_wave * .001,
                       phase_averaging=False, clipping='none', spatial_display='Gouraud color interpolation on explicit element-local GLL subtriangles; native point values unchanged'),
        animation_playback='100 ms per saved frame as a screen preview; native timestamps shown exactly, final physical frame interval can be shorter',
        objective_units='J denotes objective, not joules: L2 has m^2 units; processed objective is dimensionless',
        plotter_sha256=sha(Path(__file__)), numpy_version=np.__version__, matplotlib_version=mpl.__version__,
        gradients=gradient_scales, data_scales=display_scales,
        taylor='all six steps and five directions, both objectives; residuals normalized by their own nonzero |g dot p| for display only',
        sources=[dict(file=str(path), bytes=path.stat().st_size, sha256=sha(path)) for path in inputs])
    (output / 'native_visual_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
