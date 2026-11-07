#!/usr/bin/env python3
"""Plot the complete, water-bearing Marmousi-II truth and initial inputs.

No propagation or inversion occurs here. All 78 x 200 prepared samples are
shown; native z-up arrays are reversed to the original depth-down orientation.
The signed differences are initial minus truth. Raw model arrays are excluded
from the public figure bundle; source hashes and display transforms are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffspecfem-models-matplotlib")
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
AGL_URL = "https://www.agl.uh.edu/downloads/downloads.htm"
UPSTREAM_URL = "https://github.com/liufeng2317/ADFWI/blob/bv1.1/examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop/01_forward.ipynb"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def parse_geometry(case):
    """Read the declared case geometry, rather than drawing invented marks."""
    data = case / "DATA"
    text = (data / "SOURCE").read_text()
    def source_value(name):
        value = re.search(rf"(?m)^{name}\s*=\s*([^\s#]+)", text)[1]
        return float(value.replace("d", "e").replace("D", "e"))
    source = np.array([source_value("xs") + 5000.0, 3465.0 - source_value("zs")]) / 1000.0
    stations = [line.split() for line in (data / "STATIONS").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    receivers = np.array([[float(row[2]) + 5000.0, 3465.0 - float(row[3])] for row in stations]) / 1000.0
    if len(stations) != 41 or not np.allclose(receivers[:, 1], 0.54):
        raise ValueError("Expected the declared 41-receiver full P-SV acquisition")
    if not np.array_equal(source, np.array([9.5, 0.54])):
        raise ValueError("Full P-SV source geometry changed; review the figure contract")
    return source, receivers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "data/prepared/marmousi2_full_psv.npz")
    parser.add_argument("--case", type=Path, default=ROOT / "examples/psv_marmousi2_full")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "figures/marmousi2/models_full_psv")
    parser.add_argument("--qa-scripts", type=Path, default=ROOT / "scripts/figure_qa")
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    with np.load(args.model, allow_pickle=False) as archive:
        prepared = {key: archive[key].copy() for key in archive.files}
    prep_meta = json.loads(str(prepared["metadata_json"]))
    x = prepared["x_original_m"] / 1000.0
    depth = prepared["depth_original_m"] / 1000.0
    if x.shape != (200,) or depth.shape != (78,):
        raise ValueError("Requires the complete 78 x 200 paper model including water")
    if not np.array_equal(x[[0, -1]], np.array([5.0, 13.955])) or not np.array_equal(depth[[0, -1]], np.array([0.0, 3.465])):
        raise ValueError("Unexpected original Marmousi-II coordinates")
    if not (np.diff(x) > 0).all() or not (np.diff(depth) > 0).all():
        raise ValueError("Original plotting coordinates must strictly increase")
    fields = {}
    for key in ("vp_true", "vp_initial", "vs_true", "vs_initial", "rho_true", "rho_initial"):
        field = np.ascontiguousarray(prepared[key][::-1])
        if field.shape != (78, 200) or not np.isfinite(field).all():
            raise ValueError(f"Incomplete or nonfinite model: {key}")
        if array_sha256(field) != prep_meta["full_paper_field_sha256_le_f64"][key]:
            raise ValueError(f"Model does not match its full-paper numeric hash: {key}")
        fields[key] = field
    water = prepared["water_mask"][::-1]
    if int(water.sum()) != 2200 or not water[:11].all() or water[11:].any():
        raise ValueError("The complete flat 11-row water layer must be retained")
    for state in ("true", "initial"):
        if not np.array_equal(fields[f"vs_{state}"] == 0.0, water):
            raise ValueError("The true and initial water domains differ")
        if not np.all(fields[f"rho_{state}"] == 2450.0):
            raise ValueError("This figure contract expects the ADFWI constant-density model")
    if prep_meta["initial_smoothing_sigma_m"] != 150.0 or prep_meta["water_cells_removed"] != 0:
        raise ValueError("Requires full-water model inputs with the declared 150 m smoothing")
    source, receivers = parse_geometry(args.case)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    contract_path = ROOT / "figures/marmousi2/models_contract.json"

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.55, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    width_mm = 183.0
    height_mm = 116.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    mm_to_x = lambda mm: mm / width_mm
    mm_to_y = lambda mm: mm / height_mm
    fig.text(mm_to_x(12), mm_to_y(111), "Marmousi-II: full water-bearing model inputs", fontsize=9, weight="bold")
    fig.text(mm_to_x(12), mm_to_y(106.5), "78 × 200 samples · 45 m grid · Gaussian initial model, σ = 150 m", fontsize=6.5)
    legend_handles = [
        Line2D([], [], marker="*", linestyle="none", color="#d97812", markeredgecolor="#242424", markeredgewidth=0.35, markersize=5.5, label="Source, depth 0.54 km"),
        Line2D([], [], marker="v", linestyle="none", color="white", markeredgecolor="#222222", markeredgewidth=0.4, markersize=3, label="41 receivers, depth 0.54 km"),
        Line2D([], [], color="#4b4b4b", linestyle=(0, (3, 2)), linewidth=0.6, label="Seafloor, depth 0.45 km"),
    ]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(mm_to_x(11), mm_to_y(103)),
               ncol=3, handlelength=1.6, columnspacing=1.35, handletextpad=0.5, borderaxespad=0)
    axes, colorbar_axes, panel_metadata = [], [], []
    plot_width_mm = 48.6666666667
    plot_height_mm = plot_width_mm * (depth[-1] - depth[0]) / (x[-1] - x[0])
    letters = np.array(list("abcdef")).reshape(2, 3)
    for row, parameter in enumerate(("vp", "vs")):
        true, initial = fields[f"{parameter}_true"] / 1000.0, fields[f"{parameter}_initial"] / 1000.0
        difference = initial - true
        absolute_norm = Normalize(vmin=float(min(true.min(), initial.min())), vmax=float(max(true.max(), initial.max())))
        delta_limit = float(np.max(np.abs(difference)))
        delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)
        top_mm = (91.0, 49.0)[row]
        bottom_mm = top_mm - plot_height_mm
        for column, (state, values) in enumerate((("Truth", true), ("Initial", initial), ("Initial − truth", difference))):
            left_mm = 12.0 + column * (plot_width_mm + 10.5)
            ax = fig.add_axes([mm_to_x(left_mm), mm_to_y(bottom_mm), mm_to_x(plot_width_mm), mm_to_y(plot_height_mm)])
            ax.set_aspect("equal", adjustable="box")
            letter = str(letters[row, column])
            axes.append(ax)
            image = ax.pcolormesh(x, depth, values, shading="nearest", rasterized=True,
                                 cmap="RdBu_r" if column == 2 else "viridis",
                                 norm=delta_norm if column == 2 else absolute_norm)
            # Every sample is shown; cell edges are the midpoints between samples.
            # Limits are the declared original first/last model coordinates.
            ax.set(xlim=(float(x[0]), float(x[-1])), ylim=(float(depth[-1]), float(depth[0])))
            ax.set_xticks([5, 8, 11, 13.955], labels=["5", "8", "11", "13.955"])
            ax.set_yticks([0, 1, 2, 3.465], labels=["0", "1", "2", "3.465"])
            ax.tick_params(length=2.2, width=0.5, pad=2)
            ax.set_xlabel("Original x (km)", labelpad=2)
            if column == 0:
                ax.set_ylabel("Depth (km)", labelpad=2)
            else:
                ax.tick_params(labelleft=False)
            ax.axhline(0.45, color="white", linewidth=0.9, zorder=3)
            ax.axhline(0.45, color="#4b4b4b", linewidth=0.45, linestyle=(0, (3, 2)), zorder=4)
            ax.scatter(receivers[:, 0], receivers[:, 1], marker="v", s=3.8, facecolor="white",
                       edgecolor="#222222", linewidth=0.32, zorder=5)
            ax.scatter(*source, marker="*", s=27, facecolor="#d97812", edgecolor="#202020", linewidth=0.4, zorder=6)
            ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                        ha="left", va="bottom", fontsize=8, fontweight="bold")
            ax.annotate(f"{parameter.capitalize()} · {state}", xy=(0, 1), xycoords="axes fraction", xytext=(11, 5), textcoords="offset points",
                        ha="left", va="bottom", fontsize=7)
            cax = fig.add_axes([mm_to_x(left_mm), mm_to_y(top_mm - 28.0), mm_to_x(plot_width_mm), mm_to_y(1.5)])
            colorbar_axes.append(cax)
            cb = fig.colorbar(image, cax=cax, orientation="horizontal")
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(length=1.8, width=0.4, pad=1.6, labelsize=5.7)
            if column == 2:
                cb.set_ticks([-delta_limit, 0.0, delta_limit], labels=[f"−{delta_limit:.2f}", "0", f"{delta_limit:.2f}"])
            elif parameter == "vp":
                cb.set_ticks([1.5, 3, 4.7])
            else:
                cb.set_ticks([0, 1.4, 2.752], labels=["0", "1.4", "2.752"])
            cb.set_label(f"{'Δ' if column == 2 else ''}{parameter.capitalize()} (km/s)", fontsize=6, labelpad=1.5)
            panel_metadata.append({
                "panel": letter, "field": parameter, "state": state, "shape_zx": [78, 200],
                "display_units": "km/s", "display_range": [float(values.min()), float(values.max())],
                "color_limits": [-delta_limit, delta_limit] if column == 2 else [absolute_norm.vmin, absolute_norm.vmax],
                "colormap": "RdBu_r" if column == 2 else "viridis", "clipping": False,
                "display_value_sha256_le_f64": array_sha256(values),
            })
    fig.text(mm_to_x(12), mm_to_y(8.0), "ρ = 2450 kg/m³ everywhere in truth and initial, including water. No density update or inverted model is shown.", fontsize=6)
    source_note = fig.text(mm_to_x(12), mm_to_y(3.5), "Source: University of Houston / AGL · Martin, Wiley & Marfurt. All 11 water rows retained; no Vs floor.", fontsize=5.8)
    source_note.set_url(AGL_URL)
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=list("abcdef"),
        row_groups=[["a", "b", "c"], ["d", "e", "f"]],
        column_groups=[["a", "d"], ["b", "e"], ["c", "f"]],
        exclude_axes=colorbar_axes,
        json_out=args.output_prefix.with_name(args.output_prefix.name + "_alignment.json"),
        overlay_svg=args.output_prefix.with_name(args.output_prefix.name + "_alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
    )
    # Fixed physical canvas; no tight-bbox rescaling. Explicit suffixes also
    # make the requested export contract auditable from the plotting source.
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), dpi=600)
    fig.savefig(args.output_prefix.with_suffix(".svg"), dpi=600)
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        path = args.output_prefix.with_suffix("." + extension)
        outputs[extension] = {"filename": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
    plt.close(fig)
    metadata = {
        "figure": "Full water-bearing Marmousi-II model inputs: truth, initial and signed difference",
        "source": "University of Houston / AGL; Gary S. Martin, Robert Wiley and Kurt J. Marfurt",
        "official_source_url": AGL_URL, "adfwi_example_url": UPSTREAM_URL,
        "source_archives": prep_meta["sources"],
        "prepared_model_archive_sha256": file_sha256(args.model),
        "full_paper_field_sha256_le_f64": prep_meta["full_paper_field_sha256_le_f64"],
        "source_data_distribution": "Model arrays are local preparation inputs and are not included in this public figure bundle",
        "panel_metadata": panel_metadata,
        "integrity": {"all_samples_retained": True, "samples_per_panel": 15600, "excluded_samples": 0,
                      "water_samples_each_state": 2200, "vs_water_true_and_initial": 0,
                      "rho_constant_kg_per_m3": 2450, "initial_smoothing_sigma_m": 150,
                      "difference": "initial minus truth", "inversion_result": False},
        "coordinate_transform": "native field[::-1,:] restores original increasing depth; velocities divided by1000 only",
        "display_sampling": "Input-sample grid, not a native GLL coefficient field. Nearest sample cells use midpoint edges; the last zero-Vs sample at depth450m has a display cell ending at472.5m. Axes are limited to original first/last sample coordinates; all samples are drawn, with no display resampling, smoothing, clipping or gamma adjustment",
        "native_sem_representation": "The coupled native SEM has a conforming interface at depth450m. Its first positive-Vs solid input sample is at495m and is extended one-sided by constant border clamping from495m to450m. This differs from the input-sample cell coloring and does not alter or remove any original prepared sample. The dashed450m line indicates the native physical interface, not the edge of the nearest-sample water color band.",
        "original_extent_km": {"x": [5, 13.955], "depth": [0, 3.465]},
        "geometry": {"source_original_x_depth_km": source.tolist(), "receiver_count": len(receivers),
                     "receiver_x_extent_km": [float(receivers[:, 0].min()), float(receivers[:, 0].max())],
                     "receiver_depth_km": 0.54, "seafloor_depth_km": 0.45,
                     "source_and_receivers": "Read from the declared full P-SV case, not from a completed inversion"},
        "statistics": "Not applicable: deterministic model fields, no replicate estimates or statistical tests",
        "backend": "Python/matplotlib", "matplotlib_version": mpl.__version__, "numpy_version": np.__version__,
        "canvas_mm": [width_mm, height_mm], "png_dpi": 600, "editable_pdf_svg_text": True,
        "alignment_verdict": alignment.get("verdict", alignment.get("status")),
        "figure_contract_sha256": file_sha256(contract_path),
        "plotting_script_sha256": file_sha256(Path(__file__)), "outputs": outputs,
    }
    meta_path = args.output_prefix.with_name(args.output_prefix.name + "_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"prefix": str(args.output_prefix), "panels": 6, "all_samples_retained": True,
                      "water_samples_each_state": 2200, "alignment": metadata["alignment_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
