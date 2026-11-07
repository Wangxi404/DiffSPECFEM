#!/usr/bin/env python3
"""Prepare the complete ADFWI-paper Marmousi-II model for coupled P-SV work.

The download, SI conversion, exact 45 m grid and masked 150 m Gaussian recipe
follow HybridADFWI's public ``hybridadfwi.models`` implementation. The default
full-psv mode retains all 78 x 200 samples, including all 11 true water rows.
The historical solid-sh mode removes water only after the full preparation.
Neither mode introduces a positive Vs floor. Input arrays and downloaded
archives stay in /data, which is excluded from the repository.

The source preparation routine is adapted from HybridADFWI contributors (2026),
MIT licensed; the preserved notice is in LICENSE.HybridADFWI-MIT alongside this
script. This code notice does not grant rights in the third-party model data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://www.agl.uh.edu/downloads/"
SOURCE_HASHES = {
    "vp_marmousi-ii.segy.gz": "2c7a0c4170c95a15620c2146573711c61a4cef8dc27338ef6c211a81ffa68920",
    "vs_marmousi-ii.segy.gz": "c88bec68fc4bd7c6e624737a20ecd3241c90e10512103e28e61b344fb5baed7d",
    "density_marmousi-ii.segy.gz": "9832e20ce1f79470fa8c100e79d66566cf8c1c162baa4289f37e7ee0b96f1e67",
}
GRID = {"nx": 200, "nz": 78, "dx": 45.0, "dz": 45.0, "ox": 5000.0, "oz": 0.0}
FIELD_KEYS = tuple(f"{field}_{state}" for field in ("vp", "vs", "rho") for state in ("true", "initial"))
UPSTREAM_EXAMPLE = "https://github.com/liufeng2317/ADFWI/blob/bv1.1/examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop/01_forward.ipynb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash numeric values, independently of ZIP serialization timestamps."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def download_sources(raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for filename, expected in SOURCE_HASHES.items():
        path = raw_dir / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "DiffSPECFEM-Marmousi-preparation"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                if sha256(temporary) != expected:
                    raise ValueError(f"Downloaded source checksum mismatch: {filename}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Source checksum mismatch for {path}: {actual}; expected {expected}")
        records[filename] = {"url": BASE_URL + filename, "sha256": actual, "bytes": path.stat().st_size}
    return records


def read_resample(path: Path, depth: np.ndarray, x: np.ndarray) -> tuple:
    import obspy

    traces = obspy.read(str(path), format="SEGY")
    raw_xz = np.stack([trace.data for trace in traces]).astype(np.float64)
    source_x = np.linspace(0.0, 17000.0, raw_xz.shape[0])
    source_depth = np.linspace(0.0, 3500.0, raw_xz.shape[1])
    interpolator = RegularGridInterpolator((source_x, source_depth), raw_xz, method="linear", bounds_error=True)
    zz, xx = np.meshgrid(depth, x, indexing="ij")
    values = interpolator(np.stack((xx, zz), axis=-1)) * 1000.0
    return np.ascontiguousarray(values, dtype=np.float64), raw_xz.shape


def smooth_solid(field: np.ndarray, water: np.ndarray, sigma_grid: tuple) -> np.ndarray:
    if np.all(field == field.flat[0]) or not any(sigma_grid):
        return field.copy()
    solid = (~water).astype(np.float64)
    weights = gaussian_filter(solid, sigma_grid, mode="nearest")
    result = gaussian_filter(field * solid, sigma_grid, mode="nearest") / np.maximum(weights, np.finfo(np.float64).tiny)
    result[water] = field[water]
    return np.ascontiguousarray(result, dtype=np.float64)


def prepare_paper(raw_dir: Path) -> tuple:
    sources = download_sources(raw_dir)
    x = GRID["ox"] + np.arange(GRID["nx"], dtype=np.float64) * GRID["dx"]
    depth = GRID["oz"] + np.arange(GRID["nz"], dtype=np.float64) * GRID["dz"]
    fields, shapes = {}, {}
    for key, prefix in (("vp", "vp"), ("vs", "vs"), ("rho", "density")):
        fields[key], shapes[key] = read_resample(raw_dir / f"{prefix}_marmousi-ii.segy.gz", depth, x)
    fields["rho"] = np.full_like(fields["rho"], 2450.0)
    water = fields["vs"] == 0.0
    sigma = (150.0 / GRID["dz"], 150.0 / GRID["dx"])
    full = {"x": x, "z": depth, "water_mask": water}
    for key, values in fields.items():
        full[key + "_true"] = values
        full[key + "_initial"] = smooth_solid(values, water, sigma)
    return full, {"sources": sources, "source_shapes_xz": shapes}


def load_reference(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        full = {key: data[key].copy() for key in (*FIELD_KEYS, "water_mask", "x", "z")}
    for key in FIELD_KEYS:
        if full[key].dtype != np.float64 or full[key].shape != (78, 200):
            raise ValueError(f"Reference {key} must be the float64 78 x 200 paper array")
    return full


def compare_reference(full: dict, reference_path: Path) -> dict:
    reference = load_reference(reference_path)
    comparisons = {}
    for key in (*FIELD_KEYS, "water_mask", "x", "z"):
        a, b = full[key], reference[key]
        comparisons[key] = {
            "same_shape": a.shape == b.shape,
            "same_dtype": a.dtype == b.dtype,
            "exact_numeric_equality": bool(np.array_equal(a, b)),
            "same_array_bytes": a.tobytes(order="C") == b.tobytes(order="C"),
            "max_absolute_difference": float(np.max(np.abs(a.astype(float) - b.astype(float)))),
        }
        if not all(comparisons[key][name] for name in ("same_shape", "same_dtype", "exact_numeric_equality", "same_array_bytes")):
            raise ValueError(f"Regenerated array differs from reference: {key}: {comparisons[key]}")
    return {"reference_archive_sha256": sha256(reference_path), "comparisons": comparisons}


def write_native_solid_tomography(path: Path, x: np.ndarray, z: np.ndarray, vp: np.ndarray, vs: np.ndarray, rho: np.ndarray) -> dict:
    """Native five-column tomography: x-fast rows on increasing x and z."""
    xx, zz = np.meshgrid(x, z)
    rows = np.column_stack((xx.ravel(), zz.ravel(), vp.ravel(), vs.ravel(), rho.ravel()))
    with path.open("w") as stream:
        stream.write(f"{x[0]:.17g} {z[0]:.17g} {x[-1]:.17g} {z[-1]:.17g}\n")
        stream.write(f"{x[1]-x[0]:.17g} {z[1]-z[0]:.17g}\n{x.size} {z.size}\n")
        stream.write(" ".join(f"{value:.17g}" for field in (vp, vs, rho) for value in (field.min(), field.max())) + "\n")
        np.savetxt(stream, rows, fmt="%.17g")
    reread = np.loadtxt(path, skiprows=4)
    if not np.array_equal(rows, reread):
        raise ValueError("Native tomography text serialization changed input numeric values")
    return {"filename": path.name, "sha256": sha256(path), "rows": int(rows.shape[0]),
            "columns": ["x_native_m", "z_native_m", "vp_m_per_s", "vs_m_per_s", "rho_kg_per_m3"],
            "serialization_exact_numeric_roundtrip": True}


def write_full_model(full: dict, source_metadata: dict, output: Path, reference_check: dict | None = None) -> dict:
    """Preserve every paper sample, with only the native coordinate reversal.

    This creates inputs; it does not claim that a P-SV native adjoint has been
    implemented or validated. The native coupled solver must represent the
    discontinuous acoustic/elastic interface separately from its material
    interpolation. Zero water Vs values must never be replaced with a floor.
    """
    fields = {key: np.ascontiguousarray(full[key][::-1]) for key in FIELD_KEYS}
    for key, values in fields.items():
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"Nonphysical full model values in {key}")
        if not key.startswith("vs_") and np.any(values == 0.0):
            raise ValueError(f"Vp and density must be positive: {key}")
    if not np.array_equal(full["water_mask"], full["vs_true"] == 0.0):
        raise ValueError("Water mask does not match true zero-Vs samples")
    if not np.array_equal(full["vs_initial"] == 0.0, full["water_mask"]):
        raise ValueError("Initial smoothing changed the water domain")
    x_original, depth = full["x"], full["z"]
    x_native = x_original - x_original[0]
    z_native = depth[-1] - depth[::-1]
    water = np.ascontiguousarray(full["water_mask"][::-1])
    # Preserve the complete arrays above. The coupled native model uses a
    # separate fixed-water material and these unmodified solid samples. The
    # native reader clamps its top edge to depth495 m when queried at the
    # conforming seafloor at depth450 m, a declared one-sided 45 m extension.
    solid = {key: np.ascontiguousarray(full[key][11:][::-1]) for key in FIELD_KEYS}
    solid_z = full["z"][-1] - full["z"][11:][::-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    domain_output = output.with_name(output.stem + "_solid_domain.npz")
    domain_metadata = {
        "purpose": "Solid material input domain of the complete water-coupled P-SV model",
        "full_model_filename": output.name, "native_array_axes": "z_up,x",
        "native_x_extent_m": [0.0, 8955.0], "native_z_sample_extent_m": [0.0, 2970.0],
        "original_depth_sample_extent_m": [495.0, 3465.0],
        "physical_solid_depth_extent_m": [450.0, 3465.0],
        "interface_interpolation": "Native constant border extension of the first solid sample at depth495 m to the seafloor at depth450 m; no blending with water zeros and no Vs floor",
        "solid_field_sha256_le_f64": {key: array_sha256(solid[key]) for key in FIELD_KEYS},
    }
    np.savez_compressed(domain_output, **solid, x_native=x_native, z_native=solid_z,
                        x_original_m=x_original, depth_original_m=depth[11:],
                        metadata_json=np.array(json.dumps(domain_metadata, sort_keys=True)))
    tomography = {}
    for state in ("true", "initial"):
        path = output.with_name(output.stem + f"_solid_{state}.xyz")
        tomography[state] = write_native_solid_tomography(path, x_native, solid_z,
                                                        solid[f"vp_{state}"], solid[f"vs_{state}"], solid[f"rho_{state}"])
    metadata = {
        "name": "Marmousi-II, complete ADFWI-paper model for coupled P-SV",
        "physics_scope": "Full acoustic water plus isotropic elastic solid P-SV",
        "execution_status": "Model preparation verified; native coupled P-SV adjoint validation is separate",
        "upstream_example": UPSTREAM_EXAMPLE,
        **source_metadata,
        "source_array_axes": "x,depth", "source_extent_m": {"x": [0, 17000], "depth": [0, 3500]},
        "source_units": {"vp": "km/s", "vs": "km/s", "rho": "g/cm^3"},
        "source_to_SI_multiplier": 1000.0,
        "units": {"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"},
        "full_paper_grid": GRID, "full_paper_shape_zx": [78, 200],
        "full_paper_x_extent_m": [float(x_original[0]), float(x_original[-1])],
        "full_paper_depth_extent_m": [float(depth[0]), float(depth[-1])],
        "resampling": "bilinear on exact nominal 45 m grid, no extrapolation",
        "adfwi_coordinate_note": "ADFWI notebook uses linspace over nx*dx; the common HybridADFWI paper arrays use (nx-1)*dx to match the solver spacing exactly.",
        "density_mode": "constant2450", "initial_smoothing_sigma_m": 150.0,
        "smoothing": "Gaussian normalized within solid, mode nearest; restore all true water samples; complete paper grid",
        "water_cells": int(water.sum()), "water_rows_original": np.flatnonzero(np.any(full["water_mask"], axis=1)).tolist(),
        "water_sample_depth_extent_m": [0.0, 450.0], "first_all_solid_sample_depth_m": 495.0,
        "water_cells_removed": 0, "water_rows_removed": [],
        "native_shape_zx": list(fields["vs_true"].shape),
        "native_array_axes": "z_up,x", "native_water_mask_axes": "z_up,x",
        "native_x_extent_m": [float(x_native[0]), float(x_native[-1])],
        "native_z_extent_m": [float(z_native[0]), float(z_native[-1])],
        "native_coordinate_transform": "x_native=x_AGL-5000; z_native=3465-depth_AGL; native arrays=flipud(full_paper_array)",
        "dtype": "float64", "model_values_modified": False, "vs_floor_applied": False,
        "interface_note": "The paper grid has Vs=0 at depth450 m and Vs>0 at depth495 m. Native acoustic/elastic element classification and shared interface values require an explicit coupled-domain representation; these prepared arrays do not silently alter that transition.",
        "native_coupled_domain_mapping": {
            "water": {"physical_depth_extent_m": [0.0, 450.0], "native_z_extent_m": [3015.0, 3465.0],
                      "material_type": "fixed acoustic", "vp_m_per_s": 1500.0, "vs_m_per_s": 0.0, "rho_kg_per_m3": 2450.0,
                      "active_inversion_parameters": []},
            "solid": {**domain_metadata, "filename": domain_output.name, "sha256": sha256(domain_output),
                      "material_type": "tomography isotropic elastic; positive numeric A=1 for initial native domain classification",
                      "active_inversion_parameters": ["vp", "vs", "rho"], "tomography_files": tomography},
            "difference_from_finite_difference_grid_representation": "Full prepared arrays remain exactly unchanged. The native SEM has a conforming interface at depth450 m and a one-sided solid extension across the 45 m interval before the first positive-Vs sample at depth495 m; the finite-difference sample grid instead keeps its adjacent water and solid nodal samples.",
        },
        "full_paper_field_sha256_le_f64": {key: array_sha256(full[key]) for key in FIELD_KEYS},
        "native_field_sha256_le_f64": {key: array_sha256(fields[key]) for key in FIELD_KEYS},
        "native_field_ranges_SI": {key: [float(value.min()), float(value.max())] for key, value in fields.items()},
        "data_source": "Allied Geophysical Laboratories, University of Houston; Gary S. Martin, Robert Wiley and Kurt J. Marfurt",
        "official_download_page": "https://www.agl.uh.edu/downloads/downloads.htm",
        "data_terms_page": "https://www.agl.uh.edu/resources-data.php",
        "data_distribution": "Third-party input data are not relicensed by this repository. Raw SEG-Y, prepared NPZ and native tomography inputs are local ignored files; regenerate from the official sources.",
    }
    if reference_check is not None:
        metadata["reference_prepared_verification"] = reference_check
    np.savez_compressed(output, **fields, water_mask=water, x_native=x_native, z_native=z_native,
                        x_original_m=x_original, depth_original_m=depth,
                        metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    metadata["native_archive_sha256"] = sha256(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def write_solid_crop(full: dict, source_metadata: dict, output: Path, reference_check: dict | None = None) -> dict:
    water = full["water_mask"]
    wet_rows = np.flatnonzero(np.any(water, axis=1))
    first_solid = int(wet_rows[-1] + 1) if len(wet_rows) else 0
    if first_solid != 11 or not water[:first_solid].all() or water[first_solid:].any():
        raise ValueError("This case requires the verified flat 11-row water layer; inspect changed data before cropping")
    fields = {}
    for key in FIELD_KEYS:
        if not np.isfinite(full[key]).all():
            raise ValueError(f"Nonfinite model values in {key}")
        fields[key] = np.ascontiguousarray(full[key][first_solid:][::-1])
        if np.any(fields[key] <= 0.0):
            raise ValueError(f"Native solid SH requires strictly positive {key}")
    x_original = full["x"]
    depth = full["z"][first_solid:]
    x_native = x_original - x_original[0]
    z_native = depth[-1] - depth[::-1]
    metadata = {
        "name": "Marmousi-II, ADFWI-paper solid-only SH crop",
        "upstream_example": UPSTREAM_EXAMPLE,
        **source_metadata,
        "source_array_axes": "x,depth", "source_extent_m": {"x": [0, 17000], "depth": [0, 3500]},
        "source_units": {"vp": "km/s", "vs": "km/s", "rho": "g/cm^3"},
        "source_to_SI_multiplier": 1000.0,
        "units": {"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"},
        "full_paper_grid": GRID,
        "full_paper_shape_zx": [78, 200],
        "full_paper_x_extent_m": [float(x_original[0]), float(x_original[-1])],
        "full_paper_depth_extent_m": [0.0, 3465.0],
        "resampling": "bilinear on exact nominal 45 m grid, no extrapolation",
        "adfwi_coordinate_note": "ADFWI notebook uses linspace over nx*dx; the common HybridADFWI paper arrays use (nx-1)*dx to match the solver spacing exactly.",
        "density_mode": "constant2450", "initial_smoothing_sigma_m": 150.0,
        "smoothing": "Gaussian normalized within solid, mode nearest; restore true water; smooth full paper arrays before cropping",
        "water_cells_removed": int(water.sum()), "water_rows_removed": wet_rows.tolist(),
        "removed_water_depth_samples_m": [0.0, 450.0], "first_all_solid_row": first_solid,
        "solid_shape_zx": list(fields["vs_true"].shape),
        "solid_depth_extent_m": [float(depth[0]), float(depth[-1])],
        "native_array_axes": "z_up,x", "native_x_extent_m": [float(x_native[0]), float(x_native[-1])],
        "native_z_extent_m": [float(z_native[0]), float(z_native[-1])],
        "native_coordinate_transform": "x_native=x_AGL-5000; z_native=3465-depth_AGL; native arrays=flipud(full_paper_array[11:,:])",
        "dtype": "float64",
        "model_values_modified_by_crop": False,
        "vs_floor_applied": False,
        "boundary_limitation": "Water and coupled P-SV propagation are unsupported by this discrete SH adjoint. The crop top is a traction-free artificial solid boundary; it is not a simulation of the original water-solid interface.",
        "full_paper_field_sha256_le_f64": {key: array_sha256(full[key]) for key in FIELD_KEYS},
        "native_field_sha256_le_f64": {key: array_sha256(fields[key]) for key in FIELD_KEYS},
        "native_field_ranges_SI": {key: [float(value.min()), float(value.max())] for key, value in fields.items()},
        "data_source": "Allied Geophysical Laboratories, University of Houston; Gary S. Martin, Robert Wiley and Kurt J. Marfurt",
        "official_download_page": "https://www.agl.uh.edu/downloads/downloads.htm",
        "data_terms_page": "https://www.agl.uh.edu/resources-data.php",
        "data_distribution": "Third-party input data are not relicensed by this repository. Raw SEG-Y, prepared NPZ and native tomography inputs are local ignored files; regenerate from the official sources.",
    }
    if reference_check is not None:
        metadata["reference_prepared_verification"] = reference_check
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **fields, x_native=x_native, z_native=z_native,
                        x_original_m=x_original, depth_original_m=depth,
                        metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    metadata["native_archive_sha256"] = sha256(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full-psv", "solid-sh"), default="full-psv",
                        help="full-psv preserves the full water-bearing model; solid-sh reproduces the historical separate SH crop")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/marmousi2")
    parser.add_argument("--output", type=Path, help="Default data/prepared/marmousi2_full_psv.npz, or marmousi2_solid.npz for solid-sh mode")
    parser.add_argument("--reference-prepared", type=Path, help="Verify regenerated full arrays byte-for-byte against a locally prepared ADFWI-paper NPZ")
    parser.add_argument("--audit-output", type=Path, help="Also write provenance, hashes and comparisons without distributing input arrays")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "data/prepared" / ("marmousi2_full_psv.npz" if args.mode == "full-psv" else "marmousi2_solid.npz")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    full, source_metadata = prepare_paper(args.raw_dir)
    reference = compare_reference(full, args.reference_prepared) if args.reference_prepared else None
    writer = write_full_model if args.mode == "full-psv" else write_solid_crop
    metadata = writer(full, source_metadata, args.output, reference)
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "mode": args.mode,
                      "native_shape_zx": metadata.get("native_shape_zx", metadata.get("solid_shape_zx")),
                      "water_rows_removed": metadata["water_rows_removed"],
                      "reference_exact": reference is not None, "native_archive_sha256": metadata["native_archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
