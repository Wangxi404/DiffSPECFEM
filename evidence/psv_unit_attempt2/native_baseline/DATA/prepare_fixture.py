#!/usr/bin/env python3
"""Generate an optional nonuniform coupled-unit solid-domain test fixture.

Par_file is the uniform two-material case. Par_file_tomography selects this
analytic fixture instead, while retaining the same fixed acoustic water and
geometry. This is synthetic regression data, independent of Marmousi-II.
"""
from pathlib import Path
import json
import numpy as np


def write_tomography(path, x, z, fields, state):
    xx, zz = np.meshgrid(x, z)
    values = [fields[f"{key}_{state}"] for key in ("vp", "vs", "rho")]
    rows = np.column_stack((xx.ravel(), zz.ravel(), *(value.ravel() for value in values)))
    with path.open("w") as stream:
        stream.write(f"{x[0]:.17g} {z[0]:.17g} {x[-1]:.17g} {z[-1]:.17g}\n")
        stream.write(f"{x[1]-x[0]:.17g} {z[1]-z[0]:.17g}\n{x.size} {z.size}\n")
        stream.write(" ".join(f"{value:.17g}" for field in values for value in (field.min(), field.max())) + "\n")
        np.savetxt(stream, rows, fmt="%.17g")
    if not np.array_equal(np.loadtxt(path, skiprows=4), rows):
        raise ValueError("Tomography numeric serialization mismatch")


def main():
    output = Path(__file__).resolve().parent
    x, z = np.linspace(0.0, 2000.0, 9), np.linspace(0.0, 1600.0, 9)
    xx, zz = np.meshgrid(x / 2000.0, z / 1600.0)
    fields = {
        "vp_initial": 3000.0 * (1 + 0.08 * np.sin(np.pi * xx) * np.sin(np.pi * zz)),
        "vs_initial": 1700.0 * (1 + 0.06 * np.cos(np.pi * xx) * np.sin(np.pi * zz)),
        "rho_initial": 2500.0 * (1 + 0.04 * np.sin(2 * np.pi * xx) * np.cos(np.pi * zz)),
    }
    fields["vp_true"] = fields["vp_initial"] * (1 + 0.025 * np.cos(2 * np.pi * xx) * np.sin(np.pi * zz))
    fields["vs_true"] = fields["vs_initial"] * (1 - 0.035 * np.sin(np.pi * xx) * np.sin(2 * np.pi * zz))
    fields["rho_true"] = fields["rho_initial"] * (1 + 0.02 * np.cos(2 * np.pi * xx) * np.cos(np.pi * zz))
    minimum_lambda = {}
    for state in ("initial", "true"):
        vp, vs, rho = (fields[f"{key}_{state}"] for key in ("vp", "vs", "rho"))
        if not all(np.isfinite(field).all() and (field > 0).all() for field in (vp, vs, rho)):
            raise ValueError("Nonphysical synthetic unit fixture")
        lam = rho * (vp ** 2 - 2 * vs ** 2)
        if not (lam > 0).all():
            raise ValueError("Negative first Lamé parameter in synthetic fixture")
        minimum_lambda[state] = float(lam.min())
        write_tomography(output / f"unit_tomo_{state}.xyz", x, z, fields, state)
    metadata = {
        "name": "Analytic nonuniform coupled P-SV unit solid domain",
        "source": "Deterministic trigonometric functions in prepare_fixture.py; no third-party input data",
        "array_axes": "native_z_up,x", "shape_zx": [9, 9],
        "x_extent_m": [0.0, 2000.0], "z_extent_m": [0.0, 1600.0],
        "water": {"native_z_extent_m": [1600.0, 2000.0], "vp": 1500.0, "vs": 0.0, "rho": 1000.0, "active": False},
        "active_parameters": ["vp", "vs", "rho"],
        "units": {"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"},
        "minimum_first_lame_parameter_pa": minimum_lambda,
        "field_ranges_SI": {key: [float(field.min()), float(field.max())] for key, field in fields.items()},
        "tomography_template": "Use Par_file_tomography as Par_file; copy unit_tomo_initial.xyz or unit_tomo_true.xyz to tomo_file.xyz",
        "native_validation_status": "Inputs generated and Lamé positivity checked; no native simulation claimed",
    }
    np.savez_compressed(output / "unit_model.npz", x_native=x, z_native=z, **fields,
                        metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    (output / "unit_model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output / "unit_model.npz"), "shape": [9, 9],
                      "minimum_first_lame_parameter_pa": minimum_lambda}, indent=2))


if __name__ == "__main__":
    main()
