#!/usr/bin/env python3
"""NumPy-only gradient, Hermitian-adjoint, and observable Taylor checks.

All results are JSON serializable. Invalid input raises ValueError; a valid but
uninformative numerical experiment returns passed=False. Different floating
precisions are allowed and reported, but integer/object gradients are rejected.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fc" or not array.size:
        raise ValueError(f"{name} must contain nonempty floating/complex data")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _tolerances(atol, rtol):
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("atol and rtol must be finite and nonnegative")


def _norm(value):
    scale = float(np.max(np.abs(value)))
    return scale * float(np.linalg.norm(value / scale)) if scale else 0.0


def _pair(reference, candidate):
    """Flatten matching arrays, including structured arrays, without field loss."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    if a.dtype.names or b.dtype.names:
        if not a.dtype.names or set(a.dtype.names) != set(b.dtype.names or ()):
            raise ValueError("structured field names do not match")
        fields = [_pair(a[name], b[name]) for name in a.dtype.names]
        return tuple(np.concatenate([field[i] for field in fields]) for i in (0, 1))
    a, b = _numeric(a, "reference"), _numeric(b, "candidate")
    if a.dtype.kind != b.dtype.kind:
        raise ValueError("reference and candidate must both be real or both complex")
    dtype = np.result_type(a.dtype, b.dtype, np.float64)
    return a.astype(dtype).ravel(), b.astype(dtype).ravel()


def compare_gradients(reference, candidate, *, atol=1e-10, rtol=1e-5):
    """Compare matching numeric/structured arrays; near-zero pairs cannot pass."""
    _tolerances(atol, rtol)
    a, b = _pair(reference, candidate)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = b - a
        na, nb, nd = (_norm(v) for v in (a, b, delta))
        ia, ib, di = (float(np.max(np.abs(v))) for v in (a, b, delta))
    if not np.isfinite([na, nb, nd, ia, ib, di]).all():
        raise ValueError("comparison overflowed; rescale the arrays")
    informative = ia > atol and ib > atol
    passed = informative and nd <= atol * np.sqrt(a.size) + rtol * na and di <= atol + rtol * ia
    return {"passed": bool(passed), "informative": bool(informative),
            "reason": "ok" if passed else ("uninformative near-zero gradient" if not informative else "gradient mismatch"),
            "shape": list(np.shape(reference)), "reference_dtype": str(np.asarray(reference).dtype),
            "candidate_dtype": str(np.asarray(candidate).dtype),
            "dtype_match": np.asarray(reference).dtype == np.asarray(candidate).dtype,
            "relative_l2": nd / na if na else None,
            "relative_linf": di / ia if ia else None,
            "cosine": float(np.real(np.vdot(a / na, b / nb))) if na and nb else None,
            "absolute_linf": di, "reference_l2": na, "candidate_l2": nb}


def _scalar(value):
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def check_adjoint(forward_action, transpose_action, x, y, atol=1e-10, rtol=1e-6):
    """Check <Ax,y> = <x,A*y> using np.vdot (conjugating its first input).

    transpose_action must implement the Hermitian adjoint for complex operators.
    Zero/near-zero bilinear forms are reported as uninformative.
    """
    _tolerances(atol, rtol)
    x, y = _numeric(x, "x"), _numeric(y, "y")
    ax = _numeric(forward_action(x), "forward_action(x)")
    aty = _numeric(transpose_action(y), "transpose_action(y)")
    if ax.shape != y.shape or aty.shape != x.shape:
        raise ValueError("action output shapes must match y and x, respectively")
    lhs, rhs = np.vdot(ax, y), np.vdot(x, aty)
    scale, error = max(abs(lhs), abs(rhs)), abs(lhs - rhs)
    if not np.isfinite([scale, error]).all():
        raise ValueError("adjoint products overflowed; rescale the probes")
    informative = scale > atol
    return {"passed": bool(informative and error <= atol + rtol * scale),
            "informative": bool(informative), "lhs": _scalar(lhs), "rhs": _scalar(rhs),
            "absolute_error": float(error), "relative_error": float(error / scale) if scale else None}


def taylor_test(evaluate, m, g, direction, steps=None, *, atol=1e-10, rtol=1e-6,
                min_order=1.8, max_order=2.2, roundoff_factor=64.0):
    """Test a real objective and g under Re(vdot(g, direction)).

    Raw positive-side first-order remainders and central derivative errors are
    retained at every supplied step. Passing requires two consecutive observable
    O(h**2) intervals AND two consecutive reliable central derivative matches.
    Linear/flat objectives can therefore be inconclusive, never false successes.
    Steps must be positive, distinct, and strictly decreasing; no fitting,
    residual-floor subtraction, step removal, or direction normalization occurs.
    """
    _tolerances(atol, rtol)
    m, g, direction = (_numeric(v, n) for v, n in ((m, "m"), (g, "g"), (direction, "direction")))
    if m.shape != g.shape or m.shape != direction.shape:
        raise ValueError("m, g, and direction shapes must match")
    if not 0 < min_order <= max_order or not np.isfinite([min_order, max_order, roundoff_factor]).all() or roundoff_factor < 1:
        raise ValueError("invalid order bounds or roundoff_factor")
    steps = np.asarray(np.logspace(-1, -7, 13) if steps is None else steps, dtype=float)
    if steps.ndim != 1 or len(steps) < 3 or not np.isfinite(steps).all() or np.any(steps <= 0) or np.any(np.diff(steps) >= 0):
        raise ValueError("provide at least three finite, positive, decreasing steps")

    def objective(model):
        value = np.asarray(evaluate(model))
        if value.shape or value.dtype.kind not in "fiu" or not np.isfinite(value):
            raise ValueError("evaluate must return a finite real scalar")
        eps = np.finfo(value.dtype).eps if value.dtype.kind == "f" else np.finfo(float).eps
        return float(value), eps

    f0, eps0 = objective(m)
    eps = max(eps0, *(np.finfo(v.real.dtype).eps for v in (m, g, direction)))
    slope = float(np.real(np.vdot(g, direction)))
    ng, nm, nv = (_norm(v) for v in (g, m, direction))
    if not np.isfinite([slope, ng, nm, nv]).all():
        raise ValueError("Taylor products overflowed; rescale inputs")
    informative = nv > 0 and abs(slope) > max(atol, roundoff_factor * eps * ng * nv)
    rows = []
    for h in steps:
        plus, minus = m + h * direction, m - h * direction
        fp, ep = objective(plus)
        fm, em = objective(minus)
        local_eps = max(eps, ep, em)
        remainder = abs(fp - f0 - h * slope)
        central = (fp - fm) / (2 * h)
        floor = roundoff_factor * local_eps * (abs(fp) + abs(f0) + abs(h * slope))
        central_floor = roundoff_factor * local_eps * (abs(fp) + abs(fm)) / (2 * h)
        if not np.isfinite([remainder, central, floor, central_floor]).all():
            raise ValueError("Taylor diagnostics overflowed; rescale the objective or steps")
        tolerance = atol + rtol * max(abs(slope), abs(central))
        resolved = (h * nv > roundoff_factor * local_eps * max(nm, np.finfo(float).tiny)
                    and not np.array_equal(plus, m) and not np.array_equal(minus, m))
        rows.append({"step": float(h), "f_plus": fp, "f_minus": fm,
                     "first_order_residual": float(remainder), "central_directional_derivative": float(central),
                     "central_directional_error": float(abs(central - slope)),
                     "roundoff_floor": float(floor), "central_roundoff_floor": float(central_floor),
                     "perturbation_resolved": bool(resolved), "observable": bool(resolved and remainder > floor),
                     "central_matches": bool(resolved and central_floor <= tolerance and abs(central - slope) <= tolerance + central_floor),
                     "first_order_order": None, "central_error_order": None})
    for a, b in zip(rows, rows[1:]):
        for residual, order in (("first_order_residual", "first_order_order"), ("central_directional_error", "central_error_order")):
            if a[residual] > 0 and b[residual] > 0:
                b[order] = float((np.log(a[residual]) - np.log(b[residual])) / (np.log(a["step"]) - np.log(b["step"])))
    intervals = [a["observable"] and b["observable"] and b["first_order_order"] is not None
                 and min_order <= b["first_order_order"] <= max_order for a, b in zip(rows, rows[1:])]
    second_order = any(a and b for a, b in zip(intervals, intervals[1:]))
    central_agrees = any(a["central_matches"] and b["central_matches"] for a, b in zip(rows, rows[1:]))
    passed = bool(informative and second_order and central_agrees)
    reason = "ok" if passed else ("uninformative gradient direction" if not informative else
             "central directional derivative disagrees" if not central_agrees else "insufficient observable second-order intervals")
    return {"passed": passed, "informative": bool(informative), "reason": reason,
            "f0": f0, "directional_derivative": slope, "machine_epsilon": float(eps),
            "second_order_observed": bool(second_order), "central_agrees": bool(central_agrees), "steps": rows}


def _load(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            return {name: loaded[name] for name in loaded.files}
    return {"gradient": loaded}


def _file_metadata(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest(), "size_bytes": Path(path).stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="compare NPY/NPZ gradient arrays")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--keys", nargs="+", help="exact array keys; default: shared gradient_* (or NPY gradient)")
    compare.add_argument("--atol", type=float, default=1e-10)
    compare.add_argument("--rtol", type=float, default=1e-5)
    compare.add_argument("--output", help="write JSON here; also print it to stdout")
    args = parser.parse_args(argv)
    report = {"passed": False}
    try:
        ref, cand = _load(args.reference), _load(args.candidate)
        keys = args.keys or sorted(k for k in ref.keys() & cand.keys() if k.startswith("gradient_") or k == "gradient")
        if not keys:
            raise ValueError("no shared gradient_* arrays; select physical parameter keys with --keys")
        if any(k not in ref or k not in cand for k in keys):
            raise ValueError("every selected key must occur in both files")
        report.update(reference=_file_metadata(args.reference), candidate=_file_metadata(args.candidate),
                      atol=args.atol, rtol=args.rtol, keys=keys)
        report["comparisons"] = {k: compare_gradients(ref[k], cand[k], atol=args.atol, rtol=args.rtol) for k in keys}
        report["passed"] = all(r["passed"] for r in report["comparisons"].values())
    except (OSError, ValueError, TypeError) as exc:
        report["error"] = str(exc)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
