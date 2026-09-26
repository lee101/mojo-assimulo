"""Correctness-gated benchmark for mojo-assimulo.

Every case verifies numerical agreement with the vectorised NumPy reference
before timing, so a regression in the Mojo kernels shows up as a correctness
failure rather than a suspiciously good number. The baselines are the fastest
reasonable NumPy formulations of the same algebra, not Python loops.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "tests"))

import reference as R  # noqa: E402
from mojo_assimulo import Dopri5, _lib as k  # noqa: E402
from mojo_assimulo import tableau as T  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_rk_combo(n: int = 1 << 18, ncoef: int = 5):
    """The five-term stage update, Mojo against `coef @ K[rows]` in NumPy.

    The NumPy form is the fastest fair equivalent: one fused matrix-vector
    product over a gathered view, which is what a NumPy implementation would
    actually use.
    """
    rng = np.random.default_rng(0)
    K = np.ascontiguousarray(rng.standard_normal((7, n)))
    y = np.ascontiguousarray(rng.standard_normal(n))
    name = ["A51", "A61", "A71", "E", "D"][ncoef - 2]
    rows, coef = T.as_rows(name), T.as_float(name)
    dst = np.empty(n, dtype=np.float64)
    h = 0.125

    k.rk_combo(n, h, y, 0, K, rows, coef, dst)
    want = y + h * (coef @ K[rows])
    assert np.allclose(dst, want, rtol=1e-11), "rk_combo mismatch"

    def mojo():
        k.rk_combo(n, h, y, 0, K, rows, coef, dst)

    def numpy():
        return y + h * (coef @ K[rows])

    return f"stage combo n={n} m={ncoef}", _time(numpy), _time(mojo)


def bench_err_norm(n: int = 1 << 18):
    """The local error vector and its weighted RMS norm."""
    rng = np.random.default_rng(1)
    K = np.ascontiguousarray(rng.standard_normal((7, n)))
    y = np.ascontiguousarray(rng.standard_normal(n))
    y1 = y + 0.01 * rng.standard_normal(n)
    atol = np.full(n, 1e-9)
    rtol = np.full(n, 1e-6)
    rows, coef = T.as_rows("E"), T.as_float("E")
    kerr = np.empty(n, dtype=np.float64)
    h = 0.01

    got = k.err_norm(n, h, K, rows, coef, y, y1, atol, rtol, kerr)
    want_vec, want_norm = R.err_norm(K, h, y, y1, atol, rtol)
    assert np.allclose(kerr, want_vec, rtol=1e-9), "error vector mismatch"
    assert abs(got - want_norm) <= 1e-9 * want_norm, "error norm mismatch"

    def mojo():
        return k.err_norm(n, h, K, rows, coef, y, y1, atol, rtol, kerr)

    def numpy():
        vec, nrm = R.err_norm(K, h, y, y1, atol, rtol)
        return vec, nrm

    return f"err vector+norm n={n}", _time(numpy), _time(mojo)


def _spiral_system(n, seed=2):
    """A dense linear system y' = A y with a known exponential solution."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    A = np.outer(a, b) / n
    return A


def bench_integrate(n: int = 2000, t_end: float = 2.0):
    """A whole DOPRI5 run, Mojo driver against the NumPy reference driver.

    Both drive the same algorithm with the same Python right-hand side, so the
    comparison isolates the state-vector algebra. The right-hand side is a
    Python callback in both, exactly as it is in Assimulo, so the per-stage
    callback cost is common to both and is not what this measures.
    """
    A = _spiral_system(n)
    y0 = np.linspace(0.5, 1.5, n)

    def rhs(t, y):
        return A @ y

    got = Dopri5(rhs, y0, rtol=1e-8, atol=1e-10, maxsteps=100000)
    tlist, ylist = got.integrate(t_end)
    rt, ry, _ = R.dopcor(rhs, 0.0, y0, t_end, 1e-8, 1e-10)
    assert len(tlist) == len(rt), (len(tlist), len(rt))
    assert np.allclose(ylist[-1], ry[-1], rtol=1e-5, atol=1e-7), "final state"

    def mojo():
        Dopri5(rhs, y0, rtol=1e-8, atol=1e-10, maxsteps=100000).integrate(t_end)

    def numpy():
        R.dopcor(rhs, 0.0, y0, t_end, 1e-8, 1e-10)

    return (f"dopri5 integrate n={n}", _time(numpy, 3), _time(mojo, 3),
            f"{got.statistics['nsteps']} steps")


def bench_interpolate(n: int = 1 << 18, points: int = 64):
    """Shampine dense output, Mojo against the same nested Horner in NumPy."""
    rng = np.random.default_rng(3)
    cont = np.ascontiguousarray(rng.standard_normal(5 * n))
    dst = np.empty(n, dtype=np.float64)
    thetas = np.linspace(0.0, 1.0, points)

    def np_interp(theta):
        t1 = 1.0 - theta
        return (cont[0:n]
                + theta * (cont[n:2 * n]
                           + t1 * (cont[2 * n:3 * n]
                                   + theta * (cont[3 * n:4 * n]
                                              + t1 * cont[4 * n:5 * n]))))

    k.dopri5_interp(n, cont, 0.37, dst)
    assert np.allclose(dst, np_interp(0.37), rtol=1e-12), "interp mismatch"

    def mojo():
        for theta in thetas:
            k.dopri5_interp(n, cont, float(theta), dst)

    def numpy():
        for theta in thetas:
            np_interp(float(theta))

    return f"dense output n={n} x{points}", _time(numpy), _time(mojo)


def main():
    print(f"{'case':<34}{'reference':>13}{'mojo-assimulo':>16}{'ratio':>9}")
    print("-" * 74)
    for fn in (bench_rk_combo, bench_err_norm, bench_interpolate,
               bench_integrate):
        out = fn()
        label, ref, got = out[:3]
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref * 1e3:>11.2f}ms{got * 1e3:>14.2f}ms"
              f"{ratio:>8.2f}x")
        if len(out) > 3:
            print(f"{'':<34}{out[3]}")


if __name__ == "__main__":
    main()
