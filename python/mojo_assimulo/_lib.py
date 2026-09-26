"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-assimulo.so"

_D = ctypes.c_double
_I = ctypes.c_int64
_C = ctypes.c_int

_SIGNATURES = {
    "asu_rk_combo": [_I, _D, _I, _I, _I, _I, _I, _I, _I],
    "asu_lincomb": [_I, _I, _I, _I, _I, _I],
    "asu_l2_norm": [_I, _I, _I],
    "asu_err_norm": [_I, _D, _I, _I, _I, _I, _I, _I, _I, _I, _I, _I],
    "asu_dopri5_hnew": [_D, _D, _D, _D, _D, _D, _D, _D, _D, _C, _C, _I],
    "asu_dopri5_cont_d": [_I, _D, _I, _I, _I, _I, _I],
    "asu_dopri5_cont_abc": [_I, _D, _I, _I, _I, _I, _I],
    "asu_dopri5_interp": [_I, _I, _D, _I],
    "asu_hinit_norms": [_I, _I, _I, _I, _I, _I],
    "asu_hinit_der2": [_I, _I, _I, _I, _I, _I, _D, _I],
    "asu_hinit_guess": [_D, _D, _D, _D, _I],
    "asu_hinit_step": [_D, _D, _D, _D, _D, _D, _I],
    "asu_stiffness": [_I, _I, _I, _I, _I, _D, _D, _I],
    "asu_rk34_error": [_I, _D, _I, _I, _I, _I, _I, _I, _D, _I],
    "asu_hermite_interp": [_I, _D, _I, _I, _I, _I, _D, _I],
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, argtypes in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = argtypes
    return lib


lib = _load()


def f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def rows(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


# --------------------------------------------------------------------------
# Scalar-output kernels
# --------------------------------------------------------------------------


def l2_norm(x) -> float:
    x = f64(x)
    out = np.zeros(1, dtype=np.float64)
    lib.asu_l2_norm(x.size, addr(x), addr(out))
    return float(out[0])


def dopri5_hnew(h, err, facold, beta, safe, fac1, fac2, hmax, posneg,
                accepted, prev_rejected) -> tuple:
    """PI step-size prediction. Returns (hnew, facold_new)."""
    out = np.zeros(2, dtype=np.float64)
    lib.asu_dopri5_hnew(
        float(h), float(err), float(facold), float(beta), float(safe),
        float(fac1), float(fac2), float(hmax), float(posneg),
        1 if accepted else 0, 1 if prev_rejected else 0, addr(out),
    )
    return float(out[0]), float(out[1])


def stiffness(k2, k6, y1, ysti, h, hlamb) -> float:
    """Returns the updated hlambda; stnum and stden are also written by the
    kernel but only hlambda carries state between steps."""
    k2, k6, y1, ysti = f64(k2), f64(k6), f64(y1), f64(ysti)
    out = np.zeros(3, dtype=np.float64)
    lib.asu_stiffness(
        k2.size, addr(k2), addr(k6), addr(y1), addr(ysti), float(h),
        float(hlamb), addr(out),
    )
    return float(out[2])


def hinit_norms(y, f0, atol, rtol) -> tuple:
    y, f0, atol, rtol = f64(y), f64(f0), f64(atol), f64(rtol)
    out = np.zeros(2, dtype=np.float64)
    lib.asu_hinit_norms(y.size, addr(y), addr(f0), addr(atol), addr(rtol),
                        addr(out))
    return float(out[0]), float(out[1])


def hinit_der2(y, f0, f1, atol, rtol, h) -> float:
    y, f0, f1, atol, rtol = f64(y), f64(f0), f64(f1), f64(atol), f64(rtol)
    out = np.zeros(1, dtype=np.float64)
    lib.asu_hinit_der2(
        y.size, addr(y), addr(f0), addr(f1), addr(atol), addr(rtol),
        float(h), addr(out),
    )
    return float(out[0])


def hinit_guess(dnf, dny, hmax, posneg) -> float:
    out = np.zeros(1, dtype=np.float64)
    lib.asu_hinit_guess(float(dnf), float(dny), float(hmax), float(posneg),
                         addr(out))
    return float(out[0])


def hinit_step(dnf, der2, h, hmax, posneg, iord=5.0) -> float:
    out = np.zeros(1, dtype=np.float64)
    lib.asu_hinit_step(
        float(dnf), float(der2), float(h), float(hmax), float(posneg),
        float(iord), addr(out),
    )
    return float(out[0])


# --------------------------------------------------------------------------
# Vector kernels
# --------------------------------------------------------------------------


def rk_combo(n, h, y, yoff, ks, r, coef, dst):
    """dst[:] = y[yoff:yoff+n] + h * sum_j coef[j] * ks[r[j]]."""
    y = f64(y)
    ks = f64(ks).reshape(-1, n)
    r, coef = rows(r), f64(coef)
    lib.asu_rk_combo(
        n, float(h), addr(y), addr(ks), addr(r), addr(coef), coef.size,
        int(yoff), addr(dst),
    )
    return dst


def lincomb(n, ks, r, coef, dst):
    ks = f64(ks).reshape(-1, n)
    r, coef = rows(r), f64(coef)
    lib.asu_lincomb(n, addr(ks), addr(r), addr(coef), coef.size, addr(dst))
    return dst


def err_norm(n, h, ks, r, coef, y, y1, atol, rtol, kerr):
    """Writes the error vector into `kerr` and returns the scaled norm."""
    y, y1, atol, rtol = f64(y), f64(y1), f64(atol), f64(rtol)
    ks = f64(ks).reshape(-1, n)
    r, coef = rows(r), f64(coef)
    out = np.zeros(1, dtype=np.float64)
    lib.asu_err_norm(
        n, float(h), addr(ks), addr(r), addr(coef), coef.size, addr(y),
        addr(y1), addr(atol), addr(rtol), addr(kerr), addr(out),
    )
    return float(out[0])


def dopri5_cont_d(n, h, ks, r, dcoef, cont):
    ks = f64(ks).reshape(-1, n)
    r, dcoef = rows(r), f64(dcoef)
    lib.asu_dopri5_cont_d(n, float(h), addr(ks), addr(r), addr(dcoef),
                          dcoef.size, addr(cont))
    return cont


def dopri5_cont_abc(n, h, y, y1, k1, k2, cont):
    y, y1, k1, k2 = f64(y), f64(y1), f64(k1), f64(k2)
    lib.asu_dopri5_cont_abc(n, float(h), addr(y), addr(y1), addr(k1),
                            addr(k2), addr(cont))
    return cont


def dopri5_interp(n, cont, theta, dst):
    cont = f64(cont)
    lib.asu_dopri5_interp(n, addr(cont), float(theta), addr(dst))
    return dst


def rk34_error(n, h, y, ks, r, coef, atol, rtol, dst):
    y, atol = f64(y), f64(atol)
    ks = f64(ks).reshape(-1, n)
    r, coef = rows(r), f64(coef)
    lib.asu_rk34_error(n, float(h), addr(y), addr(ks), addr(r), addr(coef),
                       coef.size, addr(atol), float(rtol), addr(dst))
    return dst


def hermite_interp(n, theta, y, y_next, f_low, f_high, h, dst):
    y, y_next = f64(y), f64(y_next)
    f_low, f_high = f64(f_low), f64(f_high)
    lib.asu_hermite_interp(n, float(theta), addr(y), addr(y_next),
                           addr(f_low), addr(f_high), float(h), addr(dst))
    return dst
