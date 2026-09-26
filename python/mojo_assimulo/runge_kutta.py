"""Explicit Runge-Kutta solvers with the vector algebra in Mojo.

`Dopri5` mirrors the driver loop of Assimulo's `Dopri5` (which calls the
DOPRI5 FORTRAN core) with the right-hand side supplied as a Python callback.
`RungeKutta4` and `RungeKutta34` mirror the `_step` methods of the same-named
Assimulo solvers. The loop, the step-size policy and the right-hand side stay
in Python for the same reason they do upstream: the right-hand side is user
code, and the step-size policy is control flow, not arithmetic.
"""

import numpy as np

from . import _lib as _k
from .tableau import C, as_float, as_rows

# The DOPRI5 FORTRAN tests for stiffness every NSTIFF accepted points.
NSTIFF = 6

_ROW0 = np.array([0], dtype=np.int64)
_ROW1 = np.array([1], dtype=np.int64)
_ROW2 = np.array([2], dtype=np.int64)
_ONE = np.array([1.0], dtype=np.float64)
_HALF = np.array([0.5], dtype=np.float64)


def _t(name):
    return as_rows(name), as_float(name)


class _BaseSolver:
    """Shared tolerance handling, statistics and right-hand side dispatch."""

    def __init__(self, rhs, y0, rtol, atol):
        self.rhs = rhs
        self.y0 = np.ascontiguousarray(y0, dtype=np.float64)
        self.n = self.y0.size
        self.rtol = float(rtol)
        atol = np.atleast_1d(np.asarray(atol, dtype=np.float64))
        if atol.size == 1:
            atol = np.full(self.n, atol[0])
        if atol.size != self.n:
            raise ValueError("atol must be scalar or of length n")
        self.atol = np.ascontiguousarray(atol)
        self.statistics = {
            "nsteps": 0, "nfcns": 0, "nerrfails": 0, "nlstfail": 0,
        }
        self.tlist = []
        self.ylist = []
        # Every accepted (t, y), independent of the reporting mode, so callers
        # can compare dense output against the piecewise solution.
        self.step_points = []

    def _f(self, t, y, dst):
        out = np.asarray(self.rhs(t, y), dtype=np.float64)
        if out.shape != dst.shape:
            raise ValueError(
                f"rhs returned shape {out.shape}, expected {dst.shape}"
            )
        np.copyto(dst, out)
        self.statistics["nfcns"] += 1
        return dst


class Dopri5(_BaseSolver):
    """Explicit Runge-Kutta of order (4)5 with step-size control and
    continuous output, after Dormand and Prince.

    Every state-vector operation of the DOPCOR loop is a call into
    `src/kernels.mojo`; this class is the driver around them.
    """

    def __init__(self, rhs, y0, t0=0.0, rtol=1e-6, atol=1e-6, safe=0.9,
                 fac1=0.2, fac2=10.0, beta=0.04, maxh=np.inf,
                 maxsteps=100000, inith=0.0):
        super().__init__(rhs, y0, rtol, atol)
        self.t0 = float(t0)
        self.safe = float(safe)
        self.fac1 = float(fac1)
        self.fac2 = float(fac2)
        self.beta = float(beta)
        self.maxh = float(maxh)
        self.maxsteps = int(maxsteps)
        self.inith = float(inith)
        self.rtol_vec = np.full(self.n, self.rtol, dtype=np.float64)

        n = self.n
        self.K = np.zeros((7, n), dtype=np.float64)
        self.Y2 = np.zeros((2, n), dtype=np.float64)   # (y, y1)
        self.Y1 = np.zeros(n, dtype=np.float64)
        self.YSTI = np.zeros(n, dtype=np.float64)
        self.KERR = np.zeros(n, dtype=np.float64)
        self.CONT = np.zeros(5 * n, dtype=np.float64)
        self.OUT = np.zeros(n, dtype=np.float64)

        self._rows = {
            name: _t(name)
            for name in ("A21", "A31", "A41", "A51", "A61", "A71", "E", "D")
        }
        self._hlamb = 0.0
        self._interp_t = None
        self._interp_h = None
        self._k1_valid = False

    def _attempt(self, t, y, h):
        """One full DOPCOR attempt. Returns (x + h, error norm)."""
        n, K, Y2 = self.n, self.K, self.Y2
        np.copyto(Y2[0], y)
        xph = t + h
        # DOPCOR reuses the previous step's K2 as this step's K1 (FCN is only
        # called again when IRTRN >= 2), so the first attempt is the only one
        # that needs the extra right-hand side evaluation.
        if not self._k1_valid:
            np.copyto(K[0], self._f(t, y, self.OUT))
            self._k1_valid = True
        _k.rk_combo(n, h, Y2, 0, K, *self._rows["A21"], self.Y1)
        np.copyto(K[1], self._f(t + C[0] * h, self.Y1, self.OUT))
        _k.rk_combo(n, h, Y2, 0, K, *self._rows["A31"], self.Y1)
        np.copyto(K[2], self._f(t + C[1] * h, self.Y1, self.OUT))
        _k.rk_combo(n, h, Y2, 0, K, *self._rows["A41"], self.Y1)
        np.copyto(K[3], self._f(t + C[2] * h, self.Y1, self.OUT))
        _k.rk_combo(n, h, Y2, 0, K, *self._rows["A51"], self.Y1)
        np.copyto(K[4], self._f(t + C[3] * h, self.Y1, self.OUT))
        _k.rk_combo(n, h, Y2, 0, K, *self._rows["A61"], self.YSTI)
        np.copyto(K[5], self._f(xph, self.YSTI, self.OUT))
        _k.rk_combo(n, h, Y2, 0, K, *self._rows["A71"], self.Y1)
        np.copyto(K[6], self._f(xph, self.Y1, self.OUT))

        # FORTRAN loop 40 runs on every attempt, before the error test.
        _k.dopri5_cont_d(n, h, K, *self._rows["D"], self.CONT)
        np.copyto(Y2[1], self.Y1)
        err = _k.err_norm(
            n, h, K, *self._rows["E"], Y2[0], Y2[1], self.atol, self.rtol_vec,
            self.KERR,
        )
        return xph, err

    def _initial_step(self, t, y, hmax, posneg):
        """HINIT, the FORTRAN's first-guess step size.

        Three kernel calls: the scaled sums, the first guess, and the final
        MIN(100*|H|, H1, HMAX) after the explicit Euler probe.
        """
        n = self.n
        np.copyto(self.Y1, self._f(t, y, self.OUT))
        dnf, dny = _k.hinit_norms(y, self.Y1, self.atol, self.rtol_vec)
        h = _k.hinit_guess(dnf, dny, hmax, posneg)
        # One explicit Euler step y + h*F0, then a second-derivative estimate.
        _k.rk_combo(n, h, y, 0, self.Y1, _ROW0, _ONE, self.YSTI)
        np.copyto(self.OUT, self._f(t + h, self.YSTI, self.OUT))
        der2 = _k.hinit_der2(y, self.Y1, self.OUT, self.atol, self.rtol_vec, h)
        return _k.hinit_step(dnf, der2, h, hmax, posneg, 5.0)

    def integrate(self, tf, output_list=None):
        t, y = self.t0, self.y0.copy()
        n = self.n
        posneg = 1.0 if tf - t >= 0.0 else -1.0
        hmax = abs(self.maxh)
        if self.inith == 0.0:
            h = self._initial_step(t, y, hmax, posneg)
        else:
            h = self.inith
        self._hlamb = 0.0
        facold = 1.0e-4
        reject = False
        last = False
        naccpt = 0
        nsteps = 0
        out_idx = 0
        # With an explicit output list the result is exactly that list, so the
        # initial point is not prepended.
        if output_list is None:
            self.tlist = [t]
            self.ylist = [y.copy()]
        else:
            self.tlist = []
            self.ylist = []
        self._interp_t, self._interp_h = None, None

        while True:
            if nsteps > self.maxsteps:
                raise RuntimeError(
                    f"more than maxsteps={self.maxsteps} steps are needed"
                )
            if 0.1 * abs(h) <= abs(t) * 2.220446049250313e-16:
                raise RuntimeError("step size too small")
            if (t + 1.01 * h - tf) * posneg > 0.0:
                h = tf - t
                last = True
            nsteps += 1
            self.statistics["nsteps"] += 1

            xph, err = self._attempt(t, y, h)
            hnew, facold = _k.dopri5_hnew(
                h, err, facold, self.beta, self.safe, self.fac1, self.fac2,
                hmax, posneg, err <= 1.0, reject,
            )

            if err <= 1.0:
                _k.dopri5_cont_abc(n, h, y, self.Y1, self.K[0], self.K[6],
                                   self.CONT)
                t_old = t
                y = self.Y1.copy()
                t = xph
                naccpt += 1
                self.step_points.append((t, y.copy()))
                self._interp_t, self._interp_h = t_old, h
                # K2new becomes the next step's K1.
                np.copyto(self.K[0], self.K[6])
                self._k1_valid = True
                if naccpt % NSTIFF == 0:
                    self._hlamb = _k.stiffness(
                        self.K[6], self.K[5], self.Y1, self.YSTI, h,
                        self._hlamb,
                    )
                if last:
                    if output_list is None:
                        self.tlist.append(t)
                        self.ylist.append(y.copy())
                    break
                if output_list is None:
                    self.tlist.append(t)
                    self.ylist.append(y.copy())
                else:
                    while (out_idx < len(output_list)
                           and output_list[out_idx] <= t):
                        tau = output_list[out_idx]
                        self.tlist.append(tau)
                        self.ylist.append(self.interpolate(tau))
                        out_idx += 1
                reject = False
            else:
                self.statistics["nerrfails"] += 1
                reject = True
                last = False
            h = hnew

        if output_list is not None:
            while out_idx < len(output_list):
                self.tlist.append(output_list[out_idx])
                self.ylist.append(self.interpolate(output_list[out_idx]))
                out_idx += 1
        return self.tlist, self.ylist

    def interpolate(self, time):
        """Shampine dense output across the last accepted step."""
        if self._interp_h is None or self._interp_h == 0.0:
            raise RuntimeError("no step has been taken yet")
        theta = (time - self._interp_t) / self._interp_h
        return _k.dopri5_interp(self.n, self.CONT, theta, self.OUT).copy()


class RungeKutta4(_BaseSolver):
    """Classical fixed-step Runge-Kutta of order 4."""

    def __init__(self, rhs, y0, h=0.01):
        super().__init__(rhs, y0, 1.0, 1.0)
        self.h = float(h)
        n = self.n
        self.K = np.zeros((4, n), dtype=np.float64)
        self.Y1 = np.zeros(n, dtype=np.float64)
        self.OUT = np.zeros(n, dtype=np.float64)
        self._rk = _t("RK4")

    def step(self, t, y, h):
        n, K = self.n, self.K
        np.copyto(K[0], self._f(t, y, self.OUT))
        _k.rk_combo(n, h, y, 0, K, _ROW0, _HALF, self.Y1)
        np.copyto(K[1], self._f(t + h / 2.0, self.Y1, self.OUT))
        _k.rk_combo(n, h, y, 0, K, _ROW1, _HALF, self.Y1)
        np.copyto(K[2], self._f(t + h / 2.0, self.Y1, self.OUT))
        _k.rk_combo(n, h, y, 0, K, _ROW2, _ONE, self.Y1)
        np.copyto(K[3], self._f(t + h, self.Y1, self.OUT))
        y_next = np.empty(n, dtype=np.float64)
        _k.rk_combo(n, h, y, 0, K, *self._rk, y_next)
        self.statistics["nsteps"] += 1
        return t + h, y_next

    def integrate(self, tf, h=None):
        t, y = 0.0, self.y0.copy()
        h = self.h if h is None else float(h)
        self.tlist = [t]
        self.ylist = [y.copy()]
        while t + h < tf:
            h = min(h, abs(tf - t))
            t, y = self.step(t, y, h)
            self.tlist.append(t)
            self.ylist.append(y.copy())
        t, y = self.step(t, y, min(h, abs(tf - t)))
        self.tlist.append(t)
        self.ylist.append(y.copy())
        return self.tlist, self.ylist


class RungeKutta34(_BaseSolver):
    """Adaptive Bogacki-Shampine 3(2) with a fourth-order update, a scaled
    error norm, an order-4 step controller and Hermite dense output.

    Step rejection is not implemented, matching the upstream solver.
    """

    def __init__(self, rhs, y0, rtol=1e-4, atol=1e-6, h=0.01, maxsteps=10000):
        super().__init__(rhs, y0, rtol, atol)
        self.h = float(h)
        self.maxsteps = int(maxsteps)
        n = self.n
        self.K = np.zeros((5, n), dtype=np.float64)   # Y1, Y2, Y3, Y4, Z3
        self.Y1 = np.zeros(n, dtype=np.float64)
        self.OUT = np.zeros(n, dtype=np.float64)
        self.YLOW = np.zeros(n, dtype=np.float64)
        self._upd = _t("BS23_UPDATE")
        self._err = _t("BS23_ERROR")
        self._z3 = _t("BS23_Z3")

    def _step(self, t, y, h):
        n, K = self.n, self.K
        _k.rk_combo(n, h, y, 0, K, _ROW0, _HALF, self.Y1)
        np.copyto(K[1], self._f(t + h / 2.0, self.Y1, self.OUT))
        _k.rk_combo(n, h, y, 0, K, _ROW1, _HALF, self.Y1)
        np.copyto(K[2], self._f(t + h / 2.0, self.Y1, self.OUT))
        _k.rk_combo(n, h, y, 0, K, *self._z3, self.Y1)
        np.copyto(K[4], self._f(t + h, self.Y1, self.OUT))
        _k.rk_combo(n, h, y, 0, K, _ROW2, _ONE, self.Y1)
        np.copyto(K[3], self._f(t + h, self.Y1, self.OUT))
        self.statistics["nfcns"] += 4

        err_vec = np.empty(n, dtype=np.float64)
        _k.rk34_error(n, h, y, K, *self._err, self.atol, self.rtol, err_vec)
        error = _k.l2_norm(err_vec)

        y_next = np.empty(n, dtype=np.float64)
        _k.rk_combo(n, h, y, 0, K, *self._upd, y_next)
        t_next = t + h
        np.copyto(self.YLOW, K[0])
        np.copyto(K[0], self._f(t_next, y_next, self.OUT))
        self.statistics["nfcns"] += 1

        def interpolate(time):
            theta = (time - t) / (t_next - t)
            dst = np.empty(n, dtype=np.float64)
            return _k.hermite_interp(
                n, theta, y, y_next, self.YLOW, K[0], t_next - t, dst
            )

        self.interpolate = interpolate
        return t_next, y_next, error

    @staticmethod
    def adjust_stepsize(h, error):
        if error == 0.0:
            fac = 2.0
        else:
            fac = min((1.0 / error) ** 0.25, 2.0)
        return h * fac

    def integrate(self, tf):
        t, y = 0.0, self.y0.copy()
        h = min(self.h, abs(tf - t))
        np.copyto(self.K[0], self._f(t, y, self.OUT))
        self.tlist = [t]
        self.ylist = [y.copy()]
        for _ in range(self.maxsteps):
            if not t + h < tf:
                break
            t, y, error = self._step(t, y, h)
            self.statistics["nsteps"] += 1
            self.tlist.append(t)
            self.ylist.append(y.copy())
            h = min(self.adjust_stepsize(h, error), abs(tf - t))
        else:
            raise RuntimeError(
                "final time not reached within maximum number of steps"
            )
        t, y, _ = self._step(t, y, h)
        self.statistics["nsteps"] += 1
        self.tlist.append(t)
        self.ylist.append(y.copy())
        return self.tlist, self.ylist
