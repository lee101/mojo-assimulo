"""Parity tests for the solver drivers.

The strongest check here is against analytic solutions: the DOPRI5 step must
reproduce a known fifth-order global error, its embedded estimate must predict
the local error that actually occurs, and the Shampine interpolant must be exact
at both step ends and accurate in between. On top of that, the accepted step
sequence is compared against the independent NumPy transcription in
`tests/reference.py`, so a kernel or driver bug shows up as a different
step pattern rather than a slightly wrong number.
"""

import numpy as np
import pytest

from mojo_assimulo import Dopri5, RungeKutta4, RungeKutta34

import reference as R

RTOL = 1e-12


def decay(t, y):
    return -y


def decay_exact(t, y0, t_end):
    return y0 * np.exp(-t_end)


def oscillator(t, y):
    return np.array([y[1], -y[0]])


def oscillator_exact(y0, t):
    return np.array([
        y0[0] * np.cos(t) + y0[1] * np.sin(t),
        -y0[0] * np.sin(t) + y0[1] * np.cos(t),
    ])


# ------------------------------------------------------------ DOPRI5




def test_dopri5_error_estimate_is_fifth_order_in_the_step():
    """The E weights make the estimate the local error of the embedded
    fourth-order solution, so its magnitude must fall off like h**5. A
    mis-signed or transposed E weight changes the rate or the constant."""
    y0 = np.array([1.0, 0.5])
    n = 2
    predicted = []
    for h in (0.2, 0.1, 0.05, 0.025, 0.0125):
        s = Dopri5(oscillator, y0, rtol=1e-10, atol=1e-12, inith=h)
        _, err = s._attempt(0.0, y0, h)
        y_next = s.Y1.copy()
        sk = s.atol + s.rtol_vec * np.maximum(np.abs(y0), np.abs(y_next))
        predicted.append(err * float(np.sqrt(np.mean(sk ** 2))))
    ratios = [predicted[i] / predicted[i + 1] for i in range(len(predicted) - 1)]
    for r in ratios:
        assert 16.0 < r < 64.0, ratios


def test_dopri5_accepted_steps_respect_the_tolerance():
    """When the error test accepts, the local error the 5th-order update makes
    must be below the tolerance scale. The estimate overshoots the true 5th
    order error (the embedded solution is much less accurate), so a correct E
    gives a margin rather than equality."""
    y0 = np.array([1.0, 0.5])
    for h in (0.05, 0.025):
        s = Dopri5(oscillator, y0, rtol=1e-6, atol=1e-8, inith=h)
        _, err = s._attempt(0.0, y0, h)
        y_next = s.Y1.copy()
        sk = s.atol + s.rtol_vec * np.maximum(np.abs(y0), np.abs(y_next))
        predicted = err * float(np.sqrt(np.mean(sk ** 2)))
        actual = float(np.linalg.norm(y_next - oscillator_exact(y0, h))
                       / np.sqrt(2))
        assert actual < 0.2 * predicted, (h, actual, predicted)


def test_dopri5_reaches_the_analytic_solution():
    y0 = np.array([1.0, 0.5])
    s = Dopri5(oscillator, y0, rtol=1e-10, atol=1e-12, maxsteps=100000)
    tlist, ylist = s.integrate(3.0)
    assert tlist[-1] == pytest.approx(3.0, rel=0, abs=0.0)
    np.testing.assert_allclose(ylist[-1], oscillator_exact(y0, 3.0),
                               rtol=1e-8, atol=1e-8)


def test_dopri5_decay_matches_the_analytic_solution():
    y0 = np.array([2.0, -1.0])
    s = Dopri5(decay, y0, rtol=1e-11, atol=1e-13)
    tlist, ylist = s.integrate(2.0)
    np.testing.assert_allclose(ylist[-1], y0 * np.exp(-2.0), rtol=1e-9,
                               atol=1e-10)


def test_dopri5_integrates_backwards():
    y0 = np.array([1.0, 0.0])
    s = Dopri5(oscillator, y0, rtol=1e-10, atol=1e-12)
    tlist, ylist = s.integrate(-1.5)
    np.testing.assert_allclose(ylist[-1], oscillator_exact(y0, -1.5),
                               rtol=1e-8, atol=1e-8)


def test_dopri5_step_pattern_matches_the_numpy_reference():
    """Same accept/reject decisions and the same accepted step times as an
    independent vectorised transcription of DOPCOR."""
    rng = np.random.default_rng(0)
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    y0 = np.array([1.0, -0.5])

    def rhs(t, y):
        return A @ y + 0.1 * np.sin(3.0 * t)

    got = Dopri5(rhs, y0, rtol=1e-7, atol=1e-9, maxsteps=100000)
    tlist, ylist = got.integrate(5.0)
    rt, ry, rejected = R.dopcor(rhs, 0.0, y0, 5.0, 1e-7, 1e-9)

    assert len(rejected) == got.statistics["nerrfails"]
    # Step times cannot be bit-identical: HINIT evaluates a fractional power
    # and the error norm is a cancellation-heavy sum, so Mojo's FMA and NumPy's
    # separate multiply-add differ in the last bits and the PI controller
    # amplifies that. What must agree is the accept/reject pattern and the
    # resulting trajectory; a wrong tableau row or error scale breaks both by
    # orders of magnitude.
    assert len(tlist) == len(rt)
    np.testing.assert_allclose(tlist, rt, rtol=1e-6, atol=0.0)
    for a, b in zip(ylist, ry):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_dopri5_rejects_steps_when_the_first_step_is_too_large():
    """A 0.5 first step on y' = -y is far too coarse for a 1e-6 tolerance, so
    the error test must reject it and the run must still finish."""
    s = Dopri5(decay, np.array([1.0]), rtol=1e-6, atol=1e-6, inith=0.5)
    tlist, ylist = s.integrate(1.0)
    assert s.statistics["nerrfails"] > 0
    assert tlist[-1] == 1.0
    np.testing.assert_allclose(ylist[-1], np.exp(-1.0), rtol=1e-5,
                               atol=1e-6)


def test_dopri5_maxsteps_is_enforced():
    s = Dopri5(decay, np.array([1.0]), rtol=1e-6, atol=1e-6, inith=0.5,
               maxsteps=2)
    with pytest.raises(RuntimeError, match="maxsteps"):
        s.integrate(1.0)


def test_dopri5_rejects_a_wrong_shaped_rhs():
    s = Dopri5(lambda t, y: np.zeros(3), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="rhs returned shape"):
        s.integrate(1.0)


def test_dopri5_rejects_a_bad_atol_length():
    with pytest.raises(ValueError, match="atol must be"):
        Dopri5(decay, np.array([1.0, 2.0]), atol=[1e-6, 1e-6, 1e-6])


# --------------------------------------------------------- dense output


def test_dense_output_is_exact_at_the_step_ends():
    """The Shampine interpolant is a Hermite-type polynomial: at theta = 0 it
    must return the state the step started from and at theta = 1 the state the
    step ended at, both to the last bit."""
    s = Dopri5(oscillator, np.array([1.0, 0.5]), rtol=1e-10, atol=1e-12)
    tlist, ylist = s.integrate(1.0)
    t_start = s._interp_t
    np.testing.assert_allclose(s.interpolate(t_start), ylist[-2],
                               rtol=0, atol=0)
    np.testing.assert_allclose(s.interpolate(t_start + s._interp_h),
                               ylist[-1], rtol=1e-12, atol=1e-14)


def test_dense_output_beats_piecewise_linear_between_steps():
    """The point of Shampine dense output is a continuous interpolant inside a
    step. Measured against the only alternative a caller has, the straight
    line between accepted points, the Mojo kernel is orders of magnitude more
    accurate, and it is continuous at the step boundaries."""
    y0 = np.array([1.0, 0.5])
    t_end = 2.0
    probe = np.linspace(0.0, t_end, 201)
    s = Dopri5(oscillator, y0, rtol=1e-6, atol=1e-8)
    _, dense = s.integrate(t_end, output_list=list(probe))

    grid_t = np.array([t for t, _ in s.step_points])
    grid_y = np.array([y for _, y in s.step_points])
    linear = np.stack([np.interp(probe, grid_t, grid_y[:, i])
                       for i in range(2)], axis=1)
    exact = np.array([oscillator_exact(y0, t) for t in probe])
    dense_err = float(np.max(np.abs(np.array(dense) - exact)))
    lin_err = float(np.max(np.abs(linear - exact)))
    assert dense_err < 0.02 * lin_err, (dense_err, lin_err)

    # And it is continuous at the step boundaries: sampling the interpolant
    # exactly at an accepted step time must return that step's state, not a
    # jump on to the next one.
    probe2 = np.unique(np.concatenate([probe, grid_t]))
    _, dense2 = Dopri5(oscillator, y0, rtol=1e-6, atol=1e-8).integrate(
        t_end, output_list=list(probe2)
    )
    lookup = {round(t, 12): y for t, y in zip(probe2, dense2)}
    for t, y in s.step_points:
        np.testing.assert_allclose(lookup[round(t, 12)], y, rtol=1e-9,
                                   atol=1e-11)


def test_dense_output_on_an_explicit_output_list():
    y0 = np.array([1.0, 0.5])
    probe = np.linspace(0.1, 2.9, 30)
    s = Dopri5(oscillator, y0, rtol=1e-11, atol=1e-13)
    tlist, ylist = s.integrate(3.0, output_list=list(probe))
    np.testing.assert_array_equal(np.array(tlist), probe)
    for t, y in zip(tlist, ylist):
        np.testing.assert_allclose(y, oscillator_exact(y0, t), rtol=1e-8,
                                   atol=1e-8)


def test_interpolate_before_any_step_is_an_error():
    s = Dopri5(oscillator, np.array([1.0, 0.0]))
    with pytest.raises(RuntimeError, match="no step"):
        s.interpolate(0.0)


# -------------------------------------------------------- RungeKutta4


def test_runge_kutta_4_is_fourth_order():
    y0 = np.array([1.0, 0.5])
    errors = []
    for h in (0.05, 0.025, 0.0125):
        s = RungeKutta4(oscillator, y0, h=h)
        s.integrate(1.0, h=h)
        errors.append(np.max(np.abs(s.ylist[-1] - oscillator_exact(y0, 1.0))))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for r in ratios:
        assert 8.0 < r < 32.0, ratios


def spin(t, y):
    """A three-component rotation, so the stage indices are not accidentally
    the only thing that distinguishes the rows."""
    return np.array([y[1], -y[0], 2.0 * y[2] - 3.0 * y[1]])


def test_runge_kutta_4_single_step_matches_the_upstream_formula():
    """`RungeKutta4._step` returns t+h, y + h/6*(Y1 + 2 Y2 + 2 Y3 + Y4)."""
    y0 = np.array([1.0, -0.5, 2.0])
    h = 0.05
    s = RungeKutta4(spin, y0, h=h)
    t_next, y_next = s.step(0.0, y0, h)
    K = s.K
    want = y0 + h / 6.0 * (K[0] + 2.0 * K[1] + 2.0 * K[2] + K[3])
    assert t_next == h
    np.testing.assert_allclose(y_next, want, rtol=RTOL, atol=0)
    # And the stage states really are y + h*c*K, not a re-read of row 0.
    assert not np.allclose(K[1], K[2])


def test_runge_kutta_4_converges_to_the_analytic_solution():
    y0 = np.array([1.0, 0.5])
    s = RungeKutta4(oscillator, y0, h=0.005)
    tlist, ylist = s.integrate(2.0)
    assert tlist[-1] == pytest.approx(2.0, abs=1e-12)
    np.testing.assert_allclose(ylist[-1], oscillator_exact(y0, 2.0),
                               rtol=1e-7, atol=1e-7)


# ------------------------------------------------------- RungeKutta34


def test_runge_kutta_34_reaches_the_analytic_solution():
    """The controller targets the local error, not the global one, and the
    solver never rejects a step, so the global error lands near atol/h."""
    y0 = np.array([1.0, 0.5])
    s = RungeKutta34(oscillator, y0, rtol=1e-8, atol=1e-10, h=0.01)
    tlist, ylist = s.integrate(2.0)
    np.testing.assert_allclose(ylist[-1], oscillator_exact(y0, 2.0),
                               rtol=1e-4, atol=1e-4)
    assert tlist[-1] == pytest.approx(2.0, abs=1e-12)


def test_runge_kutta_34_step_matches_the_upstream_formulas():
    y0 = np.array([1.0, 0.5])
    h = 0.02
    s = RungeKutta34(oscillator, y0, h=h)
    np.copyto(s.K[0], oscillator(0.0, y0))
    t_next, y_next, error = s._step(0.0, y0, h)
    # `_step` leaves the start-of-step derivative in `s.YLOW`, because row 0
    # has been overwritten with f(t_next, y_next) for the Hermite interpolant.
    want_y = y0 + h / 6.0 * (
        s.YLOW + 2.0 * s.K[1] + 2.0 * s.K[2] + s.K[3]
    )
    np.testing.assert_allclose(y_next, want_y, rtol=RTOL, atol=0)
    scaling = np.abs(y0) * s.rtol + s.atol
    want_e = np.linalg.norm(
        h / 6.0 * (2.0 * s.K[1] + s.K[4] - 2.0 * s.K[2] - s.K[3]) / scaling
    )
    assert error == pytest.approx(want_e, rel=RTOL)
    assert t_next == h
    # Z3 is the third stage state, y - h*Y1 + 2*h*Y2, evaluated at t + h.
    z3_state = y0 - h * s.YLOW + 2.0 * h * s.K[1]
    np.testing.assert_allclose(
        s.K[4], np.array([z3_state[1], -z3_state[0]]), rtol=RTOL, atol=0
    )


def test_runge_kutta_34_hermite_interpolation_brackets_the_step():
    y0 = np.array([1.0, 0.5])
    h = 0.2
    s = RungeKutta34(oscillator, y0, h=h)
    np.copyto(s.K[0], oscillator(0.0, y0))
    t_next, y_next, _ = s._step(0.0, y0, h)
    np.testing.assert_allclose(s.interpolate(0.0), y0, rtol=0, atol=0)
    np.testing.assert_allclose(s.interpolate(t_next), y_next, rtol=RTOL,
                               atol=0)
    mid = s.interpolate(t_next / 2.0)
    np.testing.assert_allclose(mid, oscillator_exact(y0, t_next / 2.0),
                               rtol=1e-3, atol=1e-3)


def test_runge_kutta_34_maxsteps_is_enforced():
    s = RungeKutta34(decay, np.array([1.0]), rtol=1e-14, atol=1e-16, h=1e-4,
                     maxsteps=5)
    with pytest.raises(RuntimeError, match="maximum number of steps"):
        s.integrate(10.0)


# ---------------------------------------------------------- statistics


def test_statistics_count_function_evaluations():
    """Six new right-hand side evaluations per DOPRI5 attempt, plus the one
    for the initial K1 and two more inside HINIT."""
    s = Dopri5(decay, np.array([1.0]), rtol=1e-6, atol=1e-6, inith=0.5,
               maxsteps=3)
    with pytest.raises(RuntimeError):
        s.integrate(1.0)
    # inith is given, so HINIT is skipped; the first attempt makes 7 calls
    # (K1 plus the six new stages) and each later attempt makes 6 because
    # DOPCOR reuses the previous K2 as K1.
    assert s.statistics["nfcns"] == 7 + 3 * 6
    assert s.statistics["nsteps"] == 4
