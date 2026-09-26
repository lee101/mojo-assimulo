"""Parity tests for the Mojo kernel layer, one per kernel.

Assimulo itself does not build on this interpreter, so the baseline here is a
vectorised NumPy transcription of the same formulas (`tests/reference.py`),
written independently of the kernel loop structure, plus exact checks against
the rational Butcher coefficients from `assimulo/thirdparty/hairer/dopri5.f`.

Mojo emits FMA, so the vector comparisons carry a relative tolerance. Integer
and index results are compared exactly.
"""

import numpy as np
import pytest

from mojo_assimulo import _lib as k
from mojo_assimulo import tableau as T

import reference as R

# FMA reassociation in the Mojo dot products is the only source of disagreement
# with NumPy; a relative 1e-12 is several orders above the observed drift and
# far below any real algorithmic error.
RTOL = 1e-12


def _ks(rng, m, n):
    return np.ascontiguousarray(rng.standard_normal((m, n)))


# ---------------------------------------------------------------- tableaus


def test_tableau_coefficients_are_the_fortran_rationals():
    from fractions import Fraction

    expected = {
        "A21": [Fraction(1, 5)],
        "A31": [Fraction(3, 40), Fraction(9, 40)],
        "A41": [Fraction(44, 45), Fraction(-56, 15), Fraction(32, 9)],
        "A51": [Fraction(19372, 6561), Fraction(-25360, 2187),
                Fraction(64448, 6561), Fraction(-212, 729)],
        "A61": [Fraction(9017, 3168), Fraction(-355, 33),
                Fraction(46732, 5247), Fraction(49, 176),
                Fraction(-5103, 18656)],
        "A71": [Fraction(35, 384), Fraction(500, 1113), Fraction(125, 192),
                Fraction(-2187, 6784), Fraction(11, 84)],
        "E": [Fraction(71, 57600), Fraction(-71, 16695), Fraction(71, 1920),
              Fraction(-17253, 339200), Fraction(22, 525), Fraction(-1, 40)],
        "D": [Fraction(-12715105075, 11282082432),
              Fraction(87487479700, 32700410799),
              Fraction(-10690763975, 1880347072),
              Fraction(701980252875, 199316789632),
              Fraction(-1453857185, 822651844),
              Fraction(69997945, 29380423)],
    }
    for name, fracs in expected.items():
        got = T.ALL_TABLEAUS[name][1]
        assert len(got) == len(fracs), name
        np.testing.assert_array_equal(
            T.as_float(name),
            np.array([float(f) for f in fracs]),
        )
        assert [float(f) for f in fracs] == [float(f) for f in got]


def test_stage_nodes_match_cdopri():
    np.testing.assert_array_equal(
        T.C, np.array([0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0])
    )


def test_a71_satisfies_the_fifth_order_conditions():
    """The five b weights of A71 are pinned by sum(b*c**j) == 1/(j+1) for
    j = 0..4, which is exactly the order-5 condition set for an explicit
    method. A single transposed or mis-signed weight breaks all of them."""
    b = T.as_float("A71")
    # The five stages A71 consumes are K1, K3, K4, K5, K6, evaluated at the
    # node times 0, C3, C4, C5, C6.
    c = np.array([0.0, T.C[1], T.C[2], T.C[3], 1.0])
    for j in range(5):
        got = float(np.sum(b * c ** j))
        want = 1.0 / (j + 1)
        assert got == pytest.approx(want, rel=1e-13), (j, got, want)


def test_embedded_pair_is_one_order_lower():
    """The error weights E are the difference between the 5th-order update and
    the embedded 4th-order one, so sum(E*b) must be exactly zero: the estimate
    would otherwise report a nonzero error for an exactly integrated
    polynomial."""
    e = T.as_float("E")
    b5 = T.as_float("A71")
    # b4 = b5 - E over the five shared stages, plus E7 on K2new, which is
    # evaluated at t + h like the sixth stage.
    b4 = np.concatenate([b5 - e[:5], -e[5:]])
    c = np.array([0.0, T.C[1], T.C[2], T.C[3], 1.0, 1.0])
    assert float(np.sum(b4)) == pytest.approx(1.0, rel=1e-14)
    for j in range(4):
        assert float(np.sum(b4 * c ** j)) == pytest.approx(
            1.0 / (j + 1), rel=1e-12
        ), (j, float(np.sum(b4 * c ** j)))
    # b5 and b4 are both consistent, so the difference weights sum to zero.
    assert float(np.sum(e)) == pytest.approx(0.0, abs=1e-15)


# ------------------------------------------------------ rk_combo / lincomb


def test_rk_combo_matches_numpy_gather(rng=None):
    rng = np.random.default_rng(1)
    n, m = 37, 7
    K = _ks(rng, m, n)
    y = _ks(rng, 2, n)
    flat = y.reshape(-1)
    for name in ("A21", "A31", "A41", "A51", "A61", "A71", "E", "D"):
        rows, coef = T.ALL_TABLEAUS[name]
        rows = np.array(rows, dtype=np.int64)
        coef = np.array([float(c) for c in coef])
        for h in (0.0, 1.0, -0.375):
            for yoff in (0, n):
                dst = np.empty(n, dtype=np.float64)
                k.rk_combo(n, h, y, yoff, K, rows, coef, dst)
                want = R.combo(flat[yoff:yoff + n], h, K,
                               (rows.tolist(), coef.tolist()))
                np.testing.assert_allclose(dst, want, rtol=RTOL, atol=0)


def test_lincomb_matches_numpy(rng=None):
    rng = np.random.default_rng(2)
    n = 23
    K = _ks(rng, 7, n)
    rows = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)
    coef = T.as_float("E")
    dst = np.empty(n, dtype=np.float64)
    k.lincomb(n, K, rows, coef, dst)
    want = np.zeros(n)
    for r, c in zip(rows, coef):
        want += c * K[r]
    np.testing.assert_allclose(dst, want, rtol=RTOL, atol=0)


def test_rk_combo_gathers_the_requested_rows():
    """A wrong gather index (the classic transpose bug) is invisible to a
    single-coefficient call only if that row is also wrong, so pin it down."""
    n = 4
    K = np.arange(3 * n, dtype=np.float64).reshape(3, n)
    y = np.zeros(n)
    dst = np.empty(n)
    k.rk_combo(n, 1.0, y, 0, K, np.array([2], np.int64), np.array([1.0]), dst)
    np.testing.assert_array_equal(dst, K[2])
    dst2 = np.empty(n)
    k.rk_combo(n, 1.0, y, 0, K, np.array([0], np.int64), np.array([1.0]), dst2)
    np.testing.assert_array_equal(dst2, K[0])


# ------------------------------------------------------------- err_norm


def test_err_norm_matches_unfused_reference():
    rng = np.random.default_rng(3)
    n = 64
    K = _ks(rng, 7, n)
    y = _ks(rng, 1, n)[0]
    y1 = _ks(rng, 1, n)[0]
    atol = np.abs(rng.standard_normal(n)) * 1e-3 + 1e-9
    rtol = np.full(n, 1e-6)
    rows, coef = T.as_rows("E"), T.as_float("E")
    kerr = np.empty(n)
    got = k.err_norm(n, 0.375, K, rows, coef, y, y1, atol, rtol, kerr)
    want_vec, want_norm = R.err_norm(K, 0.375, y, y1, atol, rtol)
    np.testing.assert_allclose(kerr, want_vec, rtol=RTOL, atol=0)
    assert got == pytest.approx(want_norm, rel=RTOL)


def test_err_norm_uses_the_larger_of_y_and_y1_in_the_scale():
    """The FORTRAN scale is atol + rtol*max(|y|,|y1|). Both components carry
    error here and only the second one has |y1| > |y|, so dropping the max
    changes the norm by a visible amount."""
    n = 2
    K = np.zeros((7, n))
    K[0] = [1.0, 1.0]
    y = np.array([1.0, 1.0])
    y1 = np.array([1.0, 9.0])
    atol = np.full(n, 1e-8)
    rtol = np.full(n, 1e-2)
    rows, coef = T.as_rows("E"), T.as_float("E")
    kerr = np.zeros(n)
    got = k.err_norm(n, 1.0, K, rows, coef, y, y1, atol, rtol, kerr)
    sk = atol + rtol * np.maximum(np.abs(y), np.abs(y1))
    want = np.sqrt(np.sum((kerr / sk) ** 2) / n)
    assert got == pytest.approx(want, rel=RTOL)
    sk_y = atol + rtol * np.abs(y)
    assert not np.isclose(
        want, np.sqrt(np.sum((kerr / sk_y) ** 2) / n), rtol=1e-3
    )


# ---------------------------------------------------------------- hnew


def test_dopri5_hnew_matches_the_fortran_branches():
    cases = [
        # (h, err, facold, accepted, prev_rejected, hmax)
        (0.1, 0.5, 1e-4, True, False, np.inf),
        (0.1, 0.5, 1e-4, True, False, 0.02),      # hmax clip
        (0.1, 0.5, 1e-4, True, True, np.inf),     # previous rejection cap
        (0.1, 4.0, 1e-4, False, False, np.inf),   # rejection branch
        (0.1, 1e-9, 1e-4, True, False, np.inf),   # facold floor
        (-0.1, 0.5, 1e-4, True, False, np.inf),   # backwards integration
    ]
    for h, err, facold, acc, rej, hmax in cases:
        for safe, fac1, fac2, beta in (
            (0.9, 0.2, 10.0, 0.04), (0.8, 0.1, 6.0, 0.0), (0.95, 0.3, 4.0, 0.2)
        ):
            got = k.dopri5_hnew(h, err, facold, beta, safe, fac1, fac2, hmax,
                                1.0 if h >= 0 else -1.0, acc, rej)
            want = R.hnew(h, err, facold, beta, safe, fac1, fac2, hmax,
                          1.0 if h >= 0 else -1.0, acc, rej)
            assert got[0] == pytest.approx(want[0], rel=RTOL)
            assert got[1] == pytest.approx(want[1], rel=RTOL)


def test_dopri5_hnew_clamps_growth_to_fac2():
    """An err of 1e-30 would grow h by ~1e6 without the fac2 clamp."""
    h, err = 1.0, 1e-30
    hnew, _ = k.dopri5_hnew(h, err, 1e-4, 0.04, 0.9, 0.2, 10.0, np.inf, 1.0,
                            True, False)
    assert hnew == pytest.approx(h * 10.0, rel=1e-12)


# ------------------------------------------------------------ dense output


def test_cont_blocks_match_numpy():
    rng = np.random.default_rng(4)
    n = 11
    K = _ks(rng, 7, n)
    y, y1 = _ks(rng, 1, n)[0], _ks(rng, 1, n)[0]
    h = 0.125
    cont = np.zeros(5 * n)
    k.dopri5_cont_d(n, h, K, T.as_rows("D"), T.as_float("D"), cont)
    k.dopri5_cont_abc(n, h, y, y1, K[0], K[6], cont)
    want = R.cont_blocks(n, h, y, y1, K[0], K[6], K, T.ALL_TABLEAUS["D"])
    np.testing.assert_allclose(cont, want, rtol=RTOL, atol=0)


def test_cont_blocks_are_packed_in_the_fortran_slots():
    """Blocks 2 and 3 are the only easy thing to transpose, and a transpose
    leaves every slot non-zero, so pin the layout with an exact identity."""
    n = 1
    y = np.array([2.0])
    y1 = np.array([5.0])
    k1 = np.array([3.0])
    k2 = np.array([11.0])
    h = 0.5
    cont = np.zeros(5 * n)
    k.dopri5_cont_abc(n, h, y, y1, k1, k2, cont)
    # ydiff = 3, bspl = 0.5*3 - 3 = -1.5, -0.5*11 + 3 + 1.5 = -1.0
    assert cont[0] == 2.0
    assert cont[1] == 3.0
    assert cont[2] == pytest.approx(-1.5)
    assert cont[3] == pytest.approx(-1.0)
    assert cont[4] == 0.0


def test_interpolation_reproduces_both_endpoints():
    rng = np.random.default_rng(5)
    n = 9
    K = _ks(rng, 7, n)
    y, y1 = _ks(rng, 1, n)[0], _ks(rng, 1, n)[0]
    h = 0.2
    cont = np.zeros(5 * n)
    k.dopri5_cont_d(n, h, K, T.as_rows("D"), T.as_float("D"), cont)
    k.dopri5_cont_abc(n, h, y, y1, K[0], K[6], cont)
    dst = np.empty(n)
    k.dopri5_interp(n, cont, 0.0, dst)
    np.testing.assert_allclose(dst, y, rtol=0, atol=0)
    k.dopri5_interp(n, cont, 1.0, dst)
    # theta = 1 collapses the nested FMA form to c0 + c1 + c3, which is y1 up
    # to reassociation, not bit-for-bit.
    np.testing.assert_allclose(dst, y1, rtol=1e-15, atol=0)
    for theta in (0.1, 0.37, 0.5, 0.83, 0.99):
        k.dopri5_interp(n, cont, theta, dst)
        np.testing.assert_allclose(
            dst, R.interp(cont, theta, n), rtol=RTOL, atol=0
        )


# --------------------------------------------------------------- hinit


def test_hinit_matches_reference():
    rng = np.random.default_rng(6)
    n = 20
    y = _ks(rng, 1, n)[0]
    f0 = _ks(rng, 1, n)[0]
    f1 = _ks(rng, 1, n)[0]
    atol = np.full(n, 1e-6)
    rtol = np.full(n, 1e-3)
    dnf, dny = k.hinit_norms(y, f0, atol, rtol)
    sk = atol + rtol * np.abs(y)
    assert dnf == pytest.approx(float(np.sum((f0 / sk) ** 2)), rel=RTOL)
    assert dny == pytest.approx(float(np.sum((y / sk) ** 2)), rel=RTOL)
    for hmax, posneg in ((np.inf, 1.0), (0.01, 1.0), (np.inf, -1.0)):
        guess = k.hinit_guess(dnf, dny, hmax, posneg)  # noqa: E501
        want_guess = abs(min(0.01 * np.sqrt(dny / dnf), hmax))
        assert abs(guess) == pytest.approx(want_guess, rel=RTOL)
        assert np.sign(guess) == (1.0 if posneg >= 0 else -1.0)
        der2 = k.hinit_der2(y, f0, f1, atol, rtol, guess)
        want_der2 = float(np.sqrt(np.sum(((f1 - f0) / sk) ** 2))) / guess
        assert der2 == pytest.approx(want_der2, rel=RTOL)
        got = k.hinit_step(dnf, der2, guess, hmax, posneg)
        want = R.hinit(dnf, der2, guess, hmax, posneg)
        # pow() in the two implementations differs in the last bits.
        assert got == pytest.approx(want, rel=1e-11, abs=1e-300)


# ----------------------------------------------------------- stiffness


def test_stiffness_matches_reference():
    rng = np.random.default_rng(8)
    n = 13
    k2, k6 = _ks(rng, 1, n)[0], _ks(rng, 1, n)[0]
    y1, ysti = _ks(rng, 1, n)[0], _ks(rng, 1, n)[0]
    h, hlamb = 0.3, 1.25
    got = k.stiffness(k2, k6, y1, ysti, h, hlamb)
    want = R.stiffness(k2, k6, y1, ysti, h, hlamb)[2]
    assert got == pytest.approx(want, rel=RTOL)


def test_stiffness_keeps_hlamb_when_stden_is_zero():
    """DOPCOR only refreshes HLAMB when STDEN > 0, so a step where Y1 == YSTI
    must leave the previous estimate untouched."""
    n = 3
    z = np.zeros(n)
    assert k.stiffness(z, z, z, z, 0.5, 7.0) == 7.0
    # stnum = stden = n, so hlambda = h.
    assert k.stiffness(z, np.ones(n), np.ones(n), z, 0.5, 7.0) == pytest.approx(
        0.5, rel=RTOL
    )


# ------------------------------------------------------------- l2 norm


def test_l2_norm_matches_numpy():
    rng = np.random.default_rng(9)
    for n in (1, 5, 1000):
        x = _ks(rng, 1, n)[0]
        assert k.l2_norm(x) == pytest.approx(
            float(np.linalg.norm(x)), rel=RTOL
        )


# ------------------------------------------------------------- rk34


def test_rk34_error_matches_the_upstream_expression():
    """`RungeKutta34._step` computes the error vector as
    h/6 * (2 Y2 + Z3 - 2 Y3 - Y4) / (|y| * rtol + atol)."""
    rng = np.random.default_rng(10)
    n = 17
    K = _ks(rng, 5, n)
    y = _ks(rng, 1, n)[0]
    atol = np.abs(rng.standard_normal(n)) * 1e-5 + 1e-9
    rtol = 1e-4
    h = 0.03
    dst = np.empty(n)
    k.rk34_error(n, h, y, K, T.as_rows("BS23_ERROR"),
                 T.as_float("BS23_ERROR"), atol, rtol, dst)
    rows, coef = T.ALL_TABLEAUS["BS23_ERROR"]
    acc = np.zeros(n)
    for r, c in zip(rows, coef):
        acc += float(c) * K[r]
    scaling = np.abs(y) * rtol + atol
    np.testing.assert_allclose(dst, h / 6.0 * acc / scaling, rtol=RTOL,
                               atol=0)


def test_rk34_error_scales_with_the_tolerance():
    """With rtol = 0 the scale is exactly atol, so doubling atol must halve
    the error vector componentwise."""
    rng = np.random.default_rng(11)
    n = 8
    K = _ks(rng, 5, n)
    y = _ks(rng, 1, n)[0]
    atol = np.full(n, 1e-3)
    a = np.empty(n)
    b = np.empty(n)
    k.rk34_error(n, 0.02, y, K, T.as_rows("BS23_ERROR"),
                 T.as_float("BS23_ERROR"), atol, 0.0, a)
    k.rk34_error(n, 0.02, y, K, T.as_rows("BS23_ERROR"),
                 T.as_float("BS23_ERROR"), 2 * atol, 0.0, b)
    np.testing.assert_allclose(a, 2.0 * b, rtol=RTOL, atol=0)


def test_hermite_interp_matches_the_upstream_expression():
    """Direct parity with the literal Python in `RungeKutta34._step`."""
    rng = np.random.default_rng(12)
    n = 21
    y = _ks(rng, 1, n)[0]
    y_next = _ks(rng, 1, n)[0]
    f_low = _ks(rng, 1, n)[0]
    f_high = _ks(rng, 1, n)[0]
    h = 0.07
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        dst = np.empty(n)
        k.hermite_interp(n, theta, y, y_next, f_low, f_high, h, dst)
        thetha = theta
        want = (
            (1 - thetha) * y + thetha * y_next + thetha * (thetha - 1) * (
                (1 - 2 * thetha) * (y_next - y)
                + (thetha - 1) * h * f_low
                + thetha * h * f_high
            )
        )
        np.testing.assert_allclose(dst, want, rtol=RTOL, atol=0)


def test_hermite_interp_hits_both_endpoints_exactly():
    n = 4
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_next = y + 0.5
    f = np.full(n, 2.0)
    dst = np.empty(n)
    k.hermite_interp(n, 0.0, y, y_next, f, f, 0.5, dst)
    np.testing.assert_array_equal(dst, y)
    k.hermite_interp(n, 1.0, y, y_next, f, f, 0.5, dst)
    np.testing.assert_array_equal(dst, y_next)


def test_rungekutta34_step_size_controller_matches_upstream():
    from mojo_assimulo import RungeKutta34

    for h in (0.01, 0.5, 3.0):
        assert RungeKutta34.adjust_stepsize(h, 0.0) == 2.0 * h
        for err in (1e-8, 0.1, 1.0, 100.0, 1e6):
            want = h * min((1.0 / err) ** 0.25, 2.0)
            assert RungeKutta34.adjust_stepsize(h, err) == pytest.approx(
                want, rel=1e-15
            )
