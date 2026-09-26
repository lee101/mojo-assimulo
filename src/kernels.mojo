"""Explicit Runge-Kutta core algebra for `mojo-assimulo`.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric. The shared library
owns no memory: the Python driver in `python/mojo_assimulo` owns every buffer
and calls these kernels once per stage, per state vector.

The algebra is transcribed from the reference FORTRAN that upstream Assimulo
ships in `assimulo/thirdparty/hairer/dopri5.f` (subroutines DOPCOR, HINIT,
CONTD5) and from the pure-Python `_step` methods of
`assimulo/solvers/runge_kutta.py`. The driver loop and the step-size policy
stay in Python exactly as they do in Assimulo itself, where the right-hand side
is a user-supplied Python callback.

Stage derivatives live in one `(7, n)` buffer addressed by an explicit row
index array rather than a fixed stride, because the DOPRI5 error and dense
output combinations re-use a non-contiguous subset of the stage rows.
"""

from std.math import abs, fma, pow, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def max_of(a: Float64, b: Float64) -> Float64:
    if b > a:
        return b
    return a


def min_of(a: Float64, b: Float64) -> Float64:
    if b < a:
        return b
    return a

def koff(rows: IPtr, j: Int, n: Int) -> Int:
    """Byte-free element offset of stage row `j` inside an (m, n) buffer."""
    return Int(rows[unsafe_offset=j]) * n


# --------------------------------------------------------------------------
# Generic gathered vector combination: the workhorse of every RK stage.
# --------------------------------------------------------------------------


@export("asu_rk_combo")
def asu_rk_combo(
    n: Int, h: Float64, y_addr: Int, k_addr: Int, rows_addr: Int,
    coef_addr: Int, ncoef: Int, yoff: Int, dst_addr: Int
) abi("C"):
    """Compute dst[i] = y[yoff + i] + h * sum_j coef[j] * K[rows[j], i].

    `yoff` selects the row of `y` to add, which is how the same kernel produces
    both an intermediate stage state and the new solution vector.
    """
    var y = fp(y_addr)
    var k = fp(k_addr)
    var rows = ip(rows_addr)
    var coef = fp(coef_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = Float64(0.0)
    for j in range(ncoef):
        var r = koff(rows, j, n)
        var c = coef[unsafe_offset=j]
        for i in range(n):
            dst[unsafe_offset=i] += c * k[unsafe_offset=r + i]
    for i in range(n):
        dst[unsafe_offset=i] = y[unsafe_offset=yoff + i] + h * dst[
            unsafe_offset=i
        ]


@export("asu_lincomb")
def asu_lincomb(
    n: Int, k_addr: Int, rows_addr: Int, coef_addr: Int, ncoef: Int,
    dst_addr: Int
) abi("C"):
    """Compute dst[i] = sum_j coef[j] * K[rows[j], i], with no y and no h."""
    var k = fp(k_addr)
    var rows = ip(rows_addr)
    var coef = fp(coef_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = Float64(0.0)
    for j in range(ncoef):
        var r = koff(rows, j, n)
        var c = coef[unsafe_offset=j]
        for i in range(n):
            dst[unsafe_offset=i] += c * k[unsafe_offset=r + i]


@export("asu_l2_norm")
def asu_l2_norm(n: Int, x_addr: Int, dst_addr: Int) abi("C"):
    """Write dst[0] = sqrt(sum_i x[i]**2)."""
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    var acc = Float64(0.0)
    for i in range(n):
        acc = fma(x[unsafe_offset=i], x[unsafe_offset=i], acc)
    dst[unsafe_offset=0] = sqrt(acc)


# --------------------------------------------------------------------------
# DOPRI5: local error vector and its weighted RMS norm (DOPCOR loops 28, 42).
# --------------------------------------------------------------------------


@export("asu_err_norm")
def asu_err_norm(
    n: Int, h: Float64, k_addr: Int, rows_addr: Int, coef_addr: Int,
    ncoef: Int, y_addr: Int, y1_addr: Int, atol_addr: Int, rtol_addr: Int,
    kerr_addr: Int, dst_addr: Int
) abi("C"):
    """Write the embedded error vector and its scaled RMS norm.

        kerr[i] = h * sum_j coef[j] * K[rows[j], i]
        sk      = atol[i] + rtol[i] * max(|y[i]|, |y1[i]|)
        dst[0]  = sqrt(sum_i (kerr[i] / sk)**2 / n)

    The FORTRAN fills the whole error vector before accumulating the norm; the
    two passes are fused here because the error component at index i depends on
    nothing but index i, so the fused result is identical.
    """
    var k = fp(k_addr)
    var rows = ip(rows_addr)
    var coef = fp(coef_addr)
    var y = fp(y_addr)
    var y1 = fp(y1_addr)
    var atol = fp(atol_addr)
    var rtol = fp(rtol_addr)
    var kerr = fp(kerr_addr)
    var dst = fp(dst_addr)
    # The norm accumulates in place, so the scalar output starts at zero.
    dst[unsafe_offset=0] = Float64(0.0)
    for i in range(n):
        var e = Float64(0.0)
        for j in range(ncoef):
            e = fma(coef[unsafe_offset=j], k[unsafe_offset=koff(rows, j, n) + i], e)
        e = e * h
        kerr[unsafe_offset=i] = e
        var mag = max_of(abs(y[unsafe_offset=i]), abs(y1[unsafe_offset=i]))
        var sk = atol[unsafe_offset=i] + rtol[unsafe_offset=i] * mag
        var r = e / sk
        dst[unsafe_offset=0] += r * r
    dst[unsafe_offset=0] = sqrt(dst[unsafe_offset=0] / Float64(n))


# --------------------------------------------------------------------------
# DOPRI5: PI-stabilised step-size selection (DOPCOR lines 462-468, 519-534).
# --------------------------------------------------------------------------


@export("asu_dopri5_hnew")
def asu_dopri5_hnew(
    h: Float64, err: Float64, facold: Float64, beta: Float64, safe: Float64,
    fac1: Float64, fac2: Float64, hmax: Float64, posneg: Float64,
    accepted: Int, prev_rejected: Int, dst_addr: Int
) abi("C"):
    """Predict the next step size and the next facold.

    dst[0] = hnew, dst[1] = facold. The branch structure is the FORTRAN's: the
    accepted path clips to hmax and folds in the previous-rejection cap, the
    rejected path never grows the step by more than 1/fac1.
    """
    var dst = fp(dst_addr)
    var expo1 = 0.2 - beta * 0.75
    var fac11 = pow(err, expo1)
    var facc1 = 1.0 / fac1
    var facc2 = 1.0 / fac2
    var lim = max_of(facc2, min_of(facc1, fac11 / pow(facold, beta) / safe))
    var hnew = h / lim
    var facold_new = facold
    if accepted == 1:
        facold_new = max_of(err, 1.0e-4)
        if abs(hnew) > hmax:
            hnew = posneg * hmax
        if prev_rejected == 1:
            hnew = posneg * min_of(abs(hnew), abs(h))
    else:
        hnew = h / min_of(facc1, fac11 / safe)
    dst[unsafe_offset=0] = hnew
    dst[unsafe_offset=1] = facold_new


# --------------------------------------------------------------------------
# DOPRI5: Shampine (1986) continuous output.
# --------------------------------------------------------------------------


@export("asu_dopri5_cont_d")
def asu_dopri5_cont_d(
    n: Int, h: Float64, k_addr: Int, rows_addr: Int, coef_addr: Int,
    ncoef: Int, cont_addr: Int
) abi("C"):
    """Fill the fifth CONT block, the Shampine dense-output coefficients.

    FORTRAN loop 40, executed unconditionally at the end of every attempted
    step, before the accept/reject test.
    """
    var k = fp(k_addr)
    var rows = ip(rows_addr)
    var coef = fp(coef_addr)
    var cont = fp(cont_addr)
    for i in range(n):
        var acc = Float64(0.0)
        for j in range(ncoef):
            acc = fma(coef[unsafe_offset=j], k[unsafe_offset=koff(rows, j, n) + i], acc)
        cont[unsafe_offset=4 * n + i] = h * acc


@export("asu_dopri5_cont_abc")
def asu_dopri5_cont_abc(
    n: Int, h: Float64, y_addr: Int, y1_addr: Int, k1_addr: Int,
    k2_addr: Int, cont_addr: Int
) abi("C"):
    """Fill the first four CONT blocks once a step has been accepted.

    FORTRAN loop 43. CONT holds five n-long blocks laid out contiguously; this
    kernel writes blocks 0..3 and leaves block 4 to asu_dopri5_cont_d.
    """
    var y = fp(y_addr)
    var y1 = fp(y1_addr)
    var k1 = fp(k1_addr)
    var k2 = fp(k2_addr)
    var cont = fp(cont_addr)
    for i in range(n):
        var yi = y[unsafe_offset=i]
        var ydiff = y1[unsafe_offset=i] - yi
        var bspl = h * k1[unsafe_offset=i] - ydiff
        cont[unsafe_offset=i] = yi
        cont[unsafe_offset=n + i] = ydiff
        cont[unsafe_offset=2 * n + i] = bspl
        cont[unsafe_offset=3 * n + i] = -h * k2[unsafe_offset=i] + ydiff - bspl


@export("asu_dopri5_interp")
def asu_dopri5_interp(
    n: Int, cont_addr: Int, theta: Float64, dst_addr: Int
) abi("C"):
    """CONTD5: evaluate the dense solution at theta = (t - t_old) / h."""
    var cont = fp(cont_addr)
    var dst = fp(dst_addr)
    var theta1 = 1.0 - theta
    for i in range(n):
        var inner = fma(
            theta1, cont[unsafe_offset=4 * n + i], cont[unsafe_offset=3 * n + i]
        )
        var mid = fma(theta, inner, cont[unsafe_offset=2 * n + i])
        var outer = fma(theta1, mid, cont[unsafe_offset=n + i])
        dst[unsafe_offset=i] = fma(theta, outer, cont[unsafe_offset=i])


# --------------------------------------------------------------------------
# DOPRI5: initial step-size guess (HINIT).
# --------------------------------------------------------------------------


@export("asu_hinit_norms")
def asu_hinit_norms(
    n: Int, y_addr: Int, f0_addr: Int, atol_addr: Int, rtol_addr: Int,
    dst_addr: Int
) abi("C"):
    """Write dst[0] = dnf and dst[1] = dny, the sums behind the HINIT guess."""
    var y = fp(y_addr)
    var f0 = fp(f0_addr)
    var atol = fp(atol_addr)
    var rtol = fp(rtol_addr)
    var dst = fp(dst_addr)
    dst[unsafe_offset=0] = Float64(0.0)
    dst[unsafe_offset=1] = Float64(0.0)
    for i in range(n):
        var sk = atol[unsafe_offset=i] + rtol[unsafe_offset=i] * abs(
            y[unsafe_offset=i]
        )
        var a = f0[unsafe_offset=i] / sk
        var b = y[unsafe_offset=i] / sk
        dst[unsafe_offset=0] += a * a
        dst[unsafe_offset=1] += b * b


@export("asu_hinit_der2")
def asu_hinit_der2(
    n: Int, y_addr: Int, f0_addr: Int, f1_addr: Int, atol_addr: Int,
    rtol_addr: Int, h: Float64, dst_addr: Int
) abi("C"):
    """Write dst[0] = der2, the scaled norm of the second-derivative estimate."""
    var y = fp(y_addr)
    var f0 = fp(f0_addr)
    var f1 = fp(f1_addr)
    var atol = fp(atol_addr)
    var rtol = fp(rtol_addr)
    var dst = fp(dst_addr)
    dst[unsafe_offset=0] = Float64(0.0)
    for i in range(n):
        var sk = atol[unsafe_offset=i] + rtol[unsafe_offset=i] * abs(
            y[unsafe_offset=i]
        )
        var d = (f1[unsafe_offset=i] - f0[unsafe_offset=i]) / sk
        dst[unsafe_offset=0] += d * d
    dst[unsafe_offset=0] = sqrt(dst[unsafe_offset=0]) / h


@export("asu_hinit_guess")
def asu_hinit_guess(
    dnf: Float64, dny: Float64, hmax: Float64, posneg: Float64, dst_addr: Int
) abi("C"):
    """Write dst[0], the HINIT first guess, clipped to hmax and signed."""
    var dst = fp(dst_addr)
    # Function arguments are immutable, so the guess is built in a local.
    var hh = 1.0e-6
    if dnf > 1.0e-10 and dny > 1.0e-10:
        hh = sqrt(dny / dnf) * 0.01
    hh = abs(min_of(hh, hmax))
    dst[unsafe_offset=0] = hh if posneg >= 0.0 else -hh


@export("asu_hinit_step")
def asu_hinit_step(
    dnf: Float64, der2: Float64, h: Float64, hmax: Float64, posneg: Float64,
    iord: Float64, dst_addr: Int
) abi("C"):
    """Write dst[0], the HINIT step size, given the probe step actually used.

    This is the second half of HINIT: the caller has already taken the explicit
    Euler probe with step `h` and evaluated der2, so the final guess is
    MIN(100*|H|, H1, HMAX) signed by posneg.
    """
    var dst = fp(dst_addr)
    var der12 = max_of(abs(der2), sqrt(dnf))
    var h1 = max_of(abs(h) * 1.0e-3, 1.0e-6)
    if der12 > 1.0e-15:
        h1 = pow(0.01 / der12, 1.0 / iord)
    var best = min_of(abs(h) * 100.0, min_of(h1, hmax))
    dst[unsafe_offset=0] = best if posneg >= 0.0 else -best


# --------------------------------------------------------------------------
# DOPRI5: stiffness detection sums (DOPCOR loop 64).
# --------------------------------------------------------------------------


@export("asu_stiffness")
def asu_stiffness(
    n: Int, k2_addr: Int, k6_addr: Int, y1_addr: Int, ysti_addr: Int,
    h: Float64, hlamb: Float64, dst_addr: Int
) abi("C"):
    """Write stnum, stden and the updated hlamb.

    hlamb is only refreshed when stden > 0, matching the FORTRAN, which leaves
    the previous estimate in place otherwise.
    """
    var k2 = fp(k2_addr)
    var k6 = fp(k6_addr)
    var y1 = fp(y1_addr)
    var ysti = fp(ysti_addr)
    var dst = fp(dst_addr)
    dst[unsafe_offset=0] = Float64(0.0)
    dst[unsafe_offset=1] = Float64(0.0)
    for i in range(n):
        var d = k2[unsafe_offset=i] - k6[unsafe_offset=i]
        var e = y1[unsafe_offset=i] - ysti[unsafe_offset=i]
        dst[unsafe_offset=0] += d * d
        dst[unsafe_offset=1] += e * e
    if dst[unsafe_offset=1] > 0.0:
        var updated = h * sqrt(dst[unsafe_offset=0] / dst[unsafe_offset=1])
        dst[unsafe_offset=2] = updated
    else:
        dst[unsafe_offset=2] = hlamb


# --------------------------------------------------------------------------
# RungeKutta34: scaled error vector and Hermite dense output.
# --------------------------------------------------------------------------


@export("asu_rk34_error")
def asu_rk34_error(
    n: Int, h: Float64, y_addr: Int, k_addr: Int, rows_addr: Int,
    coef_addr: Int, ncoef: Int, atol_addr: Int, rtol: Float64, dst_addr: Int
) abi("C"):
    """Scaled local error of the Bogacki-Shampine 3(2) method.

    Mirrors RungeKutta34._step: the error vector is
    h/6 * (2 Y2 + Z3 - 2 Y3 - Y4) / (|y| * rtol + atol).
    """
    var y = fp(y_addr)
    var k = fp(k_addr)
    var rows = ip(rows_addr)
    var coef = fp(coef_addr)
    var atol = fp(atol_addr)
    var dst = fp(dst_addr)
    var sixth = h / 6.0
    for i in range(n):
        var acc = Float64(0.0)
        for j in range(ncoef):
            acc = fma(coef[unsafe_offset=j], k[unsafe_offset=koff(rows, j, n) + i], acc)
        var scaling = abs(y[unsafe_offset=i]) * rtol + atol[unsafe_offset=i]
        dst[unsafe_offset=i] = sixth * acc / scaling


@export("asu_hermite_interp")
def asu_hermite_interp(
    n: Int, theta: Float64, y_addr: Int, ynext_addr: Int, flow_addr: Int,
    fhigh_addr: Int, h: Float64, dst_addr: Int
) abi("C"):
    """The cubic Hermite interpolant built by RungeKutta34._step."""
    var y = fp(y_addr)
    var ynext = fp(ynext_addr)
    var flow = fp(flow_addr)
    var fhigh = fp(fhigh_addr)
    var dst = fp(dst_addr)
    var t1 = 1.0 - theta
    var inner_a = 1.0 - 2.0 * theta
    var inner_b = (theta - 1.0) * h
    var inner_c = theta * h
    var outer = theta * (theta - 1.0)
    for i in range(n):
        var yi = y[unsafe_offset=i]
        var d = ynext[unsafe_offset=i] - yi
        var inner = fma(inner_a, d, inner_b * flow[unsafe_offset=i])
        inner = fma(inner_c, fhigh[unsafe_offset=i], inner)
        var base = t1 * yi + theta * ynext[unsafe_offset=i]
        dst[unsafe_offset=i] = fma(outer, inner, base)
