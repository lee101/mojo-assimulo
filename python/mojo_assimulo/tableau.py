"""Butcher tableaus, transcribed from the Assimulo sources.

`DOPRI5` comes from `CDOPRI` in `assimulo/thirdparty/hairer/dopri5.f`; the
Runge-Kutta-4 and Bogacki-Shampine weights come from the literal Python of
`RungeKutta4._step` and `RungeKutta34._step` in
`assimulo/solvers/runge_kutta.py`. They are plain Python data, exactly as they
are in Assimulo, and the parity tests check them against the original
rationals.

Stage row indices refer to the physical layout of the driver's `(m, n)` stage
buffer. For DOPRI5 that buffer holds K1, K2, K3, K4, K5, K6, K2new in rows 0
to 6, which is why the A71, E and D combinations skip row 1: the FORTRAN
overwrites K2 with the second evaluation at the end of the step.
"""

from fractions import Fraction

import numpy as np

# Dormand-Prince 5(4). Stage node times C2..C5; C6 and C7 are both 1.0.
C = np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=np.float64)

A21 = ([0], [Fraction(1, 5)])
A31 = ([0, 1], [Fraction(3, 40), Fraction(9, 40)])
A41 = ([0, 1, 2], [Fraction(44, 45), Fraction(-56, 15), Fraction(32, 9)])
A51 = (
    [0, 1, 2, 3],
    [
        Fraction(19372, 6561),
        Fraction(-25360, 2187),
        Fraction(64448, 6561),
        Fraction(-212, 729),
    ],
)
A61 = (
    [0, 1, 2, 3, 4],
    [
        Fraction(9017, 3168),
        Fraction(-355, 33),
        Fraction(46732, 5247),
        Fraction(49, 176),
        Fraction(-5103, 18656),
    ],
)
A71 = (
    [0, 2, 3, 4, 5],
    [
        Fraction(35, 384),
        Fraction(500, 1113),
        Fraction(125, 192),
        Fraction(-2187, 6784),
        Fraction(11, 84),
    ],
)

# Embedded 4th-order error weights E1 E3 E4 E5 E6 E7. The sixth entry is K2new.
E = (
    [0, 2, 3, 4, 5, 6],
    [
        Fraction(71, 57600),
        Fraction(-71, 16695),
        Fraction(71, 1920),
        Fraction(-17253, 339200),
        Fraction(22, 525),
        Fraction(-1, 40),
    ],
)

# Shampine (1986) dense-output weights D1 D3 D4 D5 D6 D7.
D = (
    [0, 2, 3, 4, 5, 6],
    [
        Fraction(-12715105075, 11282082432),
        Fraction(87487479700, 32700410799),
        Fraction(-10690763975, 1880347072),
        Fraction(701980252875, 199316789632),
        Fraction(-1453857185, 822651844),
        Fraction(69997945, 29380423),
    ],
)

# Classical Runge-Kutta 4: y + h/6 * (Y1 + 2 Y2 + 2 Y3 + Y4).
RK4 = (
    [0, 1, 2, 3],
    [Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6)],
)

# Bogacki-Shampine 3(2) third stage: y - h*Y1 + 2*h*Y2. Rows are
# Y1, Y2, Y3, Y4, Z3.
BS23_Z3 = ([0, 1], [Fraction(-1), Fraction(2)])

# Bogacki-Shampine 3(2) update, identical in form to the RK4 weights.
BS23_UPDATE = RK4

# Bogacki-Shampine 3(2) scaled error: h/6 * (2 Y2 + Z3 - 2 Y3 - Y4). The
# 1/6 lives in the kernel, so the weights here are the bare ones.
BS23_ERROR = (
    [1, 4, 2, 3],
    [Fraction(2), Fraction(1), Fraction(-2), Fraction(-1)],
)

ALL_TABLEAUS = {
    "A21": A21, "A31": A31, "A41": A41, "A51": A51, "A61": A61, "A71": A71,
    "E": E, "D": D, "RK4": RK4, "BS23_Z3": BS23_Z3,
    "BS23_UPDATE": BS23_UPDATE, "BS23_ERROR": BS23_ERROR,
}


def as_float(name) -> np.ndarray:
    """The coefficients of tableau `name` as float64."""
    return np.array([float(c) for c in ALL_TABLEAUS[name][1]], dtype=np.float64)


def as_rows(name) -> np.ndarray:
    """The stage rows that tableau `name` consumes, as int64."""
    return np.array(ALL_TABLEAUS[name][0], dtype=np.int64)
