"""Vectorised NumPy transcription of the algorithms under test.

These are independent implementations of the same formulas the Mojo kernels
evaluate, written with whole-array NumPy operations instead of scalar loops.
They are the parity baseline: any kernel bug shows up as a disagreement with
these, and any driver bug shows up as a disagreement in the accept/reject
pattern or the accepted step times.
"""

import numpy as np

from mojo_assimulo.tableau import (
    A21, A31, A41, A51, A61, A71, D, E, C,
)


def combo(y, h, K, tableau):
    """y + h * sum_j coef[j] * K[row_j]."""
    rows, coef = tableau
    acc = np.zeros_like(y)
    for r, c in zip(rows, coef):
        acc = acc + float(c) * K[r]
    return y + h * acc


def err_norm(K, h, y, y1, atol, rtol):
    """The DOPRI5 embedded error vector and its scaled RMS norm."""
    kerr = h * combo(np.zeros_like(y), 1.0, K, E)
    sk = atol + rtol * np.maximum(np.abs(y), np.abs(y1))
    return kerr, float(np.sqrt(np.sum((kerr / sk) ** 2) / y.size))


def hnew(h, err, facold, beta, safe, fac1, fac2, hmax, posneg, accepted,
         prev_rejected):
    """DOPCOR lines 462-468 and 519-534, in Python control flow."""
    expo1 = 0.2 - beta * 0.75
    fac11 = err ** expo1
    fac = fac11 / facold ** beta
    fac = max(1.0 / fac2, min(1.0 / fac1, fac / safe))
    out = h / fac
    facold_new = facold
    if accepted:
        facold_new = max(err, 1.0e-4)
        if abs(out) > hmax:
            out = posneg * hmax
        if prev_rejected:
            out = posneg * min(abs(out), abs(h))
    else:
        out = h / min(1.0 / fac1, fac11 / safe)
    return out, facold_new


def cont_blocks(n, h, y, y1, k1, k2, K, dcoef):
    """The five CONT blocks of DOPCOR loops 40 and 43, in one array."""
    cont = np.zeros(5 * n, dtype=np.float64)
    ydiff = y1 - y
    bspl = h * k1 - ydiff
    cont[0:n] = y
    cont[n:2 * n] = ydiff
    cont[2 * n:3 * n] = bspl
    cont[3 * n:4 * n] = -h * k2 + ydiff - bspl
    rows, coef = dcoef
    acc = np.zeros(n, dtype=np.float64)
    for r, c in zip(rows, coef):
        acc = acc + float(c) * K[r]
    cont[4 * n:5 * n] = h * acc
    return cont


def interp(cont, theta, n):
    """CONTD5."""
    theta1 = 1.0 - theta
    return (cont[0:n]
            + theta * (cont[n:2 * n]
                       + theta1 * (cont[2 * n:3 * n]
                                   + theta * (cont[3 * n:4 * n]
                                              + theta1 * cont[4 * n:5 * n]))))


def hinit(dnf, der2, h, hmax, posneg, iord=5.0):
    """The second half of HINIT, given the probe step the caller actually used.

        DER12 = MAX(|DER2|, SQRT(DNF))
        H1    = (0.01/DER12)**(1/IORD)   (or max(|H|*1e-3, 1e-6) when tiny)
        H     = MIN(100*|H|, H1, HMAX), signed by POSNEG
    """
    der12 = max(abs(der2), np.sqrt(dnf))
    if der12 <= 1.0e-15:
        h1 = max(abs(h) * 1.0e-3, 1.0e-6)
    else:
        h1 = (0.01 / der12) ** (1.0 / iord)
    best = min(100.0 * abs(h), h1, hmax)
    return best if posneg >= 0.0 else -best


def stiffness(k2, k6, y1, ysti, h, hlamb):
    stnum = float(np.sum((k2 - k6) ** 2))
    stden = float(np.sum((y1 - ysti) ** 2))
    if stden > 0.0:
        hlamb = h * np.sqrt(stnum / stden)
    return stnum, stden, hlamb


def dopcor(rhs, t0, y0, tf, rtol, atol, safe=0.9, fac1=0.2, fac2=10.0,
           beta=0.04, hmax=np.inf, maxsteps=100000, inith=0.0):
    """A NumPy transcription of the DOPCOR driver loop.

    Returns (tlist, ylist, rejected_step_times). Written from the same formulas
    the Mojo driver uses, so the two must agree step for step.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    atol = np.broadcast_to(np.asarray(atol, dtype=np.float64), y0.shape).copy()
    rtol = np.full(y0.shape, float(rtol), dtype=np.float64)
    n = y0.size
    t = float(t0)
    y = y0.copy()
    posneg = 1.0 if tf - t >= 0 else -1.0
    hmax = abs(hmax)

    k0 = np.asarray(rhs(t, y), dtype=np.float64)
    sk = atol + rtol * np.abs(y)
    dnf = float(np.sum((k0 / sk) ** 2))
    dny = float(np.sum((y / sk) ** 2))
    h = 1.0e-6 if dnf <= 1e-10 or dny <= 1e-10 else np.sqrt(dny / dnf) * 0.01
    h = abs(min(h, hmax)) * (1.0 if posneg > 0 else -1.0)
    y1 = y + h * k0
    k1 = np.asarray(rhs(t + h, y1), dtype=np.float64)
    der2 = float(np.sqrt(np.sum(((k1 - k0) / sk) ** 2))) / h
    h = hinit(dnf, der2, h, hmax, posneg)

    facold = 1.0e-4
    tlist, ylist, rejected = [t], [y.copy()], []
    last = False
    reject = False
    nsteps = 0
    while True:
        if nsteps > maxsteps:
            raise RuntimeError("too many steps")
        if (t + 1.01 * h - tf) * posneg > 0.0:
            h = tf - t
            last = True
        nsteps += 1
        xph = t + h
        K = np.zeros((7, n), dtype=np.float64)
        K[0] = np.asarray(rhs(t, y), dtype=np.float64)
        K[1] = np.asarray(rhs(t + C[0] * h, combo(y, h, K, A21)),
                          dtype=np.float64)
        K[2] = np.asarray(rhs(t + C[1] * h, combo(y, h, K, A31)),
                          dtype=np.float64)
        K[3] = np.asarray(rhs(t + C[2] * h, combo(y, h, K, A41)),
                          dtype=np.float64)
        K[4] = np.asarray(rhs(t + C[3] * h, combo(y, h, K, A51)),
                          dtype=np.float64)
        K[5] = np.asarray(rhs(xph, combo(y, h, K, A61)), dtype=np.float64)
        K[6] = np.asarray(rhs(xph, combo(y, h, K, A71)), dtype=np.float64)
        _, err = err_norm(K, h, y, combo(y, h, K, A71), atol, rtol)
        hnew_, facold = hnew(h, err, facold, beta, safe, fac1, fac2, hmax,
                             posneg, err <= 1.0, reject)
        if err <= 1.0:
            y = combo(y, h, K, A71)
            t = xph
            reject = False
            if last:
                tlist.append(t)
                ylist.append(y.copy())
                break
            tlist.append(t)
            ylist.append(y.copy())
        else:
            rejected.append(t)
            reject = True
            last = False
        h = hnew_
    return tlist, ylist, rejected
