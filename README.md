# mojo-assimulo

`mojo-assimulo` is the explicit Runge-Kutta core of
[Assimulo](https://jmodeler.sourceforge.net/Assimulo/Assimulo.html) with the
state-vector algebra compiled to Mojo. It covers the vector operations of
Dormand-Prince 5(4) with step-size control and continuous output, of classical
Runge-Kutta 4, and of the adaptive Bogacki-Shampine 3(2) solver that Assimulo
calls `RungeKutta34`.

The Python package is named `mojo_assimulo`, so it installs alongside the real
`assimulo` and never shadows it.

```python
import numpy as np
import mojo_assimulo as mas

def decay(t, y):
    return -y

s = mas.Dopri5(decay, np.array([1.0]), rtol=1e-10, atol=1e-12)
tlist, ylist = s.integrate(1.0)
ylist[-1][0]          # 0.3678794411714424, i.e. exp(-1)
s.interpolate(0.5)    # Shampine dense output inside the last accepted step
```

## What was ported, and why

Assimulo is a Python-2 era Cython package that ships the Hairer FORTRAN
solvers (`assimulo/thirdparty/hairer/dopri5.f`) and the SUNDIALS C library.
The parts of it that are *arithmetic over the state vector* are concentrated in
DOPCOR, the Dormand-Prince core: the stage states, the embedded error vector and
its weighted norm, the PI step-size controller, the Shampine dense-output
coefficients, the interpolation, the initial step-size guess and the stiffness
detection sums. Every one of those is a loop over the state dimension and is
ported. The driver loop, the step-size *policy* and the right-hand side stay in
Python, exactly as they do upstream, where the right-hand side is a user
callback.

Upstream Assimulo does not build on this interpreter: its sdist is Cython for
Python 2.7/3.5 with bundled Fortran, and `pip install` fails while resolving
build requirements. The parity tests therefore compare against

* the exact Butcher coefficients transcribed from `CDOPRI` in `dopri5.f`, held
  as `fractions.Fraction` and checked against the floating-point tableaus;
* an independent vectorised NumPy transcription of the same algorithms
  (`tests/reference.py`), including the whole DOPCOR driver loop;
* analytic solutions with known error rates (the single-step convergence rate,
  the decay and harmonic-oscillator solutions, and the local error magnitude
  that the embedded estimator predicts).

## Covered subset

| area | implemented API |
| --- | --- |
| Dormand-Prince 5(4) | `Dopri5` with `rtol`, `atol`, `safe`, `fac1`, `fac2`, `beta`, `maxh`, `maxsteps`, `inith`; `integrate`, `interpolate`, `statistics` |
| DOPRI5 kernels | `rk_combo`, `lincomb`, `l2_norm`, `err_norm`, `dopri5_hnew`, `dopri5_cont_d`, `dopri5_cont_abc`, `dopri5_interp`, `hinit_norms`, `hinit_der2`, `hinit_guess`, `hinit_step`, `stiffness` |
| Runge-Kutta 4 | `RungeKutta4` with fixed step `h`; `step`, `integrate` |
| Bogacki-Shampine 3(2) | `RungeKutta34` with `rtol`, `atol`, `h`, `maxsteps`; `adjust_stepsize`, Hermite `interpolate` |
| BS23 kernels | `rk_combo`, `rk34_error`, `l2_norm`, `hermite_interp` |
| Tableaus | `mojo_assimulo.tableau` with the `Fraction` coefficients and `as_float` / `as_rows` accessors |

## Not implemented

Everything below is present upstream and is *not* ported. It is control flow,
IO or linear algebra that Mojo would not help with, and saying so is more
useful than a stub.

* **Implicit solvers.** `IDAS`, `CVODE`, `KIN`, `LSODA`, `ODASSL`, `Radau5`,
  `GLIMDA`, `DASP3`, `Radar5`, `RODAS`, `SDIRK_DAE` and the `euler`/`kinsol`
  Cython wrappers. These are dense/banded LU, Newton and BDF iteration; the
  heavy linear algebra lives in LAPACK/BLAS, and porting a sparse Newton solve
  to Mojo without LAPACK would be slower, not faster.
* **Jacobians and sensitivity equations.** `_jac`, `assimulo.problem_algebraic`
  (IDAs) and the `--sparse` options. These are finite-difference and symbolic
  differentiation assembly, i.e. Python-level recursion.
* **Event detection.** `event_locator`, state-event bisection and
  `Explicit_ODE.report_solution`. Pure control flow on scalars.
* **Result objects and IO.** `Result_<Solver>`, `write_data`, `read_data`,
  `to_file`, and the `assimulo.events` module. Pure IO.
* **Problem and result plumbing.** `Explicit_Problem`, `FirstOrderProblem`,
  options dictionaries, `print_statistics`, logging, and the `_leny` bookkeeping.
* **Complex arithmetic.** Assimulo supports complex-valued problems in some
  solvers; this port is `float64` only and rejects anything wider.

## Install and build

The repository pins its own Mojo toolchain:

```bash
bash build/build.sh          # -> dist/libmojo-assimulo.so
PYTHONPATH=python python -m pytest tests -q
```

`build/build.sh` compiles the single compilation unit `src/kernels.mojo` with
`mojo build --emit shared-lib`. The shared library owns no memory: every buffer
crosses the C ABI as a 64-bit address and is rebuilt as
`Pointer[Float64, AnyOrigin[mut=True]]` inside the kernel, which is what keeps
the exported symbols non-parametric.

## Tests

45 parity tests, all passing. They are written to fail on a plausible bug:
a transposed stage row, a wrong gather index, a swapped pair of CONT blocks, a
dropped `max(|y|,|y1|)` in the error scale, a missing `hmax` clip in the step
controller, a mis-signed `E` or `D` weight, and a stage that re-reads row 0
instead of the previous stage are all caught. Exact equality is asserted only
for indices, endpoints and the exact rational coefficients; every floating-point
comparison carries a tolerance, because Mojo emits FMA.

## Performance

Best-of-five wall clock, same process, against the fastest reasonable NumPy
formulation of the same algebra. Every case checks agreement before timing.

| case | NumPy reference | mojo-assimulo | result |
| --- | ---: | ---: | ---: |
| stage combo, n=262144, 5 terms | 65.30 ms | 3.63 ms | 18.00x faster |
| error vector + weighted norm, n=262144 | 11.42 ms | 7.15 ms | 1.60x faster |
| dense output, n=262144 x 64 | 194.79 ms | 126.91 ms | 1.53x faster |
| full DOPRI5 run, n=2000, 4 steps | 1189.54 ms | 1009.21 ms | 1.18x faster |

The stage-combo win is the clearest: NumPy has to materialise `K[rows]` before
the matrix-vector product, and that gather plus allocation dominates. The
whole-integration win is small on purpose — that case is dominated by the
Python right-hand side callback, which is common to both sides and is exactly
the work Mojo cannot touch here.

Reproduce with:

```bash
python bench/bench.py
```

## How it works

Stage derivatives live in one `(m, n)` buffer addressed by an explicit row
index array, not a fixed stride, because the DOPRI5 error and dense-output
combinations re-use a non-contiguous subset of the stage rows: the FORTRAN
overwrites `K2` with the second evaluation at the end of the step, so `A71`, `E`
and `D` all skip row 1. One gathered kernel therefore serves every stage, the
error estimate and the dense-output coefficients.

The kernels are memory-bound gathers over contiguous state vectors, so they are
plain serial loops. Threading was measured to be slower on this class of work.

## License

MIT, per this repository's `LICENSE`. Upstream Assimulo is
LGPL-2.1-or-later; no upstream code is vendored here, only its
published algorithms, reimplemented.
