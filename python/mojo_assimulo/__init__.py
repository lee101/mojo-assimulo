"""mojo-assimulo: the explicit Runge-Kutta core of Assimulo in Mojo.

The Python package is named `mojo_assimulo`, so it installs alongside the real
`assimulo` and never shadows it. Upstream Assimulo is a Python-2 era Cython
package with bundled Fortran and C sources; it does not build on this
interpreter, so the parity tests here compare against analytic solutions, a
vectorised NumPy transcription of the same algorithms, and the exact Butcher
coefficients from `assimulo/thirdparty/hairer/dopri5.f`.
"""

from .runge_kutta import Dopri5, RungeKutta4, RungeKutta34
from .tableau import ALL_TABLEAUS, as_float, as_rows

__all__ = [
    "Dopri5",
    "RungeKutta4",
    "RungeKutta34",
    "ALL_TABLEAUS",
    "as_float",
    "as_rows",
]
__version__ = "0.1.0"
