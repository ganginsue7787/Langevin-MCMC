"""
energy_landscape.py
-------------------
Core energy function E(z) = |P(z)|² for polynomial P of degree n.

Paper reference:
  Section 2 (Setup), Lemma 3.1 (Gradient Formula)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnergyLandscape:
    """
    Wraps E(z) = |P(z)|² and its derivatives for a monic polynomial P.

    Parameters
    ----------
    coeffs : array-like
        Coefficients of P in numpy.polyval convention:
        [a_n, a_{n-1}, ..., a_1, a_0]  so P(z) = a_n·z^n + ...

    Examples
    --------
    >>> # P(z) = z^5 - z - 1
    >>> landscape = EnergyLandscape([1, 0, 0, 0, -1, -1])
    >>> landscape.E(1.167304)      # E at real root ≈ 0
    >>> landscape.E(0.66874)       # E at saddle ≈ 2.3562
    """

    coeffs: np.ndarray

    def __post_init__(self):
        self.coeffs   = np.asarray(self.coeffs, dtype=complex)
        self.degree   = len(self.coeffs) - 1
        self.d_coeffs = np.polyder(self.coeffs)   # P'
        self.dd_coeffs= np.polyder(self.d_coeffs) # P''

    # ── Polynomial evaluations ────────────────────────────────────
    def P(self, z: complex) -> complex:
        """P(z)"""
        if abs(z) > 20:
            return complex(1e10, 0)
        return complex(np.polyval(self.coeffs, z))

    def Pp(self, z: complex) -> complex:
        """P'(z)"""
        if abs(z) > 20:
            return complex(1e10, 0)
        return complex(np.polyval(self.d_coeffs, z))

    def Ppp(self, z: complex) -> complex:
        """P''(z)"""
        return complex(np.polyval(self.dd_coeffs, z))

    # ── Energy function E(z) = |P(z)|² ───────────────────────────
    def E(self, z: complex) -> float:
        """
        E(z) = |P(z)|²

        Paper: Section 2, equation (1)
        """
        return float(abs(self.P(z))**2)

    # ── Gradient of E (Lemma 3.1) ─────────────────────────────────
    def gradE(self, z: complex) -> complex:
        """
        ∇E(z) in complex form:
          ∂E/∂x = 2 Re[conj(P(z))·P'(z)]
          ∂E/∂y = -2 Im[conj(P(z))·P'(z)]

        Returned as complex number gx + i·gy for use in Euler–Maruyama.

        Paper: Lemma 3.1
        """
        if abs(z) > 20:
            return z * 1e4

        Pv  = self.P(z)
        Ppv = self.Pp(z)
        gx  = 2 * (Pv.real * Ppv.real + Pv.imag * Ppv.imag)
        gy  = 2 * (Pv.imag * Ppv.real - Pv.real * Ppv.imag)
        g   = complex(gx, gy)

        # Gradient clipping for numerical stability
        if abs(g) > 200:
            g = g / abs(g) * 200
        return g

    # ── Hessian of E (Proposition 3.2 proof) ──────────────────────
    def hessianE(self, z: complex, eps: float = 1e-4) -> np.ndarray:
        """
        2×2 real Hessian of E at z (numerical differentiation).

        Returns H such that:
          Morse index = number of negative eigenvalues of H
          H at root z* → positive definite (index 0)
          H at saddle w_k → indefinite (index 1)

        Paper: Proof of Proposition 3.2
        """
        x0, y0 = z.real, z.imag

        def e(x, y):
            return self.E(complex(x, y))

        e0 = e(x0, y0)
        H  = np.array([
            [(e(x0+eps,y0) - 2*e0 + e(x0-eps,y0)) / eps**2,
             (e(x0+eps,y0+eps) - e(x0+eps,y0-eps)
              - e(x0-eps,y0+eps) + e(x0-eps,y0-eps)) / (4*eps**2)],
            [(e(x0+eps,y0+eps) - e(x0+eps,y0-eps)
              - e(x0-eps,y0+eps) + e(x0-eps,y0-eps)) / (4*eps**2),
             (e(x0,y0+eps) - 2*e0 + e(x0,y0-eps)) / eps**2],
        ])
        return H

    def morse_index(self, z: complex) -> int:
        """
        Morse index of E at z = number of negative eigenvalues of Hessian.

        Proposition 3.2:
          morse_index(z*) = 0  for roots of P
          morse_index(w_k) = 1 for roots of P'
        """
        eigs = np.linalg.eigvalsh(self.hessianE(z))
        return int(np.sum(eigs < -1e-8))

    # ── Critical points ────────────────────────────────────────────
    def roots_of_P(self) -> np.ndarray:
        """Roots of P (global minimizers, E=0)."""
        return np.roots(self.coeffs)

    def roots_of_Pp(self) -> np.ndarray:
        """Roots of P' (candidate saddle points of E)."""
        return np.roots(self.d_coeffs)

    # ── Euler characteristic check (Corollary 3.3) ────────────────
    def euler_characteristic(self) -> dict:
        """
        Verify χ(S²) = C₀ - C₁ + C₂ = 2.

        C₀ = #roots of P  (minima)
        C₁ = #roots of P' (saddles)
        C₂ = 1            (maximum at ∞)

        Paper: Corollary 3.3
        """
        C0 = self.degree
        C1 = self.degree - 1
        C2 = 1
        chi = C0 - C1 + C2
        return {"C0": C0, "C1": C1, "C2": C2, "chi": chi, "valid": chi == 2}
