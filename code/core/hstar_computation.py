"""
hstar_computation.py
--------------------
Explicit computation of H* and C(P) from polynomial coefficients.

Paper reference:
  Definition 4.1  (H* = max_k |P(w_k)|²)
  Proposition 4.2 (Gauss–Lucas bound on H*)
  Theorem 5.1     (Eyring–Kramers pre-exponential C(P))
"""

import numpy as np
from typing import Tuple
from .energy_landscape import EnergyLandscape


def compute_hstar(landscape: EnergyLandscape) -> Tuple[float, float, dict]:
    """
    Compute the maximal saddle height H* and Eyring–Kramers constant C(P).

    H* = max_k |P(w_k)|²
    where w_k are roots of P' (saddle points of E).

    C(P) = |λ⁻(w*)| / (2π) · √(det H_E(z*)) / √(|det H_E(w*)|)

    Parameters
    ----------
    landscape : EnergyLandscape

    Returns
    -------
    H_star : float
        Maximal saddle height.
    C_P : float
        Eyring–Kramers pre-exponential constant.
    details : dict
        Per-saddle energies, dominant saddle, Hessian eigenvalues.

    Examples
    --------
    >>> landscape = EnergyLandscape([1, 0, 0, 0, -1, -1])  # z^5 - z - 1
    >>> H_star, C_P, details = compute_hstar(landscape)
    >>> print(f"H* = {H_star:.4f}")   # → 2.3562
    >>> print(f"C(P) = {C_P:.4f}")   # → 2.6370

    Paper: Definition 4.1, Proposition 4.2, Theorem 5.1
    """
    saddles = landscape.roots_of_Pp()
    roots   = landscape.roots_of_P()

    # ── Saddle energies ───────────────────────────────────────────
    saddle_energies = []
    for w in saddles:
        Ew = landscape.E(w)
        # Filter: must not be a root of P
        is_root = any(abs(w - r) < 1e-5 for r in roots)
        if not is_root and Ew > 1e-8:
            saddle_energies.append((w, Ew))

    if not saddle_energies:
        return float('nan'), float('nan'), {}

    # ── H* = maximum saddle energy ────────────────────────────────
    w_star, H_star = max(saddle_energies, key=lambda x: x[1])

    # ── C(P) via Eyring–Kramers (Theorem 5.1) ────────────────────
    # Choose closest root to w_star as z*
    z_star = min(roots, key=lambda r: abs(r - w_star))

    H_zstar = landscape.hessianE(z_star)
    H_wstar = landscape.hessianE(w_star)

    eigs_z  = np.linalg.eigvalsh(H_zstar)
    eigs_w  = np.linalg.eigvalsh(H_wstar)

    det_z   = max(np.prod(np.maximum(eigs_z, 1e-12)), 1e-12)
    det_w   = abs(np.prod(eigs_w)) + 1e-12
    neg_eig = eigs_w[eigs_w < -1e-8]
    lam_neg = abs(neg_eig[0]) if len(neg_eig) > 0 else 1.0

    C_P = lam_neg / (2 * np.pi) * np.sqrt(det_z) / np.sqrt(det_w)

    details = {
        "saddle_energies": saddle_energies,
        "w_star":          w_star,
        "z_star":          z_star,
        "H_star":          H_star,
        "C_P":             C_P,
        "lambda_neg":      lam_neg,
        "det_H_zstar":     det_z,
        "det_H_wstar":     det_w,
        "eigs_z":          eigs_z,
        "eigs_w":          eigs_w,
    }
    return H_star, C_P, details


def mixing_time_bound(H_star: float, C_P: float, beta: float,
                      epsilon: float = 0.01, M: float = 10.0) -> float:
    """
    Upper bound on mixing time (Corollary 5.2):

        t_mix(ε) ≤ (1/λ₁(β)) · log(M/ε)
                 ≈ (1/C_P) · exp(β·H*) · log(M/ε)

    Parameters
    ----------
    H_star, C_P : float  — from compute_hstar()
    beta        : float  — inverse temperature
    epsilon     : float  — TV accuracy (default 0.01)
    M           : float  — initial density bound (default 10.0)

    Returns
    -------
    t_mix : float  — mixing time upper bound (continuous time units)

    Paper: Corollary 5.2
    """
    gap   = C_P * np.exp(-beta * H_star)
    t_mix = (1.0 / max(gap, 1e-15)) * np.log(M / epsilon)
    return t_mix


def critical_beta(H_star: float) -> float:
    """
    Critical inverse temperature β_c = 1/H*.

    For β < β_c: global mixing is achievable (success rate → 100%).
    For β > β_c: exponential trapping (success rate → 0%).

    Paper: Section 7, EXP-3.
    """
    return 1.0 / H_star if H_star > 0 else float('inf')


def optimal_annealing_rate(C_P: float, H_star: float,
                           beta: float) -> float:
    """
    Maximum allowable annealing rate for global coverage (Theorem 8.1):

        dβ/dt ≤ C(P) · exp(-β · H*)

    Parameters
    ----------
    C_P, H_star : float — from compute_hstar()
    beta        : float — current inverse temperature

    Returns
    -------
    max_rate : float — maximum dβ/dt that guarantees global exploration

    Paper: Theorem 8.1
    """
    return C_P * np.exp(-beta * H_star)


def gauss_lucas_bound(landscape: EnergyLandscape) -> float:
    """
    Upper bound on H* via Gauss–Lucas theorem (Proposition 4.2):

        H*(P) ≤ max_{z ∈ Conv(roots of P)} |P(z)|²

    Since all roots of P' lie in Conv(roots of P), H* is bounded by the
    maximum of E on the convex hull.

    Paper: Proposition 4.2
    """
    roots = landscape.roots_of_P()
    # Sample convex hull boundary
    n_sample = 500
    samples  = []
    for i in range(len(roots)):
        for j in range(i+1, len(roots)):
            for t in np.linspace(0, 1, 20):
                samples.append(t * roots[i] + (1-t) * roots[j])

    return max(landscape.E(z) for z in samples) if samples else float('nan')
