"""
langevin_mcmc.py
----------------
Euler–Maruyama discretization of overdamped Langevin diffusion:

    dZ_t = -∇E(Z_t) dt + √(2/β) dW_t

Paper reference:
  Section 5.1 (Langevin Diffusion)
  Corollary 5.2 (Mixing time upper bound)
  Section 7 (Numerical validation)
"""

import numpy as np
from typing import List, Optional, Tuple
from .energy_landscape import EnergyLandscape


def run_single_chain(
    landscape:  EnergyLandscape,
    beta:       float,
    roots_true: np.ndarray,
    max_steps:  int  = 80_000,
    dt:         float = 0.003,
    threshold:  float = 0.05,
    seed:       int  = 0,
    z0:         Optional[complex] = None,
) -> Tuple[set, List[Tuple[int, float]], Optional[int]]:
    """
    Run one Langevin chain. Track which roots are visited.

    The correct observable for global mixing (Section 7.1) is the
    *all-roots coverage time*: steps until all roots have been visited.
    Single-root convergence does NOT measure global mixing.

    Parameters
    ----------
    landscape   : EnergyLandscape
    beta        : float — inverse temperature
    roots_true  : (n,) complex array — known roots of P
    max_steps   : int
    dt          : float — Euler–Maruyama step size (default 0.003)
    threshold   : float — E(z) < threshold → declare root visited
    seed        : int
    z0          : complex or None — initial point (random if None)

    Returns
    -------
    visited      : set of root indices visited
    traj_E       : list of (step, E(z)) snapshots
    coverage_step: step at which all roots were first visited (None if not)

    Paper: Algorithm 1 (Euler–Maruyama), Section 7.1
    """
    rng   = np.random.RandomState(seed)
    noise = np.sqrt(2.0 * dt / beta)

    if z0 is None:
        r0  = rng.uniform(1.5, 3.0)
        th0 = rng.uniform(0, 2 * np.pi)
        z   = r0 * np.exp(1j * th0)
    else:
        z = z0

    n_roots   = len(roots_true)
    visited   = set()
    traj_E    = []

    for step in range(max_steps):
        # Euler–Maruyama update
        g    = landscape.gradE(z)
        dW   = (rng.randn() + 1j * rng.randn()) / np.sqrt(2.0)
        z_new = z - dt * g + noise * dW
        if abs(z_new) < 20:
            z = z_new

        Ev = landscape.E(z)

        # Snapshot every 1000 steps
        if step % 1000 == 0:
            traj_E.append((step, Ev))

        # Visit detection
        if Ev < threshold:
            nr = int(np.argmin(np.abs(z - roots_true)))
            visited.add(nr)
            if len(visited) == n_roots:
                return visited, traj_E, step

    return visited, traj_E, None


def run_coverage(
    landscape:  EnergyLandscape,
    beta:       float,
    n_chains:   int   = 40,
    max_steps:  int   = 80_000,
    dt:         float = 0.003,
    threshold:  float = 0.05,
    seed_base:  int   = 0,
) -> Tuple[float, float, List[int]]:
    """
    Run n_chains independent Langevin chains and collect coverage statistics.

    Returns
    -------
    mean_coverage : float — mean all-roots coverage time (steps)
    success_rate  : float — fraction of chains achieving full coverage
    times         : list  — per-chain coverage time (max_steps if failed)

    Paper: Section 7.2 (Table 2), EXP-3
    """
    roots_true = landscape.roots_of_P()
    times      = []

    for i in range(n_chains):
        visited, _, cov_step = run_single_chain(
            landscape, beta, roots_true,
            max_steps=max_steps, dt=dt,
            threshold=threshold,
            seed=seed_base + i * 31,
        )
        times.append(cov_step if cov_step is not None else max_steps)

    success   = [t for t in times if t < max_steps]
    mean_cov  = float(np.mean(success)) if success else float(max_steps)
    rate      = len(success) / n_chains

    return mean_cov, rate, times


def adaptive_beta_schedule(
    H_star:    float,
    C_P:       float,
    beta_init: float = 0.2,
    beta_final:float = 2.0,
    n_steps:   int   = 10_000,
) -> np.ndarray:
    """
    Optimal annealing schedule satisfying Theorem 8.1:

        dβ/dt ≤ C(P) · exp(-β(t) · H*)

    Integrates this ODE with forward Euler to produce β(t).

    Parameters
    ----------
    H_star, C_P : float — from compute_hstar()
    beta_init   : float — starting inverse temperature
    beta_final  : float — target inverse temperature
    n_steps     : int   — number of discrete time steps

    Returns
    -------
    beta_schedule : (n_steps,) array

    Paper: Theorem 8.1
    """
    dt_anneal = 1.0
    betas     = [beta_init]
    b         = beta_init

    for _ in range(n_steps - 1):
        rate  = C_P * np.exp(-b * H_star)        # max allowed dβ/dt
        b_new = b + 0.95 * rate * dt_anneal       # 0.95 safety margin
        b_new = min(b_new, beta_final)
        betas.append(b_new)
        if b_new >= beta_final:
            betas.extend([beta_final] * (n_steps - len(betas)))
            break

    return np.array(betas[:n_steps])


def run_adaptive_chain(
    landscape:    EnergyLandscape,
    H_star:       float,
    C_P:          float,
    n_steps:      int   = 50_000,
    dt:           float = 0.003,
    seed:         int   = 0,
) -> Tuple[complex, float, np.ndarray]:
    """
    Langevin chain with adaptive annealing schedule (Theorem 8.1).

    Returns final state, final E(z), and beta schedule used.

    Paper: Theorem 8.1 (necessary and sufficient annealing condition)
    """
    rng      = np.random.RandomState(seed)
    schedule = adaptive_beta_schedule(H_star, C_P, n_steps=n_steps)

    r0 = rng.uniform(1.5, 3.0)
    z  = r0 * np.exp(1j * rng.uniform(0, 2*np.pi))

    for step, beta in enumerate(schedule):
        noise = np.sqrt(2.0 * dt / beta)
        g     = landscape.gradE(z)
        dW    = (rng.randn() + 1j * rng.randn()) / np.sqrt(2.0)
        z_new = z - dt * g + noise * dW
        if abs(z_new) < 20:
            z = z_new

    return z, landscape.E(z), schedule
