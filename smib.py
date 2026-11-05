"""Plant definitions, linearization, LQR design, linear cost and the pulse-test simulation."""
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parent
PLANTS = {'primary': (0.2, 0.1, 0.5), 'light': (0.1, 0.01, 1.0)}  # (M, D, P_max)
Q = np.diag([10.0, 1.0])
R_NOMINAL = 0.1
DRIVES = {'primary': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], 'light': [0.2, 0.4, 0.5, 0.6, 0.8]}
T = np.linspace(0.0, 20.0, 1000)
ON = T >= 2.0


def read_csv(name):
    with (ROOT / 'results' / name).open() as f:
        return list(csv.DictReader(f))


def lin(M, D, Pmax, Pm):
    """Equilibrium angle and Jacobians A, B of the swing equation."""
    d = np.arcsin(Pm / Pmax)
    return d, np.array([[0.0, 1.0], [-Pmax * np.cos(d) / M, -D / M]]), np.array([[0.0], [1.0 / M]])


def lqr(A, B, R=R_NOMINAL):
    R = np.array([[R]])
    return np.linalg.solve(R, B.T @ solve_continuous_are(A, B, Q, R))


def cost(A, B, K, R=R_NOMINAL):
    """J(K) = tr(P_K) on the linearization (A, B)."""
    Ac = A - B @ K
    if np.max(np.linalg.eigvals(Ac).real) >= 0:
        return np.inf
    return np.trace(solve_continuous_lyapunov(Ac.T, -(Q + K.T @ np.array([[R]]) @ K)))


def damping_ratios(A, B, K):
    p = np.linalg.eigvals(A - B @ K)
    return np.sort(-p.real / abs(p))


def simulate(M, D, Pmax, Pm, K):
    """Pulse test on the nonlinear plant: dP_m=0.1 on [1, 2] s, feedback u=-K(x-x*) from t=2 s."""
    d_eq = np.arcsin(Pm / Pmax)
    x_eq = np.array([d_eq, 0.0])

    def rhs(dist, ctrl):
        def f(t, x):
            u = -(K @ (x - x_eq)).item() if ctrl else 0.0
            return [x[1], (Pm - Pmax * np.sin(x[0]) - D * x[1] + dist + u) / M]
        return f

    x0, xs = x_eq.copy(), []
    for a, b, dist, ctrl in [(0.0, 1.0, 0.0, False), (1.0, 2.0, 0.1, False), (2.0, 20.0, 0.0, True)]:
        te = T[(T >= a) & (T < b)] if b < 20.0 else T[T >= a]
        xs.append(solve_ivp(rhs(dist, ctrl), (a, b), x0, t_eval=te, method='DOP853', rtol=1e-10, atol=1e-12).y)
        x0 = solve_ivp(rhs(dist, ctrl), (a, b), x0, method='DOP853', rtol=1e-10, atol=1e-12).y[:, -1]
    return np.concatenate(xs, axis=1)


def response_discrepancy(x_hat, x_star, x_eq):
    """e_c: largest per-state relative l2 difference on t in [2, 20] s."""
    return max(np.linalg.norm(x_hat[i, ON] - x_star[i, ON]) / np.linalg.norm(x_star[i, ON] - x_eq[i]) for i in range(2))
