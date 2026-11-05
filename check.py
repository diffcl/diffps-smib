"""Check the package: recompute the stored metrics, then regenerate every table and figure.

1. From the parameter estimates, recompute the LQR gain error, the linear cost excess and,
   by nonlinear simulation, the response discrepancy e_c of all 150 parameter-based cases.
2. From the stored NODE and DP Jacobians, recompute the R=0.1 gain errors, linear cost excesses,
   physical-row Jacobian errors and, by simulation, e_c; and the R>=1 gain and damping-ratio errors.
   The R>=1 nonlinear metrics (e_c, nonlinear cost gap, phase slips) are stored simulation outputs
   and are not recomputed here.
3. Re-simulate the Fig. lqr-dp traces and compare them with results/lqr_dp_traces.npz.
4. Regenerate the tables and figures and compare them byte for byte with reference/.
Prints PASS and exits with 0 on success; stops at the first failure otherwise.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

import make_article
import smib
from smib import PLANTS, read_csv

DJ_ATOL = 1e-14  # dJ = J/J* - 1 carries float64 rounding of order 1e-15 that depends on the BLAS kernel
TRACE_ATOL = 1e-7  # BLAS-kernel differences in K* move the traces by up to about 4e-9


def close(a, b, rtol, what, atol=1e-15):
    assert abs(a - b) <= rtol * abs(b) + atol, f'{what}: recomputed {a!r}, stored {b!r}'


def check_parameter_cases():
    est = {(r['method'], r['noise'], r['draw']): r for r in read_csv('identification_estimates.csv')}
    M, D, Pmax = PLANTS['light']
    reference = {}
    cases = read_csv('parameter_control_cases.csv')
    for r in cases:
        Pm = float(r['Pm'])
        e = est[r['method'], r['noise'], r['draw']]
        M_hat, D_hat = float(e['M_hat']), float(e['D_hat'])
        d_eq, A, B = smib.lin(M, D, Pmax, Pm)
        _, A_hat, B_hat = smib.lin(M_hat, D_hat, Pmax, Pm)
        K_star, K_hat = smib.lqr(A, B), smib.lqr(A_hat, B_hat)
        what = f"{r['method']} {r['noise']}_r{r['draw']} Pm={Pm}"
        close(np.linalg.norm(K_hat - K_star) / np.linalg.norm(K_star), float(r['eps_K']), 1e-6, what + ' eps_K')
        close(smib.cost(A, B, K_hat) / smib.cost(A, B, K_star) - 1, float(r['dJ']), 1e-4, what + ' dJ', atol=DJ_ATOL)
        if Pm not in reference:
            reference[Pm] = smib.simulate(M, D, Pmax, Pm, K_star)
        x_hat = smib.simulate(M, D, Pmax, Pm, K_hat)
        e_c = smib.response_discrepancy(x_hat, reference[Pm], np.array([d_eq, 0.0]))
        close(e_c, float(r['e_c']), 1e-4, what + ' e_c', atol=1e-10)  # integrator floor
    return len(cases)


def jacobians(r):
    return (np.array([[float(r['A11']), float(r['A12'])], [float(r['A21']), float(r['A22'])]]),
            np.array([[float(r['B1'])], [float(r['B2'])]]))


def check_r01_cases():
    rows = read_csv('r01_cases.csv')
    reference = {}
    for r in rows:
        M, D, Pmax = PLANTS[r['plant']]
        Pm = float(r['Pm'])
        d_eq, A, B = smib.lin(M, D, Pmax, Pm)
        A_hat, B_hat = jacobians(r)
        K_star, K_hat = smib.lqr(A, B), smib.lqr(A_hat, B_hat)
        what = f"{r['plant']} {r['model']} {r['starts']} s{r['sigma']} r{r['draw']} seed{r['seed']} Pm={r['Pm']}"
        close(np.linalg.norm(A_hat[1] - A[1]) / np.linalg.norm(A[1]), float(r['eps_A_phys']), 1e-6, what + ' eps_A_phys')
        close(np.linalg.norm(K_hat - K_star) / np.linalg.norm(K_star), float(r['eps_K']), 1e-6, what + ' eps_K')
        close(smib.cost(A, B, K_hat) / smib.cost(A, B, K_star) - 1, float(r['dJ']), 1e-4, what + ' dJ', atol=DJ_ATOL)
        if (r['plant'], Pm) not in reference:
            reference[r['plant'], Pm] = smib.simulate(M, D, Pmax, Pm, K_star)
        e_c = smib.response_discrepancy(smib.simulate(M, D, Pmax, Pm, K_hat), reference[r['plant'], Pm], np.array([d_eq, 0.0]))
        close(e_c, float(r['e_c']), 1e-4, what + ' e_c', atol=1e-10)  # integrator floor
    return rows


def check_weight_cases(r01_rows):
    jac = {(r['plant'], r['model'], r['starts'], float(r['sigma']), float(r['draw']), float(r['seed']), float(r['Pm'])): jacobians(r)
           for r in r01_rows if r['model'] != 'dp'}
    n = 0
    for r in read_csv('weight_cases.csv'):
        if r['protocol'] != 'pulse':
            continue
        key = (r['plant'], r['model'], r['starts'], float(r['sigma']), float(r['draw']), float(r['seed']), float(r['Pm']))
        A_hat, B_hat = jac[key]
        M, D, Pmax = PLANTS[r['plant']]
        _, A, B = smib.lin(M, D, Pmax, float(r['Pm']))
        R = float(r['R'])
        K_star, K_hat = smib.lqr(A, B, R), smib.lqr(A_hat, B_hat, R)
        zs, zh = smib.damping_ratios(A, B, K_star), smib.damping_ratios(A, B, K_hat)
        what = ' '.join(f'{k}={v}' for k, v in zip(['plant', 'model', 'starts', 'sigma', 'draw', 'seed', 'Pm'], key)) + f' R={R}'
        close(np.linalg.norm(K_hat - K_star) / np.linalg.norm(K_star), float(r['eps_K']), 1e-6, what + ' eps_K')
        close(np.linalg.norm(zh - zs) / np.linalg.norm(zs), float(r['damping_relative_error']), 1e-6, what + ' e_zeta')
        n += 1
    return n


def check_dp_traces():
    stored = np.load(smib.ROOT / 'results' / 'lqr_dp_traces.npz')
    d_eq, traces, shown = make_article.dp_response_traces()
    assert (shown['noise'], shown['draw']) == (str(stored['noise']), str(stored['draw'])), 'Fig. lqr-dp case differs'
    close(d_eq, float(stored['d_eq']), 1e-12, 'Fig. lqr-dp d_eq')
    diff = np.max(np.abs(np.array(traces) - stored['traces']))
    assert diff <= TRACE_ATOL, f'Fig. lqr-dp traces differ from results/lqr_dp_traces.npz by {diff:.1e}'
    return diff


def check_outputs():
    reference = smib.ROOT / 'reference'
    with tempfile.TemporaryDirectory(prefix='smib-access-') as folder:
        make_article.generate(folder)
        names = sorted(p.name for p in reference.iterdir())
        for name in names:
            assert (Path(folder) / name).read_bytes() == (reference / name).read_bytes(), f'{name} differs from reference/'
    return names


if __name__ == '__main__':
    n_param = check_parameter_cases()
    print(f'parameter-based control: {n_param} cases recomputed (gain, linear cost, simulated e_c)')
    r01_rows = check_r01_cases()
    print(f'R=0.1 NODE and DP cases: {len(r01_rows)} recomputed from stored Jacobians (gain, linear cost, simulated e_c)')
    print(f'R>=1 NODE gains: {check_weight_cases(r01_rows)} recomputed from stored Jacobians')
    print(f'Fig. lqr-dp traces: max difference {check_dp_traces():.1e} from results/lqr_dp_traces.npz')
    names = check_outputs()
    print(f'{len(names)} table and figure files identical to reference/: ' + ', '.join(names))
    print('PASS')
    sys.exit(0)
