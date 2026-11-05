"""Regenerate the computed tables and figures of the article from data/ and results/.

Usage: python make_article.py [output_folder]   (default: out/)
Tables are written as LaTeX files and figures as PDF/PNG files, with the names used in the manuscript.
No training or fitting is performed. Fig. lqr-dp is drawn from the stored traces in
results/lqr_dp_traces.npz; save_dp_traces() rewrites that file by simulating the nonlinear plant.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import smib
from smib import DRIVES, PLANTS, read_csv

STARTS = {'MS': 'measured', 'ES': 'estimated'}


def g3(x):
    return f'{x:.3g}'


def median_pct(values):
    return g3(100 * float(np.median([float(v) for v in values])))


DJ_FLOOR = 1e-14  # dJ = J/J* - 1 cannot resolve smaller cost excesses in float64


def median_dj_pct(values):
    m = float(np.median([float(v) for v in values]))
    return r'$<10^{-12}$' if abs(m) < DJ_FLOOR else g3(100 * m)


# ---------------------------------------------------------------- Table 2: identification errors
def table_noise_realizations():
    M0, D0 = PLANTS['light'][:2]
    est = read_csv('identification_estimates.csv')

    def ms(v):
        m, s = v.mean(), v.std()
        e = int(np.floor(np.log10(m)))
        return rf"$({m / 10**e:.2f}\pm{s / 10**e:.2f})\times10^{{{e}}}$"

    rows = []
    for method in ['dp', 'pinn', 'oe']:
        for k, kind in enumerate(['mult', 'add']):
            sel = [r for r in est if r['method'] == method and r['noise'] == kind]
            assert len(sel) == 5
            eM = np.array([abs(float(r['M_hat']) - M0) / M0 for r in sel])
            eD = np.array([abs(float(r['D_hat']) - D0) / D0 for r in sel])
            lead = rf"\multirow{{2}}{{*}}{{{method.upper()}}}" if k == 0 else ''
            rows.append(rf"{lead} & {'Multiplicative' if kind == 'mult' else 'Additive'} & {ms(eM)} & {ms(eD)} \\")
        rows.append(r'\midrule')
    rows[-1] = r'\bottomrule'
    return r"""\begin{table}[t]
\centering
\caption{Absolute relative errors $|\hat\theta-\theta|/\theta$ in $\theta_M$ and $\theta_D$ at $\sigma=0.05$, mean $\pm$ population standard deviation over five noise realizations (one identification per realization). Multiplicative: $\tilde{y}=y(1+\sigma\eta)$; additive: $\tilde{y}=y+\sigma\,y_{\mathrm{rms}}\eta$, with the same draw $\eta$ for both.}
\label{tab:noise_realizations}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcc}
\toprule
Model & Noise & $\theta_M$ & $\theta_D$ \\
\midrule
""" + '\n'.join(rows) + r"""
\end{tabular}
\end{table}
"""


# ---------------------------------------------------------------- Table 3: control from parameter estimates
def table_baseline_control():
    cases = read_csv('parameter_control_cases.csv')

    def fmt(v):
        m, s = np.mean(v), np.std(v)
        e = int(np.floor(np.log10(abs(m))))
        return rf"$({m / 10**e:.1f}\pm{s / 10**e:.1f}){{\cdot}}10^{{{e}}}$"

    rows = []
    for method in ['dp', 'pinn', 'oe']:
        sel = [r for r in cases if r['method'] == method]
        assert len(sel) == 50
        rows.append(f"{method.upper()} & " + ' & '.join(fmt(np.array([float(r[m]) for r in sel]))
                                                       for m in ['eps_K', 'dJ', 'e_c']) + r' \\')
    return r"""\begin{table}[!t]
\centering
\caption{Control accuracy of LQR gains on the light plant with
$Q=\operatorname{diag}(10,1)$ and $R=0.1$.
Each method estimates parameters at $P_m=0.5$ from each of the ten
$\sigma=0.05$ data sets in Table~\ref{tab:noise_realizations}:
five noise draws, each used for both multiplicative and additive noise.
Each estimate $(\hat M,\hat D)$ is then used to design LQR gains at
$P_m\in\{0.2,0.4,0.5,0.6,0.8\}$, which are compared with
the corresponding true gains $K^*$.
Each row reports the mean $\pm$ standard deviation over these
$10\times5=50$ cases.}
\label{tab:op-seed}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
Method & $\epsilon_K$ & $\Delta J/J^*$ & $e_c$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ---------------------------------------------------------------- R = 0.1 cases (NODE and DP)
def r01(model, plant, starts, sigma):
    return [r for r in read_csv('r01_cases.csv') if r['model'] == model and r['plant'] == plant
            and r['starts'] == starts and float(r['sigma']) == sigma and float(r['Pm']) in DRIVES[plant]]


def table_structure():
    rows = []
    for plant in ['light', 'primary']:
        for starts, sigma in [('MS', 0.0), ('MS', 0.05), ('ES', 0.05)]:
            for model, label in [('node', 'NODE'), ('structured', 'Structured NODE')]:
                sel = r01(model, plant, starts, sigma)
                rows.append(' & '.join([plant, label, starts, str(sigma), str(len(sel)),
                                        *(median_pct(r[m] for r in sel) for m in ['eps_A_phys', 'eps_K', 'dJ', 'e_c'])])
                            + r' \\')
    return r"""\begin{table*}[!t]
\centering
\footnotesize
\caption{Median errors of the unconstrained (NODE) and structured NODEs at $R=0.1$. Starts: MS, measured segment starts with $m=10$; ES, estimated segment starts. Cases: seeds $\times$ data sets $\times$ drives. Every gain stabilizes the true linearized system, and every nonlinear simulation completes.}
\label{tab:r2-structure}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccc}
\toprule
 & & & & & \multicolumn{4}{c}{Median (\%)} \\
\cmidrule(lr){6-9}
Plant & Model & Starts & $\sigma$ & Cases & $\epsilon_{A,\mathrm{phys}}$ & $\epsilon_K$ & $\Delta J/J^*$ & $e_c$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


def table_dp_starts():
    rows = []
    for starts, sigma in [('MS', 0.0), ('MS', 0.05), ('ES', 0.05)]:
        sel = r01('dp', 'primary', starts, sigma)
        runs = {(r['seed'], r['draw']): r['D_error'] for r in sel}  # one damping estimate per fit
        rows.append(' & '.join([starts, str(sigma), median_pct(runs.values()),
                                *(median_pct(r[m] for r in sel) for m in ['eps_A_phys', 'eps_K']),
                                median_dj_pct(r['dJ'] for r in sel), median_pct(r['e_c'] for r in sel)]) + r' \\')
    return r"""\begin{table}[!t]
\centering
\footnotesize
\caption{DP errors with MS and ES on the primary plant at $R=0.1$. ES gives the same fit across seeds. $<10^{-12}$: below the double-precision resolution of $\Delta J/J^*$.}
\label{tab:dp-starts}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
 & & \multicolumn{5}{c}{Median (\%)} \\
\cmidrule(lr){3-7}
Starts & $\sigma$ & $\hat D/D-1$ & $\epsilon_{A,\mathrm{phys}}$ & $\epsilon_K$ & $\Delta J/J^*$ & $e_c$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ---------------------------------------------------------------- R >= 1 cases (NODE)
def weight_groups():
    """Pair unconstrained and structured NODEs on identical scenarios.

    A pair enters the response and cost statistics only if neither NODE controller nor the
    reference controller K* slips. Gain statistics use the pulse rows, one per gain.
    """
    cells = defaultdict(dict)
    for r in read_csv('weight_cases.csv'):
        key = (r['plant'], float(r['sigma']), r['starts'], float(r['R']))
        ident = (float(r['seed']), float(r['draw']), float(r['Pm']), r['protocol'], r['fraction'])
        pair = cells[key].setdefault(ident, {})
        assert r['model'] not in pair
        pair[r['model']] = r
    groups = {}
    for key, scenarios in cells.items():
        pairs = list(scenarios.values())
        assert all(set(p) == {'node', 'structured'} for p in pairs)
        slip = lambda p: any(p[m][k] == 'True' for m in p for k in ['phase_slip', 'reference_phase_slip'])
        groups[key] = dict(pairs=pairs, gains=[p for p in pairs if p['node']['protocol'] == 'pulse'],
                           common=[p for p in pairs if not slip(p)])
    return groups


def table_low_gain(groups):
    rows = []
    for plant, sigma, starts in sorted({k[:3] for k in groups}, key=lambda a: (a[0], a[1], STARTS[a[2]])):
        name = plant.capitalize() + (', clean' if sigma == 0 else r', $\sigma=0.05$') + f', {starts}'
        for R in [1, 10, 100, 1000]:
            g = groups[plant, sigma, starts, float(R)]
            med = lambda pop, m, k: 100 * float(np.median([float(p[m][k]) for p in pop]))
            pair = lambda pop, k: ' / '.join(g3(med(pop, m, k)) for m in ['node', 'structured'])
            mark = r'$\dagger$' if med(g['gains'], 'structured', 'eps_K') > med(g['gains'], 'node', 'eps_K') else ''
            group = (r'\midrule' if rows else '') + r'\multirow{4}{*}{' + name + '}' if R == 1 else ''
            rows.append(' & '.join([group, str(R), pair(g['gains'], 'eps_K') + mark,
                                    pair(g['gains'], 'damping_relative_error'),
                                    pair(g['common'], 'e_c'), pair(g['common'], 'dJ_nl')]) + r' \\')
    head = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
            r'\caption{NODE errors at larger input penalties. Entries: unconstrained / structured medians (\%). $\dagger$: structured gain error is larger. $e_\zeta=0$ occurs when both closed loops have only real eigenvalues and does not imply equal decay rates.}',
            r'\label{tab:control-weights}', r'\begin{tabular}{lrcccc}', r'\toprule',
            r'Plant / data / starts & $R$ & $\epsilon_K$ & $e_\zeta$ & $e_c$ & $\Delta J_{\rm nl}/J^*_{\rm nl}$ \\',
            r'\midrule']
    return '\n'.join(head + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


# ---------------------------------------------------------------- figures
STYLES = {'node': dict(color='#333333', marker='o', linestyle='--', label='Unconstrained'),
          'structured': dict(color='#0072B2', marker='s', linestyle='-', label='Structured')}


def figure_style():
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def draw_medians(pairs, model, metric):
    """One median per noise draw, in draw order."""
    draws = sorted({float(p[model]['draw']) for p in pairs})
    return [100 * float(np.median([float(p[model][metric]) for p in pairs if float(p[model]['draw']) == d])) for d in draws]


def figure_structure_protocol(output):
    """Fig. structure_protocol: noisy primary-plant gain errors at R=0.1, one point per draw."""
    figure_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.25), layout='constrained')
    for model, style in STYLES.items():
        for i, starts in enumerate(['MS', 'ES']):
            sel = r01(model, 'primary', starts, 0.05)
            values = draw_medians([{model: r} for r in sel], model, 'eps_K')
            x = i + (-.13 if model == 'node' else .13)
            jit = np.linspace(-.055, .055, len(values))
            ax.scatter(x + jit, values, s=20, facecolors='white', edgecolors=style['color'], marker=style['marker'],
                       label=style['label'] if i == 0 else None)
            ax.plot([x - .08, x + .08], [np.median(values)] * 2, color=style['color'], lw=2)
    ax.set(yscale='log', ylabel='Gain error (%)', xticks=[0, 1],
           xticklabels=['Measured starts (MS)', 'Estimated starts (ES)'])
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'structure_protocol.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def figure_weight_response(groups, output):
    """Fig. weight_response: noisy-data gain and response errors at R>=1, one point per draw."""
    figure_style()
    configs = [('primary', 'MS', 'Primary, MS'), ('primary', 'ES', 'Primary, ES'),
               ('light', 'MS', 'Light, MS'), ('light', 'ES', 'Light, ES')]
    weights = [1, 10, 100, 1000]
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.6), layout='constrained')
    for col, (plant, starts, title) in enumerate(configs):
        for row, (metric, population) in enumerate([('eps_K', 'gains'), ('e_c', 'common')]):
            ax = axes[row, col]
            for model, style in STYLES.items():
                medians = []
                for R in weights:
                    vals = draw_medians(groups[plant, 0.05, starts, float(R)][population], model, metric)
                    medians.append(float(np.median(vals)))
                    ax.scatter(np.array(R) * np.geomspace(.91, 1.09, len(vals)), vals, s=9, facecolors='white',
                               edgecolors=style['color'], marker=style['marker'], lw=.6, zorder=3)
                ax.plot(weights, medians, color=style['color'], linestyle=style['linestyle'], lw=1, label=style['label'])
            ax.set(xscale='log', yscale='log', xticks=weights)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel('Input weight R')
            if col == 0:
                ax.set_ylabel('Gain error (%)' if row == 0 else 'Response error (%)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=2, fontsize=7)
    fig.savefig(output / 'weight_response.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def dp_response_traces():
    """Fig. lqr-dp: the DP estimate with the largest e_c at P_m=0.5 among the ten noisy data sets."""
    M, D, Pmax = PLANTS['light']
    Pm = 0.5
    cases = [r for r in read_csv('parameter_control_cases.csv') if r['method'] == 'dp' and float(r['Pm']) == Pm]
    shown = max(cases, key=lambda r: float(r['e_c']))
    est = next(r for r in read_csv('identification_estimates.csv')
               if r['method'] == 'dp' and r['noise'] == shown['noise'] and r['draw'] == shown['draw'])
    M_hat, D_hat = float(est['M_hat']), float(est['D_hat'])
    d_eq, A, B = smib.lin(M, D, Pmax, Pm)
    _, A_hat, B_hat = smib.lin(M_hat, D_hat, Pmax, Pm)
    K_star, K_hat = smib.lqr(A, B), smib.lqr(A_hat, B_hat)
    traces = [smib.simulate(M, D, Pmax, Pm, K) for K in (np.zeros_like(K_star), K_star, K_hat)]
    traces.append(smib.simulate(M_hat, D_hat, Pmax, Pm, K_hat))
    return d_eq, traces, shown


def save_dp_traces():
    """Write results/lqr_dp_traces.npz, the simulated Fig. lqr-dp traces read by figure_lqr_dp."""
    d_eq, traces, shown = dp_response_traces()
    np.savez(smib.ROOT / 'results' / 'lqr_dp_traces.npz', t=smib.T, d_eq=d_eq, traces=np.array(traces),
             noise=shown['noise'], draw=shown['draw'])


def plot_lqr_diff(t, traces, labels, figpath, ylabel, dlabel, target, control_start, disturbance_period):
    """Upper panel: the four traces. Lower panel: |trace - (true system, K*)| for the two K_hat traces."""
    unc, star, hat, model = [np.asarray(x, dtype=float) for x in traces]
    mpl.rc('xtick', labelsize=18)
    mpl.rc('ytick', labelsize=18)
    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, figsize=(8, 8),
                                 gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.08))
    styles = [(':', 'black', 1.5), ('--', 'maroon', 1.5), ('-.', 'teal', 1.5), ('-', 'blue', 1.5)]
    for y, lab, (ls, c, lw) in zip([unc, star, hat, model], labels, styles):
        ax.plot(t, y, ls, color=c, lw=lw, label=lab)
    ax.axhline(y=target, linewidth=1.5, color='gray', linestyle=':')
    for a in (ax, bx):
        a.axvspan(disturbance_period[0], disturbance_period[1], color='red', alpha=0.1)
        a.axvline(x=control_start, linewidth=1.5, color='green', linestyle=':')
        a.xaxis.set_minor_locator(AutoMinorLocator(2))
        a.grid(True, which='major', linestyle='-', alpha=0.4)
        a.tick_params(which='major', length=6, width=1.0)
        a.tick_params(which='minor', length=3, width=0.8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=16, ncol=2, columnspacing=1.0,
              handlelength=1.8, frameon=False)
    for y, (ls, c, lw) in [(hat, styles[2]), (model, styles[3])]:
        d = np.abs(y - star)
        d[d == 0] = np.nan  # identical before the traces separate
        bx.semilogy(t, d, ls, color=c, lw=lw)
    bx.set_ylim(bottom=1e-9)
    bx.set_ylabel(dlabel, fontsize=18)
    bx.set_xlabel('t(s)', fontsize=18)
    fig.align_ylabels([ax, bx])
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    mpl.rcdefaults()


def figure_lqr_dp(output):
    stored = np.load(smib.ROOT / 'results' / 'lqr_dp_traces.npz')
    d_eq, traces = float(stored['d_eq']), stored['traces']
    labels = ['Uncontrolled (true)', r'True system, $K^*$', r'True system, $\hat{K}$', r'DP model, $\hat{K}$']
    for i, (s, yl, dl) in enumerate([('delta', r'$\delta(rad)$', r'$|\delta-\delta_{K^*}|$'),
                                     ('omega', r'$\omega(rad/s)$', r'$|\omega-\omega_{K^*}|$')]):
        plot_lqr_diff(smib.T, [x[i] for x in traces], labels, output / f'lqr-dp-{s}.png', yl, dl,
                      d_eq if i == 0 else 0.0, 2.0, [1.0, 2.0])


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = weight_groups()
    tables = {'table_noise_realizations.tex': table_noise_realizations(),
              'table_baseline_control.tex': table_baseline_control(),
              'table_structure.tex': table_structure(),
              'table_dp_starts.tex': table_dp_starts(),
              'table_low_gain.tex': table_low_gain(groups)}
    for name, text in tables.items():
        (output / name).write_text(text)
    figure_lqr_dp(output)
    figure_structure_protocol(output)
    figure_weight_response(groups, output)


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else smib.ROOT / 'out'
    generate(out)
    print('Wrote', out)
