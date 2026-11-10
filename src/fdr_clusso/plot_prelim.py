# Figures for PRELIM_RESULTS.md, drawn from grid_prelim/*.json and nothing else.
#
# Same plain-SVG approach and palette as plot_stage6.py (no plotting
# dependency). Three charts: FDP by cell, TPR by cell, and the FDP-vs-TPR trade
# with an arrow from CLUSSO to FDR-CLUSSO in each cell.
#
#   python plot_prelim.py            # writes figures/prelim_*.svg
#   python plot_prelim.py --table    # also prints the markdown tables

import argparse
import glob
import json
import os

import numpy as np

from plot_stage6 import ci
from plot_stage6 import svg as _svg

# plot_stage6's palette has no line class for the second series; add it here.
EXTRA = ('<style>.s2l{stroke:#eb6834} '
         '@media (prefers-color-scheme: dark){.s2l{stroke:#d95926}}</style>')


def svg(w, h, body, title):
    return _svg(w, h, EXTRA + '\n' + body, title)

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, 'figures')
Q = 0.10
METHODS = [('baseline', 'CLUSSO', 's2'), ('exact_k', 'FDR-CLUSSO', 's1')]


def load(pattern=os.path.join(_HERE, 'grid_prelim', '*.json')):
    cells = []
    for p in glob.glob(pattern):
        d = json.load(open(p))
        a = d['args']
        cells.append(dict(q=a['q_features'], rho=a['rho'], n=a['n'],
                          per=d['per_cohort'], meta=d['meta'], args=a))
    return sorted(cells, key=lambda c: (c['q'], c['rho']))


def label(c):
    return f"q = {c['q']}, ρ = {c['rho']:g}"


def dot_ci_chart(cells, metric, title, xmax, xlabel, ticks, target=None):
    """One row per cell; each method a dot with its 95% CI."""
    W, left, right, top, rh = 720, 130, 30, 64, 46
    H = top + rh * len(cells) + 56
    pw = W - left - right
    X = lambda v: left + pw * v / xmax
    b = [f'<text class="ink" x="20" y="28" font-size="15" font-weight="600">{title}</text>',
         f'<circle class="s2" cx="{W - 250}" cy="24" r="5"/>'
         f'<text class="ink2" x="{W - 240}" y="28" font-size="12">CLUSSO</text>',
         f'<circle class="s1" cx="{W - 170}" cy="24" r="5"/>'
         f'<text class="ink2" x="{W - 160}" y="28" font-size="12">FDR-CLUSSO</text>']
    y_end = top + rh * len(cells)
    for t in ticks:
        b.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 8}" y2="{y_end}"/>')
        b.append(f'<text class="mute" x="{X(t):.1f}" y="{y_end + 18}" font-size="11" '
                 f'text-anchor="middle">{t:g}</text>')
    b.append(f'<text class="ink2" x="{left + pw / 2:.1f}" y="{y_end + 40}" font-size="12" '
             f'text-anchor="middle">{xlabel}</text>')
    if target is not None:
        b.append(f'<line class="tgt" x1="{X(target):.1f}" x2="{X(target):.1f}" y1="{top - 14}" '
                 f'y2="{y_end}" stroke-dasharray="4 3" stroke-width="1.5"/>')
        b.append(f'<text class="ink2" x="{X(target) + 4:.1f}" y="{top - 16}" font-size="11">'
                 f'target {target:g}</text>')
    for i, c in enumerate(cells):
        yc = top + rh * i + rh / 2
        b.append(f'<text class="ink" x="{left - 12}" y="{yc + 4:.1f}" font-size="12" '
                 f'text-anchor="end">{label(c)}</text>')
        for j, (m, name, cls) in enumerate(METHODS):
            mu, lo, hi = ci([r[m][metric] for r in c['per']])
            lo, hi = max(lo, 0), min(hi, xmax)
            y = yc + (j - 0.5) * 12
            b.append(f'<line class="{cls}l" x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y:.1f}" '
                     f'y2="{y:.1f}" stroke-width="2" stroke-linecap="round"/>')
            b.append(f'<circle class="{cls} ring" cx="{X(mu):.1f}" cy="{y:.1f}" r="5" '
                     f'stroke-width="1.5"><title>{name}, {label(c)}: {mu:.3f} '
                     f'[{lo:.3f}, {hi:.3f}]</title></circle>')
    return svg(W, H, '\n'.join(b), title)


# Where a cell's label sits relative to its CLUSSO point (dx, dy, anchor), for
# the cells whose default spot (above right) hits an arrow or another label.
LABEL_AT = {(20, 0.0): (6, 18, 'start'), (20, 0.6): (6, 18, 'start'),
            (100, 0.6): (9, 4, 'start')}


def trade_chart(cells):
    """Mean FDP (x) vs mean TPR (y); arrow CLUSSO -> FDR-CLUSSO per cell."""
    W, H, left, right, top, bot = 720, 460, 70, 30, 52, 56
    pw, ph = W - left - right, H - top - bot
    xmax = 0.30
    X = lambda v: left + pw * v / xmax
    Y = lambda v: top + ph * (1 - v / 100)
    title = 'What FDR control costs: mean FDP vs mean TPR, per cell'
    b = [f'<text class="ink" x="20" y="28" font-size="15" font-weight="600">{title}</text>',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" '
         'markerHeight="9" markerUnits="userSpaceOnUse" orient="auto"><path class="gray" d="M0,0 L10,5 L0,10 z"/>'
         '</marker></defs>']
    for t in (0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        b.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + ph}"/>')
        b.append(f'<text class="mute" x="{X(t):.1f}" y="{top + ph + 18}" font-size="11" '
                 f'text-anchor="middle">{t:.2f}</text>')
    for t in (0, 25, 50, 75, 100):
        b.append(f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        b.append(f'<text class="mute" x="{left - 8}" y="{Y(t) + 4:.1f}" font-size="11" '
                 f'text-anchor="end">{t}</text>')
    b.append(f'<line class="tgt" x1="{X(Q):.1f}" x2="{X(Q):.1f}" y1="{top}" y2="{top + ph}" '
             f'stroke-dasharray="4 3" stroke-width="1.5"/>')
    b.append(f'<text class="ink2" x="{X(Q) + 4:.1f}" y="{top + ph - 6}" font-size="11">target 0.10</text>')
    b.append(f'<text class="ink2" x="{left + pw / 2:.1f}" y="{H - 14}" font-size="12" '
             f'text-anchor="middle">mean false discovery proportion (lower is better)</text>')
    b.append(f'<text class="ink2" x="18" y="{top + ph / 2:.1f}" font-size="12" text-anchor="middle" '
             f'transform="rotate(-90 18 {top + ph / 2:.1f})">mean TPR % (higher is better)</text>')
    for c in cells:
        p = {m: (np.mean([r[m]['fdp'] for r in c['per']]),
                 np.mean([r[m]['tpr'] for r in c['per']])) for m, _, _ in METHODS}
        (x0, y0), (x1, y1) = p['baseline'], p['exact_k']
        b.append(f'<line class="grayl" x1="{X(x0):.1f}" y1="{Y(y0):.1f}" x2="{X(x1):.1f}" '
                 f'y2="{Y(y1):.1f}" stroke-width="1.3" marker-end="url(#ah)" opacity="0.8"/>')
        for m, name, cls in METHODS:
            x, y = p[m]
            b.append(f'<circle class="{cls} ring" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5.5" '
                     f'stroke-width="1.5"><title>{name}, {label(c)}: FDP {x:.3f}, TPR {y:.1f}'
                     f'</title></circle>')
        # Label each cell at its CLUSSO point, where the points are spread out.
        dx, dy, anchor = LABEL_AT.get((c['q'], c['rho']), (6, -10, 'start'))
        b.append(f'<text class="ink2" x="{X(x0) + dx:.1f}" y="{Y(y0) + dy:.1f}" font-size="11" '
                 f'text-anchor="{anchor}">'
                 f'{label(c)}</text>')
    lx, ly = left + pw - 110, top + ph - 40
    b.append(f'<circle class="s2" cx="{lx}" cy="{ly}" r="5"/>'
             f'<text class="ink2" x="{lx + 10}" y="{ly + 4}" font-size="11">CLUSSO</text>')
    b.append(f'<circle class="s1" cx="{lx}" cy="{ly + 18}" r="5"/>'
             f'<text class="ink2" x="{lx + 10}" y="{ly + 22}" font-size="11">FDR-CLUSSO</text>')
    return svg(W, H, '\n'.join(b), title)


def tables(cells):
    from scipy import stats  # comes with scikit-learn
    out = ['| Cell | Method | mean FDP [95% CI] | FDP ≤ 0.10 | TPR % | FPR % | |S| | bias (median) '
           '| in-sample MSE (median) | min per run |',
           '|---|---|---|---|---|---|---|---|---|---|']
    names = [('baseline', 'CLUSSO'), ('exact_k', '**FDR-CLUSSO**'), ('naive', 'Naive'),
             ('oracle', 'Full Information')]
    for c in cells:
        for m, name in names:
            r = c['per']
            mu, lo, hi = ci([x[m]['fdp'] for x in r])
            f = np.array([x[m]['fdp'] for x in r])
            out.append(f"| {label(c)} | {name} | {mu:.3f} [{max(lo, 0):.3f}, {hi:.3f}] | "
                       f"{int((f <= Q).sum())}/{len(f)} | {np.mean([x[m]['tpr'] for x in r]):.1f} | "
                       f"{np.mean([x[m]['fpr'] for x in r]):.1f} | {np.mean([x[m]['n_sel'] for x in r]):.1f} | "
                       f"{np.median([x[m]['bias'] for x in r]):.3f} | {np.median([x[m]['mse'] for x in r]):.0f} | "
                       f"{np.mean([x[m]['sec'] for x in r]) / 60:.2f} |")
    out += ['', '| Cell | Δ FDP (FDR − CLUSSO) [95% CI] | p | Δ TPR [95% CI] | p | '
            'FDR-CLUSSO TPR lower / same / higher |', '|---|---|---|---|---|---|']
    for c in cells:
        r = c['per']
        row = [label(c)]
        for k in ('fdp', 'tpr'):
            d = np.array([x['exact_k'][k] - x['baseline'][k] for x in r])
            mu, lo, hi = ci(d)
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            fmt = '.3f' if k == 'fdp' else '.1f'
            row += [f'{mu:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]', f'{p:.1g}']
        d = np.array([x['exact_k']['tpr'] - x['baseline']['tpr'] for x in r])
        row.append(f'{(d < 0).sum()} / {(d == 0).sum()} / {(d > 0).sum()}')
        out.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args()
    cells = load()
    os.makedirs(FIG, exist_ok=True)
    figs = {
        'prelim_fdp.svg': dot_ci_chart(cells, 'fdp', 'False discovery proportion by cell (mean, 95% CI)',
                                       0.4, 'false discovery proportion', [0, 0.1, 0.2, 0.3, 0.4], target=Q),
        'prelim_tpr.svg': dot_ci_chart(cells, 'tpr', 'True positive rate by cell (mean, 95% CI)',
                                       100, 'TPR %', [0, 25, 50, 75, 100]),
        'prelim_trade.svg': trade_chart(cells),
    }
    for name, s in figs.items():
        open(os.path.join(FIG, name), 'w', encoding='utf-8').write(s)
        print('wrote', os.path.join('figures', name))
    if a.table:
        print(tables(cells))


if __name__ == '__main__':
    main()
