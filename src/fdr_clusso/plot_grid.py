# Small-multiple figures in the layout of the Random CLUSSO paper's Figs 1-5
# (Rubin et al. 2026, pages 6-9): one figure per metric, a panel per (q, rho),
# x = n, one line per method, medians over replications.
#
# Reads any set of stage4_clusso_fdr.py result JSONs -- one per grid cell --
# and draws whatever cells exist. An incomplete grid is fine: missing cells are
# empty panels or gaps, and a lone n is a single point.
#
#     python plot_grid.py grid/*.json --out figures/grid
#     python plot_grid.py stage6_exactk_fine_reps100.json --stat mean
#
# rho: our data generator has no feature correlation yet, so every current run
# is rho = 0. A run that records `rho` in its args is placed by it. GRID_PLAN.md
# lists what has to change before the rho columns can fill.
#
# Plain SVG, no plotting dependency, same palette as plot_stage6.py.

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Method key in per_cohort -> (label, css class, dashed?, marker). Fixed order
# and fixed color per method, whichever methods a run happens to contain.
METHODS = [
    ('exact_k', "FDR-CLUSSO (paper's rule)", 's1', False, 'circle'),
    ('baseline', 'CLUSSO', 's2', False, 'square'),
    ('naive', 'Naive', 's3', True, 'diamond'),
    ('oracle', 'Full information (oracle)', 's4', True, 'triangle'),
    ('fixed_lambda', 'FDR-CLUSSO one-λ', 's5', False, 'circle-open'),
]

# Metric key -> (title, y label, fixed y range or None, scale factor).
# TPR/FPR as percents, like the paper. MSE prefers held-out when a run has it.
METRICS = {
    'tpr': ('True positive rate', 'TPR (%)', (0, 100), 1.0),
    'fpr': ('False positive rate', 'FPR (%)', (0, 100), 1.0),
    'fdp': ('False discovery proportion', 'FDP', (0, 1), 1.0),
    'bias': ('Bias of β̂ (L1, both normalized)', 'Bias', None, 1.0),
    'mse': ('Prediction MSE', 'MSE', None, 1.0),
    'sec': ('Time per method', 'minutes', None, 1 / 60.0),
}

STYLE = """<style>
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .ink2{fill:#52514e} .mute{fill:#898781}
  .grid{stroke:#e1e0d9} .axis{stroke:#c3c2b7} .frame{fill:none;stroke:#c3c2b7}
  .s1{fill:#2a78d6;stroke:#2a78d6} .s2{fill:#eb6834;stroke:#eb6834}
  .s3{fill:#1baf7a;stroke:#1baf7a} .s4{fill:#eda100;stroke:#eda100}
  .s5{fill:#e87ba4;stroke:#e87ba4} .ln{fill:none} .ring{stroke:#fcfcfb}
  .tgt{stroke:#52514e}
  text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-variant-numeric:tabular-nums}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .ink{fill:#ffffff} .ink2{fill:#c3c2b7}
    .grid{stroke:#2c2c2a} .axis{stroke:#383835} .frame{stroke:#383835}
    .s1{fill:#3987e5;stroke:#3987e5} .s2{fill:#d95926;stroke:#d95926}
    .s3{fill:#199e70;stroke:#199e70} .s4{fill:#c98500;stroke:#c98500}
    .s5{fill:#d55181;stroke:#d55181} .ring{stroke:#1a1a19} .tgt{stroke:#c3c2b7}
  }
</style>"""


def load_runs(paths):
    """
    One cell per file, keyed by (q, rho, n). Two files for the same cell and
    method: the one with more replications wins, and the clash is reported, so
    a 13-step and a 40-step run of one cell never silently average together.
    """
    cells = defaultdict(dict)       # (q, rho, n) -> method -> list of dicts
    source = {}
    # A figure is one data design. core's generator and the Random CLUSSO
    # design both have rho = 0 cells, and mixing them would silently plot two
    # different simulations as one line.
    designs = {json.load(open(p))['args'].get('design', 'clusso') for p in paths}
    if len(designs) > 1:
        sys.exit(f'refusing to mix data designs in one figure: {sorted(designs)}. '
                 'Plot each design separately.')
    for p in paths:
        d = json.load(open(p))
        a = d['args']
        key = (int(a['q_features']), float(a.get('rho', 0.0)), int(a['n']))
        pc = d['per_cohort']
        for m, *_ in METHODS:
            if m not in pc[0]:
                continue
            prev = source.get((key, m))
            if prev and len(cells[key][m]) >= len(pc):
                print(f'  skip {m} at q={key[0]} rho={key[1]} n={key[2]} from '
                      f'{os.path.basename(p)}: {os.path.basename(prev)} has '
                      f'as many reps', file=sys.stderr)
                continue
            if prev:
                print(f'  replace {m} at q={key[0]} rho={key[1]} n={key[2]}: '
                      f'{os.path.basename(p)} has more reps than '
                      f'{os.path.basename(prev)}', file=sys.stderr)
            cells[key][m] = [c[m] for c in pc]
            source[(key, m)] = p
    return cells


def value(records, metric, stat):
    if metric == 'mse':
        # Held-out when present (--holdout runs), in-sample otherwise.
        k = 'test_mse' if records and records[0].get('test_mse') is not None else 'mse'
    else:
        k = metric
    v = [r.get(k) for r in records]
    if not v or any(x is None for x in v):
        return None, k
    a = np.asarray(v, float) * METRICS[metric][3]
    return float(np.median(a) if stat == 'median' else np.mean(a)), k


def marker(shape, x, y, cls, tip):
    t = f'<title>{tip}</title>'
    if shape == 'square':
        return f'<rect class="{cls} ring" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" stroke-width="1.5">{t}</rect>'
    if shape == 'diamond':
        return (f'<path class="{cls} ring" stroke-width="1.5" d="M{x:.1f},{y - 5:.1f} L{x + 5:.1f},{y:.1f} '
                f'L{x:.1f},{y + 5:.1f} L{x - 5:.1f},{y:.1f} Z">{t}</path>')
    if shape == 'triangle':
        return (f'<path class="{cls} ring" stroke-width="1.5" d="M{x:.1f},{y - 5:.1f} L{x + 5:.1f},{y + 4:.1f} '
                f'L{x - 5:.1f},{y + 4:.1f} Z">{t}</path>')
    if shape == 'circle-open':
        return (f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="4" fill-opacity="0" '
                f'stroke-width="2">{t}</circle>')
    return f'<circle class="{cls} ring" cx="{x:.1f}" cy="{y:.1f}" r="4.5" stroke-width="1.5">{t}</circle>'


def nice_max(v):
    if v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


def figure(cells, metric, stat, n_axis):
    qs = sorted({k[0] for k in cells})
    rhos = sorted({k[1] for k in cells})
    title, ylab, yfix, _ = METRICS[metric]
    pw, ph, gx, gy = 230, 150, 34, 58
    left, top = 64, 92
    present = [m for m in METHODS if any(m[0] in cells[k] for k in cells)]
    W = max(left + len(rhos) * (pw + gx) + 10, 640)
    H = top + len(qs) * (ph + gy) + 30 + 20 * len(present)
    nmin, nmax = n_axis
    used_keys = set()

    b = [f'<text class="ink" x="20" y="28" font-size="15" font-weight="600">{title}</text>',
         f'<text class="ink2" x="20" y="46" font-size="12">{stat.capitalize()} over replications '
         f'per cell. x = n (subjects).</text>',
         f'<text class="ink2" x="20" y="62" font-size="12">Panels: rows q (features), '
         f'columns ρ (feature correlation).@@MSENOTE@@</text>']
    for ri, q in enumerate(qs):
        # One y-scale per ROW when the metric has no natural range: MSE and
        # time grow with q by orders of magnitude, as in the paper's Fig 4.
        if yfix:
            y0, y1 = yfix
        else:
            vals = [value(cells[k][m], metric, stat)[0] for k in cells if k[0] == q
                    for m in cells[k]]
            y0, y1 = 0.0, nice_max(max([v for v in vals if v is not None], default=1.0))
        for ci, rho in enumerate(rhos):
            px, py = left + ci * (pw + gx), top + ri * (ph + gy)
            X = lambda n: px + pw * (n - nmin) / max(nmax - nmin, 1)
            Y = lambda v: py + ph - ph * (v - y0) / max(y1 - y0, 1e-12)
            b.append(f'<text class="ink2" x="{px + pw / 2}" y="{py - 8}" font-size="12" '
                     f'text-anchor="middle">ρ = {rho:g}, q = {q}</text>')
            for t in np.linspace(y0, y1, 5):
                b.append(f'<line class="grid" x1="{px}" x2="{px + pw}" y1="{Y(t):.1f}" '
                         f'y2="{Y(t):.1f}" stroke-width="1"/>')
                if ci == 0:
                    lab = f'{t:g}' if abs(t) < 1e4 else f'{t:.0e}'
                    b.append(f'<text class="mute" x="{px - 6}" y="{Y(t) + 4:.1f}" '
                             f'font-size="10" text-anchor="end">{lab}</text>')
            for t in sorted({nmin, nmax} | {k[2] for k in cells}):
                b.append(f'<text class="mute" x="{X(t):.1f}" y="{py + ph + 14}" font-size="10" '
                         f'text-anchor="middle">{t}</text>')
            b.append(f'<rect class="frame" x="{px}" y="{py}" width="{pw}" height="{ph}" stroke-width="1"/>')
            if metric == 'fdp':
                b.append(f'<line class="tgt" x1="{px}" x2="{px + pw}" y1="{Y(0.10):.1f}" '
                         f'y2="{Y(0.10):.1f}" stroke-width="1" stroke-dasharray="4 3"/>')
            pts_n = sorted(k[2] for k in cells if k[0] == q and k[1] == rho)
            if not pts_n:
                b.append(f'<text class="mute" x="{px + pw / 2}" y="{py + ph / 2}" font-size="12" '
                         f'text-anchor="middle">not run</text>')
                continue
            for m, lab, cls, dashed, shape in present:
                pts = []
                for n in pts_n:
                    recs = cells[(q, rho, n)].get(m)
                    if recs is None:
                        continue
                    v, key = value(recs, metric, stat)
                    if v is None:
                        continue
                    used_keys.add(key)
                    pts.append((n, v, len(recs)))
                if len(pts) > 1:
                    d = ' '.join(f'{"M" if i == 0 else "L"}{X(n):.1f},{Y(v):.1f}'
                                 for i, (n, v, _) in enumerate(pts))
                    dash = ' stroke-dasharray="5 4"' if dashed else ''
                    b.append(f'<path class="{cls} ln" d="{d}" stroke-width="2"{dash}/>')
                for n, v, r in pts:
                    b.append(marker(shape, X(n), Y(v), cls,
                                    f'{lab}, q={q}, ρ={rho:g}, n={n}: {v:.4g} ({r} reps)'))
        b.append(f'<text class="ink2" x="16" y="{top + ri * (ph + gy) + ph / 2}" font-size="12" '
                 f'text-anchor="middle" transform="rotate(-90 16 {top + ri * (ph + gy) + ph / 2})">'
                 f'{ylab}</text>')
    ly = top + len(qs) * (ph + gy) + 4
    for i, (m, lab, cls, dashed, shape) in enumerate(present):
        y = ly + 20 * i
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        b.append(f'<line class="{cls}" x1="{left}" x2="{left + 30}" y1="{y}" y2="{y}" stroke-width="2"{dash}/>')
        b.append(marker(shape, left + 15, y, cls, lab))
        b.append(f'<text class="ink2" x="{left + 40}" y="{y + 4}" font-size="12">{lab}</text>')
    note = ''
    if metric == 'mse':
        note = (' Held-out test MSE.' if used_keys == {'test_mse'} else
                ' In-sample MSE, as in the paper.' if used_keys == {'mse'} else
                ' MIXED held-out and in-sample MSE across cells.')
    body = '\n'.join(b).replace('@@MSENOTE@@', note)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
            f'height="{H}" role="img" aria-label="{title}">\n<title>{title}</title>\n{STYLE}\n'
            f'<rect class="bg" width="{W}" height="{H}" rx="8"/>\n{body}\n</svg>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='result JSONs (globs ok), one per grid cell')
    ap.add_argument('--out', default=os.path.join(_HERE, 'figures', 'grid'))
    ap.add_argument('--stat', choices=('median', 'mean'), default='median',
                    help='median is what the Random CLUSSO paper reports')
    ap.add_argument('--n-axis', type=int, nargs=2, default=(200, 1000),
                    help="x range; the paper's is 200-1000")
    ap.add_argument('--metrics', nargs='+', default=list(METRICS))
    args = ap.parse_args()

    paths = sorted({p for g in args.runs for p in glob.glob(g)})
    if not paths:
        sys.exit('no run files matched')
    cells = load_runs(paths)
    os.makedirs(args.out, exist_ok=True)
    print(f'{len(paths)} files -> {len(cells)} cells: '
          + ', '.join(f'q={k[0]} rho={k[1]:g} n={k[2]}' for k in sorted(cells)))
    for metric in args.metrics:
        path = os.path.join(args.out, f'grid_{metric}.svg')
        open(path, 'w', encoding='utf-8').write(figure(cells, metric, args.stat, args.n_axis))
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
