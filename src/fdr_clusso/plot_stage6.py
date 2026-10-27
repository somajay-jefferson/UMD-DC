# Figures for RESULTS.md, drawn from the run files and nothing else.
#
# CLAUDE.md: numbers on pages come from real runs, not typed. Every value here
# is read from stage6_exactk_fine_reps100.json (40-step ladder) and
# stage5_reps100.json (13-step ladder, the only run with the one-lambda arm).
#
# Plain SVG via the standard library -- the project carries no plotting
# dependency, and three charts don't justify adding one. Colors are the
# reference palette's first two categorical slots, with their own dark-mode
# steps, so the figures read on GitHub's light and dark themes alike.
#
# R^2 needs Var(Y), which the runs don't store. It is recomputed by regenerating
# each cohort from its seed (same draws, see stage1_noise.make_cohort_full) and
# cached in figures/stage6_var_y.json, so only the first call is slow.

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, 'figures')
Q = 0.10

STYLE = """<style>
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .ink2{fill:#52514e} .mute{fill:#898781}
  .grid{stroke:#e1e0d9} .axis{stroke:#c3c2b7} .s1{fill:#2a78d6} .s1l{stroke:#2a78d6}
  .s2{fill:#eb6834} .gray{fill:#898781} .grayl{stroke:#898781} .ring{stroke:#fcfcfb}
  .tgt{stroke:#52514e}
  text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-variant-numeric:tabular-nums}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .ink{fill:#ffffff} .ink2{fill:#c3c2b7}
    .grid{stroke:#2c2c2a} .axis{stroke:#383835} .s1{fill:#3987e5} .s1l{stroke:#3987e5}
    .s2{fill:#d95926} .ring{stroke:#1a1a19} .tgt{stroke:#c3c2b7}
  }
</style>"""


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{title}">\n'
            f'<title>{title}</title>\n{STYLE}\n'
            f'<rect class="bg" width="{w}" height="{h}" rx="8"/>\n{body}\n</svg>\n')


def rounded_bar(x, y, w, h, cls, r=4, tip=''):
    """Rounded top only -- the data end -- flat on the baseline."""
    r = min(r, w / 2, h)
    d = (f'M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
         f'H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z')
    t = f'<title>{tip}</title>' if tip else ''
    return f'<path class="{cls}" d="{d}">{t}</path>'


def ci(a):
    a = np.asarray(a, float)
    se = a.std(ddof=1) / np.sqrt(a.size)
    return a.mean(), a.mean() - 1.96 * se, a.mean() + 1.96 * se


def load():
    s6 = json.load(open(os.path.join(_HERE, 'stage6_exactk_fine_reps100.json')))
    s5 = json.load(open(os.path.join(_HERE, 'stage5_reps100.json')))
    return s6, s5


# --- figure 1: FDP by method --------------------------------------------------

def fig_fdp(s6, s5):
    p6, p5 = s6['per_cohort'], s5['per_cohort']
    rows = [
        ('Naive averaging', [c['naive']['fdp'] for c in p6], False),
        ('CLUSSO alone', [c['baseline']['fdp'] for c in p6], False),
        ('FDR-CLUSSO, 13-step ladder', [c['exact_k']['fdp'] for c in p5], False),
        ('FDR-CLUSSO one-λ, 13-step', [c['fixed_lambda']['fdp'] for c in p5], False),
        ("FDR-CLUSSO, paper's rule (40-step)", [c['exact_k']['fdp'] for c in p6], True),
        ('Oracle (true tubules)', [c['oracle']['fdp'] for c in p6], False),
    ]
    W, left, right, top, rh = 720, 250, 70, 56, 38
    H = top + rh * len(rows) + 52
    xmax = 0.7
    X = lambda v: left + (W - left - right) * v / xmax
    b = [f'<text class="ink" x="20" y="28" font-size="15" font-weight="600">'
         f'False discovery proportion by method, mean and 95% interval (100 datasets)</text>']
    for t in np.arange(0, xmax + 1e-9, 0.1):
        b.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 8}" '
                 f'y2="{top + rh * len(rows) - 8}" stroke-width="1"/>')
        b.append(f'<text class="mute" x="{X(t):.1f}" y="{top + rh * len(rows) + 10}" '
                 f'font-size="12" text-anchor="middle">{t:.1f}</text>')
    b.append(f'<line class="tgt" x1="{X(Q):.1f}" x2="{X(Q):.1f}" y1="{top - 14}" '
             f'y2="{top + rh * len(rows) - 4}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    b.append(f'<text class="ink2" x="{X(Q) + 5:.1f}" y="{top - 16}" font-size="12">'
             f'target 0.10</text>')
    for i, (name, vals, hi) in enumerate(rows):
        y = top + rh * i + 10
        m, lo, up = ci(vals)
        dot, line = ('s1', 's1l') if hi else ('gray', 'grayl')
        weight = '600' if hi else '400'
        b.append(f'<text class="{"ink" if hi else "ink2"}" x="{left - 14}" y="{y + 4}" '
                 f'font-size="13" font-weight="{weight}" text-anchor="end">{name}</text>')
        b.append(f'<line class="{line}" x1="{X(lo):.1f}" x2="{X(up):.1f}" y1="{y}" y2="{y}" '
                 f'stroke-width="2" stroke-linecap="round"/>')
        b.append(f'<circle class="{dot} ring" cx="{X(m):.1f}" cy="{y}" r="5" stroke-width="2">'
                 f'<title>{name}: mean {m:.3f}, 95% interval [{lo:.3f}, {up:.3f}]</title></circle>')
        b.append(f'<text class="{"ink" if hi else "ink2"}" x="{X(up) + 9:.1f}" y="{y + 4}" '
                 f'font-size="12" font-weight="{weight}">{m:.3f}</text>')
    b.append(f'<text class="mute" x="{left}" y="{H - 12}" font-size="12">'
             f'FDP = share of a method’s picks that are not real. Read with power: '
             f'picking less always lowers it.</text>')
    return svg(W, H, '\n'.join(b), 'False discovery proportion by method')


# --- figure 2: per-dataset FDP, CLUSSO alone vs FDR-CLUSSO -------------------

def fig_spread(s6):
    p6 = s6['per_cohort']
    panels = [('CLUSSO alone', [c['baseline']['fdp'] for c in p6], 's2'),
              ("FDR-CLUSSO, paper's rule", [c['exact_k']['fdp'] for c in p6], 's1')]
    width = 0.05
    edges = np.arange(0, 0.85 + 1e-9, width)
    W, left, right, ph, gap, top = 720, 56, 24, 130, 46, 60
    H = top + 2 * ph + gap + 56
    X = lambda v: left + (W - left - right) * v / edges[-1]
    # Bins include their RIGHT edge, so FDP = 0.10 exactly (1 false pick in 10,
    # common here) counts as on target, left of the dashed line. FDP = 0 goes in
    # the first bin.
    def right_closed(v):
        idx = np.searchsorted(edges, np.asarray(v, float) - 1e-12, side='left') - 1
        return np.bincount(np.clip(idx, 0, len(edges) - 2), minlength=len(edges) - 1)
    counts = [right_closed(v) for _, v, _ in panels]
    cmax = int(np.ceil(max(c.max() for c in counts) / 10.0) * 10)
    b = [f'<text class="ink" x="20" y="28" font-size="15" font-weight="600">'
         f'FDP in each of the 100 datasets</text>',
         f'<text class="ink2" x="20" y="46" font-size="12">FDR is the average of these. '
         f'Any single dataset can land well above or below it.</text>']
    for k, ((name, vals, cls), cnt) in enumerate(zip(panels, counts)):
        y0 = top + k * (ph + gap)
        base = y0 + ph
        Y = lambda c: base - (ph - 18) * c / cmax
        for t in range(0, cmax + 1, 10):
            b.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{Y(t):.1f}" '
                     f'y2="{Y(t):.1f}" stroke-width="1"/>')
            b.append(f'<text class="mute" x="{left - 8}" y="{Y(t) + 4:.1f}" font-size="11" '
                     f'text-anchor="end">{t}</text>')
        for j, c in enumerate(cnt):
            if c == 0:
                continue
            x = X(edges[j]) + 1
            w = X(edges[j + 1]) - X(edges[j]) - 2
            b.append(rounded_bar(x, Y(c), w, base - Y(c), cls,
                                 tip=f'{name}: {c} datasets with FDP in '
                                     f'{"[" if j == 0 else "("}{edges[j]:.2f}, '
                                     f'{edges[j + 1]:.2f}]'))
        b.append(f'<line class="axis" x1="{left}" x2="{W - right}" y1="{base}" y2="{base}" '
                 f'stroke-width="1"/>')
        m = float(np.mean(vals))
        b.append(f'<line class="tgt" x1="{X(Q):.1f}" x2="{X(Q):.1f}" y1="{y0 + 4}" '
                 f'y2="{base}" stroke-width="1.5" stroke-dasharray="4 3"/>')
        b.append(f'<text class="ink" x="{W - right}" y="{y0 + 10}" font-size="13" '
                 f'font-weight="600" text-anchor="end">{name}</text>')
        b.append(f'<text class="ink2" x="{W - right}" y="{y0 + 26}" font-size="12" '
                 f'text-anchor="end">mean {m:.3f} · at or under 0.10 in '
                 f'{int(np.sum(np.asarray(vals) <= Q + 1e-12))} · FDP 0 in {int(np.sum(np.asarray(vals) == 0))} '
                 f'· above 0.20 in {int(np.sum(np.asarray(vals) > 0.2))}</text>')
    for t in np.arange(0, edges[-1] + 1e-9, 0.1):
        b.append(f'<text class="mute" x="{X(t):.1f}" y="{top + 2 * ph + gap + 18}" '
                 f'font-size="12" text-anchor="middle">{t:.1f}</text>')
    b.append(f'<text class="mute" x="{X(Q) + 5:.1f}" y="{top + 2 * ph + gap + 36}" '
             f'font-size="12">dashed line: target 0.10 · bars: number of datasets, '
             f'bins of width {width}</text>')
    return svg(W, H, '\n'.join(b), 'FDP per dataset, CLUSSO alone versus FDR-CLUSSO')


# --- figure 3: R^2 per dataset ------------------------------------------------

def var_y(n_cohorts, seed0, n, q, sparsity):
    path = os.path.join(FIG, 'stage6_var_y.json')
    if os.path.exists(path):
        saved = json.load(open(path))
        if saved.get('key') == [n_cohorts, seed0, n, q, sparsity]:
            return np.array(saved['var_y'])
    from stage1_noise import make_cohort_full
    v = [float(make_cohort_full(n, q, sparsity, seed0 + i)['Y'].var())
         for i in range(n_cohorts)]
    json.dump({'key': [n_cohorts, seed0, n, q, sparsity], 'var_y': v}, open(path, 'w'))
    return np.array(v)


def fig_r2(s6):
    p6, a = s6['per_cohort'], s6['args']
    vy = var_y(len(p6), a['seed'], a['n'], a['q_features'], a['sparsity'])
    r_base = 1 - np.array([c['baseline']['mse'] for c in p6]) / vy
    r_fdr = 1 - np.array([c['exact_k']['mse'] for c in p6]) / vy
    lo = np.floor(min(r_base.min(), r_fdr.min()) * 50) / 50
    W = H = 460
    pad_l, pad_b, pad_t, pad_r = 64, 56, 64, 24
    S = lambda v: (v - lo) / (1 - lo)
    X = lambda v: pad_l + (W - pad_l - pad_r) * S(v)
    Y = lambda v: H - pad_b - (H - pad_b - pad_t) * S(v)
    d = r_fdr - r_base
    b = [f'<text class="ink" x="20" y="28" font-size="15" font-weight="600">'
         f'Prediction is unchanged: R² per dataset</text>',
         f'<text class="ink2" x="20" y="46" font-size="12">Each dot is one dataset. '
         f'On the diagonal = identical fit. Largest gap {np.abs(d).max():.3f}.</text>']
    ticks = np.arange(lo, 1.0 + 1e-9, 0.05 if 1 - lo > 0.2 else 0.02)
    for t in ticks:
        b.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pad_t}" '
                 f'y2="{H - pad_b}" stroke-width="1"/>')
        b.append(f'<line class="grid" x1="{pad_l}" x2="{W - pad_r}" y1="{Y(t):.1f}" '
                 f'y2="{Y(t):.1f}" stroke-width="1"/>')
        b.append(f'<text class="mute" x="{X(t):.1f}" y="{H - pad_b + 16}" font-size="11" '
                 f'text-anchor="middle">{t:.2f}</text>')
        b.append(f'<text class="mute" x="{pad_l - 8}" y="{Y(t) + 4:.1f}" font-size="11" '
                 f'text-anchor="end">{t:.2f}</text>')
    b.append(f'<line class="tgt" x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(1):.1f}" '
             f'y2="{Y(1):.1f}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    for i, (u, v) in enumerate(zip(r_base, r_fdr)):
        b.append(f'<circle class="s1 ring" cx="{X(u):.1f}" cy="{Y(v):.1f}" r="4" '
                 f'stroke-width="2"><title>dataset {i}: CLUSSO alone {u:.3f}, '
                 f'FDR-CLUSSO {v:.3f}</title></circle>')
    b.append(f'<text class="ink2" x="{(pad_l + W - pad_r) / 2}" y="{H - 14}" font-size="12" '
             f'text-anchor="middle">R², CLUSSO alone</text>')
    b.append(f'<text class="ink2" x="18" y="{(pad_t + H - pad_b) / 2}" font-size="12" '
             f'text-anchor="middle" transform="rotate(-90 18 {(pad_t + H - pad_b) / 2})">'
             f'R², FDR-CLUSSO</text>')
    return svg(W, H, '\n'.join(b), 'R squared per dataset, CLUSSO alone versus FDR-CLUSSO'), \
        {'r2_base_mean': float(r_base.mean()), 'r2_fdr_mean': float(r_fdr.mean()),
         'r2_max_abs_diff': float(np.abs(d).max()), 'r2_min': float(min(r_base.min(), r_fdr.min()))}


def main():
    os.makedirs(FIG, exist_ok=True)
    s6, s5 = load()
    out = {}
    for name, text in (('stage6_fdp_by_method.svg', fig_fdp(s6, s5)),
                       ('stage6_fdp_per_dataset.svg', fig_spread(s6))):
        open(os.path.join(FIG, name), 'w', encoding='utf-8').write(text)
    text, out['r2'] = fig_r2(s6)
    open(os.path.join(FIG, 'stage6_r2.svg'), 'w', encoding='utf-8').write(text)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
