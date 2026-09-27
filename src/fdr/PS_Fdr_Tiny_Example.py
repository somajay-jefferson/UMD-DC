# The smallest PS-Fdr run that still works, sized so every number on the
# documentation page can be checked by hand.
#
# 50 subjects, 8 predictors, 2 of them real, B = 20, M = 20.  At that size Pi
# lands on twentieths, D(u) is a two-step calculation, and the whole permuted
# null fits in one 20 x 8 table.
#
#   cd src/fdr && python PS_Fdr_Tiny_Example.py
#   cd src/fdr && python PS_Fdr_Tiny_Example.py --html frag.html
#
# --html writes the <section> that docs/ps-fdr.html embeds, so every table on
# that page comes out of this run instead of being transcribed by hand.

import argparse
import sys

import numpy as np

import ps_fdr as P
from ps_fdr import lasso_support_cv

N, PP, B, M, Q = 50, 8, 20, 20, 0.2
BETA = np.array([2.0, 1.2] + [0.0] * 6)      # only predictors 1 and 2 are real
SEED = 7

SUB = str.maketrans('12345678', '₁₂₃₄₅₆₇₈')


def fj(j):
    """Predictor j as it is labelled on the page: f with a subscript."""
    return 'f' + str(j).translate(SUB)


def make_data(seed=SEED, n=N, beta=BETA):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(beta)))
    return X, X @ beta + rng.normal(0.0, 1.0, n)


def run(seed=SEED):
    """One tiny run, keeping the masks of every stability-selection pass.

    The spy wrapper asks for masks that ps_fdr does not return for the null
    arm.  It cannot disturb the result: same calls, same order, no extra
    randomness drawn.
    """
    orig = P.stability_selection
    log = []

    def spy(*a, **kw):
        want = kw.pop('return_masks', False)
        Pi, counts, masks = orig(*a, return_masks=True, **kw)
        log.append(masks)
        return (Pi, counts, masks) if want else (Pi, counts)

    P.stability_selection = spy
    try:
        X, y = make_data(seed)
        r = P.ps_fdr(X, y, q=Q, B=B, M=M, seed=seed, n_folds=5)
    finally:
        P.stability_selection = orig

    r['X'], r['y'] = X, y
    r['masks_real'] = log[0]
    r['masks_null'] = log[1:]
    r['cv'] = np.flatnonzero(lasso_support_cv(X, y, n_folds=5,
                                              random_state=seed))
    return r


def sweep_rows(r):
    """One row per distinct cutoff, with the raw permuted-cell count."""
    rows, seen = [], set()
    for s in r['sweep']:
        if s['cutoff'] in seen:
            continue
        seen.add(s['cutoff'])
        rows.append({'delta': s['delta'], 'cutoff': s['cutoff'],
                     'n_pos': s['n_pos'],
                     'cells': int((r['Z_null'] >= s['cutoff']).sum()),
                     'e0': s['e0'], 'fdr': s['fdr'],
                     'selected': sorted(int(i) + 1 for i in s['selected'])})
    return rows


def walk(r, delta):
    """The step-down scan of Algorithm step (1) at one Delta."""
    out, first = [], None
    for i in range(PP):
        bar = r['Z_bar'][i] + delta
        ok = bool(r['Z'][i] >= bar)
        if ok and first is None:
            first = i + 1
        out.append({'j': i + 1, 'z': r['Z'][i], 'bar': bar, 'ok': ok,
                    'star': ok and first == i + 1})
    return out


def resolution_probe(seeds=(7, 8, 9)):
    """Where the toy stops working: B sets the resolution at the top of Pi."""
    out = []
    for p_ in (6, 8, 12):
        for b_ in (10, 20, 50):
            sels = []
            for s in seeds:
                beta = np.zeros(p_)
                beta[:2] = [2.0, 1.2]
                rng = np.random.default_rng(s)
                X = rng.standard_normal((N, p_))
                y = X @ beta + rng.normal(0, 1, N)
                res = P.ps_fdr(X, y, q=Q, B=b_, M=M, seed=s, n_folds=5)
                sels.append(sorted(int(i) + 1 for i in res['selected']))
            out.append({'p': p_, 'B': b_, 'sels': sels})
    return out


# ----------------------------------------------------------------- printing

def report(r):
    Pi, order = r['Pi'], r['order']
    print(f"n={N} p={PP} B={B} M={M} q={Q} beta*={BETA.tolist()} nu={1/B}")
    print(f"\nCV-tuned lasso alone selects "
          f"{[int(j) + 1 for j in r['cv']]}   (truth [1, 2])")

    print(f"\n--- step 1: {B} bootstrap fits ---")
    print("boot  " + "  ".join(f"f{j+1}" for j in range(PP)))
    for b in range(B):
        print(f"  {b+1:>2}   "
              + "   ".join('1' if x else '.' for x in r['masks_real'][b])
              + f"   |S|={int(r['masks_real'][b].sum())}")
    print("hits  " + "   ".join(f"{int(h)}"
                                for h in r['masks_real'].sum(axis=0)))
    print("Pi_j  " + " ".join(f"{v:.2f}" for v in Pi))
    print(f"median k = {r['k']}")

    m0 = r['masks_null'][0].sum(axis=0) / B
    print("\n--- step 2: permutation 1, expanded ---")
    print("Pi~_j  " + " ".join(f"{v:.2f}" for v in m0))
    print("sorted " + " ".join(f"{v:.2f}" for v in np.sort(m0)))

    print(f"\n--- step 2: all {M} permutations, by rank ---")
    for m in range(M):
        print(f"  {m+1:>2}   " + "  ".join(f"{v:.2f}" for v in r['Pi_null'][m]))
    print("Pibar  " + "  ".join(f"{v:.2f}" for v in r['Pi_bar']))

    print("\n--- step 3: both arms by rank ---")
    print(f"{'rank':>5}{'feat':>6}{'real':>6}{'Pi':>7}{'Pibar':>8}"
          f"{'Z':>8}{'Zbar':>8}{'gap':>9}")
    for i in range(PP):
        j = order[i]
        print(f"{i+1:>5}{j+1:>6}{'yes' if BETA[j] else '.':>6}"
              f"{Pi[j]:>7.2f}{r['Pi_bar'][i]:>8.2f}{r['Z'][i]:>8.2f}"
              f"{r['Z_bar'][i]:>8.2f}{r['Z'][i] - r['Z_bar'][i]:>+9.2f}")

    print("\n--- step 4: sweep ---")
    for s in sweep_rows(r):
        print(f"  Delta={s['delta']:>6.2f}  Z={s['cutoff']:>6.2f}"
              f"  N+={s['n_pos']:>2}  cells={s['cells']:>4}"
              f"  e0={s['e0']:>5.2f}  Fdr={s['fdr']:.3f}  {s['selected']}")
    print(f"\nat q={Q}: {sorted(int(i) + 1 for i in r['selected'])}"
          f"   Fdr-hat={r['fdr_hat']:.3f}")


# --------------------------------------------------------------------- html

def html(r, probe):
    """The <section> that docs/ps-fdr.html embeds."""
    Pi, order, nu = r['Pi'], r['order'], 1.0 / B
    real = BETA != 0
    rows = sweep_rows(r)
    best = rows[-1]
    mid = [s for s in rows if s['n_pos'] == 3][0]
    cv = [int(j) + 1 for j in r['cv']]
    cv_false = [j for j in cv if not real[j - 1]]
    m0 = r['masks_null'][0].sum(axis=0) / B
    o = []
    w = o.append

    w('<!-- ============ TINY EXAMPLE — generated, do not hand-edit ============ -->')
    w('<!-- cd src/fdr && python PS_Fdr_Tiny_Example.py --html frag.html        -->')
    w('  <section>')
    w('    <h2>Fifty subjects, eight predictors, nothing left out</h2>')
    w('    <h3>Small enough to check by hand</h3>')
    w('')
    w('    <div class="prose">')
    w('      <p>')
    w('        The example further down runs at thirty predictors, which is big enough to '
      'behave like a real problem and far too big to verify with a pencil. This one is the '
      'opposite. Fifty subjects, eight predictors, two of them real, and the loop counts '
      'turned down to <math><mrow><mi>B</mi><mo>=</mo><mn>20</mn></mrow></math> and '
      '<math><mrow><mi>M</mi><mo>=</mo><mn>20</mn></mrow></math>. Nothing is summarised: '
      'every table is printed in full, and every number in it can be recomputed from the '
      'table above it.')
    w('      </p>')
    w('      <p>')
    w('        Each stage carries the sentence from He et al. that it implements.')
    w('      </p>')
    w('    </div>')
    w('')
    w('    <div class="scroll-x">')
    w('      <table class="alpha">')
    w('        <thead><tr><th>the setup</th><th>value</th><th>in the paper</th></tr></thead>')
    w('        <tbody>')
    for lab, val, sym in [
            ('subjects', str(N), '<math><mi>n</mi></math>'),
            ('predictors', str(PP), '<math><mi>p</mi></math>'),
            ('the real two',
             f'{fj(1)} (&#x3B2;&thinsp;=&thinsp;2.0), {fj(2)} (&#x3B2;&thinsp;=&thinsp;1.2)',
             '&ldquo;informative variables with non&#8209;zero coefficients&rdquo;'),
            ('the other six', '&#x3B2;&thinsp;=&thinsp;0',
             '&ldquo;null&rdquo; &mdash; &ldquo;no association with the outcome variable&rdquo;'),
            ('bootstraps', str(B), '<math><mi>B</mi></math>'),
            ('permutations', str(M), '<math><mi>M</mi></math>'),
            ('target level', str(Q), '<math><mi>q</mi></math>'),
            ('&#x3BD; = 1/B', f'{nu}',
             '&ldquo;a small positive number&rdquo;')]:
        w(f'          <tr><td>{lab}</td><td>{val}</td><td class="dim">{sym}</td></tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')
    w('')
    w('    <div class="note">')
    w('      <span><strong>What one cross&#8209;validated lasso does here.</strong> Fitted '
      'once on all fifty subjects it selects <b>'
      + ', '.join(fj(j) for j in cv) + '</b>. Two are real and <b class="fp">'
      + ', '.join(fj(j) for j in cv_false) + '</b> is noise. That is &sect;2.2 in '
      'miniature, and it is the false positive everything below has to remove.</span>')
    w('    </div>')

    # ---- step 1 -------------------------------------------------------
    w('')
    w('    <div class="exhead"><span class="exnum">1</span> Bootstrap, fit, write down who survived</div>')
    w('')
    w('    <div class="quote">')
    w('      &ldquo;we bootstrap (sample with replacement to form a new sample that is also '
      'of size <math><mi>n</mi></math>) multiple (<math><mi>B</mi></math>) times. For each '
      'resampled data&hellip; we implement the Lasso and denote the selected index set by '
      '<math><msup><mi>S&#770;</mi><mrow><mo>(</mo><mi>b</mi><mo>)</mo></mrow></msup></math>.&rdquo;')
    w('    </div>')
    w('')
    w('    <div class="scroll-x">')
    w('      <table class="design grid">')
    w('        <thead><tr><th>fit</th>'
      + ''.join(f'<th>{fj(j + 1)}</th>' for j in range(PP))
      + '<th>|S&#770;|</th></tr></thead>')
    w('        <tbody>')
    for b in range(B):
        cells = ''.join(
            f'<td class="{"on" if x else "off"}">{"1" if x else "&#183;"}</td>'
            for x in r['masks_real'][b])
        w(f'          <tr><td class="dim">{b + 1}</td>{cells}'
          f'<td class="dim">{int(r["masks_real"][b].sum())}</td></tr>')
    w('          <tr class="tot"><td>hits</td>'
      + ''.join(f'<td>{int(h)}</td>' for h in r['masks_real'].sum(axis=0))
      + '<td></td></tr>')
    w('          <tr class="tot"><td>&#928;<sub>j</sub></td>'
      + ''.join(f'<td class="{"v1" if real[j] else "dim"}">{Pi[j]:.2f}</td>'
                for j in range(PP))
      + '<td></td></tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')
    w('')
    w('    <div class="note">')
    w('      <span><strong>The bottom row is the only thing carried forward.</strong> '
      '&ldquo;The selection frequency is then computed as the empirical probability that '
      'each variable is selected.&rdquo; Coefficients, the fitted '
      '<math><mi>&#x3BB;</mi></math> and the prediction error are all thrown away. Support '
      f'sizes ran {int(r["counts"].min())} to {int(r["counts"].max())}, median '
      f'<b>{r["k"]}</b> &mdash; and that median is the number step 2 freezes.</span>')
    w('    </div>')

    # ---- step 2 -------------------------------------------------------
    w('')
    w('    <div class="exhead"><span class="exnum">2</span> Permute the outcome, do all of it again</div>')
    w('')
    w('    <div class="quote">')
    w('      &ldquo;we randomly permute the outcomes <math><mi>M</mi></math> times&hellip; '
      'On each permuted dataset&hellip; we implement the stability selection '
      'procedure&rdquo; &mdash; and &ldquo;To avoid under&#8209;estimation of '
      '<math><msub><mi>e</mi><mn>0</mn></msub></math>, instead of implementing '
      'cross&#8209;validation to select the number of variables on each permuted sample, we '
      'fix the number of selected variables.&rdquo;')
    w('    </div>')
    w('')
    w('    <div class="prose"><p>')
    w('      Every row below is a whole stability&#8209;selection pass, not one fit. Here is '
      f'permutation 1 unpacked &mdash; another {B}&#8209;fit grid exactly like the one above, '
      'collapsed to one line:')
    w('    </p></div>')
    w('')
    w('    <div class="scroll-x">')
    w('      <table class="design">')
    w('        <thead><tr><th>permutation 1</th>'
      + ''.join(f'<th>{fj(j + 1)}</th>' for j in range(PP)) + '</tr></thead>')
    w('        <tbody>')
    w(f'          <tr><td class="dim">hits over its {B} fits</td>'
      + ''.join(f'<td>{int(h)}</td>'
                for h in r['masks_null'][0].sum(axis=0)) + '</tr>')
    w('          <tr><td class="dim">&#928;&#771;<sub>j</sub> &mdash; by predictor</td>'
      + ''.join(f'<td>{v:.2f}</td>' for v in m0) + '</tr>')
    w('          <tr class="tot"><td>sorted &mdash; by <em>rank</em></td>'
      + ''.join(f'<td>{v:.2f}</td>' for v in np.sort(m0)) + '</tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')
    w('')
    w('    <div class="prose"><p>')
    w('      That sorted line is row 1 below. All twenty rows, each one '
      f'{B} more fits &mdash; {M * B} in this step alone:')
    w('    </p></div>')
    w('')
    w('    <div class="scroll-x">')
    w('      <table class="design">')
    w('        <thead><tr><th>perm</th>'
      + ''.join(f'<th>({j + 1})</th>' for j in range(PP)) + '</tr></thead>')
    w('        <tbody>')
    for m in range(M):
        cells = ''.join(
            f'<td class="{"hit" if r["Pi_null"][m, j] == 1.0 else ""}">'
            f'{r["Pi_null"][m, j]:.2f}</td>' for j in range(PP))
        w(f'          <tr><td class="dim">{m + 1}</td>{cells}</tr>')
    w('          <tr class="tot"><td>&#928;&#772;<sub>(j)</sub></td>'
      + ''.join(f'<td>{v:.2f}</td>' for v in r['Pi_bar']) + '</tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')
    w('')
    w('    <div class="note">')
    w('      <span><strong>Those column headings are ranks, not predictors.</strong> '
      '&ldquo;We order the selection frequencies&hellip; and then define '
      '<math><msub><mover accent="true"><mi>&#928;</mi><mo>&#772;</mo></mover>'
      '<mrow><mo>(</mo><mi>j</mi><mo>)</mo></mrow></msub></math>, the permuted counterpart '
      'of <math><msub><mi>&#928;</mi><mrow><mo>(</mo><mi>j</mi><mo>)</mo></mrow></msub></math>.&rdquo; '
      f'In permutation 1 the {np.sort(m0)[-1]:.2f} belongs to '
      f'{fj(int(np.argmax(m0)) + 1)}; in the next permutation the top slot belongs to '
      'someone else. Once the outcome is shuffled no predictor means anything, so identity '
      'is discarded and only the shape of the ordered frequencies survives.</span>')
    w('    </div>')
    w('')
    w('    <div class="note" style="margin-top:12px;">')
    w(f'      <span><strong>Every permuted fit selected exactly {r["k"]} variables.</strong> '
      'That is the frozen count, and it is what stops the null arm from looking emptier '
      'than the real one. Even so, read the right&#8209;hand column: with nothing real in '
      'the data at all, the most stable predictor still averages '
      f'<b>{r["Pi_bar"][-1]:.2f}</b>. That is the bar a real predictor has to clear.</span>')
    w('    </div>')

    # ---- step 3 -------------------------------------------------------
    w('')
    w('    <div class="exhead"><span class="exnum">3</span> Put both arms on one scale</div>')
    w('')
    w('    <div class="quote">')
    w('      &ldquo;These normalized statistics share the virtue of the original SAM '
      'statistics and further tease apart variables by providing larger values for most of '
      'the informative variables and smaller values for most of the non&#8209;informative '
      'ones.&rdquo;')
    w('    </div>')
    w('')
    us = sorted(set(np.round(Pi, 2)) | set(np.round(r['Pi_bar'], 2)))
    lo = min(u / ((u * (1 - u)) ** 0.5 + nu) for u in us if u < 1)
    hi_ = max(u / ((u * (1 - u)) ** 0.5 + nu) for u in us if u < 1)
    w('    <div class="scroll-x">')
    w('      <table class="alpha">')
    w('        <thead><tr><th class="gk">u</th>'
      '<th class="gk">&#8730;(u(1&#8722;u))</th><th class="gk">+ &#x3BD;</th>'
      '<th class="gk">D(u)</th></tr></thead>')
    w('        <tbody>')
    for u in us:
        s = (u * (1 - u)) ** 0.5
        cls = ' class="here"' if u == 1.0 else ''
        w(f'          <tr{cls}><td>{u:.2f}</td><td>{s:.3f}</td>'
          f'<td>{s + nu:.3f}</td><td>{u / (s + nu):.2f}</td></tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')
    w('')
    w('    <div class="note">')
    w('      <span><strong>The last row is where the separation comes from.</strong> At '
      '<math><mrow><mi>u</mi><mo>=</mo><mn>1</mn></mrow></math> the square root vanishes '
      'and <math><mi>D</mi></math> becomes '
      f'<math><mrow><mn>1</mn><mo>/</mo><mi>&#x3BD;</mi><mo>=</mo><mn>{int(1 / nu)}</mn></mrow></math>. '
      f'Everything short of unanimous is squashed between {lo:.2f} and {hi_:.2f}. Being '
      'selected in every single resample is a different regime from being selected in all '
      'but one.</span>')
    w('    </div>')
    w('')
    w('    <div class="scroll-x">')
    w('      <table class="design">')
    w('        <thead><tr><th>rank (j)</th><th>predictor</th><th>real</th>'
      '<th class="gk">&#928;<sub>(j)</sub></th>'
      '<th class="gk">&#928;&#772;<sub>(j)</sub></th>'
      '<th class="gk">Z<sub>(j)</sub></th>'
      '<th class="gk">Z&#772;<sub>(j)</sub></th>'
      '<th class="gk">gap = &#916;<sub>(j)</sub></th></tr></thead>')
    w('        <tbody>')
    for i in range(PP):
        j = order[i]
        gap = r['Z'][i] - r['Z_bar'][i]
        cls = ' class="here"' if real[j] else ''
        sign = '+' if gap > 0 else '&#8722;'
        w(f'          <tr{cls}><td>{i + 1}</td>'
          f'<td class="{"v1" if real[j] else "dim"}">{fj(j + 1)}</td>'
          f'<td class="dim">{"yes" if real[j] else "&#183;"}</td>'
          f'<td>{Pi[j]:.2f}</td><td class="dim">{r["Pi_bar"][i]:.2f}</td>'
          f'<td>{r["Z"][i]:.2f}</td><td class="dim">{r["Z_bar"][i]:.2f}</td>'
          f'<td class="{"" if gap > 0 else "fp"}">{sign}{abs(gap):.2f}</td></tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')

    # ---- step 4 -------------------------------------------------------
    w('')
    w('    <div class="exhead"><span class="exnum">4</span> Step down, cut, divide</div>')
    w('')
    w('    <div class="quote">')
    w('      &ldquo;step (1) is a step&#8209;down procedure. For the index '
      '<math><mrow><mi>j</mi><mo>=</mo><mn>1</mn><mo>,</mo><mo>&#8230;</mo><mo>,</mo>'
      '<mi>p</mi></mrow></math>, starting from the left, moving to the right, we find the '
      'first <math><mrow><mi>j</mi><mo>=</mo><msup><mi>j</mi><mo>*</mo></msup></mrow></math> '
      'such that the difference between '
      '<math><msub><mi>Z</mi><mrow><mo>(</mo><mi>j</mi><mo>)</mo></mrow></msub></math> and '
      'its permuted counterpart is above <math><mi>&#916;</mi></math>.&rdquo;')
    w('    </div>')
    w('')
    w('    <div class="prose"><p>')
    w('      Two of the scans side by side. The left one lets a third predictor through; the '
      'right one is the <math><mi>&#916;</mi></math> that ends up winning.')
    w('    </p></div>')
    w('')
    w('    <div class="split">')
    for d, panel in ((mid['delta'], 'bad'), (best['delta'], 'good')):
        cut = min(x['z'] for x in walk(r, d) if x['ok'])
        w(f'      <div class="panel {panel}">')
        w(f'        <h4>&#916; = {d:.2f}</h4>')
        w('        <table class="alpha compact">')
        w('          <thead><tr><th>j</th><th class="gk">Z<sub>(j)</sub></th>'
          '<th class="gk">Z&#772;<sub>(j)</sub> + &#916;</th><th></th></tr></thead>')
        w('          <tbody>')
        for x in walk(r, d):
            mark = ('<b>&#8592;&nbsp;j*</b>' if x['star']
                    else ('yes' if x['ok'] else '<span class="dim">no</span>'))
            w(f'            <tr><td>{x["j"]}</td><td>{x["z"]:.2f}</td>'
              f'<td class="dim">{x["bar"]:.2f}</td><td>{mark}</td></tr>')
        w('          </tbody>')
        w('        </table>')
        w(f'        <p class="eq-note">Z(&#916;) = {cut:.2f}</p>')
        w('      </div>')
    w('    </div>')
    w('')
    w('    <div class="quote" style="margin-top:22px;">')
    w('      &ldquo;Count the average number of Z values in the permuted data that are above '
      'the <math><mrow><mi>Z</mi><mo>(</mo><mi>&#916;</mi><mo>)</mo></mrow></math> cutoff. '
      'This average number serves as an estimate of the expected number of false '
      'discoveries.&rdquo;')
    w('    </div>')
    w('')
    w('    <div class="scroll-x">')
    w('      <table class="design">')
    w('        <thead><tr><th class="gk">&#916;</th><th class="gk">Z(&#916;)</th>'
      '<th class="gk">N<sub>+</sub></th><th>permuted cells &#8805; cutoff</th>'
      f'<th>of {M}&#215;{PP}</th><th class="gk">&#234;<sub>0</sub></th>'
      '<th class="gk">F&#770;dr</th><th>selected</th></tr></thead>')
    w('        <tbody>')
    for s in rows:
        cls = ' class="here"' if s is best else ''
        sel = ', '.join(fj(x) for x in s['selected'])
        w(f'          <tr{cls}><td>{s["delta"]:.2f}</td><td>{s["cutoff"]:.2f}</td>'
          f'<td>{s["n_pos"]}</td><td>{s["cells"]}</td>'
          f'<td class="dim">{M * PP}</td><td>{s["e0"]:.2f}</td>'
          f'<td class="{"" if s["fdr"] <= Q else "fp"}">{s["fdr"]:.3f}</td>'
          f'<td class="dim">{sel}</td></tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')
    w('')
    idx = np.argwhere(r['Z_null'] >= best['cutoff'])
    where = '; '.join(f'permutation {m + 1} rank ({j + 1})' for m, j in idx)
    w('    <div class="note">')
    w(f'      <span><strong>The winning row is {len(idx)} cells out of {M * PP}.</strong> '
      'These are the only places in the whole permuted null where a predictor was selected '
      f'by every single bootstrap: {where}. So '
      '<math><msub><mover accent="true"><mi>e</mi><mo>^</mo></mover><mn>0</mn></msub></math> '
      f'= {len(idx)}/{M} = {best["e0"]:.2f}, and '
      '<math><mover accent="true"><mtext>Fdr</mtext><mo>^</mo></mover></math> = '
      f'{best["e0"]:.2f}/{best["n_pos"]} = {best["fdr"]:.3f}. That one division is the '
      'guarantee.</span>')
    w('    </div>')

    # ---- result -------------------------------------------------------
    w('')
    w('    <div class="exhead"><span class="exnum">&#10003;</span> Result</div>')
    w('')
    sel = sorted(int(i) + 1 for i in r['selected'])
    w('    <div class="split">')
    w('      <div class="panel good">')
    w(f'        <h4>PS&#8209;Fdr at q = {Q}</h4>')
    w('        <table class="alpha compact"><tbody>')
    w('          <tr><td>selected</td><td><b>{'
      + ', '.join(fj(x) for x in sel) + '}</b></td></tr>')
    w('          <tr><td><math><mover accent="true"><mtext>Fdr</mtext><mo>^</mo></mover>'
      f'</math></td><td>{r["fdr_hat"]:.3f}</td></tr>')
    w('          <tr><td>false discoveries</td><td><b class="zero">'
      f'{len([x for x in sel if not real[x - 1]])}</b></td></tr>')
    w('          <tr><td>real ones missed</td><td><b class="zero">'
      f'{len([j for j in range(PP) if real[j] and j + 1 not in sel])}</b></td></tr>')
    w('        </tbody></table>')
    w('      </div>')
    w('      <div class="panel bad">')
    w('        <h4>CV&#8209;tuned lasso</h4>')
    w('        <table class="alpha compact"><tbody>')
    w('          <tr><td>selected</td><td><b>{'
      + ', '.join(fj(x) for x in cv) + '}</b></td></tr>')
    w('          <tr><td><math><mover accent="true"><mtext>Fdr</mtext><mo>^</mo></mover>'
      '</math></td><td class="dim">not available</td></tr>')
    w(f'          <tr><td>false discoveries</td><td><b class="fp">{len(cv_false)}</b></td></tr>')
    w('          <tr><td>real ones missed</td><td><b class="zero">0</b></td></tr>')
    w('        </tbody></table>')
    w('      </div>')
    w('    </div>')
    w('')
    w('    <div class="note">')
    w(f'      <span><strong>Where the false positive died.</strong> '
      f'{", ".join(fj(j) for j in cv_false)} is genuinely more stable than chance &mdash; '
      'its row in the rank table has a positive gap. It just is not stable '
      f'<em class="term">enough</em>. Letting it in means dropping the cutoff to '
      f'{mid["cutoff"]:.2f}, where {mid["cells"]} permuted cells clear the bar too and '
      '<math><mover accent="true"><mtext>Fdr</mtext><mo>^</mo></mover></math> jumps to '
      f'{mid["fdr"]:.3f}. The procedure declines and stops at {best["n_pos"]}.</span>')
    w('    </div>')

    # ---- resolution ---------------------------------------------------
    w('')
    w('    <div class="exhead"><span class="exnum">!</span> Why it does not go smaller</div>')
    w('')
    w('    <div class="prose"><p>')
    w('      The first attempt at this example was smaller still &mdash; six predictors, '
      '<math><mrow><mi>B</mi><mo>=</mo><mn>10</mn></mrow></math> &mdash; and it selected '
      '<em class="term">nothing at all</em>, at any level. Three seeds per cell:')
    w('    </p></div>')
    w('')
    w('    <div class="scroll-x">')
    w('      <table class="design">')
    w('        <thead><tr><th class="gk">p</th><th class="gk">B</th>'
      '<th>selected, three seeds</th></tr></thead>')
    w('        <tbody>')
    for row in probe:
        cells = '; '.join(
            ('{' + ', '.join(fj(x) for x in s) + '}') if s
            else '<span class="fp">nothing</span>' for s in row['sels'])
        cls = ' class="here"' if (row['p'], row['B']) == (PP, B) else ''
        w(f'          <tr{cls}><td>{row["p"]}</td><td>{row["B"]}</td>'
          f'<td class="dim">{cells}</td></tr>')
    w('        </tbody>')
    w('      </table>')
    w('    </div>')
    w('')
    w('    <div class="note">')
    w('      <span><strong><math><mi>B</mi></math> is resolution, not just precision.</strong> '
      '<math><mi>&#928;</mi></math> only takes values in steps of '
      '<math><mrow><mn>1</mn><mo>/</mo><mi>B</mi></mrow></math>, so at '
      '<math><mrow><mi>B</mi><mo>=</mo><mn>10</mn></mrow></math> a noise predictor needs '
      'just ten straight hits to reach '
      '<math><mrow><mi>&#928;</mi><mo>=</mo><mn>1</mn></mrow></math>. Under the null that '
      'happens constantly, '
      '<math><msub><mover accent="true"><mi>e</mi><mo>^</mo></mover><mn>0</mn></msub></math> '
      'never falls, and nothing clears any target. The paper&rsquo;s '
      '<math><mrow><mi>B</mi><mo>=</mo><mn>50</mn></mrow></math> and '
      '<math><mrow><mi>M</mi><mo>=</mo><mn>100</mn></mrow></math> buy room at exactly the '
      'end of the scale where the decision is made. Even here the smallest non&#8209;zero '
      '<math><mover accent="true"><mtext>Fdr</mtext><mo>^</mo></mover></math> that can be '
      f'expressed is {1 / M:.2f}/{best["n_pos"]} = {1 / M / best["n_pos"]:.3f}.</span>')
    w('    </div>')
    w('')
    w('    <div class="prose" style="margin-top:22px;"><p>')
    w('      Everything below is these same four steps at thirty predictors, where the '
      'tables stop fitting on a page.')
    w('    </p></div>')
    w('  </section>')
    return '\n'.join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--html', type=str, default=None,
                    help='write the docs section here ("-" for stdout)')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    r = run(args.seed)
    report(r)

    if args.html:
        frag = html(r, resolution_probe())
        if args.html == '-':
            print(frag)
        else:
            with open(args.html, 'w', encoding='utf-8') as fh:
                fh.write(frag)
            print(f"\nwrote {args.html}", file=sys.stderr)


if __name__ == '__main__':
    main()
