# Stage 6 report: exact-k on the fine ladder, 100 cohorts

## Headline

**Yes, as far as 100 cohorts can tell.** On its own 40-step ladder, the paper's
exact-k rule gives mean FDP **0.113**, 95% CI **[0.093, 0.132]**. The CI covers
0.10. The mean is 1.23 SE above 0.10, which is not a significant miss. So this
run doesn't show that FDR-CLUSSO fails to control FDR at 0.10. It also doesn't
prove control: the point estimate is a bit above 0.10, and 100 cohorts can't
rule out a true FDR up to about 0.13.

The script's own gate still prints FAIL, because it checks the raw mean
(0.113 > 0.10) with no allowance for noise.

The run finished cleanly: 100/100 cohorts, 1294.9 min, JSON written
2026-10-27 10:34.

## 1. Headline numbers (stage 6, 100 cohorts)

SE = sd/√100. CI = mean ± 1.96 SE. z = (mean − 0.10)/SE.

| method | mean FDP | SE | 95% CI | z | median FDP | power | mean FD | mean \|S\| | share FDP ≤ 0.10 |
|---|---|---|---|---|---|---|---|---|---|
| FDR-CLUSSO exact-k | **0.1126** | 0.0102 | [0.0926, 0.1325] | +1.23 | 0.0909 | 0.933 | 1.31 | 10.64 | 0.59 |
| bare CLUSSO (baseline) | 0.3130 | 0.0145 | [0.2845, 0.3415] | +14.64 | 0.3095 | 0.975 | 5.12 | 14.87 | 0.09 |
| naive | 0.6312 | 0.0153 | [0.6011, 0.6613] | +34.62 | 0.6970 | 0.979 | 20.57 | 30.36 | 0.00 |
| oracle | 0.0963 | 0.0144 | [0.0680, 0.1245] | −0.26 | 0.0000 | 1.000 | 1.46 | 11.46 | 0.70 |

Other exact-k means: null_top 0.876, fdr_hat 0.044, k 27.49, mean_fits 11.96.

## 2. Fine vs coarse, paired over the same 100 cohorts

Fine = stage 6 exact-k. Coarse = stage 5 exact-k. Difference = fine − coarse.

| metric | mean diff | SE | t | cohorts fine > coarse | cohorts fine < coarse |
|---|---|---|---|---|---|
| fdp | −0.0059 | 0.0029 | −1.99 | 3 | 9 |
| power | −0.0010 | 0.0010 | −1.00 | 0 | 1 |
| n_sel | −0.090 | 0.038 | −2.38 | 2 | 10 |
| null_top | +0.0043 | 0.0003 | +14.21 | 92 | 6 |
| fdr_hat | −0.0011 | 0.0022 | −0.51 | 66 | 19 |
| fd (false discoveries) | −0.080 | 0.037 | −2.18 | 2 | 9 |

- 98 of 100 cohorts differ on at least one of fdp / n_sel / null_top. Almost
  all of that is null_top, which moves a little nearly everywhere.
- Only 12 cohorts actually change their selection (fdp and n_sel both differ on
  12). Fine FDP is lower on 9 of those, higher on 3.
- Mean FDP: 0.113 (fine) vs 0.118 (coarse). The coarse ladder sets a lower
  null_top on 92 cohorts, which lets in a few extra features, mostly false ones.
  The effect is small (about 0.006 FDP, 0.08 false discoveries per cohort) but
  points one way: borderline significant (t ≈ −2).
- Split by cohort range:

| cohorts | fine FDP | coarse FDP |
|---|---|---|
| 0–19 (stage 4's) | 0.1382 | 0.1565 |
| 20–99 | 0.1061 | 0.1089 |

  The first 20 cohorts are just a harder-than-average draw. Stage 4's 0.138
  came from those 20.

## 3. Reproducibility checks

| check | result |
|---|---|
| baseline identical to stage 5 (fdp, n_sel, bias, mse, all 100) | **PASS**: 0 mismatches, max abs diff 0 |
| naive identical to stage 5 | **PASS**: 0 mismatches |
| oracle identical to stage 5 | **PASS**: 0 mismatches |
| stage 6 exact-k cohorts 0–19 vs stage4.json (fdp, n_sel, k, null_top) | **PASS**: all 20 match exactly, max abs diff 0 on all four |
| exhausted_rate = 0 everywhere | **PASS**: max 0.0 |

So the newer code reproduces stage 4 bit-for-bit on the fine ladder. That
confirms the stage 4 vs stage 5 gap on cohorts 0–19 came from the ladder alone.

## 4. Cost

| | fine (stage 6) | coarse (stage 5) | ratio |
|---|---|---|---|
| exact-k mean sec per cohort | 4548.4 (median 4507.2) | 2298.6 | **1.98×** |
| mean CLUSSO fits per permuted target | 11.96 | 4.74 | 2.52× |

- `meta.wall_minutes` = **1294.9** (about 21.6 h) at `--jobs 6`, exact-k
  only. Stage 5 took 778 min, but that included the fixed_lambda arm too
  (about 424 s per cohort), so the two wall times aren't directly comparable.
- Baseline and oracle took about 14.4 s and 12.5 s per cohort, so exact-k is
  basically the whole cost.

## 5. Recommendations (not acted on)

**Public page (`docs/fdr-clusso.html`).** The current wording ("Measured, falls
short", "it does not reach it", built on stage 4's 0.138 from 20 cohorts) is too
strong given stage 6. With 100 cohorts on the same fine ladder, the CI
[0.093, 0.132] covers 0.10, and the 20 cohorts behind 0.138 turn out to be an
unlucky subset (the other 80 average 0.106). I recommend regenerating the page
from stage 6:
`python stage4_clusso_fdr.py --from-json stage6_exactk_fine_reps100.json --html <out>`.
This would change what the public site shows. Two caveats. Stage 6 has no
fixed_lambda arm, so the page's "0.138 against 0.129" arm comparison would
need its own source (stage 5 has both arms, but on the coarse ladder) or would
have to be dropped. And the honest framing is "consistent with 0.10, point
estimate slightly above", not "controls FDR".

**Default ladder.** Fine costs 1.98× the time per cohort. What you get for it
is a small but consistent FDP drop (0.118 → 0.113, t ≈ −2), and it's the
paper's own rule, so results match stage 4 exactly. Coarse changes selection on
only 12 of 100 cohorts, and both ladders' CIs cover 0.10, so coarse doesn't
change the verdict. My recommendation: make **fine** the default for any
headline or public result, since that's what the paper specifies and coarse
shifts FDP up a little. Keep **coarse** as an opt-in for quick runs and
development, and label those runs (the `meta` block already does this).

## Proposed RUNS.md row

Replaces the current "running" row for this file:

```
| `stage6_exactk_fine_reps100.json` (+ `.log`) | `stage4_clusso_fdr.py` | 2026-10-26 12:59 → 2026-10-27 10:34 | `70c1c8c` (has `meta`, clean `src/`) | **fine, 40** | 100 reps, exact-k only (`--arm a`) + naive and oracle controls, otherwise as stage 5, `--jobs 6` | 1295 min | Exact-k FDP 0.113 [0.093, 0.132] (covers 0.10), power 0.933, bare CLUSSO 0.313, oracle 0.096. Cohorts 0–19 match `stage4.json` exactly; controls match stage 5 exactly. Vs coarse: FDP −0.006 (t −1.99), selection differs on 12/100, 1.98× the time per cohort. `stage6_report.md`. |
```
