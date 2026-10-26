# Runs

Every saved result in this directory, with what produced it. One row per file.
Add a row when a run is committed.

From 2026-10-26 on, `stage4_clusso_fdr.py` stamps a `meta` block into its JSON:
git commit, whether `src/` had uncommitted changes, the exact command, both λ
ladders, start and finish time, and wall-clock minutes. Older files predate it,
so their code version below is reconstructed from git history.

**Why this file exists:** `stage4.json` and `stage5_reps100.json` have
identical command-line settings but ran on different permuted-arm ladders (40
steps vs 13), and that difference changes exact-k's result. Nothing in the old
files said so.

| file | script | ran | code | null ladder | settings | wall time | headline |
|---|---|---|---|---|---|---|---|
| `stage1.json` | `stage1_noise.py` | 2026-08-26 | `e6429db` | — | n=300, q=50, 100 reps, 10 resamples, seed 20260817 | — | Random starts disagree with their own modal support on a median 0.53 of draws; 5 fixed starts close the objective gap to +0.0027. |
| `stage2.json` | `stage2_path.py` | 2026-08-26 | `e6429db` | — | 20 datasets, n=300, q=50, seed 20260817 | — | λ → support size is monotone on 16/20 cohorts; worst inversion 2 features. |
| `stage3.json` | `stage3_relax_k.py` | ≤ 2026-09-30 | `a889b9e` | — | 120 reps, B=50, M=100, seed 20260826 | — | **FAIL:** relaxing exact-k to a floor costs power 0.688 → 0.565. |
| `stage3b.json` | `stage3b_fixed_lambda.py` | ≤ 2026-09-30 | `a889b9e` | — | 120 reps, B=50, M=100, seed 20260902 | — | **PASS** on the plain lasso: one-λ null_top 0.928 vs exact-k 0.931. |
| `stage4.json` | `stage4_clusso_fdr.py` | 2026-09-29 | between `f30fbb6` and `3714a33` | **fine, 40** | 20 reps, n=300, q=50, 10 real, B=50, M=100, q=0.10, seed 20260929, `--jobs 12` | ~20 h | Exact-k FDP 0.138 (bare CLUSSO 0.334), power 0.900. No controls yet. Feeds `docs/fdr-clusso.html`. |
