# What to actually review

This directory is ~900 lines. Most of it is mechanical. These are the places I
made a judgement call that could be wrong, what the alternative was, and what
would prove me wrong.

Read these six. Skim the rest.

Run `python check.py` first — it either passes or it doesn't, and it will tell
you whether the mechanical parts are faithful so you can spend attention here.

---

## 1. Five starting points, and the number five

**Where:** `clusso_select.py` — `DEFAULT_N_STARTS`, `alpha_starts`

**The call:** CLUSSO's answer depends on where the optimizer starts, and the
existing code draws that start randomly. I replaced it with five fixed points.

**Why not one:** measured +2.95 median objective gap against best-of-100 random
starts. Deterministic but landing in genuinely worse solutions.

**Why not nine:** identical to five (+0.00068 both).

**Why a grid rather than five fixed random draws:** at P=2 the L1-normalisation
and sign convention (`Mainfunction_albet.py:79-81`) collapse the whole
initialisation space to two one-parameter arcs. So the paper's random restarts
are sampling an interval by accident. A grid covers it deliberately.

**Bug found in review, now fixed:** the first version generated the start set
afresh for each `n_starts`, so the sets were not nested -- `alpha_starts(2, 7)`
returned six starts, and seven starts could beat eight. `alpha_starts` now
returns a prefix of one fixed sequence, and `check.py` c11 pins both properties.

**Every number in this file was re-measured after that fix.** The old figures
(+0.0027 arc5, 16/20 monotone, worst inversion 2) were produced with the buggy
generator. Current: +0.00068, 14/20 monotone, worst inversion 3.

**What would prove me wrong:** P > 2. The arc argument is specific to two
clusters, and `alpha_starts` generalises to higher P in a way I have not
measured at all. If TEPIG or a three-cluster variant ever uses this, the number
five is unsupported.

**Also:** the +2.95 figure is one cohort at one setting. It is a large enough
gap that I did not think replication was needed to justify five over one, but it
is a single measurement.

---

## 2. The gate criterion I changed mid-experiment

**Where:** `stage1_noise.py` — the block above the final PASS/FAIL

**What happened:** I planned to gate on "does the deterministic rule pick the
same features as best-of-random". It scored 60% at the time (70% after the
`alpha_starts` fix below), which reads like a failure. I changed the criterion
to the objective gap, which passes at +0.00068.

**Why I think that is right and not goalpost-moving:** best-of-random is not
itself a fixed rule. Where several optima sit at nearly the same objective, no
deterministic rule can reproduce a stochastic one, and that is not a defect.
What stability selection actually requires is that the rule be a fixed function
of the data — which the arc grid is by construction — and that it not settle for
materially worse solutions, which is what the gap measures.

**Why you should be suspicious anyway:** I changed a threshold after seeing the
result. That is the classic way to talk yourself into a pass. The reasoning is
written into the code rather than left implicit, and the match rate is still
reported, so you can disagree with me using the same numbers.

**SETTLED by measurement.** Rather than argue it, the benchmark was run twice
with different seeds and compared to itself. Over 10 bootstrap resamples at
n=300, q=50:

    best-of-100 random starts agrees with ITSELF   80%
    arc5 agrees with best-of-100                   80%

Seeding makes best-of-random reproducible but not canonical -- seed 1000 and
seed 5000 are equally valid runs of the same procedure, and they disagree twice
in ten. So 80% is the ceiling, not a shortfall, and the deterministic rule sits
exactly on it. The original criterion was asking a fixed rule to reproduce
something that does not reproduce itself.

Caveat: 10 resamples, so both figures are 8/10. The equality is exact but the
sample is small.

**Supporting evidence:** at 70% match, the disputed features are all within a few
multiples of the 0.001 cut — i.e. carrying under 1% of total coefficient mass. That points to
near-degenerate optima rather than worse ones. This is the argument I would
attack if I wanted to attack the conclusion.

---

## 3. `_score_fold` duplicates code that already exists

**Where:** `clusso_select.py:_score_fold`

**The call:** it is a copy of the scoring inside `slasso_mse`, including its two
easy-to-get-wrong centering conventions.

**Why:** `slasso_mse` fits and scores in one call, so it cannot score a fit you
already have. The multi-start path needs exactly that.

**Why it is a problem:** this is now the third copy of that logic — R original,
`SLasso_MSE.py`, and here. `CLAUDE.md` specifically warns about the first two
drifting; I added a third.

**Mitigation:** `check.py` c2 and c3 pin it against `slasso_mse` exactly.
Currently 0.00e+00.

**The better fix I did not do:** factor the scoring out of `slasso_mse` so there
is one copy. That means editing `src/core/`, which the whole design avoids. If
you would rather take the `src/core/` edit, say so — I think it is the cleaner
end state.

---

## 4. Frozen clustering

**Where:** `clusso_select.py` module docstring

**The call:** the design tensor is built once and reused for every resample,
rather than re-clustering inside the loop.

**What it costs:** the FDR guarantee becomes conditional on that one clustering
solution. Since clustering accuracy dominates CLUSSO's error, that is a real
weakening, not a technicality.

**What it buys:** the composition is possible at all — PS-Fdr assumes a fixed
design — and compute drops from `docs/fdr-clusso.html`'s "two weeks" to under an
hour.

**Sharper objection I have not addressed:** the clustering is fitted jointly
across the whole cohort, so one subject's design row depends on every other
subject's objects. Resampling subjects is therefore not resampling independent
units. Freezing lets you *condition* on this rather than fixing it. I think that
is honest and statable; a referee might not.

**Related, and easy to miss:** the clustering step is the one place the Python
is explicitly NOT a faithful port. `CLUSSO_Functions_Project1_6_16_23.py:98-100`
calls sklearn's `GaussianMixture` "the closest Python equivalent to R's
`Mclust`" -- an approximation, not a translation. So the parity claim in
CLAUDE.md already has a hole exactly where this obstacle lives. The two
questions are the same question wearing different clothes.

**DECISION (2026-08-26): proceed as planned, frozen, and measure the cost at
the end rather than argue it now.** Stage 5 therefore owes a second arm: run it
with the TRUE clustering, which the simulations know, alongside the estimated
one. If the false discovery rate barely moves, "conditional on this clustering"
becomes a footnote with a number behind it. If it moves a lot, the size of the
problem is quantified rather than hand-waved. Either outcome is reportable;
only failing to measure it is not.

---

## 5. `at_least_k` and its anti-conservative failure

**Where:** `clusso_select.py:clusso_support_at_least_k`

**The call:** PS-Fdr pins the permuted arm at exactly k. I relaxed that to a
floor, because CLUSSO can clear a bar but cannot land on a number.

**Why a floor and not a ceiling:** the failure the fixed count prevents is the
permuted arm selecting too *few*. Overshoot only makes the null stronger.

**The dangerous part:** if the lambda grid runs out before reaching k, the
support comes back short. That makes `e0` too small and `Fdr-hat` too small —
**anti-conservative**, silently under-controlling FDR. `info['exhausted']` flags
it, `check.py` c8 verifies it is never silent, and any caller must report the
rate.

**Not yet established:** whether the exact-k → at-least-k swap is *statistically*
valid, as opposed to merely implementable. That is Stage 3, deliberately tested
on the plain lasso where both rules can run. **Nothing built on this should be
trusted until Stage 3 passes.**

**Known cost, measured:** 4 of 20 cohorts cannot go below k=13 even at
lambda=20, while targets as low as k=10 are asked for. The floor overshoots
there -- null stronger than requested, safe direction, but it costs power.
`stage2_path.py` now warns on this and fails outright above 3x overshoot; it
previously printed the number and gated on nothing.

---

## 6. The support definition inherited from `src/core/`

**Where:** `clusso_select.py:support_mask`, `threshold_binding`

**The call:** `tau=0` — take core's own nonzeros.

**Why:** `bet` is returned L1-normalised and already cut at 0.001
(`Mainfunction_albet.py:124`). Anything done downstream can only shrink the
support further; a coefficient core has zeroed cannot be recovered. So the
option space is {core's rule, stricter than core's rule}, and there is no
principled reason to be stricter by default.

**Why it is still unsatisfying:** that cut interacts badly with the
normalisation. More surviving features means every coefficient is smaller, so a
fixed absolute threshold bites harder exactly when the support is large.
Measured: it is live on **52-85%** of the lambda path at n=300, q=50. It decides
the support more often than the lasso does.

**What I did instead of fixing it:** `threshold_binding` reports when it is
plausibly deciding the answer. That is a diagnostic, not a fix.

**The real fix:** make the threshold a parameter of `Mainfunction_albet` — which
the R Random-CLUSSO variant already does (`Random_CLUSSO_Functions_Simulate_Real_Data.R:685`
takes a `thresh` argument). The Python port hardcodes it. Doing this means
editing `src/core/`.

---

## Things I am NOT asking you to review

`clusso_fit`, `clusso_objective`, `make_folds`, `cv_mse`, `clusso_fit_cv`,
`support_mask` are thin and pinned against `src/core/` by `check.py`. If the
checks pass, they are faithful. If you want to spot-check one, `clusso_objective`
is the only one with no parity check — it follows
`CLUSSO_Data_Example.py:162-166` by eye, and only ever ranks starts against each
other, so a constant error in it would not change any result.

---

## 7. The blind spot underneath everything above

Every check in `check.py` compares my code against **the Python port**. Not one
of them says anything about whether that port matches the R it was translated
from. If the translation is wrong, every number in this directory is wrong and
nothing here would notice, because it is Python being compared against Python.

That is the foundation the whole project sits on, and as far as I can tell it
has never been verified end to end.

**Verified cheaply, without R:** all 8 R files have Python counterparts, and
every R function has one (three look missing from `CLUSSO_Functions` only
because Python moved them into `SLasso_MSE.py`). `_glmnet_lasso` is
Python-only, because R calls `glmnet` directly and the port reimplements it.

**What that does not tell you:** whether the numbers agree. Only running both
settles that, and R is not installed here.

**Where drift would most likely hide, worst first:**

1. `_glmnet_lasso` (`Mainfunction_albet.py:12`) -- hand-reimplements glmnet's
   centering, scaling, and the `alpha = lambda/2` objective conversion. The most
   arithmetic per line in the port, and the least visible if slightly off.
2. `coefficient.py` -- a complete port that **nothing imports**. Unused code is
   unexercised code; a bug there could sit indefinitely.
3. The GMM-for-Mclust substitution -- documented as an approximation rather than
   a translation, so this one is known to differ.
4. `CV_make_folds` -- `np.random.choice` where R uses `sample()`. Different
   algorithms, so folds differ even at matched seeds.

**How to actually check it:** the two languages' RNGs differ, so you cannot
generate matching data from a shared seed. Write one fixed dataset to CSV, run
both implementations against it, compare coefficients. Needs R with `glmnet` and
`mclust`. Half a day, and it would either retire the question or find something
that invalidates a lot of work.

---

## 8. The permuted-arm rule changed again, and sections 5 and 6 are stale

**Where:** `clusso_select.py:clusso_support_exact_k`, `clusso_ps_fdr.py`,
`stage4_clusso_fdr.py`

**This section supersedes section 5.** Everything above it was written before
stage 3 ran. Read this first, then section 5 for the history.

**What happened.** Section 5 argued for relaxing PS-Fdr's exact-k to a floor,
because CLUSSO can clear a bar but cannot land on a number. Stage 3 measured
what that costs and **it failed**: `stage3.json` records power 0.688 -> 0.565 at
delta=3, against a gate allowing -0.05. FDR control was never the problem; the
floor overshoots, the overshoot makes the null stronger than requested, and a
stronger null costs detections.

**The fix, and it was sitting in the paper the whole time.** Keep the floor, then
**truncate to the k largest coefficients**. That is not a workaround invented
here -- it is what `ps_fdr.lasso_support_fixed_k` already does when a path step
overshoots k (`ps_fdr.py:112-113`). The lasso needs it rarely, because the path
gives fine-grained control; `stage3.json`'s `part_a` found `exact`, `atleast` and
`nearest` differ on 0 of 120 lasso cohorts. CLUSSO needs it constantly. Same
rule, different bite.

**What that retires, and it is more than it looks:**

- Section 5's whole premise. "CLUSSO cannot land on a number" is true of the
  fit and false of the selection rule, once truncation is allowed.
- Stage 2's non-monotonicity. We never needed lambda -> support size to be
  invertible. We needed it to clear a bar once, which it does.
- The fineness of `DEFAULT_LAMBDA_GRID`. Overshoot is free now, so the grid
  could be coarsened for speed at no cost in accuracy. Not done yet; `mean_fits`
  in `stage4.json` is the number that would justify it.

**What it does NOT retire, and this is the one to watch.** Grid exhaustion. If
the scan reaches the dense end without ever clearing k, the support comes back
SHORT. That makes `e0` too small and `Fdr-hat` too small -- **anti-conservative**,
and it does not announce itself. Truncation does nothing about this. Section 5's
warning stands unchanged, `info['exhausted']` still flags it, `check.py` c8 still
verifies it is never silent, and `stage4_clusso_fdr.py` **gates** on the rate
rather than merely printing it.

**`fixed_lambda` survived as a comparison arm.** One CV-tuned lambda everywhere,
nothing targeting a count. It passed on the lasso (`stage3b.json`) and it is much
cheaper -- one fit per resample instead of a scan. But it is a *deviation* from
the published procedure, and "we ported the published rule" is a defensible
sentence where "we changed the procedure and it seemed fine" is not. So exact-k
is arm A and gates; `fixed_lambda` is arm B and only reports. If they agree, the
count-fixing device was not load-bearing for CLUSSO. If they diverge, the gap is
the measurement.

**Two things I rejected, so they do not get re-proposed:**

*Retrying different initialisations until one hits k.* Determinism is not the
objection -- `alpha_starts` is a fixed nested sequence, so ordered retries stay
pure. **Comparability** is. `clusso_fit_multistart` currently picks the start
with the best objective; picking the start with the right *count* instead would
let the null arm settle into materially worse stationary points purely because
they had k nonzeros. The null arm's fits would then be drawn from a different
region of the solution space than the real arm's -- a second asymmetry stacked on
the one the paper already tolerates (real arm tunes lambda by CV, null arm sets
it by count). Truncation removes the need anyway.

*Warm-starting the lambda scan from the previous resample's answer.* It would
roughly halve arm A's cost. It would also make resample 7's answer depend on
resamples 1-6, so the selection rule would stop being a fixed function of the
data -- which is the exact property `stage1_noise.py` exists to establish and
Meinshausen-Bühlmann's framework requires. Not worth it. The scan cost is
bounded and measured.

**One thing I added that the lasso version does not have.** The truncation sort
is `kind='stable'`. `bet` comes back L1-normalised and then hard-cut at 0.001
inside `Mainfunction_albet`, which piles surviving coefficients near the
threshold and makes exact ties ordinary rather than exotic. numpy's default
quicksort would break those ties on array order, which is not data. The lasso
version at `:113` does not ask for stability, but it is not sorting
normalised-then-cut coefficients. `check.py` c13 pins it.

**Section 6 is also stale, in a smaller way.** It reports the 0.001 threshold as
a diagnostic finding. Truncation now ranks features *by* those same coefficients,
so the cut is no longer only deciding the support -- it is also deciding which
features survive truncation. That raises the stakes on section 6's "real fix"
(parameterising the threshold, which means editing `src/core/`) without settling
it. `stage4_clusso_fdr.py` reports the binding rate per run.

**What would prove this wrong:** a material `exhausted_rate` in `stage4.json`, or
arm A and arm B disagreeing in a direction that says the truncation is throwing
away the wrong features. Both are printed. Neither is argued.

---

## 9. A coarser ladder for the permuted arm, and a warm start that failed

**Where:** `clusso_select.py:NULL_LAMBDA_GRID`, `clusso_ps_fdr.py:ClussoSelector`

The scan is the dominant cost of the whole composition: `M*B` = 5,000 scans per
cohort, each walking down a lambda ladder fitting at every rung. A 20-cohort run
at the default regime took **19 hours** and was still going. Two things were
tried.

### Adopted: the permuted arm gets 13 rungs instead of 40

Overshoot is free now that `clusso_support_exact_k` truncates, so the ladder no
longer has to be fine. Measured end to end on one cohort (n=200, q=40, B=10,
M=10), fine grid against coarse:

        grid          fits/target   null_top   |S|    fdp   power
        fine (40)         15.9        0.880      8   0.000  1.000
        coarse (13)        5.9        0.880      8   0.000  1.000

**2.7x fewer fits and identical output** -- same `null_top` to three decimals,
same selected set, same FDP and power. At production B=50 M=100 the null arm
dominates, so this is roughly 2.2x end to end.

Why identical rather than merely close: the permuted arm's output is `Pi_bar`, a
curve over sorted RANKS. Permuting y destroys feature identity, which is why
`ps_fdr` sorts every permutation before averaging. The null arm needs the right
shape, not the right features. **The real arm keeps the fine grid**, because
there identity does matter.

**Not a deviation from He et al.** The paper has no lambda grid -- `lars_path`
gives exact breakpoints, so there is nothing to discretize. The grid is our
artifact for coping with CLUSSO's missing path, and its resolution sits below
the level the paper specifies.

### Rejected by measurement: warm-starting the scan

Continuation is standard -- glmnet warm-starts down its own path. Within one
scan it is also legal here, since the chain never leaves a single call, so the
rule stays a pure function of that resample's data. (Warm-starting ACROSS
resamples is the thing section 8 rejects, and for a different reason.)

Tried both ways, on the expensive cohort under null-arm conditions, over the
first 7 rungs:

        mode        fits   mean objective vs cold
        cold          35        +0.0000
        extra         41        +0.0000      <- 17% more work, zero gain
        warm-only     11     +1985 to +4916  <- far worse optima

As an **extra** candidate the warm start never won once against the five arc
starts, so it bought nothing and cost a sixth fit per rung. As a **replacement**
it was catastrophic: for scale, section 1 rejected a single cold start over a
gap of **+2.95**, and this is three orders of magnitude worse.

The reason is section 1's own argument. At P=2 the L1-normalisation collapses
the initialisation space to two one-parameter arcs, and the five-point grid
already covers them -- there is no unexplored basin for a warm start to reach.
Warm-only is bad because it abandons multistart to track one branch, which
drifts into a poor stationary point as lambda moves.

**Not shipped.** Section 7 criticises `coefficient.py` for being code nothing
exercises; a `warm_start` parameter defaulted off would be the same fault.

**What would change this:** P > 2. The arc-coverage argument is specific to two
clusters, and with a larger initialisation space a warm start could plausibly
reach basins the fixed grid misses. Worth re-measuring before any three-cluster
variant, not before.

---

## Open questions

1. ~~Is the objective-gap gate criterion defensible?~~ **Settled by measurement,
   see section 2.** The benchmark agrees with itself 80% of the time and the
   deterministic rule matches it 80% of the time. It was on the ceiling.
2. ~~Is conditional-on-one-clustering statable or fatal?~~ **Decided: proceed
   frozen, measure the cost in stage 5. See section 4.**
3. Is taking the `src/core/` edit worth it? **Still open.** My recommendation is
   no to the `_score_fold` de-duplication, since `check.py` already pins it at
   zero tolerance, and not yet to parameterising the 0.001 threshold, since that
   is currently a diagnostic finding rather than a demonstrated problem. Worth
   knowing that adding the parameter with default 0.001 would change no existing
   result -- it exposes a knob rather than altering a translation -- so it is a
   smaller violation than it sounds.
4. **New:** is validating the R-to-Python translation worth half a day and an R
   install? See section 7. Everything else here is conditional on it.
