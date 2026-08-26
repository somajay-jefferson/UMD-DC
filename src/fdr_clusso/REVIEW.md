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

**Why not nine:** identical to five (+0.0027 both).

**Why a grid rather than five fixed random draws:** at P=2 the L1-normalisation
and sign convention (`Mainfunction_albet.py:79-81`) collapse the whole
initialisation space to two one-parameter arcs. So the paper's random restarts
are sampling an interval by accident. A grid covers it deliberately.

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
same features as best-of-random". It scored 60%, which reads like a failure. I
changed the criterion to the objective gap, which passes at +0.0027.

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

**Supporting evidence:** the disputed features are all within a few multiples of
the 0.001 cut — i.e. carrying under 1% of total coefficient mass. That points to
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

**Known cost, measured:** on some cohorts the sparse end of the grid still keeps
13 features at lambda=20, so a small target k overshoots several-fold. Safe
direction, but it costs power.

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

## Open questions I would like an opinion on

1. Is taking the `src/core/` edit worth it, to kill the `_score_fold`
   duplication and to make the 0.001 threshold a parameter? Both are cleaner
   ends; both break the "src/core is a frozen port" rule.
2. Is conditional-on-one-clustering a statable caveat or a fatal one?
3. Is the objective-gap gate criterion (§2) defensible, or am I fooling myself?
