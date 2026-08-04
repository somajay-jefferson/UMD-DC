# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For what the code does and how to run it, read [README.md](README.md) — it's
kept current and this file won't repeat it. This file is about how to work in
this repo, not what's in it.

## Response style

- Keep it succinct and precise. No filler, no restating what was just
  discussed, no summarizing the repo back before acting.
- Write like a college freshman talks: plain words, short sentences,
  casual tone. Skip the jargon and the corporate-sounding phrasing.
- Confirm before anything hard to reverse or visible outside this machine:
  `git push`, creating/deleting a GitHub repo, changing repo visibility,
  force-push, rewriting history. Everything else (local edits, running
  scripts, `git add`/`commit` locally) can proceed without asking.
- If a change affects what the public docs site
  (https://somajay-jefferson.github.io/UMD-DC/) shows, say so explicitly
  rather than letting it happen silently.

## Coding conventions

- Every Python module under `src/` has a matching R file it was ported from
  (`clussocode.zip`, `Supplemental_Code.zip`, `Random_CLUSSO_*.R`). The two
  are expected to match numerically, not just behaviorally. When fixing a bug
  or changing logic in one, check whether the other needs the identical fix
  — don't silently let them drift.
- No build step, package manifest, or test suite exists. Verify a change by
  actually running the relevant worked-example script (e.g.
  `CLUSSO_Data_Example.py`, `PS_Fdr_Data_Example.py`) end to end, not just by
  reading the diff.
- Don't add a dependency beyond `numpy`, `scikit-learn`, `pandas`, `joblib`
  without checking first — the project is deliberately dependency-light.
- Numbers and figures on the `docs/` pages are generated from real runs of
  the code (e.g. `PS_Fdr_Data_Example.py --json`), not hand-authored. If a
  method changes, regenerate the corresponding numbers — don't hand-edit
  `docs/*.html` to match.

## Git / commit habits

- Never push without explicit confirmation for that push, even if a
  previous push in the same session was approved.
- Keep commits scoped to one method/module at a time; don't bundle unrelated
  CLUSSO/TEPIG/PS-Fdr changes into a single commit.
- The academic reference PDFs (`CLUSSO_Paper.pdf`,
  `False_Discovery_Rate_Control.pdf`) are intentionally committed and public
  — that decision has already been made, don't re-raise the copyright
  question on future changes.
