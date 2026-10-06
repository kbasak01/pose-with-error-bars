---
name: claim-trace
description: Trace every number and every guarantee-style word in README/docs/tables to a committed results file, before publishing or after editing README. Invoke as /claim-trace [file].
#disable-model-invocation: true
argument-hint: "[file, default README.md]"
---

Trace claims in: $ARGUMENTS (default `README.md` plus `docs/*.md` and `results/TABLES.md`).

1. Extract every number and every occurrence of: guarantee, certified, provably, safe, ensures,
   always, calibrated, valid coverage.
2. For each number: find its JSON in `results/`, compare exact value and precision. Output a table
   claim | location | artifact | match.
3. For each guarantee-style word: is it about in-distribution marginal coverage, with the
   exchangeability caveat nearby? Anything attached to a lightbox/sunlamp number is a failure.
4. Hand the table to `eval-reproducibility-auditor` for the verdict. Do not edit the README
   yourself in this skill; list the fixes.
