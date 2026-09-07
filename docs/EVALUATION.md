# What would support a “best in class” claim?

A model becomes competitive through measured held-out results, not architecture size, training-loss reduction or the presence of security-related data. This release has execution evidence only.

## Separate the questions

1. **Does it run correctly?** Dense-attention equivalence, finite gradients, label masking, graph gradients, checkpoint resume, tokenizer consistency and backend tests.
2. **Does it learn in-distribution?** Validation curves, real-data overfitting/label-shuffle sanity checks, class distribution, calibrated probabilities and error analysis.
3. **Does it generalize?** Untouched future incidents and project families, realistic prevalence, low-FPR recall, group confidence intervals, source coverage and independently adjudicated findings.

The package tests the first question and supplies tools for parts of the others. It does not answer the third with its synthetic fixtures.

## Evaluation design

Freeze global holdout membership before pre-training and tokenizer learning. Group originals/fixes/backports/forks/augmentations and cross-dataset duplicates. Track complete files containing held-out functions, not only record hashes. Pre-training source exposure matters even without vulnerability labels. Keep chronological and project-held-out evaluations separately named; default hash splits are not temporal splits.

Use train to optimize weights, validation to select checkpoints and hyperparameters, calibration to choose operating thresholds, and test for the final report. Repeatedly selecting a threshold or configuration based on test results turns it into development data.

PrimeVul's original benchmark should preserve its partition definitions when reporting a PrimeVul result. A new globally project-grouped mixture is a new benchmark, even when it contains PrimeVul examples. The splitter refuses conflicting supplied partitions when `--respect-splits` is selected.

## Metrics implemented

`evaluate` reports AUPRC, AUROC, prevalence, confusion counts, precision, recall, FPR, F1, Brier score, diagnostic recall at 1% and 0.1% FPR and a one-sided 95% zero-observed-FP bound where applicable. Optional bootstrap resamples declared case groups for an AUPRC interval. Unknown labels are excluded, not mapped to zero.

Curve-derived test recall at an FPR is a diagnostic, not the chosen deployment threshold. Calibration uses a positive temperature plus intercept on a distinct labeled pool and conservatively handles tied negative scores when selecting an empirical FPR threshold. Calibration cannot improve ranking discrimination.

The zero-FP bound assumes the relevant negative trials behave like the statistical model; correlated functions/project shifts weaken that interpretation. At very low FPR, many independently assessed negatives are needed. The built-in minimum calibration size is merely an error guard. Do a power/sample-size analysis for your claimed operating rate and sampling design.

Metrics cover only admitted, complete examples. Report rejected overlength examples and parse/graph failures, plus the fraction of the original corpus represented in evaluation. A low error rate on a small easy subset is not high repository coverage.

## Metrics still requiring experiment-specific evaluation

The package does not yet aggregate task-specific evidence localization, graph-trace correctness, CWE hierarchy metrics, verified-pair success, selective coverage-versus-risk curves or repository-wide discovery yield. The heads expose logits and the canonical schema preserves annotations for implementing those assessments. Do not call the current risk-only report a complete evaluation of those capabilities.

The CLI lists candidate evidence tokens but does not assemble or prove feasible execution paths. It suppresses output slots lacking both positive and negative supervision. Passing this supervision guard is necessary, not sufficient, for a valid evidence claim.

## Baselines and ablations

Compare at equal declared neural parameter and training-compute budgets where possible:

- Same encoder with graph adapters disabled.
- Same context policy with graph adapters enabled.
- Shuffled-edge control to test whether graph semantics matter.
- Reviewed-pair/evidence objectives enabled versus disabled.
- Publicly available comparable-size code encoders, with size/token/exposure differences explicitly reported.
- Relevant conventional analyzers, measured on the same projects and review budget.

Repeat meaningful configurations across at least three seeds before claiming robustness. Report per-project/weakness-family results, all parameter counts (including active/non-active modules), tokenizer budget, processed and unique tokens, data revisions and thresholds. Do not compare a teacher-assisted pipeline to a standalone model without disclosing teacher compute and inputs.

## Training quality gates

Before increasing compute, verify that a small real annotated subset can be fitted, a label-shuffled control does not generalize, and source/patch/generated metadata cannot reveal labels. Examine high-confidence false positives and hard missed positives. Test valid identifier renaming and context changes, not only random text corruption.

Retain the 100M architecture only if its empirical benefit justifies its graph preprocessing and memory costs. Learning-rate, loss-weight, masking and curriculum defaults are experiments, not established optima. More tokens cannot compensate for systematic mislabeling or leakage.

## Intended release language

Before actual training: “Experimental implementation; CPU execution tests passed; Apple hardware validation pending.”

After a measured training run: report its exact datasets, scopes and held-out metrics, not “finds any vulnerability.”

Only use “best in class” after defining the comparison class, testing credible alternatives under comparable conditions and reporting the reproducible evidence. No such result is supplied with this code.
