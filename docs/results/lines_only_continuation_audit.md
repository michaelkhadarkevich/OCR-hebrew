# Lines-only adapted-baseline continuation audit

Verified against local logs and checkpoint files on 9 September 2026.

The earlier README, final-test report and public curves omitted the second phase of this run. Lines-only training did continue to recorded step 20,000 using our adapted baseline. The final-test metrics were correct for the selected checkpoint; the reported run length was incomplete.

## Evidence

| Evidence | Observed value |
|---|---|
| Initial run | `output/manual_lines_10000_rotation/run` |
| Continuation | `output/manual_lines_10000_rotation_continued_20000/run` |
| Continuation config | `start_step=10000`, `steps=20000`; resumes initial `last_model.pth` and retains initial `best_model.pth` |
| Baseline settings | Batch 16, learning rate 0.0005, weight decay 0.0001, light rotation, 1,438 line training samples |
| Initial metric records | 10,000 rows, steps 1 through 10,000; all training losses finite |
| Continuation metric records | 10,000 rows, steps 10,001 through 20,000 |
| Summary | `completed_steps=20000`, `best_step=5821` |
| First nonfinite training loss | Step 18,159 |
| Nonfinite training-loss rows in continuation | 1,395; last finite training loss at step 18,745 |
| First nonfinite validation loss | Step 18,160 |
| Nonfinite validation-loss rows in continuation | 1,291; last finite validation loss at step 18,744 |
| Best checkpoint | Step 5,821, no nonfinite model tensors |
| Last checkpoint | Step 20,000, 132 model-state tensors containing nonfinite values |
| Existing final-test evaluation of continuation best | 32 lines, 13 exact, CER 0.04329608938547486, whitespace excluded |

The initial and continuation best checkpoint files have the same SHA-256:

```text
1fb8b3179de78b9817bb695323fd3993d5b37b1e700affbd2625160f2ca80c60
```

The continuation last checkpoint SHA-256 is:

```text
c6c3a4ab87095dceb69f00b9a63e81c801d1ba0ac8ef47ac9884a18a1747f124
```

The final metrics come from the existing evaluation at `output/harmonit_final_test_all_models/manual_lines_10000_rotation_continued_20000_run_best_model.pth/summary.json`. No new inference was needed to establish equivalence: the two best checkpoint files are byte-identical. This audit does not establish the root cause of the numerical failure.

## Reporting the experiment

Report **20,000 recorded training steps, selected checkpoint at step 5,821, numerical failure during the continuation**. Do not describe the run as ending at 10,000, as 20,000 healthy optimization updates, or as a paper-style experiment. The 4.33% CER and 13/32 exact-line result remain unchanged.

The public validation CSV and graph now include both training phases. The marker at step 18,159 identifies the first nonfinite training loss; the plotted validation CER values are retained as logged, including the failure interval. Matching run lengths alone does not make this a controlled comparison.
