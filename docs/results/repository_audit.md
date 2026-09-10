# Repository and experiment audit — 9 September 2026

The three published baseline final-test results are reproducible and their graph matches the source data. The audit found additional protocol and documentation problems, including a confirmed EMA implementation bug in the historical paper-style runs. Correcting code does not retroactively repair those experiments.

Scope: the fetched GitHub `main` branch at `13e24c5`, all tracked Markdown reports and result figures, their CSV sources and generating code, 25 local training-run configurations and associated manifests/metrics, and 29 saved evaluation summaries with prediction CSVs. The audit also checked all 8,432 current PNG/TXT pairs and ran fresh inference for the three central checkpoints. This is not an audit of every historical Git revision, external issue, release attachment or slide deck. Local `output/` artifacts are not published GitHub results.

Machine-readable counts and run metadata are in [`repository_audit_data.json`](repository_audit_data.json). No historical checkpoints or dataset files were changed.

## Confirmed baseline results

Fresh CUDA inference after the EMA code fix reproduced these results on the same 32 final-test lines and 716 non-whitespace characters:

| Training data | Training samples | Recorded run steps | Selected step | Edit errors | CER | Exact lines |
|---|---:|---:|---:|---:|---:|---:|
| Words only | 6,779 | 20,000 | 12,086 | 81 | 11.31% | 5/32 |
| Lines only | 1,438 | 20,000 | 5,821 | 31 | 4.33% | 13/32 |
| Lines + words | 8,217 | 20,000 | 15,341 | 22 | 3.07% | 16/32 |

These runs use batch size 16, AdamW and no EMA or warm-up/cosine schedule. Their current training/validation manifests have no missing paths or label changes. The selected lines-only checkpoint is finite, despite numerical failure in its later continuation. See the [separate continuation audit](lines_only_continuation_audit.md).

The final-test bar chart agrees with freshly recomputed predictions. Every committed validation-curve point agrees with the corresponding source metric row, including the lines-only continuation to 20,000. The CSV samples step 1 and every 100th step; the graph starts at 1,000. The best checkpoint is selected from all recorded evaluations, not just the displayed samples. Thus a graph's visible minimum need not be the exact recorded minimum.

## Findings and corrections

### 1. EMA validation executed the wrong model — confirmed code bug

The old `patch_forward_no_final_logit_norm` installed a function closing over the original model. `ModelEma` uses `deepcopy`, which copied the model but retained that same function and closure. Calling the EMA copy therefore executed the raw model instead of its own parameters. Switching the EMA copy to evaluation mode also did not switch the raw model referenced by the closure, so raw BatchNorm buffers could be updated during validation. Meanwhile the saved best state was taken from the EMA copy.

A minimal reproduction set the raw head weight to 2 and the copied head weight to 7: both forward calls returned 2. The corrected bound method returns values from each instance separately. Regression tests cover independent parameters, BatchNorm state during validation and EMA updates.

As a concrete checkpoint/log mismatch, the historical paper-style lines-only 20,000-step run records best validation CER **34.042553%** at step 19,200. Fresh evaluation of that saved checkpoint on the actual 32 nested validation samples gives **33.060556%**. The latter is an evaluation of the saved weights, not a corrected historical model-selection process.

Correction: the patch now uses a bound method, preserving the copied instance. New configurations/summaries record `training_recipe_version=global-cosine-bound-ema-v1`. Legacy EMA/cosine checkpoints are rejected for continuation into the corrected recipe. The primary baseline results are unaffected because those runs have EMA disabled.

### 2. Historical paper-style schedules differ from the documented recipe

The earlier chunk-local cosine horizon also remains visible in the saved learning rates. For the 20,000-step paper-style runs, the logged learning rate disagrees with the documented global schedule at **184/200** evaluation rows for words, **188/200** for lines, and **185/200** for mixed data. The 30,000-step continuations retain the affected history. These are actual logged-rate comparisons, not an inference from directory names alone.

Their saved final-test prediction arithmetic can still be evaluated, but their curves do not demonstrate training under the intended corrected recipe. The historical 20,000-step final CERs are 14.80%, 30.31% and 6.98% for words, lines and mixed data; these are not substitutes for the verified baseline chart and should not be presented as a faithful paper reproduction.

Correction: the README explains both defects. General plots of legacy EMA/warm-up configurations visibly include `[legacy protocol]` in their labels. The public baseline charts continue to use the baseline directories only.

### 3. The previous rerun instruction would reuse old results

Previously the PowerShell runner skipped a directory when its summary said the target step count was complete. Executing the same command would therefore not regenerate corrected results. Resuming a legacy checkpoint would also retain the invalid experimental history.

Correction: the 20,000/30,000-step runners accept `-OutputRoot` and reject incompatible summaries. The README provides a fresh output-root command. Guards were tested against the existing legacy directories without starting training. A corrected SAM/EMA training smoke step and a full-state resume step were exercised separately. A new full-length comparison has **not** been run.

### 4. Validation shares source pages with training

The 32 historical validation lines come from 16 page identifiers; **15 page identifiers also occur in central training**. This is a line-level split, not a page-level split. No byte-identical validation images occur in the central training folders, and no training word filenames identify them as crops of the held-out validation lines. The archive folder `HarmonitManual` does contain copies of many validation images, but it is not among the central runs' training folders.

The 32 final-test images have no byte-identical duplicate anywhere else under `data/`, and their page identifiers overlap neither central training nor validation. These filename/hash checks do not prove writer independence or exclude renamed/related images.

Correction: the README and final report now distinguish the actual historical split from the recommended future page-disjoint protocol. The final set has also been used retrospectively to compare multiple configurations; the winning final metric is exploratory, not an untouched once-only estimate for a preselected winner.

### 5. A separate batch-size-8 baseline is incomplete

`controlled_baseline_words_only_20000/run` has metrics only through step **600**, no completion summary and no `last_model.pth` full-state checkpoint. The `20000` in the directory name is its target, not evidence of completion. It differs from the published words-only baseline, which is `only_words_four_datasets_20000_rotation/run`, batch size 16, and reached 20,000.

Correction: the README identifies this as a separate incomplete experiment. No graph or final metric is attributed to a completed controlled comparison.

### 6. Some historical runs used different data or evaluation splits

The regularized run used 12 validation lines sampled from `HarmonitManualTrain`, 1,426 training samples, and stopped at step 2,040. Its historical summary's `final_test_cer` was calculated on `ManualTest`, not the later final set. The report's regularized row correctly uses the separate `ManualFinalTest` evaluation: 15/32 exact, CER 4.33%.

Four early word-run manifests (`hebrew_words_3000`, `hebrew_words_10000`, `hebrew_words_10000_rotation`, and `hebrew_words_10000_rotation_cleaned`) no longer exactly match current data. The first three each reference six missing image paths and 13 changed label paths; the cleaned run has eight changed label paths and no missing paths. This is historical data drift, not a reason to alter the current transcriptions. All three central run manifests match current data.

An interrupted legacy paper-word metric file also contains one malformed row; the charger-interrupted run has logs beyond its last summary step. These incomplete artifacts were not used for the published figures. Directory names and stale summaries alone are not reliable measures of completed training.

Correction: the final report distinguishes the regularized protocol and historical reproducibility limits. No missing historical samples or label versions were invented.

The auxiliary `evaluate_training_cer.py` also reloads entire configured folders rather than the saved training manifest. For a split-out validation protocol such as the regularized run, that would include the excluded 12 lines. The README now states this limitation. The existing paper-style training-set evaluations use explicit unsplit training folders, so this particular discrepancy does not apply to them.

### 7. Several implementation details were missing from the prose

- Baseline augmentation also changes contrast (probability 0.35, factor 0.8–1.2) and brightness (probability 0.25, factor 0.9–1.1), alongside the configured rotation.
- The Hebrew entry points bypass the original model's final logit normalization. Calling the original model forward directly is a different inference path.
- EMA's `0.9999` is a cap; its effective decay ramps with the update count.
- Mask ratio `0.4` is a parameter, not a guarantee that exactly 40% of unique positions are hidden; sampled spans may overlap.
- Width-limited images are compressed to fit after height scaling; padding is not the only possible width operation.
- Word segmentation is an automatic proposal followed by review, not exclusively manual word boxing. Its precision was not quantitatively established.
- `1 - CER` is the report's derived character score, not an independent per-position accuracy measurement.
- `img/HTR-VT.png` and `img/visual.png` are reference illustrations, not measured Hebrew results.

Correction: the relevant README/report passages and an `img/README.md` now explain these points. The older `manual_segmenter/` and one-off diagnostic scripts are retained as historical tools; some contain machine-specific paths. The documented portable segmentation workflow uses `tools/segmentation/`.

### 8. The line evaluator missed nested datasets

`evaluate_line_folder.py` previously searched only the top directory for PNGs. The final set is flat, so its published evaluations were unaffected, but `ManualTest` stores images in subdirectories. Evaluating that folder would report no pairs.

Correction: the evaluator searches recursively, matching the training loader's behavior. Fresh validation evaluation found all 32 samples. Final-set inference on the three central checkpoints reproduced the prior metrics.

## Validation and remaining limits

- 8,432 PNGs decode successfully; all have nonempty UTF-8 TXT partners, with no orphan TXT files in the inspected dataset folders. This checks file integrity, not semantic transcription accuracy.
- All 29 pre-existing evaluated summary/prediction pairs checked had matching edit distances, CER, exact counts and current ground-truth labels.
- Three central final-test checkpoints were rerun; all final metrics matched.
- Every committed baseline curve point was matched to its original log row; final metrics were checked against full predictions.
- Three EMA regression tests passed; corrected SAM/EMA smoke training and checkpoint resume were exercised.
- PowerShell syntax and legacy-result rejection were tested; no full paper-style experiment was restarted.

The numerical root cause of the late lines-only baseline failure remains undiagnosed. The existing AdamW training loop can still record nonfinite losses; a future training run needs explicit numerical monitoring. New page-disjoint validation would require retraining and would define a new experiment. A full corrected paper-style comparison also requires fresh training. Neither outcome can be obtained by editing a summary or replotting old logs.

### Post-audit diagnostic rerun (10 September 2026)

A fresh lines-only baseline run added explicit per-step loss, gradient and AMP
monitoring and completed all 20,000 steps without reproducing the historical
nonfinite loss. Its selected step was 11,648; final evaluation produced 24/716
edit errors (3.35% CER) and 15/32 exact lines. The last checkpoint contained no
nonfinite tensors. Eleven gradient overflows were recorded and handled by AMP
without a nonfinite loss.

This narrows the finding to a non-reproducible historical failure; it does not
identify the cause. The rerun was evaluated after repeated use of the final set
and remains supplementary diagnostic evidence. Full details and
machine-readable values are in the
[diagnostic rerun report](lines_only_diagnostic_rerun.md).

Recheck the public figures on the training machine:

```powershell
python scripts/generate_public_results.py --verify-local-sources
python -m unittest discover -s tests -p test_ema_forward.py -v
```
