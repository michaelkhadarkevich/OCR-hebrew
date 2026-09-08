# HTR Model Evaluation on HarmonitManualFinalTest

The original evaluation was performed on 3 September 2026. The three main checkpoints were independently re-evaluated on 7 September 2026, and the four-dataset words-only checkpoint was verified on 8 September 2026.

## Evaluation protocol

All saved HTR checkpoints were compared on the held-out dataset at `data/HarmonitManualFinalTest`. It contains 32 labeled text lines from pages that were not used for training or checkpoint selection.

The model alphabet has no whitespace class. Spaces are therefore removed from both the reference and prediction before all metrics are calculated. Reported CER and exact-line accuracy do not score whitespace.

## Leakage audit

SHA-256 hashes of the 32 final-test images were compared with every other image under `data` before evaluation.

- Final-test images: 32
- Identical images found elsewhere: 0
- Result: no byte-identical final-test images were found elsewhere in the repository

This hash check detects exact file duplicates. It does not by itself rule out related crops or different images originating from the same source page, so page-level separation remains the stronger safeguard.

## Evaluated checkpoints

The original scan covered 19 `best_model.pth` files, including complete experiments, smoke tests and one partial run. The four-dataset words-only checkpoint was evaluated separately. The table below includes only the 11 substantive training runs; smoke-test checkpoints are excluded because they are not meaningful trained models.

## Results from completed runs

| Model | Saved step | Exact lines | Exact-line accuracy | CER | Character accuracy |
|---|---:|---:|---:|---:|---:|
| Lines + words, continued to 20,000 steps | 15,341 | **16/32** | **50.00%** | **3.07%** | **96.93%** |
| Regularized | 1,440 | 15/32 | 46.88% | 4.33% | 95.67% |
| Lines only, 10,000 steps | 5,821 | 13/32 | 40.63% | 4.33% | 95.67% |
| Lines + words, 10,000 steps | 9,762 | 7/32 | 21.88% | 6.01% | 93.99% |
| Words only, four datasets, 20,000 steps | 12,086 | 5/32 | 15.63% | 11.31% | 88.69% |
| Words, rotation, cleaned | 5,443 | 8/32 | 25.00% | 19.41% | 80.59% |
| Words, cleaned test96 | 2,647 | 5/32 | 15.63% | 19.55% | 80.45% |
| Words, 3,000 steps | 2,596 | 3/32 | 9.38% | 26.82% | 73.18% |
| Words, rotation | 4,637 | 2/32 | 6.25% | 30.17% | 69.83% |
| Words, 10,000 steps | 8,584 | 5/32 | 15.63% | 33.52% | 66.48% |
| Words + Physics | 7,376 | 3/32 | 9.38% | 36.03% | 63.97% |

CER is the character error rate, so lower is better. Character accuracy is reported as `1 - CER`.

## Best checkpoint

The strongest checkpoint on the final test is the lines-plus-words model continued to 20,000 steps:

- Checkpoint: `output/manual_lines_plus_words_continued_20000/run/best_model.pth`
- Saved step: 15,341
- Exact lines: 16/32 (50.00%)
- CER: 3.07%
- Character accuracy: 96.93%
- Edit errors: 22 across 716 non-whitespace reference characters

On the earlier development test, this model achieved 75.00% exact-line accuracy and 97.38% character accuracy. On the new final test, it achieved 50.00% exact-line accuracy and 96.93% character accuracy. This decrease is not by itself evidence of a model failure: the final test contains unseen pages and was not involved in checkpoint selection, making it a more independent and difficult estimate of generalization.

## Main three-way comparison

| Experiment | Training data | Result on final test |
|---|---|---|
| Words only | Four datasets of segmented words, width 1024 | 5/32 exact lines (15.63%), CER 11.31% |
| Lines only | Full text lines, width 1024 | 13/32 exact lines (40.63%), CER 4.33% |
| Lines + words | Both sample granularities, width 1024 | **16/32 exact lines (50.00%), CER 3.07%** |

These three results were independently verified with `evaluate_line_folder.py`. Spaces are excluded because the checkpoints do not contain a space class.

## Interpretation and limitations

1. The 20,000-step lines-plus-words checkpoint is the strongest available model.
2. In our experiments, exposure to complete lines was strongly beneficial when the evaluation input was a complete line.
3. In the available experiments, sufficiently trained mixed data produced the best final-test result.
4. The historical three-way comparison is not a perfectly controlled ablation. All three central runs used batch size 16, but their training-step budgets, dataset sizes and effective numbers of epochs differ. The comparison supports an empirical trend, but does not prove that mixed data alone caused the improvement.
5. The final test contains only 32 lines, so the reported percentages have substantial sampling uncertainty.
6. A 3.07% CER is a strong project result, but 50.00% exact-line accuracy still calls for human review in a production workflow.

## Reproducing the evaluation

Evaluate any selected checkpoint with:

```powershell
python evaluate_line_folder.py `
  --checkpoint output/<run>/run/best_model.pth `
  --data-dir data/HarmonitManualFinalTest `
  --output-dir output/<run>_final
```

Generated evaluation files are deliberately ignored by Git because they may include local data and large artifacts. The principal local result files are:

- Model comparison: `output/harmonit_final_test_all_models/comparison.csv`
- Best-model predictions: `output/harmonit_final_test_all_models/manual_lines_plus_words_continued_20000_run_best_model.pth/predictions.csv`
- Best-model summary: `output/harmonit_final_test_all_models/manual_lines_plus_words_continued_20000_run_best_model.pth/summary.json`
