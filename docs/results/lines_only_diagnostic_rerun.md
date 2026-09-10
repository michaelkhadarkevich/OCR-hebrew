# Lines-only numerical diagnostic rerun

Run on 10 September 2026 after the historical lines-only continuation developed
a nonfinite training loss at step 18,159. The rerun started from scratch with
the same central baseline settings: 1,438 line samples, seed 123, batch size 16,
AdamW, fixed learning rate 0.0005, weight decay 0.0001, light augmentation and
20,000 requested steps.

## Result

| Measurement | Historical run | Diagnostic rerun |
|---|---:|---:|
| Recorded steps | 20,000 | 20,000 |
| Selected step | 5,821 | 11,648 |
| Best validation CER | 2.45% | 2.13% |
| Final-test CER | 4.33% (31/716 edits) | 3.35% (24/716 edits) |
| Final-test exact lines | 13/32 (40.63%) | 15/32 (46.88%) |
| Nonfinite training loss | First at step 18,159 | None |
| Nonfinite tensors in last checkpoint | 132 | 0 |

The historical failure did not recur. All 20,000 recorded training losses were
finite, including the interval around step 18,159, and every tensor in the last
checkpoint was finite.

Numerical diagnostics recorded the loss, pre-clipping gradient norm, AMP scale,
learning rate and batch paths at every step. Eleven nonfinite gradient norms
were observed at steps 1-7, 10,309, 14,338, 14,710 and 14,993. PyTorch AMP's
gradient scaler handled those overflows by skipping unsafe optimizer updates;
none produced a nonfinite loss or corrupted checkpoint.

## Interpretation

The rerun supports the main empirical finding: exposure to complete lines
performs much better on full-line evaluation than the words-only baseline. Its
3.35% final CER is close to, but still above, the existing mixed checkpoint's
3.07% CER; it produced 15 exact lines compared with 16 for mixed.

The rerun does not identify the root cause of the old failure. The failure may
have depended on the historical resume state, software state or nondeterministic
GPU execution. A single successful repeat cannot distinguish those causes.

This result is supplementary rather than a silent replacement for the original
three-run table. It was trained and evaluated after the audit and after the
final set had already been inspected repeatedly. It therefore provides
diagnostic replication evidence, not an independent once-only test estimate.
Machine-readable values are in
[`lines_only_diagnostic_rerun.json`](lines_only_diagnostic_rerun.json).
