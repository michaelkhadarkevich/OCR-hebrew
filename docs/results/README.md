# Verified result figures

These figures summarize the three central AdamW baseline experiments described in the project README.

- `verified_final_test_comparison.png` shows the independently verified results on the same 32-line final test. Whitespace is excluded from all metrics.
- `verified_validation_curves.png` shows the recorded validation CER against optimizer steps after the first 1,000 steps. The highly unstable initialization interval is omitted explicitly so that differences during the useful part of training remain visible. The curves are downsampled to keep the repository small.
- `verified_baseline_results.csv` contains the values used in the final-test figure.
- `verified_validation_curves.csv` contains the downsampled values used in the validation figure.

The historical comparison is not a fully controlled ablation. All three runs used batch size 16, AdamW, learning rate `5e-4`, weight decay `1e-4`, image size 64x1024, seed 123 and light rotation. However, the lines-only run ended at 10,000 steps, the other runs ended at 20,000 steps, and the dataset sizes differ.

Regenerate the figures from the committed CSV files:

```powershell
python scripts/generate_public_results.py
```

To rebuild the downsampled validation CSV from the original local run outputs before plotting:

```powershell
python scripts/generate_public_results.py --refresh-curves-from-output
```

The historical paper-style graphs are deliberately excluded because those stored runs predate the cosine-scheduler correction documented in the main README.
