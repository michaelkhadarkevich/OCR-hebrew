import csv
from pathlib import Path

csv_path = Path('/home/maxim/HTR-VT/output/omer_ariel_fixed_150_lr5e4/predictions.csv')
out_path = csv_path.with_name('predictions_readable_utf8_bom.txt')
with csv_path.open(encoding='utf-8', newline='') as f, out_path.open('w', encoding='utf-8-sig') as out:
    rows = list(csv.DictReader(f))
    out.write(f'Predictions: {csv_path}\n')
    out.write(f'Total rows: {len(rows)}\n')
    out.write('=' * 80 + '\n\n')
    for i, row in enumerate(rows, 1):
        out.write(f'#{i}\n')
        out.write(f'Source: {row.get("source", "")}\n')
        out.write(f'Image: {row.get("image", "")}\n')
        out.write(f'Truth: {row.get("truth", "")}\n')
        out.write(f'Prediction: {row.get("prediction", "")}\n')
        out.write(f'Edit distance: {row.get("edit_distance", "")}\n')
        out.write('-' * 80 + '\n')
print(out_path)
print('rows', len(rows))
print('nonblank', sum(1 for r in rows if r.get('prediction')))
print('unique', len(set(r.get('prediction','') for r in rows)))
