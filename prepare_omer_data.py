import csv, zipfile, shutil
from pathlib import Path

root = Path('/home/maxim/slide')
zip_path = root / 'omerreshatot_segment_crops_full.zip'
out = root / 'HTR-VT' / 'data' / 'omer'
lines = out / 'lines'
if out.exists():
    shutil.rmtree(out)
lines.mkdir(parents=True, exist_ok=True)

def repair_mojibake(text):
    raw = bytearray()
    for ch in text:
        o = ord(ch)
        if o < 256:
            raw.append(o)
        else:
            try:
                raw.extend(ch.encode('cp1251'))
            except UnicodeEncodeError:
                raw.extend(ch.encode('utf-8'))
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return text

with zipfile.ZipFile(zip_path) as z:
    names = set(z.namelist())
    with z.open('metadata.csv') as f:
        rows = list(csv.DictReader((line.decode('utf-8-sig') for line in f)))
    list_entries = []
    for row in rows:
        src0 = row['image']
        candidates = [src0, src0.replace('\\', '/'), 'lines/' + src0.replace('\\', '/')]
        src = next((c for c in candidates if c in names), None)
        if src is None:
            raise KeyError('missing image for {}'.format(src0))
        safe = row['page'] + '__' + row['line_id']
        png_name = safe + '.png'
        txt_name = safe + '.txt'
        with z.open(src) as im, open(lines / png_name, 'wb') as out_im:
            shutil.copyfileobj(im, out_im)
        text = repair_mojibake(row['text'])
        (lines / txt_name).write_text(text + '\n', encoding='utf-8')
        list_entries.append(png_name)

for split in ('train', 'val', 'test'):
    (out / (split + '.ln')).write_text('\n'.join(list_entries) + '\n', encoding='utf-8')

alphabet = sorted(set(''.join((lines / (Path(x).stem + '.txt')).read_text(encoding='utf-8').strip() for x in list_entries)))
print('prepared {} lines'.format(len(list_entries)))
print('alphabet chars {}'.format(len(alphabet)))
print('sample label:', (lines / (Path(list_entries[0]).stem + '.txt')).read_text(encoding='utf-8').strip())
