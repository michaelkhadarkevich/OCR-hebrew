import csv, zipfile, shutil
from pathlib import Path

zip_path = Path('/home/maxim/slide/ariel_page4_segment_crops.zip')
out = Path('/home/maxim/HTR-VT/data/ariel')
lines = out / 'lines'
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

added = []
with zipfile.ZipFile(zip_path) as z:
    names = set(z.namelist())
    with z.open('metadata.csv') as f:
        rows = list(csv.DictReader((line.decode('utf-8-sig') for line in f)))
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
        (lines / txt_name).write_text(repair_mojibake(row['text']) + '\n', encoding='utf-8')
        added.append(png_name)

all_entries = sorted(p.name for p in lines.glob('*.png'))
for split in ('train', 'val', 'test'):
    (out / (split + '.ln')).write_text('\n'.join(all_entries) + '\n', encoding='utf-8')

alphabet = sorted(set(''.join((lines / (Path(x).stem + '.txt')).read_text(encoding='utf-8').strip() for x in all_entries)))
print('added {} lines'.format(len(added)))
print('total {} lines'.format(len(all_entries)))
print('alphabet chars {}'.format(len(alphabet)))
print('sample added label:', (lines / (Path(added[0]).stem + '.txt')).read_text(encoding='utf-8').strip() if added else '')
