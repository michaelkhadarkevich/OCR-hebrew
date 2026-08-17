from pathlib import Path
import shutil

repo = Path('/home/maxim/HTR-VT')
src = repo / 'data' / 'omer'
dst = repo / 'data' / 'omer_visual_ltr'
if dst.exists():
    shutil.rmtree(dst)
(dst / 'lines').mkdir(parents=True, exist_ok=True)

for list_name in ['train.ln', 'val.ln', 'test.ln']:
    shutil.copy2(src / list_name, dst / list_name)

count = 0
for img in (src / 'lines').glob('*.png'):
    shutil.copy2(img, dst / 'lines' / img.name)
    txt = src / 'lines' / (img.stem + '.txt')
    label = txt.read_text(encoding='utf-8').rstrip('\n')
    reversed_label = label[::-1]
    (dst / 'lines' / (img.stem + '.txt')).write_text(reversed_label + '\n', encoding='utf-8')
    count += 1

print(f'created {count} unchanged images with reversed labels at {dst}')
print('sample original:', (src / 'lines/page_0001__eSc_line_c9d61a1e.txt').read_text(encoding='utf-8').strip())
print('sample reversed:', (dst / 'lines/page_0001__eSc_line_c9d61a1e.txt').read_text(encoding='utf-8').strip())
