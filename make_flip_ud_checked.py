from pathlib import Path
from PIL import Image, ImageOps, ImageChops
import shutil

repo = Path('/home/maxim/HTR-VT')
src = repo / 'data' / 'omer'
dst = repo / 'data' / 'omer_flip_ud_labels_normal'
if dst.exists():
    shutil.rmtree(dst)
(dst / 'lines').mkdir(parents=True, exist_ok=True)

for list_name in ['train.ln', 'val.ln', 'test.ln']:
    shutil.copy2(src / list_name, dst / list_name)

for txt in (src / 'lines').glob('*.txt'):
    shutil.copy2(txt, dst / 'lines' / txt.name)

count = 0
for img_path in (src / 'lines').glob('*.png'):
    with Image.open(img_path) as im:
        flipped = ImageOps.flip(im)
        flipped.save(dst / 'lines' / img_path.name)
    count += 1

sample = 'page_0001__eSc_line_c9d61a1e.png'
orig = Image.open(src / 'lines' / sample).convert('L')
made = Image.open(dst / 'lines' / sample).convert('L')
lr = ImageOps.mirror(orig)
ud = ImageOps.flip(orig)

def diff_score(a, b):
    diff = ImageChops.difference(a, b)
    return sum(diff.histogram()[i] * i for i in range(256))

print('created', count, 'top-bottom flipped images')
print('dataset', dst)
print('up_down_diff', diff_score(made, ud))
print('left_right_diff', diff_score(made, lr))
print('same_as_original_diff', diff_score(made, orig))
print('sample label', (dst / 'lines' / sample.replace('.png', '.txt')).read_text(encoding='utf-8').strip())
