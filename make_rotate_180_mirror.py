from pathlib import Path
from PIL import Image, ImageOps, ImageChops
import shutil

repo = Path('/home/maxim/HTR-VT')
src = repo / 'data' / 'omer'
dst = repo / 'data' / 'omer_rotate_180_mirror_labels_normal'

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
        rotated = ImageOps.flip(ImageOps.mirror(im))
        rotated.save(dst / 'lines' / img_path.name)
    count += 1

sample = 'page_0001__eSc_line_c9d61a1e.png'
orig = Image.open(src / 'lines' / sample).convert('L')
made = Image.open(dst / 'lines' / sample).convert('L')
expected_180 = orig.rotate(180)
expected_lr = ImageOps.mirror(orig)
expected_ud = ImageOps.flip(orig)

def diff_score(a, b):
    diff = ImageChops.difference(a, b)
    return sum(diff.histogram()[i] * i for i in range(256))

print('dataset', dst)
print('created_images', count)
print('diff_rotate_180', diff_score(made, expected_180))
print('diff_left_right', diff_score(made, expected_lr))
print('diff_up_down', diff_score(made, expected_ud))
print('diff_original', diff_score(made, orig))
print('sample label', (dst / 'lines' / sample.replace('.png', '.txt')).read_text(encoding='utf-8').strip())
