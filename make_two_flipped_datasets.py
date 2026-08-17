from pathlib import Path
from PIL import Image, ImageOps, ImageChops
import shutil

repo = Path('/home/maxim/HTR-VT')
src = repo / 'data' / 'omer'
configs = [
    ('omer_flip_up_down', ImageOps.flip, 'up_down'),
    ('omer_flip_left_right', ImageOps.mirror, 'left_right'),
]

sample = 'page_0001__eSc_line_c9d61a1e.png'
orig = Image.open(src / 'lines' / sample).convert('L')
expected_ud = ImageOps.flip(orig)
expected_lr = ImageOps.mirror(orig)

def diff_score(a, b):
    diff = ImageChops.difference(a, b)
    return sum(diff.histogram()[i] * i for i in range(256))

for dirname, fn, kind in configs:
    dst = repo / 'data' / dirname
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
            fn(im).save(dst / 'lines' / img_path.name)
        count += 1

    made = Image.open(dst / 'lines' / sample).convert('L')
    print(dirname)
    print('  created_images', count)
    print('  label_sample', (dst / 'lines' / sample.replace('.png', '.txt')).read_text(encoding='utf-8').strip())
    print('  diff_up_down', diff_score(made, expected_ud))
    print('  diff_left_right', diff_score(made, expected_lr))
    print('  diff_original', diff_score(made, orig))
