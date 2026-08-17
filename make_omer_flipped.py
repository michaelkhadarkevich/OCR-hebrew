from pathlib import Path
from PIL import Image
import shutil

repo = Path('/home/maxim/slide/HTR-VT')
src = repo / 'data' / 'omer'
dst = repo / 'data' / 'omer_flipped'
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
        flipped = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped.save(dst / 'lines' / img_path.name)
    count += 1

print(f'created {count} horizontally flipped images at {dst}')
print('labels copied unchanged')
