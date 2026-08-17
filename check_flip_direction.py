from pathlib import Path
from PIL import Image, ImageChops

repo = Path('/home/maxim/HTR-VT')
name = 'page_0001__eSc_line_c9d61a1e.png'
orig_path = repo / 'data/omer/lines' / name
flip_path = repo / 'data/omer_flipped/lines' / name

orig = Image.open(orig_path).convert('L')
made = Image.open(flip_path).convert('L')
lr = orig.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
ud = orig.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

for label, candidate in [('left_right', lr), ('up_down', ud), ('same_as_original', orig)]:
    diff = ImageChops.difference(made, candidate)
    stat = sum(diff.histogram()[i] * i for i in range(256))
    print(label, stat)
print('orig size', orig.size, 'flipped size', made.size)
