from pathlib import Path
p = Path('/home/maxim/HTR-VT/data/omer_flipped/lines/page_0001__eSc_line_c9d61a1e.txt')
print(p.read_text(encoding='utf-8').strip())
