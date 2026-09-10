from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path
p = Path('/home/maxim/HTR-VT/data/omer_flipped/lines/page_0001__eSc_line_c9d61a1e.txt')
print(p.read_text(encoding='utf-8').strip())
