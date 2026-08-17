from pathlib import Path
lines = Path('/home/maxim/slide/HTR-VT/data/omer/lines')
lengths = []
for p in lines.glob('*.txt'):
    s = p.read_text(encoding='utf-8').strip()
    lengths.append((len(s), p.name, s))
for item in sorted(lengths, reverse=True)[:10]:
    print(item[0], item[1], item[2])
print('count', len(lengths), 'max', max(x[0] for x in lengths))
