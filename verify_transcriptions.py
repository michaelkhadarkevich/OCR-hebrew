from pathlib import Path
import sys
from data import dataset
from utils import utils

base = Path('/home/maxim/slide/HTR-VT')
label_file = base / 'data/omer_flipped/lines/page_0001__eSc_line_c9d61a1e.txt'
raw = label_file.read_bytes()
print('python default encoding:', sys.getdefaultencoding())
print('locale preferred encoding:', __import__('locale').getpreferredencoding(False))
print('raw first bytes:', raw[:40])
print('utf8 read:', label_file.read_text(encoding='utf-8').strip())

train_ds = dataset.myLoadDS('data/omer_flipped/train.ln', 'data/omer_flipped/lines/', [512, 64])
print('dataset length:', len(train_ds))
print('dataset first label:', train_ds.tlbls[0])
print('dataset second label:', train_ds.tlbls[1])
print('alphabet size:', len(train_ds.ralph))
print('alphabet contains Hebrew resh:', 'ר' in train_ds.alph)
print('alphabet contains mojibake Ch:', 'Ч' in train_ds.alph)

converter = utils.CTCLabelConverter(train_ds.ralph.values())
encoded, lengths = converter.encode(train_ds.tlbls[:2])
print('encoded lengths:', lengths.tolist())
print('first encoded ids:', encoded[:lengths[0]].tolist())
print('decoded back:', converter.decode(encoded.cpu(), lengths.cpu()))
