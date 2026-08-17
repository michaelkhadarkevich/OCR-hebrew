from data import dataset
from utils import utils

train_ds = dataset.myLoadDS('data/omer_visual_ltr/train.ln', 'data/omer_visual_ltr/lines/', [512, 64])
print('dataset length:', len(train_ds))
print('first label:', train_ds.tlbls[0])
print('second label:', train_ds.tlbls[1])
print('alphabet size:', len(train_ds.ralph))
converter = utils.CTCLabelConverter(train_ds.ralph.values())
encoded, lengths = converter.encode(train_ds.tlbls[:2])
print('encoded lengths:', lengths.tolist())
print('decoded back:', converter.decode(encoded.cpu(), lengths.cpu()))
