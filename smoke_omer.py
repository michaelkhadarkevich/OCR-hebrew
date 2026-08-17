import torch
from data import dataset
from model import HTR_VT

ds = dataset.myLoadDS('data/omer/train.ln', 'data/omer/lines/', [512, 64])
print('dataset', len(ds), 'alphabet', len(ds.ralph))
img, label = ds[0]
x = torch.tensor(img).unsqueeze(0).float()
model = HTR_VT.create_model(nb_cls=len(ds.ralph) + 1, img_size=[64, 512])
with torch.no_grad():
    y = model(x)
print('input', tuple(x.shape), 'output', tuple(y.shape), 'label', label)
