import torch
from torch.utils.data import DataLoader
from data import dataset
from utils import utils
from model import HTR_VT


def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack([torch.from_numpy(img).float() for img in images], dim=0)
    return images, list(labels)

train_ds = dataset.myLoadDS('data/omer_flipped/train.ln', 'data/omer_flipped/lines/', [512, 64])
loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
images, labels = next(iter(loader))
converter = utils.CTCLabelConverter(train_ds.ralph.values())
text, lengths = converter.encode(labels)
model = HTR_VT.create_model(nb_cls=len(train_ds.ralph) + 1, img_size=[64, 512])
preds = model(images.float()).float()
preds_size = torch.IntTensor([preds.size(1)] * images.size(0))
loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)(preds.permute(1,0,2).log_softmax(2), text.cpu(), preds_size, lengths.cpu())
print('batch labels:')
for label in labels:
    print(label)
print('image batch shape:', tuple(images.shape))
print('pred shape:', tuple(preds.shape))
print('pred time steps:', preds.size(1))
print('target lengths:', lengths.tolist())
print('encoded target total length:', int(text.numel()))
print('ctc loss:', float(loss.item()))
