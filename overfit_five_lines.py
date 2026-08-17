import csv, types
from pathlib import Path
import editdistance
import torch
from torch.utils.data import DataLoader, Dataset
from data import dataset
from model import HTR_VT
from utils import utils

class SmallDS(Dataset):
    def __init__(self, n=5):
        self.items=[]
        for name in Path('data/omer/train.ln').read_text(encoding='utf-8').splitlines()[:n]:
            p=Path('data/omer/lines')/name
            self.items.append((str(p), p.with_suffix('.txt').read_text(encoding='utf-8').strip()))
        labels=[x[1] for x in self.items]
        alph=dataset.get_alphabet(labels)
        self.ralph=dict(zip(alph.values(), alph.keys()))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,label=self.items[i]
        img=dataset.get_images(p,512,64).transpose((2,0,1))
        return img,label,p

def collate(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack([torch.from_numpy(x).float() for x in imgs]), list(labels), list(paths)

def patch_forward_no_final_norm(model):
    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        x = self.layer_norm(x)
        x = self.patch_embed(x)
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        x = x + self.pos_embed
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        return self.head(x)
    model.forward = types.MethodType(forward, model)

def decode_all(model, loader, conv):
    model.eval(); rows=[]; total=0; length=0
    with torch.no_grad():
        for imgs, labels, paths in loader:
            preds=model(imgs).float(); sizes=torch.IntTensor([preds.size(1)]*imgs.size(0))
            lp=preds.permute(1,0,2).log_softmax(2)
            arg=lp.argmax(2).transpose(1,0).contiguous().view(-1)
            pred=conv.decode(arg.cpu(), sizes)
            for p,t,y in zip(paths,labels,pred):
                ed=editdistance.eval(y,t); total+=ed; length+=len(t); rows.append((Path(p).name,t,y,ed))
    model.train(); return total/max(1,length), rows

ds=SmallDS(5)
loader=DataLoader(ds,batch_size=1,shuffle=True,collate_fn=collate)
eval_loader=DataLoader(ds,batch_size=1,shuffle=False,collate_fn=collate)
model=HTR_VT.create_model(nb_cls=len(ds.ralph)+1, img_size=[64,512])
patch_forward_no_final_norm(model)
with torch.no_grad():
    model.head.bias.zero_(); model.head.bias[0] = -2.0
conv=utils.CTCLabelConverter(ds.ralph.values())
crit=torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
opt=torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
print('labels:')
for _,label in ds.items: print(label)
for step in range(1,301):
    for imgs, labels, _ in loader:
        text,lens=conv.encode(labels)
        preds=model(imgs).float(); sizes=torch.IntTensor([preds.size(1)]*imgs.size(0))
        loss=crit(preds.permute(1,0,2).log_softmax(2), text.cpu(), sizes, lens.cpu())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        break
    if step in (1,25,50,100,200,300):
        cer, rows=decode_all(model,eval_loader,conv)
        print('\nstep',step,'loss',float(loss.detach()),'cer',cer)
        for r in rows: print(r)
