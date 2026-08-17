import torch
from pathlib import Path
from data import dataset
from model import HTR_VT
from utils import utils

ckpt_path = Path('/home/maxim/HTR-VT/output/omer_ariel_cpu_50/htrvt_omer_ariel_cpu.pth')
ckpt = torch.load(ckpt_path, map_location='cpu')
labels=[]; items=[]
for source, list_path, lines_dir in [('omer','data/omer/train.ln','data/omer/lines'),('ariel','data/ariel/train.ln','data/ariel/lines')]:
    for name in Path(list_path).read_text(encoding='utf-8').splitlines():
        p=Path(lines_dir)/name
        lab=p.with_suffix('.txt').read_text(encoding='utf-8').strip()
        labels.append(lab); items.append((source,p,lab))
alph=dataset.get_alphabet(labels); ralph=dict(zip(alph.values(), alph.keys()))
converter=utils.CTCLabelConverter(ralph.values())
model=HTR_VT.create_model(nb_cls=ckpt['nb_cls'], img_size=[64,512])
model.load_state_dict(ckpt['state_dict'])
source,p,label=items[0]
img=dataset.get_images(str(p),512,64).transpose((2,0,1))
x=torch.tensor(img).unsqueeze(0).float()

def run(mode):
    if mode == 'eval': model.eval()
    else: model.train()
    with torch.no_grad():
        preds=model(x).float()
        sizes=torch.IntTensor([preds.size(1)])
        lp=preds.permute(1,0,2).log_softmax(2)
        arg=lp.argmax(2).transpose(1,0).contiguous().view(-1)
        pred=converter.decode(arg.cpu(), sizes)[0]
        blank=float(lp.exp()[...,0].mean())
        uniq=[(int(i), int((arg==i).sum())) for i in arg.unique()]
    print(mode, 'pred=', repr(pred), 'blank_prob=', blank, 'uniq=', uniq[:10])
print('truth:', label)
run('eval')
run('train')
