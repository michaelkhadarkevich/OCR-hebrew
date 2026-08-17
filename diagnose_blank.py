import torch
from pathlib import Path
from data import dataset
from model import HTR_VT
from utils import utils

ckpt_path = Path('/home/maxim/HTR-VT/output/omer_ariel_cpu_50/htrvt_omer_ariel_cpu.pth')
ckpt = torch.load(ckpt_path, map_location='cpu')
train_ds = dataset.myLoadDS('data/omer/train.ln', 'data/omer/lines/', [512, 64])
# rebuild combined alphabet like run_omer_ariel_cpu.py did
labels=[]
items=[]
for source, list_path, lines_dir in [('omer','data/omer/train.ln','data/omer/lines'),('ariel','data/ariel/train.ln','data/ariel/lines')]:
    for name in Path(list_path).read_text(encoding='utf-8').splitlines():
        p=Path(lines_dir)/name
        lab=p.with_suffix('.txt').read_text(encoding='utf-8').strip()
        labels.append(lab); items.append((source,p,lab))
alph = dataset.get_alphabet(labels)
ralph = dict(zip(alph.values(), alph.keys()))
chars = ['[blank]'] + list(ralph.values())
model = HTR_VT.create_model(nb_cls=ckpt['nb_cls'], img_size=[64,512])
model.load_state_dict(ckpt['state_dict'])
model.eval()
source,p,label=items[0]
img = dataset.get_images(str(p),512,64).transpose((2,0,1))
x=torch.tensor(img).unsqueeze(0).float()
with torch.no_grad():
    logits=model(x)
    probs=logits.softmax(-1)
    arg=logits.argmax(-1)[0]
print('label:', label)
print('logits shape:', tuple(logits.shape))
print('unique argmax ids/counts:', sorted([(int(i), int((arg==i).sum())) for i in arg.unique()], key=lambda x:-x[1])[:20])
print('first 40 argmax ids:', arg[:40].tolist())
print('first 40 chars:', ''.join(chars[i] if i < len(chars) and i != 0 else '_' for i in arg[:40].tolist()))
mean_probs=probs[0].mean(0)
top=torch.topk(mean_probs, 10)
print('top mean prob classes:')
for val, idx in zip(top.values.tolist(), top.indices.tolist()):
    print(idx, repr(chars[idx] if idx < len(chars) else '?'), val)
blank_mean=float(mean_probs[0])
print('blank mean prob:', blank_mean)
PY
