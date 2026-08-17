import types
from pathlib import Path
import torch
from data import dataset
from model import HTR_VT
from utils import utils

ckpt_path = Path('output/omer_ariel_cpu_50_fixed_ctc/htrvt_omer_ariel_cpu.pth')
ckpt = torch.load(ckpt_path, map_location='cpu')

labels = []
items = []
for source, list_path, lines_dir in [('omer','data/omer/train.ln','data/omer/lines'),('ariel','data/ariel/train.ln','data/ariel/lines')]:
    for name in Path(list_path).read_text(encoding='utf-8').splitlines():
        p = Path(lines_dir) / name
        lab = p.with_suffix('.txt').read_text(encoding='utf-8').strip()
        labels.append(lab)
        items.append((source, p, lab))
alph = dataset.get_alphabet(labels)
ralph = dict(zip(alph.values(), alph.keys()))
converter = utils.CTCLabelConverter(ralph.values())

source, p, label = items[0]
img = dataset.get_images(str(p), 512, 64).transpose((2,0,1))
x = torch.tensor(img).unsqueeze(0).float()

def patch_forward_no_final_norm(model):
    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        x = self.layer_norm(x)
        x = self.patch_embed(x)
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)
    model.forward = types.MethodType(forward, model)

def decode(model):
    model.eval()
    with torch.no_grad():
        preds = model(x).float()
        sizes = torch.IntTensor([preds.size(1)])
        lp = preds.permute(1,0,2).log_softmax(2)
        arg = lp.argmax(2).transpose(1,0).contiguous().view(-1)
        pred = converter.decode(arg.cpu(), sizes)[0]
        blank = float(lp.exp()[...,0].mean())
        uniq = [(int(i), int((arg==i).sum())) for i in arg.unique()]
    return pred, blank, uniq

m1 = HTR_VT.create_model(nb_cls=ckpt['nb_cls'], img_size=[64,512])
m1.load_state_dict(ckpt['state_dict'])
m2 = HTR_VT.create_model(nb_cls=ckpt['nb_cls'], img_size=[64,512])
patch_forward_no_final_norm(m2)
m2.load_state_dict(ckpt['state_dict'])
print('truth:', label)
print('checkpoint keys:', sorted(ckpt.keys()))
print('original_forward:', decode(m1))
print('patched_forward:', decode(m2))
