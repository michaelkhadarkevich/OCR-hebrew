import types
import torch
from pathlib import Path
from data import dataset
from model import HTR_VT
from utils import utils

line_name = Path('data/omer/train.ln').read_text(encoding='utf-8').splitlines()[1]
img_path = Path('data/omer/lines') / line_name
label = img_path.with_suffix('.txt').read_text(encoding='utf-8').strip()
labels = [label]
alph = dataset.get_alphabet(labels)
ralph = dict(zip(alph.values(), alph.keys()))
converter = utils.CTCLabelConverter(ralph.values())
img = dataset.get_images(str(img_path), 512, 64).transpose((2,0,1))
x = torch.tensor(img).unsqueeze(0).float()
text, lengths = converter.encode(labels)


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
        x = self.head(x)
        return x
    model.forward = types.MethodType(forward, model)


def run(name, blank_bias=None, no_final_norm=False, lr=5e-4, steps=150):
    model = HTR_VT.create_model(nb_cls=len(ralph)+1, img_size=[64,512])
    if no_final_norm:
        patch_forward_no_final_norm(model)
    if blank_bias is not None:
        with torch.no_grad():
            model.head.bias.zero_()
            model.head.bias[0] = blank_bias
    criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def decode():
        model.eval()
        with torch.no_grad():
            preds = model(x).float()
            sizes = torch.IntTensor([preds.size(1)])
            lp = preds.permute(1,0,2).log_softmax(2)
            arg = lp.argmax(2).transpose(1,0).contiguous().view(-1)
            pred = converter.decode(arg.cpu(), sizes)[0]
            blank = float(lp.exp()[...,0].mean())
            uniq = [(int(i), int((arg==i).sum())) for i in arg.unique()]
        model.train()
        return pred, blank, uniq

    print('\nRUN', name, 'target=', label)
    for step in range(1, steps+1):
        preds = model(x).float()
        sizes = torch.IntTensor([preds.size(1)])
        lp = preds.permute(1,0,2).log_softmax(2)
        loss = criterion(lp, text.cpu(), sizes, lengths.cpu())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step in (1, 10, 25, 50, 75, 100, 125, 150):
            pred, blank, uniq = decode()
            print(f'step={step} loss={float(loss.detach()):.4f} blank={blank:.4f} pred={pred!r} uniq={uniq[:8]}')

run('final_norm_on_blank_bias_-2', blank_bias=-2.0, no_final_norm=False)
run('final_norm_off_blank_bias_-2', blank_bias=-2.0, no_final_norm=True)
