import torch
from pathlib import Path
from data import dataset
from model import HTR_VT
from utils import utils

# Use one short sample so a healthy setup should overfit quickly.
line_name = Path('data/omer/train.ln').read_text(encoding='utf-8').splitlines()[1]
img_path = Path('data/omer/lines') / line_name
label = img_path.with_suffix('.txt').read_text(encoding='utf-8').strip()
labels = [label]
alph = dataset.get_alphabet(labels)
ralph = dict(zip(alph.values(), alph.keys()))
converter = utils.CTCLabelConverter(ralph.values())
chars = ['[blank]'] + list(ralph.values())

img = dataset.get_images(str(img_path), 512, 64).transpose((2,0,1))
x = torch.tensor(img).unsqueeze(0).float()
text, lengths = converter.encode(labels)

model = HTR_VT.create_model(nb_cls=len(ralph)+1, img_size=[64,512])
criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def decode():
    model.eval()
    with torch.no_grad():
        preds = model(x).float()
        preds_size = torch.IntTensor([preds.size(1)])
        log_probs = preds.permute(1,0,2).log_softmax(2)
        arg = log_probs.argmax(2).transpose(1,0).contiguous().view(-1)
        pred = converter.decode(arg.cpu(), preds_size)[0]
        blank_prob = float(log_probs.exp()[...,0].mean())
        uniq = [(int(i), int((arg==i).sum())) for i in arg.unique()]
    model.train()
    return pred, blank_prob, uniq

print('target:', label)
for step in range(1, 101):
    preds = model(x).float()
    preds_size = torch.IntTensor([preds.size(1)])
    log_probs = preds.permute(1,0,2).log_softmax(2)
    loss = criterion(log_probs, text.cpu(), preds_size, lengths.cpu())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if step == 1 or step % 10 == 0:
        pred, bp, uniq = decode()
        print(f'step={step} loss={float(loss):.4f} blank_prob={bp:.4f} pred={pred!r} uniq={uniq[:8]}')
