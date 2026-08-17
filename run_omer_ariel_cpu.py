import argparse
import csv
import types
from pathlib import Path

import editdistance
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data import dataset
from model import HTR_VT
from utils import utils


class MultiLineDataset(Dataset):
    def __init__(self, specs, img_size=(512, 64)):
        self.items = []
        self.img_size = list(img_size)
        for name, list_path, lines_dir in specs:
            list_path = Path(list_path)
            lines_dir = Path(lines_dir)
            for line in list_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                img = lines_dir / line
                txt = img.with_suffix('.txt')
                label = txt.read_text(encoding='utf-8').strip()
                self.items.append((name, str(img), label))
        labels = [label for _, _, label in self.items]
        alph = dataset.get_alphabet(labels)
        self.ralph = dict(zip(alph.values(), alph.keys()))
        self.alph = alph

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        source, img_path, label = self.items[index]
        image = dataset.get_images(img_path, self.img_size[0], self.img_size[1])
        image = image.transpose((2, 0, 1))
        return image, label, source, img_path


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


def collate(batch):
    images, labels, sources, paths = zip(*batch)
    images = torch.stack([torch.from_numpy(img).float() for img in images], dim=0)
    return images, list(labels), list(sources), list(paths)


def decode_batch(model, loader, converter, device):
    model.eval()
    rows = []
    total_ed = 0
    total_len = 0
    with torch.no_grad():
        for images, labels, sources, paths in loader:
            images = images.to(device)
            preds = model(images).float()
            preds_size = torch.IntTensor([preds.size(1)] * images.size(0))
            log_probs = preds.permute(1, 0, 2).log_softmax(2)
            _, preds_index = log_probs.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            pred_strs = converter.decode(preds_index.cpu(), preds_size)
            for source, path, pred, label in zip(sources, paths, pred_strs, labels):
                ed = editdistance.eval(pred, label)
                total_ed += ed
                total_len += len(label)
                rows.append({'source': source, 'image': path, 'truth': label, 'prediction': pred, 'edit_distance': ed})
    return total_ed / max(1, total_len), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--out-dir', default='output/omer_ariel_cpu_50')
    parser.add_argument('--blank-bias', type=float, default=None)
    parser.add_argument('--no-final-logit-norm', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device('cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ('omer', 'data/omer/train.ln', 'data/omer/lines'),
        ('ariel', 'data/ariel/train.ln', 'data/ariel/lines'),
    ]
    train_ds = MultiLineDataset(specs, img_size=(512, 64))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    eval_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    nb_cls = len(train_ds.ralph) + 1
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=[64, 512]).to(device)
    if args.no_final_logit_norm:
        patch_forward_no_final_norm(model)
        print('no_final_logit_norm=True')
    if args.blank_bias is not None:
        with torch.no_grad():
            model.head.bias.zero_()
            model.head.bias[0] = args.blank_bias
        print('blank_bias={}'.format(args.blank_bias))
    converter = utils.CTCLabelConverter(train_ds.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    source_counts = {}
    for source, _, _ in train_ds.items:
        source_counts[source] = source_counts.get(source, 0) + 1
    print('dataset={} source_counts={} alphabet={} nb_cls={}'.format(len(train_ds), source_counts, len(train_ds.ralph), nb_cls))
    initial_cer, _ = decode_batch(model, eval_loader, converter, device)
    print('initial_CER={:.4f}'.format(initial_cer))

    iterator = iter(train_loader)
    model.train()
    for step in range(1, args.iters + 1):
        try:
            images, labels, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels, _, _ = next(iterator)
        images = images.to(device)
        text, lengths = converter.encode(labels)
        preds = model(images).float()
        preds_size = torch.IntTensor([preds.size(1)] * images.size(0))
        log_probs = preds.permute(1, 0, 2).log_softmax(2)
        loss = criterion(log_probs, text.cpu(), preds_size, lengths.cpu())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 5 == 0 or step == args.iters:
            print('iter={} loss={:.4f}'.format(step, float(loss.item())))

    final_cer, rows = decode_batch(model, eval_loader, converter, device)
    print('final_CER={:.4f}'.format(final_cer))
    with open(out_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'image', 'truth', 'prediction', 'edit_distance'])
        writer.writeheader()
        writer.writerows(rows)
    torch.save({
        'state_dict': model.state_dict(),
        'alphabet': list(train_ds.ralph.values()),
        'nb_cls': nb_cls,
        'source_counts': source_counts,
        'iters': args.iters,
        'final_cer': final_cer,
    }, out_dir / 'htrvt_omer_ariel_cpu.pth')
    print('saved', out_dir / 'htrvt_omer_ariel_cpu.pth')
    print('saved', out_dir / 'predictions.csv')


if __name__ == '__main__':
    main()
