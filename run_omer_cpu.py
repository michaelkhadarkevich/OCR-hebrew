import argparse
import csv
from pathlib import Path

import editdistance
import torch
from torch.utils.data import DataLoader

from data import dataset
from model import HTR_VT
from utils import utils


def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack([torch.from_numpy(img).float() for img in images], dim=0)
    return images, list(labels)


def decode_batch(model, loader, converter, device):
    model.eval()
    rows = []
    total_ed = 0
    total_len = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).float()
            preds_size = torch.IntTensor([preds.size(1)] * images.size(0))
            log_probs = preds.permute(1, 0, 2).log_softmax(2)
            _, preds_index = log_probs.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            pred_strs = converter.decode(preds_index.cpu(), preds_size)
            for pred, label in zip(pred_strs, labels):
                ed = editdistance.eval(pred, label)
                total_ed += ed
                total_len += len(label)
                rows.append({'truth': label, 'prediction': pred, 'edit_distance': ed})
    cer = total_ed / max(1, total_len)
    return cer, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/omer')
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out-dir', default='output/omer_cpu')
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device('cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_ds = dataset.myLoadDS(str(data_dir / 'train.ln'), str(data_dir / 'lines') + '/', [512, 64])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    eval_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    nb_cls = len(train_ds.ralph) + 1
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=[64, 512]).to(device)
    converter = utils.CTCLabelConverter(train_ds.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    before_cer, _ = decode_batch(model, eval_loader, converter, device)
    print('data_dir={} dataset={} alphabet={} nb_cls={}'.format(args.data_dir, len(train_ds), len(train_ds.ralph), nb_cls))
    print('initial_CER={:.4f}'.format(before_cer))

    iterator = iter(train_loader)
    model.train()
    for step in range(1, args.iters + 1):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)

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

    after_cer, rows = decode_batch(model, eval_loader, converter, device)
    print('final_CER={:.4f}'.format(after_cer))

    with open(out_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['truth', 'prediction', 'edit_distance'])
        writer.writeheader()
        writer.writerows(rows)

    torch.save({
        'state_dict': model.state_dict(),
        'alphabet': list(train_ds.ralph.values()),
        'nb_cls': nb_cls,
        'img_size': [512, 64],
        'data_dir': args.data_dir,
        'iters': args.iters,
        'final_cer': after_cer,
    }, out_dir / 'omer_htrvt_cpu_smoke.pth')
    print('saved', out_dir / 'omer_htrvt_cpu_smoke.pth')
    print('saved', out_dir / 'predictions.csv')


if __name__ == '__main__':
    main()
