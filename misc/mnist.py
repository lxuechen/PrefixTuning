import argparse
import sys

import torch
from torch import nn, optim
import torch.nn.functional as F

from gpt2 import blocks
from lxuechen_utils import utils
from privacy_utils import privacy_engine


def create_model_and_optimizer(dx=784, dh=200, dy=10, rank=10):
    if args.low_rank:
        model = nn.Sequential(
            nn.Flatten(),
            blocks.LrkLinear(dx, dh, rank=rank),
            nn.Tanh(),
            blocks.LrkLinear(dh, dh, rank=rank),
            nn.Tanh(),
            nn.Linear(dh, dy)
        )
        params = []
        for name, param in model.named_parameters():
            if 'left' not in name and 'right' not in name:
                params.append(param)
    else:
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dx, dh),
            nn.Tanh(),
            nn.Linear(dh, dh),
            nn.Tanh(),
            nn.Linear(dh, dy),
        )
        params = list(model.parameters())
    if args.optimizer == "adam":
        optimizer = optim.Adam(params=params, lr=args.lr)
    else:
        optimizer = optim.SGD(params=params, lr=args.lr, momentum=args.momentum)

    model.to(device)
    return model, optimizer


def train(model, optimizer, epochs, global_step=0):
    for epoch in range(epochs):
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)

            if args.low_rank:
                blocks.decompose_weight(module=model)

            model.train()
            y = model(x)
            loss = F.cross_entropy(y, t)
            loss.backward()

            if args.low_rank:
                blocks.create_gradient(module=model)
                blocks.restore_weight(module=model)

            optimizer.step()

            global_step += 1

        test_xent, test_zeon = evaluate(model, test_loader)
        print(f'epoch {epoch}, global_step {global_step}, test_xent {test_xent:.4f}, test_zeon: {test_zeon:.4f}')


@torch.no_grad()
def evaluate(model, loader, eval_iters=sys.maxsize):
    xent, zeon = [], []
    for i, (x, y) in enumerate(loader):
        model.eval()
        x, y = tuple(t.to(device) for t in (x, y))
        y_hat = model(x)
        this_xent = F.cross_entropy(y_hat, y, reduction="none")
        this_zeon = torch.eq(y_hat.argmax(dim=1), y).to(torch.get_default_dtype())
        zeon.append(this_zeon)
        xent.append(this_xent)
        if i >= eval_iters:
            break
    return tuple(torch.cat(lst, dim=0).mean(dim=0).item() for lst in (xent, zeon))


def main():
    model, optimizer = create_model_and_optimizer()
    pe = privacy_engine.PrivacyEngine(
        module=model,
        max_grad_norm=args.max_grad_norm,
        noise_multiplier=args.noise_multiplier,
        batch_size=args.train_batch_size,
        sample_size=60000,
    )
    pe.attach(optimizer=optimizer)
    train(model=model, optimizer=optimizer, epochs=args.epochs)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ap = argparse.ArgumentParser()
    ap.add_argument('--low_rank', type=utils.str2bool, default=False, const=True, nargs="?")
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--noise_multiplier', type=float, default=1)
    ap.add_argument('--train_batch_size', type=int, default=500)
    ap.add_argument('--test_batch_size', type=int, default=500)
    ap.add_argument('--max_grad_norm', type=float, default=0.1)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--optimizer', type=str, default="sgd")
    args = ap.parse_args()

    train_loader, test_loader = utils.get_loader(
        root="/nlp/scr/lxuechen/data",
        data_name="mnist", data_aug=False, task="classification",
        train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
        num_workers=0
    )

    # python -m misc.mnist --low_rank
    main()
