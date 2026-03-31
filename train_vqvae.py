import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import utils

from tqdm import tqdm

from dataset import GameIRSuperResolutionDataset
from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist


def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (_, img) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    value_range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.distributed = dist.get_world_size() > 1

    dataset = GameIRSuperResolutionDataset(
        lr_dir=args.lr_path,
        hr_dir=args.hr_path,
        root=args.root,
        lr_res=args.lr_res,
        hr_res=args.hr_res,
        suffix=args.suffix,
        hr_patch_size=args.size,
        scale=2,
        augment=True,
    )
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_state = ckpt['model'] if 'model' in ckpt else ckpt
        if args.distributed:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']
        print(f"Resumed from epoch {start_epoch}")

    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(start_epoch, args.epoch):
        if args.distributed:
            sampler.set_epoch(i)

        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            state_dict = model.module.state_dict() if args.distributed else model.state_dict()
            torch.save(
                {'model': state_dict, 'optimizer': optimizer.state_dict(), 'epoch': i + 1},
                f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_path", type=str, default=None)
    parser.add_argument("--hr_path", type=str, default=None)
    parser.add_argument("--root", type=str, default=None, help="GameIR dataset root (nested mode)")
    parser.add_argument("--lr_res", type=str, default="720p", help="LR resolution folder name")
    parser.add_argument("--hr_res", type=str, default="1440p", help="HR resolution folder name")
    parser.add_argument("--suffix", type=str, default="_rgb.png", help="Image filename suffix filter")

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
