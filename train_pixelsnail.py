import argparse
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GameIRSuperResolutionDataset
from pixelsnail import PixelSNAIL
from resnet_prior import ResNetPrior
from scheduler import CycleScheduler
from vqvae import VQVAE
import distributed as dist


def train(args, epoch, loader, model, vqvae, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    acc_sum = 0.0
    batch_count = 0

    for i, (lr_img, hr_img) in enumerate(loader):
        model.zero_grad()

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        with torch.no_grad():
            _, _, _, top, bottom = vqvae.encode(hr_img)
            _, _, _, lr_top, _ = vqvae.encode(lr_img)

        top = top.long()
        bottom = bottom.long()
        lr_top = lr_top.long()

        if args.hier == 'top':
            target = top
            if args.use_lr_condition:
                out, _ = model(top, condition=lr_top)
            else:
                out, _ = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        if dist.is_primary():
            lr = optimizer.param_groups[0]['lr']
            loss_sum += loss.item()
            acc_sum += accuracy.item()
            batch_count += 1

            loader.set_description(
                (
                    f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                    f'acc: {accuracy.item():.5f}; avg loss: {loss_sum / batch_count:.5f}; '
                    f'avg acc: {acc_sum / batch_count:.5f}; lr: {lr:.5f}'
                )
            )


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.distributed = dist.get_world_size() > 1

    dataset = GameIRSuperResolutionDataset(
        lr_dir=args.lr_path,
        hr_dir=args.hr_path,
        root=args.root,
        lr_res=args.lr_res,
        hr_res=args.hr_res,
        suffix=args.suffix,
        hr_patch_size=args.size,
        scale=args.scale,
        augment=True,
        patch_per_image=4,
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

    vqvae = VQVAE()
    vqvae_ckpt = torch.load(args.vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(vqvae_ckpt['model'] if 'model' in vqvae_ckpt else vqvae_ckpt)
    vqvae = vqvae.to(device)
    vqvae.eval()

    ckpt = {}

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)

    ModelClass = ResNetPrior if args.model_type == 'resnet' else PixelSNAIL

    if args.hier == 'top':
        model = ModelClass(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block if args.use_lr_condition else 0,
            cond_res_channel=args.n_res_channel if args.use_lr_condition else 0,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = ModelClass(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        if args.distributed:
            sampler.set_epoch(i)

        train(args, i, loader, model, vqvae, optimizer, scheduler, device)

        if dist.is_primary():
            model_state = model.module.state_dict() if args.distributed else model.state_dict()
            torch.save(
                {
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                    'epoch': i + 1,
                },
                f'checkpoint/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != 'win32' else 1) % 2 ** 14
    )
    parser.add_argument('--dist_url', default=f'tcp://127.0.0.1:{port}')

    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_lr_condition', action='store_true')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['pixelsnail', 'resnet'])
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--vqvae_ckpt', type=str, required=True)
    parser.add_argument('--lr_path', type=str, default=None)
    parser.add_argument('--hr_path', type=str, default=None)
    parser.add_argument('--root', type=str, default=None, help='GameIR dataset root (nested mode)')
    parser.add_argument('--lr_res', type=str, default='720p', help='LR resolution folder name')
    parser.add_argument('--hr_res', type=str, default='1440p', help='HR resolution folder name')
    parser.add_argument('--suffix', type=str, default='_rgb.png', help='Image filename suffix filter')

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
