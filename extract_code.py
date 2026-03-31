import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GameIRSuperResolutionDataset
from vqvae import VQVAE


def extract_codes(loader, model, device):
    rows = []

    with torch.no_grad():
        for lr_img, hr_img, filename in tqdm(loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            _, _, _, hr_top, hr_bottom = model.encode(hr_img)
            _, _, _, lr_top, _ = model.encode(lr_img)

            for file, top, bottom, top_lr in zip(filename, hr_top, hr_bottom, lr_top):
                rows.append(
                    {
                        'filename': file,
                        'top': top.cpu(),
                        'bottom': bottom.cpu(),
                        'lr_top': top_lr.cpu(),
                    }
                )

    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--lr_path', type=str, default=None)
    parser.add_argument('--hr_path', type=str, default=None)
    parser.add_argument('--root', type=str, default=None, help='GameIR dataset root (nested mode)')
    parser.add_argument('--lr_res', type=str, default='720p', help='LR resolution folder name')
    parser.add_argument('--hr_res', type=str, default='1440p', help='HR resolution folder name')
    parser.add_argument('--suffix', type=str, default='_rgb.png', help='Image filename suffix filter')
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()

    device = (
        'mps'
        if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    dataset = GameIRSuperResolutionDataset(
        lr_dir=args.lr_path,
        hr_dir=args.hr_path,
        root=args.root,
        lr_res=args.lr_res,
        hr_res=args.hr_res,
        suffix=args.suffix,
        hr_patch_size=args.size,
        scale=args.scale,
        augment=False,
        return_name=True,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = VQVAE()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model = model.to(device)
    model.eval()

    rows = extract_codes(loader, model, device)
    torch.save(rows, args.out)
    print(f'Saved {len(rows)} code rows to {args.out}')
