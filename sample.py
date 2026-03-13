import argparse
import os
import pickle

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


@torch.no_grad()
def load_lr_top_condition(vqvae, lr_image_path, hr_size, scale, batch, device):
    lr_size = hr_size // scale
    lr_image = Image.open(lr_image_path).convert('RGB')
    lr_image = TF.center_crop(lr_image, [lr_size, lr_size])
    lr_tensor = TF.normalize(
        TF.to_tensor(lr_image), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    ).unsqueeze(0)
    lr_tensor = lr_tensor.repeat(batch, 1, 1, 1).to(device)

    _, _, _, lr_top, _ = vqvae.encode(lr_tensor)
    return lr_top.long()


def load_model(model, checkpoint, device):
    ckpt_path = os.path.join('checkpoint', checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    except pickle.UnpicklingError:
        # Backward-compatible path for trusted local checkpoints that stored argparse.Namespace.
        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    ckpt_args = ckpt.get('args') if isinstance(ckpt, dict) else None

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        use_lr_condition = bool(cfg_get(ckpt_args, 'use_lr_condition', False))
        model = PixelSNAIL(
            [32, 32],
            512,
            cfg_get(ckpt_args, 'channel'),
            5,
            4,
            cfg_get(ckpt_args, 'n_res_block'),
            cfg_get(ckpt_args, 'n_res_channel'),
            dropout=cfg_get(ckpt_args, 'dropout'),
            n_cond_res_block=cfg_get(ckpt_args, 'n_cond_res_block', 0)
            if use_lr_condition
            else 0,
            cond_res_channel=cfg_get(ckpt_args, 'n_res_channel', 0)
            if use_lr_condition
            else 0,
            n_out_res_block=cfg_get(ckpt_args, 'n_out_res_block', 0),
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            cfg_get(ckpt_args, 'channel'),
            5,
            4,
            cfg_get(ckpt_args, 'n_res_block'),
            cfg_get(ckpt_args, 'n_res_channel'),
            attention=False,
            dropout=cfg_get(ckpt_args, 'dropout'),
            n_cond_res_block=cfg_get(ckpt_args, 'n_cond_res_block'),
            cond_res_channel=cfg_get(ckpt_args, 'n_res_channel'),
        )
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model, ckpt_args


if __name__ == '__main__':
    device = (
        'mps'
        if torch.backends.mps.is_available()
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--lr_image', type=str)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('filename', type=str)

    args = parser.parse_args()

    model_vqvae, _ = load_model('vqvae', args.vqvae, device)
    model_top, top_args = load_model('pixelsnail_top', args.top, device)
    model_bottom, _ = load_model('pixelsnail_bottom', args.bottom, device)

    top_condition = None
    top_is_lr_conditioned = (
        top_args is not None
        and bool(cfg_get(top_args, 'use_lr_condition', False))
    )
    if top_is_lr_conditioned:
        if args.lr_image is None:
            raise ValueError(
                'Top checkpoint was trained with LR conditioning. Provide --lr_image.'
            )
        top_condition = load_lr_top_condition(
            model_vqvae, args.lr_image, args.size, args.scale, args.batch, device
        )

    top_sample = sample_model(
        model_top, device, args.batch, [32, 32], args.temp, condition=top_condition
    )
    bottom_sample = sample_model(
        model_bottom, device, args.batch, [64, 64], args.temp, condition=top_sample
    )

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, args.filename, normalize=True, value_range=(-1, 1))
