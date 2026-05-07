import argparse
from PIL import Image, ImageDraw


def get_panel_info(img, panels=3, pad=2):
    full_w, full_h = img.size
    panel_w = (full_w - pad * (panels + 1)) // panels
    panel_h = full_h - pad * 2
    return pad, panel_w, panel_h


def extract_gt_panel(img, panels=3, pad=2):
    pad, panel_w, panel_h = get_panel_info(img, panels, pad)
    gt_x_offset = pad + (panels - 1) * (panel_w + pad)
    return img.crop((gt_x_offset, pad, gt_x_offset + panel_w, pad + panel_h))


LABELS_3 = ['Bicubic', 'SR', 'GT']
LABELS_4 = ['Bicubic', 'Base SR', 'LoRA SR', 'GT']


def do_crop(img, top, left, size, output, thickness, panels=3, pad=2):
    pad, panel_w, panel_h = get_panel_info(img, panels, pad)
    full_w, full_h = img.size
    labels = LABELS_4 if panels == 4 else LABELS_3

    print(f'Full image: {full_w}×{full_h}')
    print(f'{panels} panels, each: {panel_w}×{panel_h} (padding={pad}px)')
    print(f'Cropping {size}×{size} patch at ({top}, {left})')

    if top + size > panel_h or left + size > panel_w:
        print(f'ERROR: Crop extends beyond panel bounds ({panel_w}×{panel_h})')
        return

    crops = []
    for i in range(panels):
        x_offset = pad + i * (panel_w + pad)
        box = (
            x_offset + left,
            pad + top,
            x_offset + left + size,
            pad + top + size,
        )
        crop = img.crop(box)
        crops.append(crop)
        print(f'  {labels[i]}: cropped from ({box[0]}, {box[1]}) to ({box[2]}, {box[3]})')

    gap = 4
    result_w = size * panels + gap * (panels - 1)
    result = Image.new('RGB', (result_w, size), (255, 255, 255))
    for i, crop in enumerate(crops):
        result.paste(crop, (i * (size + gap), 0))

    result.save(output)
    print(f'\nSaved crop: {output}')

    gt_panel = extract_gt_panel(img, panels, pad)
    draw = ImageDraw.Draw(gt_panel)
    for t in range(thickness):
        draw.rectangle(
            [left - t, top - t, left + size + t, top + size + t],
            outline='red'
        )

    box_path = output.rsplit('.', 1)
    box_out = f'{box_path[0]}_boxed.png'
    gt_panel.save(box_out)
    print(f'Saved boxed: {box_out}')
    print(f'Key: {" | ".join(labels)}')


def interactive_mode(image_path, size, thickness, panels, pad):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img = Image.open(image_path)
    gt_panel = extract_gt_panel(img, panels, pad)
    _, panel_w, panel_h = get_panel_info(img, panels, pad)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(gt_panel)
    ax.set_title(f'Click to place a {size}×{size} crop (close window when done)')
    ax.axis('off')

    rect = None
    click_coords = {}

    def on_click(event):
        nonlocal rect
        if event.inaxes != ax:
            return

        cx, cy = int(event.xdata), int(event.ydata)
        left = max(0, min(cx - size // 2, panel_w - size))
        top = max(0, min(cy - size // 2, panel_h - size))

        click_coords['left'] = left
        click_coords['top'] = top

        if rect:
            rect.remove()
        rect = mpatches.Rectangle(
            (left, top), size, size,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f'Crop at top={top}, left={left} — click again or close window to confirm')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

    if 'left' not in click_coords:
        print('No click detected, exiting.')
        return

    left = click_coords['left']
    top = click_coords['top']
    print(f'\nSelected crop: top={top}, left={left}')

    base = image_path.rsplit('.', 1)
    output = f'{base[0]}_crop_{top}_{left}.png'

    do_crop(img, top, left, size, output, thickness, panels, pad)


def main():
    parser = argparse.ArgumentParser(description='Crop matching patches from comparison image')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the comparison image')
    parser.add_argument('--top', type=int, default=None,
                        help='Top pixel coordinate of the crop (within a single panel)')
    parser.add_argument('--left', type=int, default=None,
                        help='Left pixel coordinate of the crop (within a single panel)')
    parser.add_argument('--size', type=int, default=200,
                        help='Size of the square crop in pixels (default 200)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: adds _crop to input name)')
    parser.add_argument('--thickness', type=int, default=3,
                        help='Red box line thickness in pixels (default 3)')
    parser.add_argument('--interactive', action='store_true',
                        help='Click on the image to choose crop location')
    parser.add_argument('--panels', type=int, default=3,
                        help='Number of panels in the image (3 for eval_specific, 4 for eval_lora_comparison)')
    parser.add_argument('--padding', type=int, default=2,
                        help='Padding between panels in pixels (2 for eval_specific, 0 for eval_lora_comparison)')
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.image, args.size, args.thickness, args.panels, args.padding)
    else:
        if args.top is None or args.left is None:
            print('ERROR: --top and --left are required in non-interactive mode.')
            print('       Use --interactive to click and choose instead.')
            return

        img = Image.open(args.image)

        if args.output:
            out_path = args.output
        else:
            base = args.image.rsplit('.', 1)
            out_path = f'{base[0]}_crop_{args.top}_{args.left}.png'

        do_crop(img, args.top, args.left, args.size, out_path, args.thickness, args.panels, args.padding)


if __name__ == '__main__':
    main()
