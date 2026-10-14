"""
Classify your own images with the trained model.

Run from the repository root:
    python -m src.predict path/to/photo.jpg
    python -m src.predict photo1.jpg photo2.png photo3.jpg

For each image it prints the model's top 3 guesses and saves a preview
(results/prediction_<name>.png) showing your image next to the 28x28 version
the model actually sees.

Tips for good results: one piece of clothing, photographed flat or on a
hanger, on a plain background that contrasts with the garment, filling most
of the photo. Fashion-MNIST only contains clean product shots, so busy
backgrounds, people wearing the clothes, or several items at once will
confuse the model.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from src import config
from src.model import FashionCNN
from src.utils import SURFACE, TEXT_COLOR, get_device, load_checkpoint


def to_fashion_mnist_format(path: Path) -> np.ndarray:
    """
    Turn an ordinary photo into a 28x28 image that looks like Fashion-MNIST:
    grayscale, garment bright on a black background, cropped to the garment,
    and centred with its longest side filling the 28 pixels.

    Returns a float array of shape (28, 28) with values in [0, 1].
    """
    image = ImageOps.exif_transpose(Image.open(path))   # respect phone rotation
    pixels = np.asarray(image.convert("L"), dtype=np.float32)

    # Fashion-MNIST garments are light on a dark background. Most photos are
    # the opposite (dark garment, light background), so invert those.
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if np.median(border) > 127:
        pixels = 255 - pixels
        border = 255 - border

    # Make the background pure black, so only the garment remains.
    background = np.median(border)
    pixels = np.clip(pixels - background, 0, None)

    # Crop to the garment. Blurring first averages out camera noise and texture,
    # so stray speckles in the background do not stretch the crop box.
    blur_radius = max(1, min(pixels.shape) / 150)
    smooth = np.asarray(
        Image.fromarray(pixels.astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_radius)),
        dtype=np.float32,
    )
    bright = np.percentile(smooth, 99)
    mask = smooth > max(15.0, 0.2 * bright)
    if mask.any():
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        pixels = pixels[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Stretch the brightness so the garment uses the full 0-255 range. The 99th
    # percentile (not the max) keeps a few very bright pixels from dimming the rest.
    top = np.percentile(pixels, 99)
    if top > 0:
        pixels = np.clip(pixels / top * 255, 0, 255)

    # Shrink so the longest side is 28 pixels, then centre it on a black 28x28 canvas.
    crop = Image.fromarray(pixels.astype(np.uint8))
    scale = 28 / max(crop.size)
    new_size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(new_size, Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))

    return np.asarray(canvas, dtype=np.float32) / 255.0


def predict(model, image_28: np.ndarray, device) -> torch.Tensor:
    """Return the probability of each of the 10 classes for one 28x28 image."""
    x = (torch.from_numpy(image_28) - config.MEAN[0]) / config.STD[0]   # same normalisation as training
    x = x.unsqueeze(0).unsqueeze(0).to(device)    # shape [1, 1, 28, 28]: a batch of one image
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return torch.softmax(logits, dim=1)[0].cpu()  # turn the 10 scores into probabilities


def save_preview(path: Path, image_28: np.ndarray, probs: torch.Tensor, out_path: Path) -> None:
    """Save the original photo, the model's 28x28 view and the top guess side by side."""
    original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    top = int(probs.argmax())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.6), facecolor=SURFACE)
    ax1.imshow(original)
    ax1.set_title("Your image", fontsize=10, color=TEXT_COLOR)
    ax2.imshow(image_28, cmap="gray", vmin=0, vmax=1)
    ax2.set_title(f"What the model sees\n→ {config.CLASSES[top]} ({probs[top]:.0%})",
                  fontsize=10, color=TEXT_COLOR)
    for ax in (ax1, ax2):
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main(paths: list) -> None:
    if not paths:
        print(__doc__)
        sys.exit(1)

    device = get_device()
    model = FashionCNN(num_classes=config.NUM_CLASSES).to(device)
    load_checkpoint(model, config.MODEL_PATH, device)

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"\n{path}: file not found")
            continue

        image_28 = to_fashion_mnist_format(path)
        probs = predict(model, image_28, device)

        print(f"\n{path.name}")
        for rank, idx in enumerate(probs.argsort(descending=True)[:3], start=1):
            print(f"  {rank}. {config.CLASSES[idx]:<12} {probs[idx]:6.1%}")

        preview = config.RESULTS_DIR / f"prediction_{path.stem}.png"
        save_preview(path, image_28, probs, preview)
        print(f"  Preview saved to {preview.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main(sys.argv[1:])
