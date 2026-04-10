from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont, ImageOps

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "images" / "spanish_uppercase"
PROFILE_X_DIR = BASE_DIR / "data" / "profiles" / "x"
PROFILE_Y_DIR = BASE_DIR / "data" / "profiles" / "y"
FEATURE_CSV = BASE_DIR / "data" / "features" / "spanish_uppercase_features.csv"

ALPHABET = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
FONT_CANDIDATES = ["times.ttf", "times new roman.ttf", "arial.ttf", "calibri.ttf"]
CANVAS_SIZE = (600, 600)
FONT_SIZE = 320


def ensure_paths() -> None:
    """Make sure all directories exist before writing files."""
    for path in (IMAGE_DIR, PROFILE_X_DIR, PROFILE_Y_DIR, FEATURE_CSV.parent):
        path.mkdir(parents=True, exist_ok=True)


def load_font(size: int = FONT_SIZE) -> ImageFont.ImageFont:
    """Try several common fonts and fall back to the default if none are available."""
    for font_name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_letter(letter: str, font: ImageFont.ImageFont) -> Image.Image:
    """
    Render a single letter on a white canvas and crop excess whitespace.

    The letter is rendered in black so the inverted image helps compute the tight
    bounding box.
    """
    canvas = Image.new("L", CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(canvas)
    text_bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (CANVAS_SIZE[0] - text_width) / 2 - text_bbox[0]
    y = (CANVAS_SIZE[1] - text_height) / 2 - text_bbox[1]
    draw.text((x, y), letter, font=font, fill=0)
    bbox = ImageOps.invert(canvas).getbbox()
    if bbox is None:
        return canvas
    pad = max(8, FONT_SIZE // 16)
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(canvas.width, bbox[2] + pad)
    bottom = min(canvas.height, bbox[3] + pad)
    return canvas.crop((left, top, right, bottom))


def generate_letters() -> None:
    """Create one PNG per letter using the chosen font."""
    font = load_font()
    for letter in ALPHABET:
        path = IMAGE_DIR / f"{letter}.png"
        image = render_letter(letter, font)
        image.save(path, optimize=True)


def normalize_mask(image: Image.Image) -> np.ndarray:
    """Convert an image to a binary mask where True denotes ink."""
    arr = np.array(image.convert("L"), dtype=np.uint8)
    return arr < 128


def quadrant_masks(mask: np.ndarray) -> Sequence[np.ndarray]:
    """Split the mask into four quarters (TL, TR, BL, BR)."""
    h, w = mask.shape
    mid_h = h // 2
    mid_w = w // 2
    TL = mask[:mid_h, :mid_w]
    TR = mask[:mid_h, mid_w:]
    BL = mask[mid_h:, :mid_w]
    BR = mask[mid_h:, mid_w:]
    return (TL, TR, BL, BR)


def weighted_center(mask: np.ndarray) -> tuple[float, float]:
    """Return (x, y) coordinates of the centroid in pixel space."""
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)
    mass = mask.sum()
    if mass == 0:
        return float(w) / 2, float(h) / 2
    sy = (mask * ys).sum()
    sx = (mask * xs).sum()
    return sx / mass, sy / mass


def axial_moments(mask: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Compute the inertial moment around horizontal and vertical axes through the centroid."""
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)
    dy2 = (ys - cy) ** 2
    dx2 = (xs - cx) ** 2
    moment_h = (mask * dy2).sum()
    moment_v = (mask * dx2).sum()
    return moment_h, moment_v


def save_profile(series: Iterable[int], target: Path, letter: str, axis_label: str) -> None:
    """Render a bar chart for a profile (per-column or per-row counts)."""
    values = np.array(list(series), dtype=int)
    if values.size == 0:
        return
    indices = np.arange(values.size)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(indices, values, width=0.9, color="#2d2d2d")
    ax.set_xlabel("Column index" if axis_label == "x" else "Row index")
    ax.set_ylabel("Black pixel count")
    ax.set_title(f"{letter}: {axis_label.upper()} profile")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    if values.size <= 30:
        ticks = indices
    else:
        step = max(1, values.size // 30)
        ticks = list(range(0, values.size, step))
        if ticks[-1] != values.size - 1:
            ticks.append(values.size - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    plt.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def compute_features() -> list[dict[str, float | int]]:
    """Walk through each letter image, compute scalar descriptors, and emit rows."""
    rows: list[dict[str, float | int]] = []
    for letter in ALPHABET:
        path = IMAGE_DIR / f"{letter}.png"
        if not path.exists():
            continue
        mask = normalize_mask(Image.open(path))
        total_mass = int(mask.sum())
        if total_mass == 0:
            continue
        TL, TR, BL, BR = quadrant_masks(mask)
        quadrants = (TL, TR, BL, BR)
        masses = [int(q.sum()) for q in quadrants]
        areas = [q.size for q in quadrants]
        densities = [
            float(m) / a if a else 0.0 for m, a in zip(masses, areas)
        ]
        cx, cy = weighted_center(mask)
        inertia_h, inertia_v = axial_moments(mask, cx, cy)
        inertia_h_norm = inertia_h / total_mass
        inertia_v_norm = inertia_v / total_mass
        h, w = mask.shape
        # Save profiles
        save_profile(
            mask.sum(axis=0),
            PROFILE_X_DIR / f"{letter}.png",
            letter,
            "x",
        )
        save_profile(
            mask.sum(axis=1),
            PROFILE_Y_DIR / f"{letter}.png",
            letter,
            "y",
        )
        center_x_norm = cx / (w - 1) if w > 1 else 0.0
        center_y_norm = cy / (h - 1) if h > 1 else 0.0
        rows.append(
            {
                "letter": letter,
                "total_mass": total_mass,
                "tl_mass": masses[0],
                "tr_mass": masses[1],
                "bl_mass": masses[2],
                "br_mass": masses[3],
                "tl_density": densities[0],
                "tr_density": densities[1],
                "bl_density": densities[2],
                "br_density": densities[3],
                "center_x": cx,
                "center_y": cy,
                "center_x_norm": center_x_norm,
                "center_y_norm": center_y_norm,
                "moment_h": inertia_h,
                "moment_v": inertia_v,
                "moment_h_norm": inertia_h_norm,
                "moment_v_norm": inertia_v_norm,
                "width": w,
                "height": h,
            }
        )
    return rows


def write_csv(rows: Sequence[dict[str, float | int]]) -> None:
    """Serialize computed descriptors into a semicolon-delimited file."""
    headers = [
        "letter",
        "total_mass",
        "tl_mass",
        "tr_mass",
        "bl_mass",
        "br_mass",
        "tl_density",
        "tr_density",
        "bl_density",
        "br_density",
        "center_x",
        "center_y",
        "center_x_norm",
        "center_y_norm",
        "moment_h",
        "moment_v",
        "moment_h_norm",
        "moment_v_norm",
        "width",
        "height",
    ]
    with FEATURE_CSV.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter=";")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row["letter"],
                    row["total_mass"],
                    row["tl_mass"],
                    row["tr_mass"],
                    row["bl_mass"],
                    row["br_mass"],
                    f"{row['tl_density']:.5f}",
                    f"{row['tr_density']:.5f}",
                    f"{row['bl_density']:.5f}",
                    f"{row['br_density']:.5f}",
                    f"{row['center_x']:.2f}",
                    f"{row['center_y']:.2f}",
                    f"{row['center_x_norm']:.5f}",
                    f"{row['center_y_norm']:.5f}",
                    f"{row['moment_h']:.2f}",
                    f"{row['moment_v']:.2f}",
                    f"{row['moment_h_norm']:.5f}",
                    f"{row['moment_v_norm']:.5f}",
                    row["width"],
                    row["height"],
                ]
            )


def main() -> None:
    """Generate glyphs, compute the requested features, and export everything."""
    ensure_paths()
    generate_letters()
    rows = compute_features()
    write_csv(rows)


if __name__ == "__main__":
    main()
