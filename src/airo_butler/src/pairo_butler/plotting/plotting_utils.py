import os
from typing import List, Optional, Tuple
import PIL
from PIL import ImageDraw, ImageFont, Image
from matplotlib import pyplot as plt
import numpy as np
import rospy as ros
from pairo_butler.utils.tools import UGENT, pyout
import rospkg


def add_info_to_image(
    image: PIL.Image,
    title: Optional[str] = None,
    info_position: str = "NW",
    info_box_padding: int = 10,
    **kwargs,
) -> PIL.Image:
    assert info_position in ("NW", "NE", "SE", "SW")

    draw = ImageDraw.Draw(image)

    text = format_info(title, **kwargs)
    font = get_monospace_font()
    text_width = max([draw.textlength(line, font=font) for line in text.split("\n")])
    text_height = font.size * len(text.split("\n"))
    box_width = text_width + 2 * info_box_padding
    box_height = text_height + 2 * info_box_padding

    box = compute_box(
        img_size=image.size,
        box_size=(box_width, box_height),
        position=info_position,
    )

    draw.rectangle(box, fill=UGENT.BLUE, outline=UGENT.WHITE)

    text_position = (box[0] + info_box_padding, box[1] + info_box_padding)
    draw.multiline_text(text_position, text, fill=UGENT.WHITE, font=font)

    return image


def compute_box(img_size: Tuple[int, int], box_size: Tuple[int, int], position: str):
    assert position in ("NW", "NE", "SE", "SW")

    img_W, img_H = img_size
    box_W, box_H = box_size

    if position == "NW":
        box = (0, 0, box_W, box_H)
    elif position == "NE":
        box = (img_W - box_W, 0, img_W, box_H)
    elif position == "SE":
        box = (img_W - box_W, img_H - box_H, img_W, img_H)
    else:
        box = (0, img_H - box_H, box_W, img_H)

    return box


def format_info(title: Optional[str] = None, **kwargs) -> str:
    key_len = max(len(key) for key in kwargs.keys())
    val_len = max(len(str(val)) for val in kwargs.values())

    text = "" if title is None else f"{title}.\n"
    for key, val in kwargs.items():
        text += "\n" if len(text) > 0 else ""
        text += (key.replace("_", " ") + ":").ljust(key_len + 2)
        text += (val).rjust(val_len)

    return text


def get_monospace_font(size: int = 14):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("airo_butler")
    font_path = os.path.join(package_path, "res", "fonts", "UbuntuMono-B.ttf")
    font = ImageFont.truetype(font_path, size=size)
    return font


def compute_fps_and_latency(timestamps: List[ros.Time]):
    now = ros.Time.now()

    if len(timestamps) == 0:
        return 0, 0

    while len(timestamps) and timestamps[0] < now - ros.Duration(secs=1):
        timestamps.pop(0)

    fps = len(timestamps)
    try:
        latency_ms: float = int((now - timestamps[-1]).to_sec() * 1000)
    except IndexError:
        latency_ms = "1000+"

    return fps, latency_ms


def overlay_heatmap_on_image(image: PIL.Image.Image, heatmap: np.ndarray):

    # Normalize heatmap to [0, 1] range based on its min and max values
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    # heatmap_normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-6)
    heatmap_normalized = heatmap / 255

    colormap = plt.get_cmap("viridis")
    heatmap_colored = colormap(heatmap_normalized)

    # Convert to PIL image and ensure same size as original
    heatmap_image = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).convert(
        "RGBA"
    )
    heatmap_image.putalpha(100)

    # Overlay the heatmap on the image
    overlay_image = Image.new("RGBA", image.size)
    overlay_image = Image.alpha_composite(overlay_image, image.convert("RGBA"))
    overlay_image = Image.alpha_composite(overlay_image, heatmap_image)

    # Convert to rgb
    background = Image.new("RGB", overlay_image.size, (255, 255, 255))
    rgb_image = Image.alpha_composite(
        background.convert("RGBA"), overlay_image
    ).convert("RGB")

    return rgb_image
