import os
from typing import Optional, Tuple
import PIL
from PIL import ImageDraw, ImageFont
from pairo_butler.utils.tools import UGENT, pyout


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
    import rospkg

    rospack = rospkg.RosPack()
    package_path = rospack.get_path("airo_butler")
    font_path = os.path.join(package_path, "res", "fonts", "UbuntuMono-B.ttf")
    font = ImageFont.truetype(font_path, size=size)
    return font
