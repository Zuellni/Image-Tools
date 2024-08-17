import random
from argparse import ArgumentParser
from pathlib import Path
from string import punctuation
from uuid import uuid4
from warnings import simplefilter

simplefilter("ignore")

parser = ArgumentParser()
parser.add_argument("-i", "--input_dir", type=Path, default="in")
parser.add_argument("-o", "--output_dir", type=Path, default="out")
parser.add_argument("-c", "--crop", type=int, default=64)
parser.add_argument("-d", "--seed", type=int, default=0)
parser.add_argument("-a", "--adjust", action="store_true")
parser.add_argument("-f", "--flip", action="store_true")
parser.add_argument("-j", "--jpeg", action="store_true")
parser.add_argument("-s", "--shuffle", action="store_true")
args = parser.parse_args()

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

progress = Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

random.seed(args.seed)
input_dir = args.input_dir
output_dir = args.output_dir
crop = args.crop
files = []

for ext in ("avif", "bmp", "jpeg", "jpg", "png", "webp"):
    files.extend(input_dir.glob(f"*.{ext}"))

output_dir.mkdir(parents=True, exist_ok=True)
before = len(files)
digits = len(str(before))
after = 0


from torchvision import io
from torchvision import transforms as T
from torchvision.transforms import functional as TF


def save_image(image, adjust=False):
    global after

    output = uuid4().hex[:16]
    after += 1

    if adjust:
        brightness = round(random.uniform(0.9, 1.1), 2)
        contrast = round(random.uniform(0.9, 1.1), 2)
        saturation = round(random.uniform(0.9, 1.1), 2)
        sharpness = round(random.uniform(0.9, 1.1), 2)

        image = image if brightness == 1 else TF.adjust_brightness(image, brightness)
        image = image if contrast == 1 else TF.adjust_contrast(image, contrast)
        image = image if saturation == 1 else TF.adjust_saturation(image, saturation)
        image = image if sharpness == 1 else TF.adjust_sharpness(image, sharpness)

    image = image.cpu()

    if args.jpeg:
        io.write_jpeg(image, str(output_dir / f"{output}.jpg"), quality=100)
    else:
        io.write_png(image, str(output_dir / f"{output}.png"), compression_level=0)

    return output


def save_text(text, output, shuffle=False):
    output = output_dir / f"{output}.txt"

    if shuffle:
        text = text.strip(punctuation).strip()
        list = [t.strip() for t in text.split(",")]
        random.shuffle(list)
        output.write_text(", ".join(list), encoding="utf-8")
    else:
        output.write_text(text, encoding="utf-8")


with progress as p:
    task = p.add_task("Cropping", total=len(files))

    for file in files:
        image = io.read_image(str(file), io.ImageReadMode.RGB).cuda()
        height, width = image.shape[1:3]
        min_size = min(height, width) // crop * crop
        max_size = max(height, width) // crop * crop
        max_size = max_size + (crop if max_size == min_size else 0)
        text = ""

        if (input := input_dir / f"{file.stem}.txt").is_file():
            text = input.read_text(encoding="utf-8")

        base = TF.resize(
            image,
            size=min_size,
            max_size=max_size,
            antialias=True,
            interpolation=T.InterpolationMode.BICUBIC,
        )

        height, width = base.shape[1:3]
        base = TF.center_crop(base, (height // crop * crop, width // crop * crop))
        output = save_image(base)
        text and save_text(text, output)

        if args.flip:
            flipped = TF.hflip(base)
            output = save_image(flipped, args.adjust)
            text and save_text(text, output, args.shuffle)

        image = TF.resize(
            image,
            size=min_size + crop,
            max_size=max_size + crop,
            antialias=True,
            interpolation=T.InterpolationMode.BICUBIC,
        )

        cropped = TF.ten_crop(image, (height, width))
        cropped = cropped[:4] + (cropped[5:9] if args.flip else [])

        for c in cropped:
            c = TF.center_crop(c, (height // crop * crop, width // crop * crop))
            output = save_image(c, args.adjust)
            text and save_text(text, output, args.shuffle)

        p.advance(task)
