from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path, default=".")
parser.add_argument("-o", "--output", type=Path, default=".")
parser.add_argument("-q", "--quality", type=int, default=100)
parser.add_argument("-s", "--size", type=int, nargs=2, default=[1920, 1920])
args = parser.parse_args()

args.output.mkdir(parents=True, exist_ok=True)
inputs = []

if args.input.is_dir():
    for ext in ("avif", "bmp", "jpeg", "jpg", "png", "webp"):
        inputs.extend(args.input.glob(f"*.{ext}"))
elif args.input.is_file():
    inputs.append(args.input)

for input in inputs:
    image = Image.open(input).convert("RGB")
    image.thumbnail(args.size)
    output = args.output / f"{input.stem}.jpg"
    index = 1

    while output.exists():
        output = args.output / f"{input.stem} ({index}).jpg"
        index += 1

    image.save(output, optimize=True, quality=args.quality)
