from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path, default=".")
parser.add_argument("-o", "--output_dir", type=Path, default=".")
parser.add_argument("-q", "--quality", type=int, default=100)
parser.add_argument("-s", "--size", type=int, nargs=2, default=[1920, 1920])
args = parser.parse_args()

input = args.input
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
suffixes = (".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp")

files = (
    [f for f in input.glob("*.*") if f.suffix in suffixes]
    if input.is_dir()
    else [input] if input.is_file() else []
)

for file in files:
    image = Image.open(file).convert("RGB")
    image.thumbnail(args.size)
    output = output_dir / f"{file.stem}.jpg"
    index = 1

    while output.exists():
        output = output_dir / f"{file.stem} ({index}).jpg"
        index += 1

    image.save(output, optimize=True, quality=args.quality)
