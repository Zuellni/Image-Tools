from argparse import ArgumentParser
from pathlib import Path
from warnings import simplefilter

simplefilter("ignore")

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torchvision import io

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path, default=".")
parser.add_argument("-o", "--output_dir", type=Path, default=".")
parser.add_argument("-s", "--score", action="store_true")
parser.add_argument("-t", "--threshold", type=float, default=0.9)
parser.add_argument("--min_height", type=int, default=512)
parser.add_argument("--min_ratio", type=float, default=0.8)
parser.add_argument("--min_white", type=float, default=0.7)
args = parser.parse_args()

progress = Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

if args.score:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification

    path = "cafeai/cafe_aesthetic"
    processor = AutoFeatureExtractor.from_pretrained(path)
    model = AutoModelForImageClassification.from_pretrained(path).eval().cuda()

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
    image = io.read_image(file, io.ImageReadMode.RGB).cuda()
    gray = image.float().mean(dim=0) / 255
    white = (gray >= args.min_ratio).float().mean(dim=1) > args.min_white

    i = torch.where(white[:-1] != white[1:])[0] + 1
    i = torch.cat((torch.tensor([0]).cuda(), i, torch.tensor([len(white)]).cuda()))

    slices = [
        image[:, i[j] : i[j + 1], :] for j in range(len(i) - 1) if not white[i[j]]
    ]

    for index, slice in enumerate(slices):
        if slice.shape[1] >= args.min_height:
            output = output_dir / f"{file.stem}_{index}.jpg"

            if args.score:
                with torch.inference_mode():
                    features = processor(slice, return_tensors="pt")
                    features = features["pixel_values"][0].unsqueeze(0)
                    score = model(features).logits.softmax(dim=-1)[0][-1].item()

                if score >= args.threshold:
                    io.write_jpeg(slice.cpu(), output, 100)
            else:
                io.write_jpeg(slice.cpu(), output, 100)
