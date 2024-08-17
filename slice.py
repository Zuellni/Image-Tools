from argparse import ArgumentParser
from pathlib import Path
from time import time
from warnings import simplefilter

simplefilter("ignore")
start = time()

parser = ArgumentParser()
parser.add_argument("-i", "--input_dir", type=Path, default=".")
parser.add_argument("-o", "--output_dir", type=Path, default=".")
parser.add_argument("-s", "--score", action="store_true")
parser.add_argument("-t", "--threshold", type=float, default=0.9)
parser.add_argument("--min_height", type=int, default=512)
parser.add_argument("--min_white", type=float, default=0.8)
parser.add_argument("--min_white_px", type=float, default=0.7)
args = parser.parse_args()

import torch
from torchvision import io

if args.score:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification

    path = "cafeai/cafe_aesthetic"
    processor = AutoFeatureExtractor.from_pretrained(path)
    model = AutoModelForImageClassification.from_pretrained(path).eval().cuda()

input_dir = args.input_dir
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
files = []

for ext in ("avif", "bmp", "jpeg", "jpg", "png", "webp"):
    files.extend(args.input_dir.glob(f"*.{ext}"))

count = len(files)
digits = len(str(count))
print(f"Processing {count} images")

for index, file in enumerate(files):
    print(f"{index + 1:0{digits}}/{count}", end="\r")
    image = io.read_image(str(file), io.ImageReadMode.RGB).cuda()

    gray = image.float().mean(dim=0) / 255
    white = (gray >= args.min_white).float().mean(dim=1) > args.min_white_px

    i = torch.where(white[:-1] != white[1:])[0] + 1
    i = torch.cat((torch.tensor([0]).cuda(), i, torch.tensor([len(white)]).cuda()))
    slices = [image[:, i[j] : i[j + 1], :] for j in range(len(i) - 1) if not white[i[j]]]

    for i, slice in enumerate(slices):
        if slice.shape[1] >= args.min_height:
            output = str(output_dir / f"{file.stem}_{i}.png")

            if args.score:
                with torch.inference_mode():
                    features = processor(slice, return_tensors="pt")
                    features = features["pixel_values"][0].unsqueeze(0)
                    score = model(features).logits.softmax(dim=-1)[0][-1].item()

                if score >= args.threshold:
                    io.write_jpeg(slice.cpu(), output, 100)
            else:
                io.write_jpeg(slice.cpu(), output, 100)

total = round(time() - start, 2)
print(f"Finished in {total} seconds")
