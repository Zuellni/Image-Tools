from argparse import ArgumentParser
from pathlib import Path
from warnings import simplefilter

simplefilter("ignore")

tasks = {
    "mdc": "<MORE_DETAILED_CAPTION>",
    "dc": "<DETAILED_CAPTION>",
    "c": "<CAPTION>",
    "ocr": "<OCR>",
}

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path, default=".")
parser.add_argument("-o", "--output_dir", type=Path, default=".")
parser.add_argument("-m", "--model", type=str, default="microsoft/Florence-2-large-ft")
parser.add_argument("-t", "--task", type=str, choices=list(tasks.keys()), default="mdc")
parser.add_argument("-T", "--tokens", type=int, default=256)
parser.add_argument("-b", "--beams", type=int, default=4)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-D", "--dtype", type=str, default="float16")
args = parser.parse_args()

import torch
from rich import print
from torchvision import io
from transformers import AutoModelForCausalLM, AutoProcessor

input = args.input
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
dtype = getattr(torch, args.dtype)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model,
    device_map=device,
    torch_dtype=dtype,
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=args.model,
    device_map=device,
    torch_dtype=dtype,
    trust_remote_code=True,
)

if input.is_dir():
    suffixes = (".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp")
    files = [f for f in input.glob("*.*") if f.suffix in suffixes]
else:
    files = [input]

for file in files:
    task = tasks[args.task]
    image = io.read_image(file, io.ImageReadMode.RGB)
    inputs = processor(text=task, images=image)
    inputs = inputs.to(device=device, dtype=dtype)

    output = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        do_sample=False,
        early_stopping=False,
        max_new_tokens=args.tokens,
        num_beams=args.beams,
    )[0]

    text = processor.decode(output, skip_special_tokens=True)
    text = "\n".join([t for t in text.splitlines() if t])
    text = " ".join(text.split())
    (output_dir / f"{file.stem}.txt").write_text(text, encoding="utf-8")
    print(f'File: "{file.name}"\nOutput: "{text}"\n')
