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
parser.add_argument("-d", "--device", type=str, choices=("cuda", "cpu"), default="cuda")
args = parser.parse_args()

import torch
from PIL import Image
from rich import print
from transformers import AutoModelForCausalLM, AutoProcessor

input = args.input
output_dir = args.output
output_dir.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model,
    device_map=device,
    torch_dtype=dtype,
    trust_remote_code=True,
).eval()

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=args.model,
    device_map=device,
    torch_dtype=dtype,
    trust_remote_code=True,
)

if input.is_dir():
    suffixes = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
    files = [f for f in input.glob("*.*") if f.suffix in suffixes]
else:
    files = [input]

with torch.inference_mode():
    images = [Image.open(f).convert("RGB") for f in files]
    text = [tasks[args.task]] * len(files)
    inputs = processor(text=text, images=images)
    inputs = inputs.to(device=device, dtype=dtype)

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        do_sample=False,
        early_stopping=False,
        max_new_tokens=256,
        num_beams=3,
    )

    outputs = processor.batch_decode(output_ids, skip_special_tokens=True)


for file, output in zip(files, outputs):
    print(f'File: "{file.name}"\nOutput: "{output}"\n')
    (output_dir / f"{file.stem}.txt").write_text(output, encoding="utf-8")
