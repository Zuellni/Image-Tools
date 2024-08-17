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
parser.add_argument("-B", "--beams", type=int, default=3)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-D", "--dtype", type=str, default="float16")
args = parser.parse_args()

import torch
from PIL import Image
from rich import print
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
    for file in files:
        text = tasks[args.task]
        images = Image.open(file).convert("RGB")
        inputs = processor(text=text, images=images)
        inputs = inputs.to(device=device, dtype=dtype)

        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            do_sample=False,
            early_stopping=False,
            max_new_tokens=args.tokens,
            num_beams=args.beams,
        )[0]

        output = processor.decode(output_ids, skip_special_tokens=True)
        (output_dir / f"{file.stem}.txt").write_text(output, encoding="utf-8")
        print(f'File: "{file.name}"\nOutput: "{output}"\n')