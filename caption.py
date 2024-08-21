import json
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
from transformers import AutoModelForCausalLM, AutoProcessor
from unidecode import unidecode

tasks = {
    "mdc": "<MORE_DETAILED_CAPTION>",
    "dc": "<DETAILED_CAPTION>",
    "c": "<CAPTION>",
    "ocr": "<OCR>",
    "sd3": "<DESCRIPTION>Describe this image in great detail.",
    "tag": "<GENERATE_PROMPT>",
}

types = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

task_list = list(tasks.keys())
type_list = list(types.keys())

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path, default=".")
parser.add_argument("-o", "--output_dir", type=Path, default=".")
parser.add_argument("-m", "--model", type=str, default="microsoft/Florence-2-large-ft")
parser.add_argument("-j", "--save_as_jsonl", action="store_true")
parser.add_argument("-b", "--batch_size", type=int, default=8)
parser.add_argument("-B", "--num_beams", type=int, default=3)
parser.add_argument("-t", "--task", type=str, choices=task_list, default=task_list[0])
parser.add_argument("-T", "--max_new_tokens", type=int, default=256)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-D", "--dtype", type=str, choices=type_list, default=type_list[0])
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

device = torch.device(args.device)
dtype = types[args.dtype]

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

input = args.input
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
suffixes = (".avif", ".bmp", ".jpeg", ".jpg", ".png", ".webp")
captions = []

files = (
    [f for f in input.glob("*.*") if f.suffix in suffixes]
    if input.is_dir()
    else [input] if input.is_file() else []
)

batches = [
    files[i : i + args.batch_size] for i in range(0, len(files), args.batch_size)
]

with progress, torch.inference_mode():
    task = progress.add_task("Captioning", total=len(files))

    for batch in batches:
        text = [tasks[args.task]] * len(batch)
        images = [io.read_image(file, io.ImageReadMode.RGB) for file in batch]
        inputs = processor(text=text, images=images)
        inputs = inputs.to(device=device, dtype=dtype)

        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            do_sample=False,
            early_stopping=False,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)

        for file, caption in zip(batch, outputs):
            caption = unidecode(caption)
            caption = "\n".join([t for t in caption.splitlines() if t])
            caption = " ".join(caption.split())
            caption = caption.rsplit(".", 1)[0] + "."
            captions.append({"file_name": file.name, "text": caption})
            progress.advance(task)

if args.save_as_jsonl:
    output = output_dir / "metadata.jsonl"
    output.write_text("\n".join([json.dumps(c) for c in captions]), encoding="utf-8")
else:
    for file, caption in zip(files, captions):
        output = output_dir / f"{file.stem}.txt"
        output.write_text(caption["text"], encoding="utf-8")
