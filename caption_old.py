from argparse import ArgumentParser
from json import dumps
from pathlib import Path
from warnings import simplefilter

simplefilter("ignore")
dir = Path(__file__).parent
taggers = {}
models = {}

for tagger in dir.glob("*.onnx"):
    taggers[tagger.stem] = tagger

for model in dir.glob("*/"):
    if (model / "config.json").is_file():
        models[model.name] = model

parser = ArgumentParser()
parser.add_argument("-i", "--input_dir", type=Path, default=".")
parser.add_argument("-o", "--output_dir", type=Path, default=".")
parser.add_argument("-m", "--model", type=str, default=None, choices=models.keys())
parser.add_argument("-t", "--tagger", type=str, default=None, choices=taggers.keys())
parser.add_argument("-c", "--cutoff", type=float, default=0.3)
parser.add_argument("-n", "--minimum", type=float, default=0.3)
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-j", "--json", action="store_true")
parser.add_argument("-l", "--limit", action="store_true")
parser.add_argument("-p", "--plain", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
args = parser.parse_args()

from rich import print
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

with progress as p:
    if args.tagger:
        from onnxruntime import InferenceSession
        from pandas import read_csv
        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms.functional import pad, resize

        tagger = taggers[args.tagger]
        session = InferenceSession(
            tagger, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        labels = read_csv(tagger.parent / f"{tagger.stem}.csv")
        input = session.get_inputs()[0]
        label = session.get_outputs()[0].name
        size = input.shape[1]

    if args.limit:
        from transformers import AutoTokenizer

        clip = AutoTokenizer.from_pretrained(dir / "tokenizer")
        max = clip.model_max_length - 2
        min = round(args.minimum * max)

    if args.model:
        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Cache,
            ExLlamaV2Config,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator

        config = ExLlamaV2Config()
        config.model_dir = models[args.model]
        config.prepare()
        config.max_seq_len = 2048

        model = ExLlamaV2(config)
        loading = p.add_task("Loading", total=len(model.modules) + 1)
        model.load(callback=lambda _, __: p.advance(loading))

        cache = ExLlamaV2Cache(model)
        tokenizer = ExLlamaV2Tokenizer(config)
        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.7
        settings.top_k = 0
        settings.top_p = 1.0
        settings.typical = 1.0
        settings.min_p = 0.0
        settings.top_a = 0.1
        settings.token_repetition_penalty = 1.1
        settings.temperature_last = True

        stop = [tokenizer.eos_token_id, tokenizer.newline_token_id, ".", "?", "!"]
        generator.set_stop_conditions(stop)

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tagger_data = []
    model_data = []
    files = []

    if args.tagger:
        for ext in ("avif", "bmp", "jpeg", "jpg", "png", "webp"):
            files.extend(input_dir.glob(f"*.{ext}"))
    else:
        files.extend(input_dir.glob("*.txt"))

    if args.tagger:
        tagging = p.add_task("Tagging", total=len(files))

    if args.model:
        captioning = p.add_task("Captioning", total=len(files))

    for file in files:
        if args.tagger:
            image = read_image(str(file), ImageReadMode.RGB).cuda()

            h, w = image.shape[1:]
            lr = (h - w) // 2 if h > w else 0
            tb = (w - h) // 2 if w > h else 0

            image = pad(image, padding=(lr, tb), fill=255)
            image = resize(image, size=(size, size), antialias=True)
            image = image.permute(1, 2, 0)
            image = image[..., [2, 1, 0]]
            image = image.unsqueeze(0).float().cpu().numpy()

            score = session.run([label], {input.name: image})[0]
            tags = labels[["name", "category"]].copy()
            tags["score"] = score[0]
            tags = dict(tags[tags["category"] == 0][["name", "score"]].values)
            tags = dict(sorted(tags.items(), key=lambda k: k[1], reverse=True))
            cutoff = args.cutoff

            while True:
                tags_str = ", ".join(
                    [
                        k.replace("_", " ").lower().strip()
                        for k, v in tags.items()
                        if v >= cutoff
                    ]
                )

                if not args.limit:
                    break
                else:
                    tokens = clip.tokenize(tags_str)
                    tokens_len = len(tokens)

                    if tokens_len > max:
                        print(f"{file.stem}: {tokens_len} > {max}")
                        cutoff += 0.1
                    elif tokens_len < min:
                        print(f"{file.stem}: {tokens_len} < {min}")
                        cutoff -= 0.1
                    else:
                        break

            if args.debug:
                print(f"Tags: {tags_str}")

            if args.save or not args.model:
                if args.json:
                    tagger_data.append({"file_name": file.name, "text": tags_str})

                if not args.json or args.plain:
                    suffix = "_tag" if args.model else ""
                    output = output_dir / f"{file.stem}{suffix}.txt"
                    output.write_text(tags_str)

            p.advance(tagging)
        else:
            tags_str = file.read_text()

        if tags_str and args.model:
            prompt = (
                "### User:\n"
                "Expand the following keywords into a short sentence: girl, bare shoulders, black clothes, black hair, breasts, cleavage, closed eyes, from above, large breasts, lips, long hair, pale skin, ponytail, sketch, solo\n\n"
                "### Assistant:\n"
                "a girl with long black hair styled into a ponytail and closed eyes, wearing a black dress with cleavage showing her pale skin and large breasts\n\n"
                "### User:\n"
                "girl, animal, brown hair, breasts, breasts apart, breasts out, frown, giant snake, large breasts, lips, nipples, no bra, snake, solo, underbust\n\n"
                "### Assistant:\n"
                "a girl with brown hair posing naked with her large breasts on display and a giant snake coiled around her\n"
                "### User:\n"
                "bangs, blue eyes, girl, holding, red flower, bouquet, long hair, blonde hair, one eye closed, pointy ears, puffy sleeves, blue background, solo, two side up, upper body, white dress\n\n"
                "### Assistant:\n"
                "a girl with long blonde hair, pointy ears and blue eyes wearing a white dress and holding a red flower against a blue background\n\n"
                "### User:\n"
                f"{tags_str}\n\n"
                "### Assistant:\n"
                "a"
            )

            prompt = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
            settings.temperature = 1.0

            while True:
                generator.begin_stream_ex(prompt, settings)
                text = ""

                while True:
                    response = generator.stream_ex()
                    text += response["chunk"]

                    if response["eos"]:
                        break

                text = text.lower().strip()

                if not args.limit:
                    break
                else:
                    tokens = clip.tokenize(text)
                    tokens_len = len(tokens)

                    if tokens_len > max:
                        print(f"{file.stem}: {tokens_len} > {max}")
                        settings.temperature -= 0.1
                    elif tokens_len < min:
                        print(f"{file.stem}: {tokens_len} < {min}")
                        settings.temperature += 0.1
                    else:
                        break

            if args.debug:
                print(f"Caption: {text}")

            if args.json:
                model_data.append({"file_name": file.name, "text": text})

            if not args.json or args.plain:
                output = output_dir / f"{file.stem}.txt"
                output.write_text(text)

            p.advance(captioning)

if tagger_data:
    output = output_dir / "tagger_data.jsonl"
    output.write_text("\n".join([dumps(d) for d in tagger_data]))

if model_data:
    output = output_dir / "model_data.jsonl"
    output.write_text("\n".join([dumps(d) for d in model_data]))
