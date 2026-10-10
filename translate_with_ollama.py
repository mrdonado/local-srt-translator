#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib import error, request

DEFAULT_MODEL = "gemma3:4b"
BATCH_SIZE = 16
DEFAULT_CONFIG_NAME = "translation_config.json"

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}


def ollama_request(ollama_url: str, endpoint: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = request.Request(f"{ollama_url}{endpoint}", data=data, headers=headers)
    with request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def choose_model(requested_model: str | None, ollama_url: str) -> str:
    try:
        tags = ollama_request(ollama_url, "/api/tags")
    except error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {ollama_url}. Start it with `ollama serve` and retry."
        ) from exc

    installed = sorted(item.get("name") for item in tags.get("models", []) if item.get("name"))
    if requested_model and requested_model in installed:
        return requested_model
    if requested_model:
        choices = ", ".join(installed) or "none found"
        raise RuntimeError(
            f"Model {requested_model!r} is not installed. Installed models: {choices}. "
            f"Pull it with `ollama pull {requested_model}` or pass --model with an installed model name."
        )

    gemma_models = [name for name in installed if name.lower().startswith("gemma")]
    if not gemma_models:
        choices = ", ".join(installed) or "none found"
        raise RuntimeError(
            f"No Gemma model is installed. Available models: {choices}. "
            "Pull one with `ollama pull gemma3:4b` or specify an installed Gemma using --model."
        )
    return next((name for name in gemma_models if name == DEFAULT_MODEL), gemma_models[0])


def split_srt(text: str) -> tuple[list[str], str]:
    newline = "\r\n" if "\r\n" in text else "\n"
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    return blocks, newline


def get_cue(block: str) -> tuple[str, str] | None:
    lines = block.splitlines()
    if len(lines) < 3 or "-->" not in lines[1]:
        return None
    return lines[0], "\n".join(lines[2:])


def translate_batch(
    model: str,
    cues: list[tuple[int, str]],
    source_language: str,
    target_language: str,
    ollama_url: str,
) -> dict[int, str]:
    prompt = (
        f"Translate every {source_language} subtitle value into natural {target_language}. "
        "Return one translation for every input id, preserving ids exactly. "
        "Preserve speaker labels, sound cues, musical notes, and inline subtitle tags. "
        "Do not add explanations or combine separate cues.\n\n"
        "Subtitle cues as JSON:\n"
        + json.dumps([{"id": cue_id, "text": text} for cue_id, text in cues], ensure_ascii=False)
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": TRANSLATION_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 8192},
        "keep_alive": "10m",
    }

    last_error = None
    for attempt in range(3):
        try:
            result = ollama_request(ollama_url, "/api/generate", payload)
            response = json.loads(result.get("response", ""))
            translations = response.get("translations", [])
            translated = {item["id"]: item["text"].strip() for item in translations}
            expected_ids = {cue_id for cue_id, _ in cues}
            if set(translated) != expected_ids or any(not text for text in translated.values()):
                raise ValueError("model returned missing, extra, or empty cue translations")
            return translated
        except (error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)

    if len(cues) > 1:
        midpoint = len(cues) // 2
        return translate_batch(
            model, cues[:midpoint], source_language, target_language, ollama_url
        ) | translate_batch(model, cues[midpoint:], source_language, target_language, ollama_url)
    cue_id = cues[0][0]
    raise RuntimeError(f"Could not translate cue {cue_id}: {last_error}") from last_error


def translate_file(
    src: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    source_language: str,
    target_language: str,
    ollama_url: str,
    output_suffix: str,
) -> Path:
    original = src.read_text(encoding="utf-8-sig")
    blocks, newline = split_srt(original)
    parsed_cues = []
    for block_index, block in enumerate(blocks):
        cue = get_cue(block)
        if cue is not None:
            parsed_cues.append((block_index, cue[0], cue[1]))

    if not parsed_cues:
        raise RuntimeError(f"No valid SRT cues found in {src.name}")

    translations: dict[int, str] = {}
    total_batches = (len(parsed_cues) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(parsed_cues), batch_size), start=1):
        batch = parsed_cues[start : start + batch_size]
        translated = translate_batch(
            model,
            [(block_index, text) for block_index, _, text in batch],
            source_language,
            target_language,
            ollama_url,
        )
        translations.update(translated)
        print(
            f"  {src.name}: batch {batch_number}/{total_batches} "
            f"({min(start + len(batch), len(parsed_cues))}/{len(parsed_cues)} cues)",
            flush=True,
        )

    for block_index, _, _ in parsed_cues:
        lines = blocks[block_index].splitlines()
        lines[2:] = translations[block_index].splitlines()
        blocks[block_index] = "\n".join(lines)

    output = (newline + newline).join(blocks) + newline
    out_path = output_dir / f"{src.stem}.{output_suffix}.srt"
    temp_path = out_path.with_name(out_path.name + ".tmp")
    temp_path.write_text(output, encoding="utf-8")
    temp_path.replace(out_path)
    return out_path


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Translate SRT files using a local Ollama model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_dir / DEFAULT_CONFIG_NAME,
        help=f"JSON configuration file (default: {DEFAULT_CONFIG_NAME})",
    )
    parser.add_argument("--model", help="Installed Ollama model; overrides the configuration")
    parser.add_argument("--batch-size", type=int, help="Subtitle cues per model request")
    parser.add_argument("--input-dir", type=Path, help="Input directory; overrides the configuration")
    parser.add_argument("--output-dir", type=Path, help="Output directory; overrides the configuration")
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Cannot read configuration file {args.config}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in configuration file {args.config}: {exc}") from exc

    config_dir = args.config.resolve().parent

    def configured_path(value: str | Path, override: bool) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute() and not override:
            path = config_dir / path
        return path.resolve()

    input_dir = configured_path(args.input_dir or config["input_dir"], args.input_dir is not None)
    output_dir = configured_path(args.output_dir or config["output_dir"], args.output_dir is not None)
    source_language = config["source_language"]
    target_language = config["target_language"]
    output_suffix = config.get("output_suffix", target_language)
    ollama_url = config.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", BATCH_SIZE)

    if not isinstance(batch_size, int) or batch_size < 1:
        parser.error("--batch-size must be at least 1")

    sources = sorted(input_dir.glob("*.srt"))
    if not sources:
        print(f"No source .srt files found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    model = choose_model(args.model or config.get("model"), ollama_url)
    print(
        f"Using local Ollama model {model}; translating {len(sources)} files "
        f"from {source_language} to {target_language}.",
        flush=True,
    )
    for file_number, src in enumerate(sources, start=1):
        print(f"[{file_number}/{len(sources)}] {src.name}", flush=True)
        out_path = translate_file(
            src,
            output_dir,
            model,
            batch_size,
            source_language,
            target_language,
            ollama_url,
            output_suffix,
        )
        print(f"Created {out_path.name}", flush=True)
    print("All subtitle files translated.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
