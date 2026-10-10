# Local SRT Translator

Translate subtitle files locally with Ollama. The tool reads every `.srt` file directly inside `input/`, translates the subtitle text with an installed Gemma model, preserves cue numbering and timestamps, and writes separate translated files into `output/`. The source files are never modified.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- An installed Gemma model, for example:

```bash
ollama pull gemma3:4b
```

The translator uses Python's standard library, so no Python packages need to be installed.

## Run

Put source subtitle files in `input/`, then run from the project directory:

```bash
python3 translate_with_ollama.py
```

When `model` is `null`, the script automatically selects an installed Gemma model. To use a particular installed model, set its exact name in `translation_config.json`, such as `gemma3:4b`.

Each output is named after its source with the configured suffix. For example:

```text
input/episode.srt
output/episode.English.srt
```

The script reports file and batch progress. It validates the returned cue IDs and retries failed batches, splitting a batch into smaller requests when necessary. Each finished file is written atomically, so an interrupted translation does not leave a partial output file.

## Configuration

Edit `translation_config.json` to change the defaults:

| Setting | Purpose | Default |
| --- | --- | --- |
| `source_language` | Language of the source subtitles | `Spanish` |
| `target_language` | Language for the translated subtitles | `English` |
| `input_dir` | Folder containing source `.srt` files | `input` |
| `output_dir` | Folder for translated `.srt` files | `output` |
| `output_suffix` | Text added before the output `.srt` extension | `English` |
| `model` | Exact installed Ollama model name, or `null` to auto-select Gemma | `null` |
| `ollama_url` | Ollama server base URL | `http://127.0.0.1:11434` |
| `batch_size` | Number of subtitle cues sent per request | `16` |

Relative input and output paths are resolved from the directory containing the configuration file. The script reads `.srt` files directly inside the input directory, not nested folders.

Optional command-line overrides are available:

```bash
python3 translate_with_ollama.py --model gemma3:4b --batch-size 8
python3 translate_with_ollama.py --input-dir ./input --output-dir ./output
python3 translate_with_ollama.py --config ./translation_config.json
```

## Notes

Machine-translated dialogue may need human review, especially for idioms, jokes, song lyrics, and overlapping speech. Check a few generated files before relying on them as final subtitles.
