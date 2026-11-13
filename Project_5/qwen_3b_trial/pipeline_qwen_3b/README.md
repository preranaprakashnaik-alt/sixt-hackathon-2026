# Project 5 pipeline

## Run Stage 1 and Stage 2 locally

The `ollama` router uses a local Ollama server and does not read `P5_API_KEY` or
send an authorization header. It uses the Ollama model tag
`qwen2.5-coder:3b`.

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. In PowerShell, download the model:

   ```powershell
   ollama pull qwen2.5-coder:3b
   ```

3. Extract each release ZIP to its own folder:

   - Stage 1: `Project_5/stage1`
   - Stage 2: `Project_5/stage2`

4. From `Project_5/pipeline`, run either stage:

   ```powershell
   # Stage 1 uses the original canonical column names.
   python run.py --stage ../stage1 --out ../public_answers_qwen.csv --votes 3

   # Stage 2 uses its renamed-column map.
   python run.py --stage ../stage2 --schema-map ../stage2_schema_map.json --out ../final_answers.csv --router ollama --votes 3
   ```

`stage2_schema_map.json` contains the Stage 2 column mapping, checked against
the release data dictionary. If a later stage changes its columns, generate a
new map with `schema_map.py --stage <stage-dir> --out <map.json> --router ollama`
and review any tables reported as `FAILED` before using it.

The Ollama server must be running locally at `http://127.0.0.1:11434`. The local
run writes model-call logs to `Project_5/logs/ollama_calls.jsonl` and the normal
per-question report beside the answer CSV. No SIXT gateway or API credential is
used by either `--router ollama` command. The model runs with a 16K context.

Local Ollama is the default for `run.py` and `schema_map.py` for either stage.
The `--router llm` mode remains available only when explicitly selected for an
OpenAI-compatible API endpoint; that mode uses `P5_API_BASE`, `P5_API_KEY`, and
`P5_MODEL`.
