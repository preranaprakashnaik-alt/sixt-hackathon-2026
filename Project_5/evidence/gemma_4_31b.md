# Model evidence: Gemma 4 31B

- Model ID as called: `bedrock_mantle/google.gemma-4-31b` on the SIXT hackathon LLM gateway (`https://llm.orange.sixt.com/v1`).
- The gateway echoes the same ID in the `model` field of every response (see `logs/model_usage.csv`, column `model_returned`).
- Publisher: Google DeepMind. Architecture: dense (not mixture-of-experts).
- Total parameters: **30.7 billion**, below the 40 billion limit.
  - Source 1: Google, "Gemma 4 model card", Dense Models table, row "Total Parameters", column "31B Dense": 30.7B. https://ai.google.dev/gemma/docs/core/model_card_4
  - Source 2: Hugging Face model page `google/gemma-4-31B`, same table. https://huggingface.co/google/gemma-4-31B
  - Source 3: Gemma 4 Technical Report, Table 1 (31B row: 550M vision encoder, 1,410M embedder, 29,290M backbone). https://arxiv.org/abs/2607.02770
- Accessed: 2026-11-13.
- Not verified by us: the exact weight revision and quantization the gateway serves. The gateway does not expose them.

## Settings

- Temperature 0.0 for the first vote and 0.4 for the second and third votes.
- 20 questions per request, `max_tokens` = 120 per question + 200.
- System prompt: recipe catalogue and allowed values, built by `build_system_prompt` in `pipeline/router.py`.

## Usage record

`logs/model_usage.csv` lists every call (timestamp, model, temperature, question ID range, tokens, seconds) without question text.
The full request/response log is kept locally and not published because it contains challenge questions.
