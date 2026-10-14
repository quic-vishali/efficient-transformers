# vLLM benchmark golden outputs

This directory contains the committed Hugging Face/PyTorch reference outputs used by the vLLM QAIC benchmark validation flow.

All reference outputs are stored in one file:

```text
goldens.json
```

The file uses this structure:

```text
family -> model -> variant -> record
```

For example:

```json
{
  "causal_lm": {
    "openai-community/gpt2": {
      "<variant>": {
        "generated_texts": ["..."],
        "prompts": ["..."],
        "config_fp": "...",
        "gen_len": 32,
        "timestamp": "...",
        "device": "cpu"
      }
    }
  }
}
```

The variant key is derived from the model, fixed input content, endpoint, and generation-affecting parameters. QAIC device and execution settings are not part of the key, so one PyTorch golden can be reused across QAIC variants.

On the first run, the benchmark creates the matching PyTorch golden and compares the vLLM `generated_texts` value against it. Later runs load that entry and perform the same comparison.
The vLLM output remains in the case `benchmark_result.json`; it is never written into the golden store as the reference.

To inspect the store:

```bash
python3 -m json.tool goldens.json
```
