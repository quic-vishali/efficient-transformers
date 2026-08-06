import torch
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor

torch.set_printoptions(
    threshold=float("inf"),
    linewidth=100000,
    precision=10,
    sci_mode=False,
)

MODEL_NAME = "facebook/wav2vec2-base-960h"
WAV2VEC2_MAX_SEQ_LEN = 480000


def capture_outputs(model, input_features):
    """Run one forward pass, return {layer_name: output_tensor} in fp32."""
    captured = {}

    def make_hook(name):
        def hook(module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                captured[name] = t.detach().float()
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in model.named_modules()]
    with torch.no_grad():
        outputs = model(input_values=input_features)
    for h in handles:
        h.remove()

    return captured, outputs.logits


def main():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, )

    print("Loading dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = ds[0]["audio"]["array"]
    sample_rate = ds[0]["audio"]["sampling_rate"]
    audio_np = torch.tensor(audio).unsqueeze(0).numpy()

    input_fp32 = processor(
        audio_np[0], return_tensors="pt", max_length=WAV2VEC2_MAX_SEQ_LEN,
        truncation=True, padding="max_length", sampling_rate=sample_rate,
    ).input_values  # float32

    input_fp16 = input_fp32.to(torch.float16)
    print(processor.tokenizer.convert_ids_to_tokens(6))
    print(processor.tokenizer.convert_ids_to_tokens(0))

    print("\nLoading fp32 model...")
    model_fp32 = AutoModelForCTC.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=False,
    ).eval()

    print("\nLoading fp16 model...")
    model_fp16 = AutoModelForCTC.from_pretrained(
        MODEL_NAME, dtype=torch.float16, attn_implementation="eager", low_cpu_mem_usage=False,
    ).eval()

    print("\nRunning fp32 forward...")
    fp32_caps, logits_fp32 = capture_outputs(model_fp32, input_fp32)

    print("Running fp16 forward...")
    fp16_caps, logits_fp16 = capture_outputs(model_fp16, input_fp16)

    # Layerwise comparison — only layers present in both
    common = [n for n in fp32_caps if n in fp16_caps]

    print(f"\n{'Layer':<70} {'min_fp32':>12} {'max_fp32':>12} {'min_fp16':>12} {'max_fp16':>12} {'min_diff':>12} {'max|diff|':>12}  NaN_fp16")
    print("-" * 155)

    first_nan = None
    first_big_diff = None
    for name in common:
        t32 = fp32_caps[name]
        print("score shape - ", t32.shape)
        t16 = fp16_caps[name]
        if t32.shape != t16.shape:
            continue
        has_nan = torch.isnan(t16).any().item()
        min32 = t32.min().item()
        max32 = t32.max().item()
        min16 = t16.min().item() if not has_nan else float("nan")
        max16 = t16.max().item() if not has_nan else float("nan")
        if not has_nan:
            diff = t32 - t16
            min_diff = diff.min().item()
            max_diff = diff.abs().max().item()
        else:
            min_diff = float("nan")
            max_diff = float("nan")

        nan_str = "*** NaN ***" if has_nan else ""
        big_diff = not has_nan and max_diff > 1.0
        marker = "  <-- large diff" if big_diff else ""

        print(f"{name:<70} {min32:>12.4f} {max32:>12.4f} {min16:>12.4f} {max16:>12.4f} {min_diff:>12.4f} {max_diff:>12.4f}  {nan_str}{marker}")

        if first_nan is None and has_nan:
            first_nan = name
        if first_big_diff is None and big_diff:
            first_big_diff = name

    print("\n" + "=" * 155)
    print(f"First NaN layer    : {first_nan}")
    print(f"First large diff   : {first_big_diff}")
    # print(logits_fp32)
    print(f"fp32 transcription : {processor.batch_decode(torch.argmax(logits_fp32, dim=-1))}")
    logits_fp16_f32 = logits_fp16.float()
    # print(logits_fp16_f32.size())
    pred16 = torch.argmax(logits_fp16_f32, dim=-1) 
    # print(pred16)
    print(pred16.shape)
    print(f"fp16 transcription : {processor.batch_decode(pred16)}")


if __name__ == "__main__":
    main()