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


def prove_first_token_survives(model_fp16, input_fp16, processor):
    attn = model_fp16.wav2vec2.encoder.layers[11].attention
    proof = {}

    def attn_hook(module, inp, _out):
        hidden = inp[0]                            # (1, T, 768) fp16
        B, T, _ = hidden.shape
        num_heads = module.num_heads
        head_dim = module.head_dim
        scale = head_dim ** -0.5

        q = module.q_proj(hidden).view(B, T, num_heads, head_dim).transpose(1, 2)
        k = module.k_proj(hidden).view(B, T, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (1, H, T, T) fp16

        # For each query token t_q, check if ALL its key scores are -inf across all heads
        # scores[0]: (H, T, T) — scores[0, h, tq, :] is the row for query token tq in head h
        # A token's softmax row is corrupted if any head has all -inf in that row
        # Actually: token output is NaN if the softmax output for that token is NaN
        # softmax output shape after attn: (T, hidden) — one vector per token

        # Check per query token: does it have -inf in all key positions for any head?
        neg_inf_mask = torch.isneginf(scores[0])  # (H, T, T)

        # For each query token, count how many heads have at least one -inf key score
        # A full row of -inf in any head means NaN softmax for that token in that head
        all_neginf_per_head_per_tq = neg_inf_mask.all(dim=-1)  # (H, T): True if all keys are -inf
        any_head_all_neginf = all_neginf_per_head_per_tq.any(dim=0)  # (T,): True if any head has all-inf row

        # Stable softmax per token
        shifted = scores - scores.amax(dim=-1, keepdim=True)  # (1, H, T, T)
        softmax_out = torch.exp(shifted) / torch.exp(shifted).sum(dim=-1, keepdim=True)
        nan_per_tq = torch.isnan(softmax_out[0]).any(dim=-1).any(dim=0)  # (T,)

        proof["T"] = T
        proof["nan_per_tq"] = nan_per_tq          # (T,) bool
        proof["any_head_all_neginf"] = any_head_all_neginf  # (T,) bool
        proof["scores_token0"] = scores[0, :, 0, :]  # (H, T_k) — all heads, query=token0
        proof["num_neginf_token0"] = torch.isneginf(scores[0, :, 0, :]).sum().item()
        proof["num_neginf_token1"] = torch.isneginf(scores[0, :, 1, :]).sum().item()

    handle = attn.register_forward_hook(attn_hook)
    with torch.no_grad():
        out = model_fp16(input_values=input_fp16)
    handle.remove()

    logits = out.logits[0]                            # (T, vocab)
    nan_logit_per_token = torch.isnan(logits).any(dim=-1)  # (T,)
    predicted_ids = torch.where(nan_logit_per_token, torch.zeros(proof["T"], dtype=torch.long), logits.argmax(dim=-1))

    print("\n" + "=" * 70)
    print("PROOF: why token 0 survives but others produce NaN")
    print("=" * 70)

    T = proof["T"]
    nan_tq = proof["nan_per_tq"]
    neginf_tq = proof["any_head_all_neginf"]

    n_nan_tokens = nan_tq.sum().item()
    n_clean_tokens = (~nan_tq).sum().item()

    print(f"\n  Total query tokens (T)          : {T}")
    print(f"  Tokens with NaN softmax         : {n_nan_tokens}")
    print(f"  Tokens with clean softmax       : {n_clean_tokens}")
    print(f"  Tokens with all-inf score row   : {neginf_tq.sum().item()}")

    print(f"\n  Token 0 softmax NaN?            : {nan_tq[0].item()}")
    print(f"  Token 0 has all-inf score row?  : {neginf_tq[0].item()}")
    print(f"  Token 0 -inf score count        : {proof['num_neginf_token0']}  (out of {12 * T} entries across all heads)")

    print(f"\n  Token 1 softmax NaN?            : {nan_tq[1].item()}")
    print(f"  Token 1 -inf score count        : {proof['num_neginf_token1']}")

    print(f"\n  Token 0 logit NaN?              : {torch.isnan(logits[0]).any().item()}")
    print(f"  Token 0 predicted id            : {logits[0].argmax().item()}")
    print(f"  Token 0 decoded                 : {repr(processor.decode([logits[0].argmax().item()]))}")

    # Show first 10 tokens: clean vs NaN
    print(f"\n  First 20 tokens — NaN status and prediction:")
    print(f"  {'token':>6}  {'softmax_nan':>12}  {'logit_nan':>10}  {'pred_id':>8}  decoded")
    print("  " + "-" * 60)
    for t in range(min(20, T)):
        l_nan = torch.isnan(logits[t]).any().item()
        pred = logits[t].argmax().item() if not l_nan else -1
        decoded = processor.decode([pred]) if pred >= 0 else "<NaN>"
        print(f"  {t:>6}  {str(nan_tq[t].item()):>12}  {str(l_nan):>10}  {pred:>8}  {repr(decoded)}")

    # Show distribution: how many consecutive clean tokens from the start
    first_nan_token = nan_tq.nonzero(as_tuple=True)[0]
    if len(first_nan_token) > 0:
        print(f"\n  First NaN token index           : {first_nan_token[0].item()}")
        print(f"  → Tokens 0..{first_nan_token[0].item()-1} are clean, token {first_nan_token[0].item()} onwards are NaN")
    else:
        print("\n  No NaN tokens found")

    print("=" * 70)


def main():
    print("Loading processor and data...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = ds[0]["audio"]["array"]
    sample_rate = ds[0]["audio"]["sampling_rate"]
    audio_np = torch.tensor(audio).unsqueeze(0).numpy()

    input_fp16 = processor(
        audio_np[0], return_tensors="pt", max_length=WAV2VEC2_MAX_SEQ_LEN,
        truncation=True, padding="max_length", sampling_rate=sample_rate,
    ).input_values.to(torch.float16)

    print("Loading fp16 model...")
    model_fp16 = AutoModelForCTC.from_pretrained(
        MODEL_NAME, dtype=torch.float16, attn_implementation="eager", low_cpu_mem_usage=False,
    ).eval()

    prove_first_token_survives(model_fp16, input_fp16, processor)


if __name__ == "__main__":
    main()
