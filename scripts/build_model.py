"""Build the browser model files for this site.

Steps:
  1. Load GPT-2 (any size) with the explainer's instrumented model code.
  2. Export it to ONNX, keeping the per-head attention tensors the UI draws.
  3. Quantize the weights to 8-bit (roughly 4x smaller download).
  4. Split into 10 MB chunks + manifest.json in static/model/<name>/.
  5. Regenerate the cached example runs (src/constants/examples/ex*.js) that
     the page shows while the real model is still downloading.

Usage (run from the repo root):
  python scripts/build_model.py --model gpt2-medium
  python scripts/build_model.py --model gpt2-medium --random-weights   # offline test

The real weights are downloaded from Hugging Face by the `transformers` package.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "utils", "model"))
from model import GPT, GPTConfig  # noqa: E402  (the explainer's instrumented GPT-2)

CHUNK_SIZE = 10 * 1024 * 1024
# Only the attention tensors the UI actually reads (AttentionMatrix.svelte and
# AttentionWeightPopover.svelte). Must match `attentionSuffixes` in src/utils/data.ts.
ATTN_SUFFIXES = ["attn", "attn_masked", "attn_dropout"]

SIZES = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
}

# Defaults the page starts with (src/store/index.ts)
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 5


class Wrapper(torch.nn.Module):
    """Flatten the model's nested output dict into the tuple ONNX needs."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head

    def forward(self, input):
        out = self.model(input)
        flat = []
        for i in range(self.n_layer):
            for j in range(self.n_head):
                head = out["block"][f"block_{i}"]["attn"][f"head_{j}"]
                flat.extend(head[s] for s in ATTN_SUFFIXES)
        flat.append(out["linear"]["output"])
        return tuple(flat)


def output_names(cfg):
    names = [
        f"block_{i}_attn_head_{j}_{s}"
        for i in range(cfg.n_layer)
        for j in range(cfg.n_head)
        for s in ATTN_SUFFIXES
    ]
    return names + ["linear_output"]


def load_model(name, random_weights):
    if random_weights:
        cfg = GPTConfig(**SIZES[name], vocab_size=50257, block_size=1024, bias=True)
        torch.manual_seed(0)
        model = GPT(cfg)
    else:
        model = GPT.from_pretrained(name)
    model.eval()
    return model


def export_onnx(model, path):
    names = output_names(model.config)
    dummy = torch.tensor([[6601, 32704, 795, 30132, 2985, 284]])
    torch.onnx.export(
        Wrapper(model),
        dummy,
        path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=names,
        dynamic_axes={"input": {0: "batch", 1: "seq"}, **{n: {0: "d0", 1: "d1", 2: "d2"} for n in names}},
        dynamo=False,
    )
    return names


def quantize(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def chunk(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = []
    with open(path, "rb") as f:
        i = 0
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            name = f"model.onnx.part{i}"
            with open(os.path.join(out_dir, name), "wb") as c:
                c.write(data)
            names.append(name)
            i += 1
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"chunks": names, "bytes": os.path.getsize(path)}, f)
    return names


# ---------- cached examples ----------

def load_tokenizer():
    try:
        from transformers import GPT2TokenizerFast

        return GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as e:  # offline test run
        print(f"  (no tokenizer available: {e.__class__.__name__}; using placeholder token labels)")
        return None


def display_token(tok, token_id):
    t = tok.decode([token_id]) if tok else f"#{token_id}"
    t = t.replace("\n", "[NEWLINE]").replace("\t", "[TAB]").replace("\r", "[CR]")
    return re.sub(r"\s{2,}", lambda m: f"[{len(m.group(0))} SPACES]", t)


def top_k_probabilities(tok, logits, k=DEFAULT_TOP_K, temperature=DEFAULT_TEMPERATURE, max_items=50):
    """Mirror of topKSampling() in src/utils/data.ts (deterministic: takes rank 0)."""
    order = np.argsort(-logits)[:max_items]
    scaled = logits[order] / temperature
    topk = np.where(np.arange(len(order)) < k, scaled, -np.inf)
    m = np.max(topk)
    exp = np.where(np.isinf(topk), 0.0, np.exp(topk - m))
    probs = exp / exp.sum()
    items = []
    for rank, tid in enumerate(order):
        items.append(
            {
                "tokenId": int(tid),
                "logit": float(logits[tid]),
                "scaledLogit": float(scaled[rank]),
                "topKLogit": None if rank >= k else float(topk[rank]),
                "rank": rank,
                "token": display_token(tok, int(tid)),
                "expLogit": float(exp[rank]),
                "probability": float(probs[rank]),
            }
        )
    return items


def to_js(value, decimals):
    """JSON-like text that allows -Infinity and rounds floats to keep files small."""
    if isinstance(value, float):
        if math.isinf(value):
            return "-Infinity" if value < 0 else "Infinity"
        if math.isnan(value):
            return "NaN"
        return repr(round(value, decimals))
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(to_js(v, decimals) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{json.dumps(k)}:{to_js(v, decimals)}" for k, v in value.items()) + "}"
    raise TypeError(type(value))


# The five starter prompts, in the same order as inputTextExample in src/store/index.ts
EXAMPLE_PROMPTS = [
    "Data visualization empowers users to",
    "Artificial Intelligence is transforming the",
    "As the spaceship was approaching the",
    "On the deserted planet they discovered a",
    "IEEE VIS conference highlights the",
]


def wrap(text, every=12):
    """Break long output into short lines. Vite's build hangs on files with
    multi-hundred-kilobyte single lines, so start a new line every 12 commas
    (never inside a quoted string, since tokens can contain commas)."""
    out, commas, in_str, escaped = [], 0, False, False
    for ch in text:
        out.append(ch)
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == ",":
            commas += 1
            if commas % every == 0:
                out.append("\n\t\t")
    return "".join(out)


def build_examples(onnx_path, names):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tok = load_tokenizer()
    ex_dir = os.path.join(REPO, "src", "constants", "examples")
    for i in range(5):
        path = os.path.join(ex_dir, f"ex{i}.js")
        prompt = EXAMPLE_PROMPTS[i]
        if tok is None:
            print(f"  skipped ex{i}.js (needs the GPT-2 tokenizer)")
            continue
        token_ids = tok.encode(prompt)
        tokens = [tok.decode([t]) for t in token_ids]
        res = sess.run(None, {"input": np.array([token_ids], dtype=np.int64)})
        by_name = dict(zip(names, res))
        logits = by_name["linear_output"].reshape(-1).astype(np.float64)
        # Drop the leading batch dimension, as reshapeArray() in src/utils/array.ts does in the browser
        outputs = {
            n: {"data": (by_name[n][0] if by_name[n].shape[0] == 1 else by_name[n]).tolist()}
            for n in names
            if n != "linear_output"
        }
        probabilities = top_k_probabilities(tok, logits)
        example = {
            "prompt": prompt,
            "tokens": tokens,
            "tokenIds": token_ids,
            "logits": [float(x) for x in logits],
            "outputs": outputs,
            "probabilities": probabilities,
            "sampled": probabilities[0],
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"// Generated by scripts/build_model.py — do not edit by hand\nexport const ex{i} = {{\n")
            for k, v in example.items():
                f.write(f"\t{k}: {wrap(to_js(v, 4))},\n")
            f.write("};\n")
        print(f"  wrote ex{i}.js ({os.path.getsize(path) / 1e6:.1f} MB): {prompt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium", choices=list(SIZES))
    ap.add_argument("--random-weights", action="store_true", help="skip the download; for testing the pipeline")
    ap.add_argument("--work-dir", default=os.path.join(REPO, "src", "utils", "model", "params_output"))
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    fp32 = os.path.join(args.work_dir, f"{args.model}.onnx")
    q8 = os.path.join(args.work_dir, f"{args.model}-quant.onnx")
    out_dir = os.path.join(REPO, "static", "model", args.model)

    print(f"1/5 loading {args.model}{' (random weights)' if args.random_weights else ''}")
    model = load_model(args.model, args.random_weights)
    print("2/5 exporting to ONNX")
    with torch.no_grad():
        names = export_onnx(model, fp32)
    del model
    print(f"    fp32: {os.path.getsize(fp32) / 1e6:.0f} MB")
    print("3/5 quantizing to 8-bit")
    quantize(fp32, q8)
    print(f"    int8: {os.path.getsize(q8) / 1e6:.0f} MB")
    print("4/5 chunking")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    chunks = chunk(q8, out_dir)
    print(f"    {len(chunks)} chunks in {os.path.relpath(out_dir, REPO)}")
    print("5/5 regenerating cached examples")
    build_examples(q8, names)
    print("done")


if __name__ == "__main__":
    main()
