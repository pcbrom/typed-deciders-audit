#!/usr/bin/env python3
"""Laya multilingual on the CPU in fp32, bf16 autocast (AVX-512 BF16) and dynamic int8 of the encoder
(x86 engine, VNNI), on the report items of pt and en with the question of the article, batches of 8
sorted by length; probabilities per item and the median time of three passes
(implementation_notes.md, CPU precision). Output: data/derived/multilingue/cpu_precisao.npz and .json.

Run with the interpreter of arquivos/modelos_decisao/.venv, with CUDA_VISIBLE_DEVICES="".
Usage: cpu_precisao.py
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import DATA, load_items, question, EXTERNAL  # noqa: E402
from laya.common import collate_items  # noqa: E402
from laya_precisao import load_agent  # noqa: E402

MODES = ("fp32", "bf16_autocast", "int8_dinamico")


def batched(agent, texts, options, bs=8):
    encoded, internals = [], []
    for t, o in zip(texts, options):
        internal = {"intent": agent._to_internal(question(o))}
        encoded.append(agent._encode_state({"utterance": t}, ["intent"], internal))
        internals.append(internal)
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i][0]["ids"]))
    probs = [None] * len(texts)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            b = collate_items([encoded[i] for i in idx], agent.tok.pad_token_id)
            logits, act = agent._forward(b)
            for row, i in enumerate(idx):
                a = agent._decode_answers(logits, act, encoded[i], ["intent"], internals[i], row)["intent"]
                probs[i] = [a["probabilities"][o] for o in options[i]]
    return np.array(probs)


def main():
    torch.set_num_threads(int(os.environ.get("LAYA_THREADS", 6)))
    flags = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    model = next((l.split(":", 1)[1].strip() for l in flags.splitlines() if l.startswith("Model name")), platform.processor())
    isa = sorted({f for f in ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "avx_vnni", "amx_bf16", "amx_tile") if f in flags.split()})
    arrays, out = {}, {"cpu": model, "isa": isa, "threads": torch.get_num_threads(), "torch": torch.__version__, "modes": {}}
    for mode in MODES:
        agent = load_agent("multilingual", mode)
        times = {}
        for loc in ("pt", "en"):
            items, _ = load_items(loc)
            texts = [items[i]["utterance"] for i in range(1000, 2000)]
            opts = [items[i]["options"] for i in range(1000, 2000)]
            batched(agent, texts[:16], opts[:16])
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                P = batched(agent, texts, opts)
                ts.append(time.perf_counter() - t0)
            arrays[f"{mode}|{loc}"] = P
            times[loc] = round(1000 * float(np.median(ts)) / len(texts), 2)
        out["modes"][mode] = {"ms_por_item_mediana": times}
        print(mode, times, flush=True)
        agent = None
    od = DATA / "derived" / "multilingue"
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "cpu_precisao.npz", **arrays)
    (od / "cpu_precisao.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
