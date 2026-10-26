"""Laya learner for the drift experiment: batched prediction and in-memory fine-tuning.

The fine-tuning recipe is the author's: full fine-tuning,
AdamW with encoder lr 2.5e-5 and head lr 1e-4, weight decay 0.01, 5% warmup plus cosine schedule,
gradient clipping at 1.0, bf16 autocast, action head frozen, cross-entropy against the gold
option. Runs on CUDA when available, otherwise on CPU (for tests only).

Run with the interpreter of arquivos/modelos_decisao/.venv, where the laya package is installed.
"""

import json
import os
import math
import random
from pathlib import Path

import numpy as np
import torch

import laya
from laya.common import collate_items

from data import INSTRUCTION

EXTERNAL = Path(os.environ.get("TDA_EXTERNAL", Path(__file__).resolve().parents[2] / "external"))   # data, weights and caches outside the repository
WEIGHTS = (EXTERNAL / "laya-pesos")
CACHE = (EXTERNAL / "laya-ajustado/drift")


class Learner:
    def __init__(self, base="multilingual", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = laya.load(str(WEIGHTS / base), device=self.device)
        self.model = self.agent.model
        for p in self.model.act_head.parameters():
            p.requires_grad_(False)
        self._enc = {}

    def encode(self, it):
        """Encoded question for an item, cached by its content (identifier, text, options and instruction), since
        MASSIVE gives the same identifier to the same utterance in every locale."""
        key = (it["id"], it["text"], tuple(it["options"]), it.get("instruction", INSTRUCTION), json.dumps(it.get("state"), sort_keys=True))
        if key not in self._enc:
            q = {"type": "choice", "instructions": it.get("instruction", INSTRUCTION), "criteria": {o: o.replace("_", " ") for o in it["options"]}}
            self.agent._check_question("q", q)
            internal = {"q": self.agent._to_internal(q)}
            enc = self.agent._encode_state(it.get("state") or {"utterance": it["text"]}, ["q"], internal)[0]
            if len(enc["markers"]) != len(it["options"]):
                raise ValueError(f"item {it['id']}: options do not fit the head budget")
            self._enc[key] = enc
        return self._enc[key]

    def _forward(self, batch):
        b = collate_items([[x] for x in batch], self.agent.tok.pad_token_id)
        dev = self.device
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits, _ = self.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                   b["marker_mask"].to(dev), b["qtype"].to(dev))
        return logits.float().masked_fill(~b["marker_mask"].to(dev), -1e4)

    @torch.no_grad()
    def predict(self, items, bs=64):
        """Probabilities over the five options of each item, in the item's option order."""
        self.model.eval()
        enc = [self.encode(it) for it in items]
        order = np.argsort([len(e["ids"]) for e in enc], kind="stable")
        out = np.zeros((len(items), len(items[0]["options"])), dtype=np.float32)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            out[idx] = torch.softmax(self._forward([enc[j] for j in idx]), -1).cpu().numpy()
        return out

    def finetune(self, items, epochs=2, batch=32, lr_encoder=2.5e-5, lr_head=1e-4, seed=0):
        """Fine-tune the current weights on labelled items; returns training seconds."""
        import time
        t0 = time.perf_counter()
        enc = []
        for it in items:
            e = dict(self.encode(it))
            e["target"] = [1.0 if j == it["gold"] else 0.0 for j in range(len(it["options"]))]
            enc.append(e)
        m = self.model
        groups = [{"params": [p for n, p in m.named_parameters() if n.startswith("encoder.") and p.requires_grad], "lr": lr_encoder},
                  {"params": [p for n, p in m.named_parameters() if not n.startswith(("encoder.", "act_head.")) and p.requires_grad], "lr": lr_head}]
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
        total = max(1, math.ceil(len(enc) / batch) * epochs)
        warm = max(1, int(0.05 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
        rng = random.Random(seed)
        m.train()
        for _ in range(epochs):
            data = sorted(enc, key=lambda e: len(e["ids"]))
            batches = [data[i:i + batch] for i in range(0, len(data), batch)]
            rng.shuffle(batches)
            for bt in batches:
                tgt = torch.tensor([e["target"] for e in bt], device=self.device)
                loss = -(tgt * torch.log_softmax(self._forward(bt), -1)).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
        m.eval()
        if self.device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def state(self):
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def load_state(self, sd):
        self.model.load_state_dict(sd)


def initial_learner(drift, init_items, epochs=3, base="multilingual"):
    """Learner with the initial fine-tune of a drift, cached on disk outside the repository."""
    lr = Learner(base)
    key = drift if drift.startswith("R") else ("D1" if drift == "D1" else "all")
    path = CACHE / f"{base}_{key}.pt"
    if path.exists():
        lr.load_state(torch.load(path, map_location="cpu"))
    else:
        lr.finetune(init_items, epochs=epochs, seed=1)
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save(lr.state(), path)
    return lr
