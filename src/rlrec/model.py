"""Политика (GPT-2 next-item) и reward model (GPT-2 по истории с откликами)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from .data import N_DUR_BUCKETS, N_RATIO_BUCKETS, PAD


def gpt2_config(n_items: int, d: int, n_layer: int, n_head: int, T: int) -> GPT2Config:
    return GPT2Config(
        vocab_size=n_items + 1, n_embd=d, n_layer=n_layer, n_head=n_head,
        n_positions=T + 8, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=PAD, eos_token_id=PAD, pad_token_id=PAD,
    )


class Policy(nn.Module):
    """GPT-2, токены — id объектов. Логиты по всему каталогу на каждой позиции."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.lm = GPT2LMHeadModel(cfg)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        attn = (items != PAD).long()
        logits = self.lm(input_ids=items, attention_mask=attn).logits
        logits[..., PAD] = -1e9
        return logits

    def last_logits(self, items: torch.Tensor) -> torch.Tensor:
        """Контекст выровнен влево с паддингом справа: берём последнюю непустую позицию."""
        h = self.lm.transformer(input_ids=items, attention_mask=(items != PAD).long()).last_hidden_state
        last = (items != PAD).sum(1) - 1
        logits = self.lm.lm_head(h[torch.arange(len(items)), last])
        logits[..., PAD] = -1e9
        return logits

    def sft_loss(self, items: torch.Tensor) -> torch.Tensor:
        logits = self.forward(items[:, :-1])
        tgt = items[:, 1:].clone()
        tgt[tgt == PAD] = -100
        tgt[items[:, :-1] == PAD] = -100
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)


class RewardModel(nn.Module):
    """Вход: история (объект + квантованная доля досмотра). Для кандидата c на шаге t+1
    предсказывает долю досмотра, продолжение сессии и возврат в течение суток."""

    HEADS = ("ratio", "cont", "ret")

    def __init__(self, cfg: GPT2Config, item_dur_bucket: torch.Tensor):
        super().__init__()
        d = cfg.n_embd
        self.item = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.rb = nn.Embedding(N_RATIO_BUCKETS, d)
        self.cand = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.dur = nn.Embedding(N_DUR_BUCKETS, d)
        self.register_buffer("item_dur", torch.cat([torch.zeros(1, dtype=torch.long), item_dur_bucket]))
        self.tr = GPT2Model(cfg)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(self.HEADS)))

    def encode(self, items, rbucket):
        x = self.item(items) + self.rb(rbucket)
        return self.tr(inputs_embeds=x, attention_mask=(items != PAD).long()).last_hidden_state

    def score(self, h, cand):
        """h: [..., d] состояние истории, cand: [...] id кандидатов -> [..., 3] логиты/значения."""
        c = self.cand(cand) + self.dur(self.item_dur[cand])
        out = self.head(torch.cat([h, c], -1))
        return out

    @staticmethod
    def to_preds(out):
        # ratio: сигмоида, масштабированная на [0, RATIO_CLIP]
        from .data import RATIO_CLIP
        return {
            "ratio": torch.sigmoid(out[..., 0]) * RATIO_CLIP,
            "cont": torch.sigmoid(out[..., 1]),
            "ret": torch.sigmoid(out[..., 2]),
        }

    def loss(self, batch):
        from .data import RATIO_CLIP
        items, rb = batch["items"], batch["rbucket"]
        h = self.encode(items[:, :-1], rb[:, :-1])
        out = self.score(h, items[:, 1:])
        m = ((items[:, 1:] != PAD) & (items[:, :-1] != PAD)).float()
        y_r = batch["ratio"][:, 1:] / RATIO_CLIP
        l_r = F.binary_cross_entropy_with_logits(out[..., 0], y_r, reduction="none")
        l_c = F.binary_cross_entropy_with_logits(out[..., 1], batch["cont"][:, 1:], reduction="none")
        l_t = F.binary_cross_entropy_with_logits(out[..., 2], batch["ret"][:, 1:], reduction="none")
        mo = m * batch["obs"][:, 1:]  # цензурированные события не учим на cont/ret
        n, no = m.sum().clamp(min=1), mo.sum().clamp(min=1)
        parts = {"ratio": (l_r * m).sum() / n, "cont": (l_c * mo).sum() / no, "ret": (l_t * mo).sum() / no}
        return parts["ratio"] + 0.5 * parts["cont"] + 0.5 * parts["ret"], parts
