from __future__ import annotations
import torch
from torch.nn import functional as F


def masked_bce(logits, targets, weights, pos_weight=1.0):
    """Unknown=-1 contributes neither positive nor negative supervision."""
    mask = targets >= 0
    safe = torch.where(mask, targets, torch.zeros_like(targets))
    pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=logits.device)
    element = F.binary_cross_entropy_with_logits(logits.float(), safe.float(), reduction="none", pos_weight=pw) * mask
    if logits.ndim > 1:
        dims = tuple(range(1, logits.ndim))
        count = mask.sum(dim=dims)
        per_item = element.sum(dim=dims) / count.clamp_min(1)
    else:
        count, per_item = mask, element
    scale = weights * (count > 0)
    return (per_item * scale).sum() / scale.sum().clamp_min(1e-8)


def security_loss(outputs, batch, settings=None):
    settings = settings or {}
    mapping = {"risk": "label", "cwe": "cwe", "family": "family", "roles": "roles", "adequacy": "adequate"}
    defaults = {"risk": 1.0, "cwe": .2, "family": .1, "roles": .5, "adequacy": .3, "edges": .3, "pair": .2, "teacher": 0.0}
    terms = {}
    for head, label in mapping.items():
        terms[head] = masked_bce(outputs[head], batch[label], batch["weight"],
                                 settings.get("positive_weight", 1.0) if head == "risk" else 1.0)
    if "edges" in outputs:
        terms["edges"] = masked_bce(outputs["edges"], batch["graph"]["edge_targets"], batch["weight"])
    pairs = batch["pairs"]
    if pairs.shape[0]:
        a = outputs["risk"].index_select(0, pairs[:, 0]).float()
        b = outputs["risk"].index_select(0, pairs[:, 1]).float()
        ranked = F.softplus(float(settings.get("pair_margin", 1.0)) - (a - b))
        invariant = (a.sigmoid() - b.sigmoid()).square()
        terms["pair"] = torch.where(pairs[:, 2] == 0, ranked, invariant).mean()
    terms["teacher"] = masked_bce(outputs["risk"], batch["teacher_probability"], batch["weight"])
    total = sum(value * float(settings.get(key, defaults[key])) for key, value in terms.items())
    return total, terms


def count_supervision(batch, previous=None):
    """Counts supervised presentations, not distinct examples; inference guards."""
    result = previous if previous is not None else {}
    for head, field in (("risk", "label"), ("cwe", "cwe"), ("family", "family"), ("roles", "roles"), ("adequacy", "adequate")):
        value = batch[field]
        size = value.shape[-1] if value.ndim > 1 else 1
        flat = value.reshape(-1, size)
        p, n = (flat > .5).sum(0).tolist(), ((flat >= 0) & (flat <= .5)).sum(0).tolist()
        old = result.setdefault(head, {"positive": [0] * size, "negative": [0] * size})
        old["positive"] = [a + int(b) for a, b in zip(old["positive"], p)]
        old["negative"] = [a + int(b) for a, b in zip(old["negative"], n)]
    return result
