"""Bidirectional SecTrace encoder. No CUDA-only or third-party graph operators.

Local attention is EXACT sliding-window attention, evaluated in query chunks.
Global attention is also query-chunked, NOT approximated. Global work remains
quadratic. Checkpointing trades recomputation for activation memory.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)).to(dtype) * self.weight.to(dtype)


def rotary(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Real-valued RoPE avoids complex-dtype requirements on MPS."""
    d = x.shape[-1]
    inv = torch.exp(-math.log(base) * torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angles = positions.float()[:, None] * inv[None, :]
    co, si = angles.cos().to(x.dtype)[None, None], angles.sin().to(x.dtype)[None, None]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * co - b * si, a * si + b * co), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, global_layer: bool):
        super().__init__()
        self.cfg, self.global_layer = cfg, global_layer
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h, hd = self.cfg.heads, d // self.cfg.heads
        q, k, v = self.qkv(x).view(b, t, 3, h, hd).permute(2, 0, 3, 1, 4).unbind(0)
        pos = torch.arange(t, device=x.device)
        q, k = rotary(q, pos, self.cfg.rope_base), rotary(k, pos, self.cfg.rope_base)
        chunks = []
        for begin in range(0, t, self.cfg.query_chunk):
            end = min(t, begin + self.cfg.query_chunk)
            kb = 0 if self.global_layer else max(0, begin - self.cfg.local_radius)
            ke = t if self.global_layer else min(t, end + self.cfg.local_radius)
            allowed = valid[:, None, None, kb:ke]
            if not self.global_layer:
                distance = (pos[begin:end, None] - pos[None, kb:ke]).abs()
                allowed = allowed & (distance <= self.cfg.local_radius)[None, None]
            # Padded query rows can otherwise have zero allowed keys. Give them
            # one harmless key and then zero their outputs, including gradients.
            qvalid = valid[:, None, begin:end, None]
            safe_key = (pos[kb:ke] == kb)[None, None, None, :]
            allowed = (allowed & qvalid) | (~qvalid & safe_key)
            z = F.scaled_dot_product_attention(
                q[:, :, begin:end], k[:, :, kb:ke], v[:, :, kb:ke],
                attn_mask=allowed, dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=False,
            )
            chunks.append(z * qvalid.to(z.dtype))
        z = torch.cat(chunks, dim=2).transpose(1, 2).contiguous().view(b, t, d)
        return self.out(z) * valid[..., None].to(z.dtype)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, index: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.attn = Attention(cfg, (index + 1) % cfg.global_every == 0)
        self.up_gate = nn.Linear(cfg.dim, 2 * cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x, valid):
        x = x + self.attn(self.n1(x), valid)
        gate, up = self.up_gate(self.n2(x)).chunk(2, -1)
        return (x + self.down(F.silu(gate) * up)) * valid[..., None].to(x.dtype)


def batched_gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather [B,N,D] at any [B,...] indices without sparse tensor kernels."""
    b, n, d = values.shape
    flat = index.reshape(b, -1)
    out = values.gather(1, flat[..., None].expand(b, flat.shape[1], d))
    return out.reshape(*index.shape, d)


def pool_nodes(h: torch.Tensor, graph: dict[str, torch.Tensor]) -> torch.Tensor:
    """Exact contiguous-span means, accumulated in FP32 for stability."""
    prefix = F.pad(h.float().cumsum(dim=1), (0, 0, 1, 0))
    lo, hi = graph["spans"].unbind(-1)
    summed = batched_gather(prefix, hi) - batched_gather(prefix, lo)
    out = summed / (hi - lo).clamp_min(1)[..., None]
    return out.to(h.dtype) * graph["node_mask"][..., None].to(h.dtype)


class GraphAdapter(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pre = RMSNorm(cfg.dim)
        self.down = nn.Linear(cfg.dim, cfg.graph_dim, bias=False)
        self.norm = RMSNorm(cfg.graph_dim)
        self.qkv = nn.Linear(cfg.graph_dim, 3 * cfg.graph_dim, bias=False)
        self.out = nn.Linear(cfg.graph_dim, cfg.graph_dim, bias=False)
        self.up = nn.Linear(cfg.graph_dim, cfg.dim, bias=False)
        self.relation = nn.Embedding(cfg.relations, cfg.graph_dim)
        self.bias = nn.Embedding(cfg.relations, cfg.graph_heads)
        self.gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, h, graph):
        z = self.norm(self.down(pool_nodes(self.pre(h), graph)))
        b, n, gd = z.shape
        nh, hd = self.cfg.graph_heads, gd // self.cfg.graph_heads
        q, k, v = self.qkv(z).chunk(3, -1)
        nbr, rel = graph["neighbors"], graph["relations"]
        rk = self.relation(rel)
        keys = (batched_gather(k, nbr) + rk).view(b, n, -1, nh, hd)
        vals = (batched_gather(v, nbr) + rk).view(b, n, -1, nh, hd)
        scores = (q.view(b, n, 1, nh, hd).float() * keys.float()).sum(-1) / math.sqrt(hd)
        scores = scores + self.bias(rel).float()
        mask = graph["neighbor_mask"][..., None]
        # Finite sentinel plus post-softmax mask avoids NaN for padded nodes.
        weights = scores.masked_fill(~mask, -1e9).softmax(dim=2) * mask
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-9)
        update = (weights.to(vals.dtype)[..., None] * vals).sum(2).reshape(b, n, gd)
        update = self.up(self.out(update)) * graph["node_mask"][..., None]
        token_nodes = graph["token_nodes"]
        token_mask = graph["token_node_mask"]
        injected = batched_gather(update, token_nodes) * token_mask[..., None]
        injected = injected.sum(2) / token_mask.sum(2).clamp_min(1)[..., None]
        return h + torch.tanh(self.gate).to(h.dtype) * injected.to(h.dtype)


class TaskHeads(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.project = nn.Linear(c.dim, c.task_dim)
        self.risk = nn.Linear(c.task_dim, 1)
        self.family = nn.Linear(c.task_dim, c.families)
        self.cwe = nn.Linear(c.task_dim, c.cwes)
        self.roles = nn.Linear(c.task_dim, c.evidence_roles)
        self.adequacy = nn.Linear(c.task_dim, 1)
        self.edge_q = nn.Linear(c.dim, c.edge_dim, bias=False)
        self.edge_k = nn.Linear(c.dim, c.edge_dim, bias=False)
        self.edge_bias = nn.Embedding(c.relations, 1)
        self.edge_dim = c.edge_dim

    def forward(self, h, target_mask, graph=None):
        w = target_mask.to(h.dtype)
        target = (h * w[..., None]).sum(1) / w.sum(1).clamp_min(1)[:, None]
        pooled = F.gelu(self.project((h[:, 0] + target) * 0.5))
        out = {"risk": self.risk(pooled).squeeze(-1),
               "family": self.family(pooled), "cwe": self.cwe(pooled),
               "adequacy": self.adequacy(pooled).squeeze(-1),
               "roles": self.roles(F.gelu(self.project(h)))}
        if graph is not None:
            nodes = pool_nodes(h, graph)
            q, k = self.edge_q(nodes), self.edge_k(nodes)
            src, dst = graph["edge_index"].unbind(-1)
            out["edges"] = (batched_gather(q, src) * batched_gather(k, dst)).sum(-1) / math.sqrt(self.edge_dim)
            out["edges"] = out["edges"] + self.edge_bias(graph["edge_relations"]).squeeze(-1)
        return out


class SecTrace(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=0)
        self.kind = nn.Embedding(32, cfg.dim)
        self.role = nn.Embedding(8, cfg.dim)
        self.language = nn.Embedding(8, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.layers))
        self.adapters = nn.ModuleDict({str(i): GraphAdapter(cfg) for i in range(cfg.layers)
                                      if (i + 1) % cfg.global_every == 0})
        self.final_norm = RMSNorm(cfg.dim)
        self.heads = TaskHeads(cfg)
        self.mlm_transform = nn.Linear(cfg.dim, cfg.dim)
        self.mlm_norm = RMSNorm(cfg.dim)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.apply(self._init)
        # Residual scaling is a starting optimization, not an accuracy guarantee.
        for b in self.blocks:
            nn.init.normal_(b.attn.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(b.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))
        with torch.no_grad():
            self.token.weight[0].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, batch: dict, task: str = "security") -> dict:
        ids, valid = batch["input_ids"], batch["attention_mask"]
        if ids.shape[1] > self.cfg.max_length:
            raise ValueError("Sequence exceeds configured model capacity")
        h = (self.token(ids) + self.kind(batch["kind_ids"]) + self.role(batch["role_ids"])
             + self.language(batch["language_ids"])) * valid[..., None]
        graph = batch.get("graph") if self.cfg.graph_enabled else None
        for i, block in enumerate(self.blocks):
            if self.training and self.cfg.checkpoint_blocks and torch.is_grad_enabled():
                h = checkpoint(block, h, valid, use_reentrant=False)
            else:
                h = block(h, valid)
            if graph is not None and str(i) in self.adapters:
                adapter = self.adapters[str(i)]
                if self.training and self.cfg.checkpoint_blocks and torch.is_grad_enabled():
                    # Bind adapter explicitly: no late-bound loop variable during backward.
                    h = checkpoint(adapter, h, graph, use_reentrant=False)
                else:
                    h = adapter(h, graph)
        h = self.final_norm(h)
        if task == "mlm":
            # Compute V-way logits only for selected positions, not all tokens.
            pos = batch["mlm_positions"]
            selected = h.reshape(-1, h.shape[-1]).index_select(0, pos)
            z = self.mlm_norm(F.gelu(self.mlm_transform(selected)))
            return {"mlm": F.linear(z, self.token.weight, self.mlm_bias)}
        if task != "security":
            raise ValueError(f"Unknown task: {task}")
        return self.heads(h, batch["target_mask"], graph)

    def set_stage(self, stage: str):
        for p in self.parameters():
            p.requires_grad_(True)
        if stage == "pretrain":
            for p in self.heads.parameters():
                p.requires_grad_(False)
        else:
            for mod in (self.mlm_transform, self.mlm_norm):
                for p in mod.parameters():
                    p.requires_grad_(False)
            self.mlm_bias.requires_grad_(False)
        if not self.cfg.graph_enabled:
            for p in self.adapters.parameters():
                p.requires_grad_(False)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        mlm_only = sum(p.numel() for p in self.mlm_transform.parameters()) + sum(p.numel() for p in self.mlm_norm.parameters()) + self.mlm_bias.numel()
        return {"training_model": total, "inference_model": total - mlm_only,
                "trainable_now": sum(p.numel() for p in self.parameters() if p.requires_grad)}
