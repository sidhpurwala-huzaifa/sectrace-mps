from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    dim: int = 640
    layers: int = 16
    heads: int = 10
    ffn_dim: int = 1664
    max_length: int = 8192
    local_radius: int = 256
    global_every: int = 4
    query_chunk: int = 128
    graph_dim: int = 128
    graph_heads: int = 4
    relations: int = 16
    task_dim: int = 256
    edge_dim: int = 64
    families: int = 12
    cwes: int = 64
    evidence_roles: int = 8
    rope_base: float = 10000.0
    dropout: float = 0.0
    checkpoint_blocks: bool = True
    graph_enabled: bool = True

    def __post_init__(self):
        assert self.dim % self.heads == 0 and (self.dim // self.heads) % 2 == 0
        assert self.graph_dim % self.graph_heads == 0
        assert self.layers > 0 and self.global_every > 0 and self.query_chunk > 0
        assert self.local_radius >= 0 and self.max_length > 0
        assert self.vocab_size >= 272
        assert 0 <= self.dropout < 1

    @classmethod
    def tiny(cls):
        return cls(vocab_size=512, dim=64, layers=4, heads=4, ffn_dim=160,
                   max_length=512, local_radius=16, global_every=2, query_chunk=16,
                   graph_dim=32, graph_heads=4, task_dim=32, edge_dim=16)

    def to_dict(self):
        return asdict(self)


def read_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError("Configuration must be a mapping")
    obj.setdefault("model", {})
    obj.setdefault("training", {})
    return obj


def model_config(obj: dict) -> ModelConfig:
    m = dict(obj.get("model", {}))
    preset = m.pop("preset", "100m")
    defaults = ModelConfig.tiny().to_dict() if preset == "tiny" else ModelConfig().to_dict()
    defaults.update(m)
    return ModelConfig(**defaults)
