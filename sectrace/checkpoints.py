from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file, save_file
from .config import ModelConfig
from .model import SecTrace
from .util import atomic_json


def resolve_checkpoint(path, prefer="best"):
    path = Path(path)
    if path.is_file() and path.name.endswith(".json"):
        ref = json.loads(path.read_text())["checkpoint"]
        return path.parent / ref
    if (path / "model.safetensors").exists():
        return path
    for key in (prefer, "latest"):
        ref = path / f"{key}.json"
        if ref.exists():
            return path / json.loads(ref.read_text())["checkpoint"]
    raise FileNotFoundError(f"No checkpoint in {path}")


def load_model(path, device="cpu", prefer="best"):
    directory = resolve_checkpoint(path, prefer)
    meta = json.loads((directory / "metadata.json").read_text())
    model = SecTrace(ModelConfig(**meta["model_config"]))
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    model.to(device)
    return model, meta, directory


def save_checkpoint(run, model, optimizer, state, metadata, best=False, keep=3):
    run = Path(run)
    parent = run / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    name = f"step_{state['step']:08d}"
    final, temp = parent / name, parent / ("." + name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    save_file({key: tensor.detach().cpu().contiguous() for key, tensor in model.state_dict().items()},
              str(temp / "model.safetensors"))
    # Trainer state contains tensors and primitive containers only. Load with
    # weights_only=True. Never load an arbitrary untrusted pickle checkpoint.
    torch.save({**state, "optimizer": optimizer.state_dict()}, temp / "trainer.pt")
    atomic_json(temp / "metadata.json", metadata)
    if final.exists():
        raise FileExistsError(f"Checkpoint already exists: {final}")
    os.replace(temp, final)
    relative = str(final.relative_to(run))
    atomic_json(run / "latest.json", {"checkpoint": relative})
    if best:
        atomic_json(run / "best.json", {"checkpoint": relative})
    protected = {name}
    if (run / "best.json").exists():
        protected.add(Path(json.loads((run / "best.json").read_text())["checkpoint"]).name)
    candidates = sorted(p for p in parent.glob("step_*") if p.is_dir())
    for old in candidates[:-max(1, keep)]:
        if old.name not in protected:
            shutil.rmtree(old)
    return final
