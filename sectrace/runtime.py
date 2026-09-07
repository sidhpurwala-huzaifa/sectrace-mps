from __future__ import annotations
from contextlib import nullcontext
import json
import os
import platform
import warnings
import psutil
import torch
from torch import nn
from torch.nn import functional as F


def select_device(request="auto", memory_fraction=0.85):
    if request == "auto":
        request = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Use native arm64 Python and an MPS-enabled PyTorch/macOS installation; `sectrace doctor` reports details.")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(request)
    if device.type == "mps":
        if not 0 < memory_fraction <= 1:
            raise ValueError("Use an MPS memory fraction in (0,1]; do not disable allocator safeguards")
        torch.mps.set_per_process_memory_fraction(memory_fraction)
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            warnings.warn("MPS CPU fallback is enabled: timings may include silent CPU operations.")
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device


def autocast_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def probe_bf16(device):
    """Capability check for actual autocast + attention backward + optimizer.

This is not a performance claim. The doctor also tests the real model. Keeping
master weights/Adam moments FP32 avoids pure-BF16 update rounding problems.
"""
    if device.type == "cpu":
        return False, "CPU defaults to FP32 for portable smoke tests"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            projection = nn.Linear(32, 96, bias=False).to(device)
            opt = torch.optim.AdamW(projection.parameters(), lr=1e-3, foreach=False)
            x = torch.randn(1, 16, 32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                q, k, v = projection(x).chunk(3, -1)
                if q.dtype != torch.bfloat16:
                    return False, "Autocast did not produce BF16 projections"
                allow = torch.ones(1, 1, 16, 16, device=device, dtype=torch.bool)
                y = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], attn_mask=allow)
                loss = y.float().square().mean()
            loss.backward()
            if not all(torch.isfinite(p.grad).all().item() for p in projection.parameters()):
                return False, "Non-finite gradients in BF16 probe"
            opt.step()
            synchronize(device)
        return True, "BF16 autocast, SDPA backward and AdamW probe passed"
    except (RuntimeError, TypeError, UserWarning, NotImplementedError) as e:
        return False, str(e)


def choose_precision(device, request="auto"):
    if request == "fp32":
        return "fp32", "Requested FP32"
    ok, message = probe_bf16(device)
    if request == "bf16" and not ok:
        raise RuntimeError(f"BF16 failed its runtime probe: {message}. Use --precision fp32.")
    return ("bf16" if ok else "fp32"), message


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def runtime_info(device=None):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__,
            "unified_or_system_ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
            "available_system_ram_gib": round(psutil.virtual_memory().available / 2**30, 2),
            "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
            "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "0"),
            "mps_prefer_metal": os.environ.get("PYTORCH_MPS_PREFER_METAL", "default")}
    if device is not None:
        info["selected_device"] = str(device)
        if device.type == "mps":
            info.update({"mps_allocated_gib": torch.mps.current_allocated_memory() / 2**30,
                         "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30,
                         "mps_recommended_max_gib": torch.mps.recommended_max_memory() / 2**30})
    return info


def seed_all(seed, device):
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    out = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        out["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
