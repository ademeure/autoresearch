"""
Attention backend selection for autoresearch.

Prefers Flash Attention 4 on Hopper/Blackwell, then Flash Attention 3, and
falls back to PyTorch SDPA when no fused backend is available.

Override at import time with AUTORESEARCH_FLASH_BACKEND=auto|fa4|fa3|sdpa.
"""

import os

import torch
import torch.nn.functional as F

_FLASH_BACKEND_ENV = "AUTORESEARCH_FLASH_BACKEND"


def _get_cuda_capability():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return None


def _normalize_window_size(window_size):
    left, right = window_size
    left = None if left is None or left < 0 else left
    right = None if right is None or right < 0 else right
    return left, right


def _get_cuda_device_index(device=None):
    get_device_index = getattr(torch.cuda, "_get_device_index", None)
    if callable(get_device_index):
        return get_device_index(device, optional=True)
    if device is None:
        return torch.cuda.current_device()
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device_obj}")
    return torch.cuda.current_device() if device_obj.index is None else device_obj.index


def _patch_fa4_current_stream():
    """Make torch.cuda.current_stream return a CUDA-handle-bearing stream when needed.

    Under torch.compile on Torch 2.10, FA4 can observe a generic torch.Stream
    without the legacy .cuda_stream attribute that flash_attn.cute expects.
    When that happens, rebuild a CUDA stream wrapper from the raw handle.
    """
    original = torch.cuda.current_stream
    if getattr(original, "_autoresearch_fa4_compat", False):
        return False

    raw_stream_getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    external_stream_cls = getattr(torch.cuda, "ExternalStream", None)
    if raw_stream_getter is None or external_stream_cls is None:
        return False

    def current_stream(device=None):
        stream = original(device)
        if hasattr(stream, "cuda_stream"):
            return stream
        device_index = _get_cuda_device_index(device)
        raw_handle = raw_stream_getter(device_index)
        return external_stream_cls(raw_handle, device=device_index)

    current_stream._autoresearch_fa4_compat = True
    current_stream._autoresearch_fa4_original = original
    torch.cuda.current_stream = current_stream
    return True


def _sdpa_attention(q, k, v, window_size):
    """SDPA attention with sliding-window support in (B, H, T, D) layout."""
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class _SDPAFlashAttention:
    name = "sdpa"

    @staticmethod
    def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
        if not causal:
            raise NotImplementedError("autoresearch only uses causal attention")
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = _sdpa_attention(q, k, v, window_size)
        return y.transpose(1, 2)


class _WrappedFlashAttention:
    def __init__(self, name, backend):
        self.name = name
        self._backend = backend

    def flash_attn_func(self, q, k, v, causal=False, window_size=(-1, -1)):
        result = self._backend(q, k, v, causal=causal, window_size=_normalize_window_size(window_size))
        if isinstance(result, tuple):
            return result[0]
        return result


class _FA3FlashAttention:
    name = "fa3"

    def __init__(self, backend):
        self._backend = backend

    def flash_attn_func(self, q, k, v, causal=False, window_size=(-1, -1)):
        return self._backend.flash_attn_func(q, k, v, causal=causal, window_size=window_size)


def _load_flash_attention_4():
    capability = _get_cuda_capability()
    if capability is None:
        return None
    major, _ = capability
    if major < 9:
        return None
    try:
        _patch_fa4_current_stream()
        from flash_attn.cute import flash_attn_func
        return _WrappedFlashAttention("fa4", flash_attn_func)
    except Exception:
        return None


def _load_flash_attention_3():
    capability = _get_cuda_capability()
    if capability is None:
        return None
    try:
        from kernels import get_kernel
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        repo = "varunneal/flash-attention-3" if capability == (9, 0) else "kernels-community/flash-attn3"
        backend = get_kernel(repo).flash_attn_interface
        return _FA3FlashAttention(backend)
    except Exception:
        return None


_fa4 = _load_flash_attention_4()
_fa3 = _load_flash_attention_3()
HAS_FA4 = _fa4 is not None
HAS_FA3 = _fa3 is not None
FA4_STREAM_COMPAT_PATCHED = getattr(torch.cuda.current_stream, "_autoresearch_fa4_compat", False)


def _resolve_flash_attention():
    requested = os.environ.get(_FLASH_BACKEND_ENV, "auto").strip().lower()
    if requested in {"", "auto", "native"}:
        if _fa4 is not None:
            return _fa4
        if _fa3 is not None:
            return _fa3
        return _SDPAFlashAttention()
    if requested == "fa4":
        if _fa4 is None:
            raise RuntimeError("AUTORESEARCH_FLASH_BACKEND=fa4 requested, but FA4 is not available")
        return _fa4
    if requested == "fa3":
        if _fa3 is None:
            raise RuntimeError("AUTORESEARCH_FLASH_BACKEND=fa3 requested, but FA3 is not available")
        return _fa3
    if requested == "sdpa":
        return _SDPAFlashAttention()
    raise ValueError(
        f"Invalid {_FLASH_BACKEND_ENV}={requested!r}; expected auto, fa4, fa3, or sdpa"
    )


flash_attn = _resolve_flash_attention()

FLASH_ATTENTION_IMPL = flash_attn.name
