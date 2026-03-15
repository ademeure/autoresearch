"""
Attention backend selection for autoresearch.

Prefers Flash Attention 4 on Hopper/Blackwell, then Flash Attention 3, and
falls back to PyTorch SDPA when no fused backend is available.
"""

import os

import torch
import torch.nn.functional as F


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


flash_attn = _load_flash_attention_4()
if flash_attn is None:
    flash_attn = _load_flash_attention_3()
if flash_attn is None:
    flash_attn = _SDPAFlashAttention()

FLASH_ATTENTION_IMPL = flash_attn.name
