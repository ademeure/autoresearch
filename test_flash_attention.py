import torch

import flash_attention as fa


def test_flash_attention_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if device == "cpu":
        assert fa.FLASH_ATTENTION_IMPL == "sdpa"

    B, T, H, D = 2, 16, 4, 32
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)

    y = fa.flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(8, 0))
    loss = y.float().sum()
    loss.backward()

    assert y.shape == (B, T, H, D)
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
