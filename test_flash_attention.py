import torch

import flash_attention as fa


def test_patch_fa4_current_stream_wraps_generic_stream(monkeypatch):
    class GenericStream:
        pass

    class ExternalStream:
        def __init__(self, handle, device=None):
            self.cuda_stream = handle
            self.device = device

    calls = {}

    def generic_current_stream(device=None):
        calls["current_stream_device"] = device
        return GenericStream()

    def raw_stream(device_index):
        calls["raw_stream_device_index"] = device_index
        return 0xC0FFEE

    monkeypatch.setattr(fa.torch.cuda, "current_stream", generic_current_stream)
    monkeypatch.setattr(fa.torch.cuda, "ExternalStream", ExternalStream)
    monkeypatch.setattr(
        fa.torch.cuda,
        "_get_device_index",
        lambda device, optional=True: 3,
        raising=False,
    )
    monkeypatch.setattr(fa.torch._C, "_cuda_getCurrentRawStream", raw_stream, raising=False)

    assert fa._patch_fa4_current_stream() is True

    stream = fa.torch.cuda.current_stream("cuda:3")
    assert isinstance(stream, ExternalStream)
    assert stream.cuda_stream == 0xC0FFEE
    assert stream.device == 3
    assert calls["current_stream_device"] == "cuda:3"
    assert calls["raw_stream_device_index"] == 3


def test_patch_fa4_current_stream_preserves_cuda_stream(monkeypatch):
    class ExistingCudaStream:
        cuda_stream = 123

    stream = ExistingCudaStream()

    monkeypatch.setattr(fa.torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(fa.torch.cuda, "ExternalStream", object)
    monkeypatch.setattr(
        fa.torch._C,
        "_cuda_getCurrentRawStream",
        lambda device_index: 0xBAD,
        raising=False,
    )

    assert fa._patch_fa4_current_stream() is True
    assert fa.torch.cuda.current_stream() is stream


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
