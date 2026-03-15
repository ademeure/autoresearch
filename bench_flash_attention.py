import argparse
import gc
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import flash_attention as auto_fa


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    seqlen: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    window_size: tuple[int, int]


BENCH_CASES = [
    Case("ar_sliding_2k", batch=16, seqlen=2048, num_heads=4, num_kv_heads=4, head_dim=128, window_size=(1024, 0)),
    Case("ar_full_2k", batch=16, seqlen=2048, num_heads=4, num_kv_heads=4, head_dim=128, window_size=(-1, -1)),
    Case("gqa_sliding_2k", batch=8, seqlen=2048, num_heads=8, num_kv_heads=2, head_dim=128, window_size=(1024, 0)),
]

STRESS_CASES = BENCH_CASES + [
    Case("long_full_4k", batch=4, seqlen=4096, num_heads=8, num_kv_heads=8, head_dim=128, window_size=(-1, -1)),
]


def _format_case(case: Case) -> str:
    return (
        f"B={case.batch} T={case.seqlen} H={case.num_heads}/{case.num_kv_heads} "
        f"D={case.head_dim} window={case.window_size}"
    )


def _make_data(case: Case, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(case.batch, case.seqlen, case.num_heads, case.head_dim, device=device, dtype=dtype)
    k = torch.randn(case.batch, case.seqlen, case.num_kv_heads, case.head_dim, device=device, dtype=dtype)
    v = torch.randn(case.batch, case.seqlen, case.num_kv_heads, case.head_dim, device=device, dtype=dtype)
    return q, k, v


def _sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
    tq = q.size(2)
    tk = k.size(2)
    window = window_size[0]
    enable_gqa = q.size(1) != k.size(1)

    if (window < 0 or window >= tq) and tq == tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    if tq == 1:
        if window >= 0 and window < tk:
            start = max(0, tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    row_idx = (tk - tq) + torch.arange(tq, device=q.device).unsqueeze(1)
    col_idx = torch.arange(tk, device=q.device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def _sdpa_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if not causal:
        raise NotImplementedError("benchmark baseline only supports causal attention")
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = _sdpa_attention(q, k, v, window_size)
    return y.transpose(1, 2)


def _run_fwd_bench_once(fn, q, k, v, window_size):
    with torch.no_grad():
        fn(q, k, v, causal=True, window_size=window_size)


def _run_fwd_checked_once(fn, q, k, v, window_size):
    with torch.no_grad():
        y = fn(q, k, v, causal=True, window_size=window_size)
    if not torch.isfinite(y).all():
        raise RuntimeError("non-finite output in forward pass")


def _run_fwdbwd_bench_once(fn, q, k, v, window_size):
    q.grad = None
    k.grad = None
    v.grad = None
    y = fn(q, k, v, causal=True, window_size=window_size)
    loss = y.float().square().mean()
    loss.backward()


def _run_fwdbwd_checked_once(fn, q, k, v, window_size):
    q.grad = None
    k.grad = None
    v.grad = None
    y = fn(q, k, v, causal=True, window_size=window_size)
    if not torch.isfinite(y).all():
        raise RuntimeError("non-finite output in forward+backward pass")
    loss = y.float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss in forward+backward pass")
    loss.backward()
    for name, grad in (("q", q.grad), ("k", k.grad), ("v", v.grad)):
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"non-finite gradient for {name}")


def _benchmark_one(name: str, fn, case: Case, mode: str, dtype: torch.dtype, warmup: int, iters: int):
    device = torch.device("cuda")
    q_data, k_data, v_data = _make_data(case, device, dtype)

    if mode == "fwd":
        q = q_data
        k = k_data
        v = v_data
        runner = _run_fwd_bench_once
        tokens_per_iter = case.batch * case.seqlen
    else:
        q = q_data.detach().clone().requires_grad_(True)
        k = k_data.detach().clone().requires_grad_(True)
        v = v_data.detach().clone().requires_grad_(True)
        runner = _run_fwdbwd_bench_once
        tokens_per_iter = case.batch * case.seqlen

    for _ in range(warmup):
        runner(fn, q, k, v, case.window_size)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner(fn, q, k, v, case.window_size)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    mean_ms = sum(times_ms) / len(times_ms)
    sorted_times = sorted(times_ms)
    p50_ms = sorted_times[len(sorted_times) // 2]
    p90_ms = sorted_times[min(len(sorted_times) - 1, math.ceil(len(sorted_times) * 0.9) - 1)]
    attn_tok_per_sec = tokens_per_iter / (mean_ms / 1000.0)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "backend": name,
        "mode": mode,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "attn_tok_per_sec": attn_tok_per_sec,
        "peak_mem_mb": peak_mem_mb,
    }


def _compare_against_sdpa(case: Case, dtype: torch.dtype):
    device = torch.device("cuda")
    q_data, k_data, v_data = _make_data(case, device, dtype)

    def run_once(fn):
        q = q_data.detach().clone().requires_grad_(True)
        k = k_data.detach().clone().requires_grad_(True)
        v = v_data.detach().clone().requires_grad_(True)
        y = fn(q, k, v, causal=True, window_size=case.window_size)
        loss = y.float().square().mean()
        loss.backward()
        return y, q.grad, k.grad, v.grad

    y_auto, qg_auto, kg_auto, vg_auto = run_once(auto_fa.flash_attn.flash_attn_func)
    y_ref, qg_ref, kg_ref, vg_ref = run_once(_sdpa_flash_attn_func)

    return {
        "out_max_abs": (y_auto - y_ref).abs().max().item(),
        "out_mean_abs": (y_auto - y_ref).abs().mean().item(),
        "q_grad_max_abs": (qg_auto - qg_ref).abs().max().item(),
        "k_grad_max_abs": (kg_auto - kg_ref).abs().max().item(),
        "v_grad_max_abs": (vg_auto - vg_ref).abs().max().item(),
    }


def _stress_case(case: Case, dtype: torch.dtype, iters: int):
    device = torch.device("cuda")
    q_data, k_data, v_data = _make_data(case, device, dtype)
    q = q_data.detach().clone().requires_grad_(True)
    k = k_data.detach().clone().requires_grad_(True)
    v = v_data.detach().clone().requires_grad_(True)

    torch.cuda.empty_cache()
    gc.collect()
    baseline_alloc = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(iters):
        _run_fwdbwd_checked_once(auto_fa.flash_attn.flash_attn_func, q, k, v, case.window_size)

    q.grad = None
    k.grad = None
    v.grad = None
    torch.cuda.synchronize()
    gc.collect()
    final_alloc = torch.cuda.memory_allocated(device)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    alloc_delta_mb = (final_alloc - baseline_alloc) / (1024 ** 2)
    return {
        "iters": iters,
        "peak_mem_mb": peak_mem_mb,
        "alloc_delta_mb": alloc_delta_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark and stress test autoresearch flash attention.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--stress-iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16

    print(f"device={device_name} capability={capability[0]}.{capability[1]}")
    print(f"auto_backend={auto_fa.FLASH_ATTENTION_IMPL}")
    print()

    for case in BENCH_CASES:
        print(f"[bench] {case.name} {_format_case(case)}")
        auto_fwd = _benchmark_one("auto", auto_fa.flash_attn.flash_attn_func, case, "fwd", dtype, args.warmup, args.iters)
        sdpa_fwd = _benchmark_one("sdpa", _sdpa_flash_attn_func, case, "fwd", dtype, args.warmup, args.iters)
        auto_fwdbwd = _benchmark_one("auto", auto_fa.flash_attn.flash_attn_func, case, "fwdbwd", dtype, args.warmup, args.iters)
        sdpa_fwdbwd = _benchmark_one("sdpa", _sdpa_flash_attn_func, case, "fwdbwd", dtype, args.warmup, args.iters)
        diff = _compare_against_sdpa(case, dtype)

        print(
            "  fwd: "
            f"auto={auto_fwd['mean_ms']:.2f}ms "
            f"sdpa={sdpa_fwd['mean_ms']:.2f}ms "
            f"speedup={sdpa_fwd['mean_ms'] / auto_fwd['mean_ms']:.2f}x "
            f"auto_peak={auto_fwd['peak_mem_mb']:.0f}MB "
            f"sdpa_peak={sdpa_fwd['peak_mem_mb']:.0f}MB"
        )
        print(
            "  fwdbwd: "
            f"auto={auto_fwdbwd['mean_ms']:.2f}ms "
            f"sdpa={sdpa_fwdbwd['mean_ms']:.2f}ms "
            f"speedup={sdpa_fwdbwd['mean_ms'] / auto_fwdbwd['mean_ms']:.2f}x "
            f"auto_peak={auto_fwdbwd['peak_mem_mb']:.0f}MB "
            f"sdpa_peak={sdpa_fwdbwd['peak_mem_mb']:.0f}MB"
        )
        print(
            "  diff: "
            f"out_max={diff['out_max_abs']:.5f} "
            f"out_mean={diff['out_mean_abs']:.5f} "
            f"q_grad_max={diff['q_grad_max_abs']:.5f} "
            f"k_grad_max={diff['k_grad_max_abs']:.5f} "
            f"v_grad_max={diff['v_grad_max_abs']:.5f}"
        )
        print()

    for case in STRESS_CASES:
        print(f"[stress] {case.name} {_format_case(case)}")
        result = _stress_case(case, dtype, args.stress_iters)
        print(
            f"  pass: iters={result['iters']} "
            f"peak={result['peak_mem_mb']:.0f}MB "
            f"alloc_delta={result['alloc_delta_mb']:.2f}MB"
        )
        print()


if __name__ == "__main__":
    main()
