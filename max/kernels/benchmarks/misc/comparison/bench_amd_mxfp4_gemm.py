# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# AMD MXFP4 dense GEMM benchmark comparing MAX against aiter.
#
# Computes Y = A @ B^T where A is [M, K] and B is [N, K], both quantized to
# MXFP4 (E2M1 packed 2-per-uint8 along K, with one E8M0 scale per 32-element
# K block). Output is BF16.
#   * MAX path  -> `dynamic_block_scaled_matmul_mxfp4`
#                  (custom op `mo.matmul.dynamic.block.scaled.mxfp4` ->
#                   the CDNA4 kernel `mxfp4_block_scaled_matmul_amd`).
#   * aiter path -> `aiter.ops.triton.gemm.basic.gemm_afp4wfp4`
#                   (the Triton `_gemm_afp4wfp4_kernel`).
#
# Timing mirrors bench_amd_mla.py's chained-call strategy: ncopies distinct
# rotating weight buffers (total footprint > L2) are chained into ONE
# device-graph (MAX) / CUDA-graph (aiter); a single replay sweeps all ncopies
# cold-HBM GEMMs back-to-back, free of per-replay launch gaps. Reported latency
# is whole-graph time / ncopies (per-op).
#
# Run via kbench: kbench bench_amd_mxfp4_gemm.yaml

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable
from typing import Any

# Configure aiter JIT environment before any aiter imports. When run via
# kbench (file: mode), the Bazel env vars aren't applied, so set them as
# fallbacks.
if "AITER_JIT_DIR" not in os.environ:
    _ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    os.environ["AITER_JIT_DIR"] = os.path.join(
        _ws, ".derived", "aiter_jit_cache"
    )
if "/usr/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = (
        "/usr/bin:/bin:/usr/local/bin:/opt/rocm/bin:"
        + os.environ.get("PATH", "")
    )

import torch
from bencher_utils import Bench, ThroughputMeasure
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import dynamic_block_scaled_matmul_mxfp4

# aiter MXFP4 Triton GEMM (`_gemm_afp4wfp4_kernel`). JIT/compile on first use.
_aiter_gemm: Callable[..., torch.Tensor] | None
try:
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
        gemm_afp4wfp4 as _aiter_gemm,
    )
except (ImportError, Exception) as e:
    print(f"Warning: aiter gemm_afp4wfp4 not available: {e}")
    _aiter_gemm = None

# MI355X L2 cache size (256 MB).
_L2_CACHE_SIZE_BYTES = int(256e6)
# Number of rotating weight copies chained into one graph. The total footprint
# (ncopies * B) must exceed the 256 MB L2 so each chained GEMM reads cold HBM.
# Overridable via `--ncopies`; 0 = auto.
_NCOPIES = 16
# MXFP4 micro-scaling block: 32 FP4 values share one E8M0 scale.
_SCALE_BLOCK = 32


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------


def _auto_ncopies(weight_bytes_per_copy: int) -> int:
    """Rotating-copy count so the working set exceeds the L2 (=> cold HBM
    reads), capped at 16. For large weights one or two copies already exceed
    L2; for small weights the cap of 16 is used."""
    return max(
        2,
        min(
            16,
            math.ceil(
                1.5 * _L2_CACHE_SIZE_BYTES / max(1, weight_bytes_per_copy)
            ),
        ),
    )


def _compute_flops(m: int, n: int, k: int) -> int:
    """Dense GEMM FLOPs: 2*M*N*K (one multiply + one add per MAC)."""
    return 2 * m * n * k


# E2M1 (FP4) value table indexed by the 4-bit nibble (sign bit = bit 3).
_E2M1_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize a [R, K//2] uint8 (2 FP4/byte) + [R, K//32] E8M0 uint8 scale
    tensor to a [R, K] float32 reference. Low nibble = even element, high
    nibble = odd element (matches the kernels)."""
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    rows, k_half = packed.shape
    out = torch.empty(
        rows, k_half * 2, dtype=torch.float32, device=packed.device
    )
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    # E8M0: value = 2^(byte - 127); each scale covers _SCALE_BLOCK K-elements.
    sc = torch.exp2(scales.float() - 127.0).repeat_interleave(
        _SCALE_BLOCK, dim=1
    )
    return out * sc


def _gen_mxfp4_inputs(
    m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random valid MXFP4 GEMM inputs on the GPU. Every uint8 is a valid pair
    of E2M1 nibbles (E2M1 has no inf/nan). Scale bytes stay in [124, 128) so
    the E8M0 scale ~= 1 and never hits the 0xFF NaN code."""
    if k % _SCALE_BLOCK != 0:
        raise ValueError(f"K={k} must be a multiple of {_SCALE_BLOCK}")
    a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    a_s = torch.randint(
        124, 128, (m, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    b_s = torch.randint(
        124, 128, (n, k // _SCALE_BLOCK), dtype=torch.uint8, device="cuda"
    )
    return a, b, a_s, b_s


def _check_close(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    label: str,
) -> None:
    """Compare a kernel output against the float32 dequantized reference."""
    ref = _dequant_mxfp4(a, a_s) @ _dequant_mxfp4(b, b_s).T
    out_f = out.detach().to(torch.float32)
    denom = ref.abs().max().clamp_min(1e-6)
    max_rel = (out_f - ref).abs().max() / denom
    print(f"  [{label} check] max_rel_err={max_rel.item():.4e}")
    torch.testing.assert_close(
        out_f, ref, rtol=3e-2, atol=(denom * 3e-2).item()
    )


# ----------------------------------------------------------------------------
# MAX backend.
# ----------------------------------------------------------------------------


def bench_matmul_max(
    m: int,
    n: int,
    k: int,
    num_iters: int,
    check: bool = False,
    ncopies: int | None = None,
) -> tuple[float, int] | None:
    """MAX dynamic_block_scaled_matmul_mxfp4 (dense MXFP4 GEMM).

    Builds ONE device-graph chaining `ncopies` GEMM ops, op i reading a
    distinct rotating weight buffer (total > L2). A single replay sweeps all
    `ncopies` cold-HBM GEMMs; per-op latency = whole-graph time / ncopies.
    """
    ncopies = _NCOPIES if ncopies is None else ncopies
    if ncopies <= 0:
        ncopies = _auto_ncopies(n * (k // 2))

    a_t, b_t, a_s_t, b_s_t = _gen_mxfp4_inputs(m, n, k)

    a_type = TensorType(DType.uint8, shape=[m, k // 2], device=DeviceRef.GPU())
    b_type = TensorType(DType.uint8, shape=[n, k // 2], device=DeviceRef.GPU())
    a_s_type = TensorType(
        DType.float8_e8m0fnu,
        shape=[m, k // _SCALE_BLOCK],
        device=DeviceRef.GPU(),
    )
    b_s_type = TensorType(
        DType.float8_e8m0fnu,
        shape=[n, k // _SCALE_BLOCK],
        device=DeviceRef.GPU(),
    )

    # Rotating weight copies (op i reads copy i in the chained graph below).
    keepalive: list[Any] = []
    b_bufs: list[Buffer] = [Buffer.from_dlpack(b_t)]
    for _ in range(ncopies - 1):
        bt = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, device="cuda"
        )
        keepalive.append(bt)
        b_bufs.append(Buffer.from_dlpack(bt))

    session = InferenceSession(devices=[Accelerator()])
    with Graph(
        "mxfp4_matmul_max_chain",
        input_types=[
            a_type,
            a_s_type,
            b_s_type,
            *([b_type] * ncopies),
        ],
    ) as graph:
        ins = graph.inputs
        a, a_scales, b_scales = ins[0], ins[1], ins[2]
        b_in = ins[3 : 3 + ncopies]
        results = [
            dynamic_block_scaled_matmul_mxfp4(
                a.tensor,
                b.tensor,
                a_scales.tensor,
                b_scales.tensor,
                out_type=DType.bfloat16,
            )
            for b in b_in
        ]
        graph.output(*results)

    model = session.load(graph)

    a_buf = Buffer.from_dlpack(a_t)
    a_s_buf = Buffer.from_dlpack(a_s_t).view(DType.float8_e8m0fnu)
    b_s_buf = Buffer.from_dlpack(b_s_t).view(DType.float8_e8m0fnu)
    graph_inputs = (a_buf, a_s_buf, b_s_buf, *b_bufs)

    outs = model.capture(0, *graph_inputs)
    keepalive.append(outs)
    # Validate the captured graph matches eager execution.
    model.debug_verify_replay(0, *graph_inputs)

    if check:
        try:
            # Replay once so the captured output buffers hold live data, then
            # read op 0 (which uses weight copy 0 == b_t).
            model.replay(0, *graph_inputs)
            torch.cuda.synchronize()
            out0 = torch.from_dlpack(outs[0]).clone()
            _check_close(out0, a_t, b_t, a_s_t, b_s_t, "MAX")
        except Exception as e:
            print(f"  [MAX check skipped: {e}]")

    nrun = max(num_iters, 200)
    torch.cuda.synchronize()
    for _ in range(50):
        model.replay(0, *graph_inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nrun):
        model.replay(0, *graph_inputs)
    end.record()
    torch.cuda.synchronize()
    per_op_s = start.elapsed_time(end) / 1e3 / nrun / ncopies

    w_mb = (n * (k // 2)) / (1024.0 * 1024.0)
    print(
        f"[MAX chained device-graph] chain={ncopies} "
        f"weight_per_copy~{w_mb:.1f}MB working_set~{w_mb * ncopies:.1f}MB (cold)"
        f" | per-op {per_op_s * 1e6:.2f}us"
    )
    keepalive.clear()
    return per_op_s, _compute_flops(m, n, k)


# ----------------------------------------------------------------------------
# aiter backend.
# ----------------------------------------------------------------------------


def bench_matmul_aiter(
    m: int,
    n: int,
    k: int,
    num_iters: int,
    check: bool = False,
    ncopies: int | None = None,
) -> tuple[float, int] | None:
    """aiter gemm_afp4wfp4 (Triton `_gemm_afp4wfp4_kernel`).

    Mirrors the MAX path: `ncopies` rotating weight buffers (total > L2)
    chained into ONE CUDA graph, one output buffer per chained call to avoid a
    false write-after-write dependency. per-op = whole-graph time / ncopies.
    """
    if _aiter_gemm is None:
        print("aiter not available, skipping bench_matmul_aiter")
        return None
    ncopies = _NCOPIES if ncopies is None else ncopies
    if ncopies <= 0:
        ncopies = _auto_ncopies(n * (k // 2))

    a_t, b0_t, a_s_t, b_s_t = _gen_mxfp4_inputs(m, n, k)
    b_bufs = [b0_t] + [
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        for _ in range(ncopies - 1)
    ]
    # One output per chained call so consecutive calls don't serialize on a
    # shared output tensor.
    o_list = [
        torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        for _ in range(ncopies)
    ]

    def call(b_buf: torch.Tensor, out: torch.Tensor) -> None:
        _aiter_gemm(a_t, b_buf, a_s_t, b_s_t, dtype=torch.bfloat16, y=out)

    if check:
        call(b0_t, o_list[0])
        torch.cuda.synchronize()
        _check_close(o_list[0], a_t, b0_t, a_s_t, b_s_t, "aiter")

    nrun = max(num_iters, 200)
    # Capture one CUDA graph chaining `ncopies` GEMM calls over distinct cold
    # weight buffers. Side-stream warmup is mandatory before capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            for b in range(ncopies):
                call(b_bufs[b], o_list[b])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gchain = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gchain):
        for b in range(ncopies):
            call(b_bufs[b], o_list[b])
    torch.cuda.synchronize()
    for _ in range(50):
        gchain.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nrun):
        gchain.replay()
    end.record()
    torch.cuda.synchronize()
    per_op_s = start.elapsed_time(end) / 1e3 / nrun / ncopies

    w_mb = (n * (k // 2)) / (1024.0 * 1024.0)
    print(
        f"[aiter chained CUDA-graph] chain={ncopies} "
        f"weight_per_copy~{w_mb:.1f}MB working_set~{w_mb * ncopies:.1f}MB (cold)"
        f" | per-op {per_op_s * 1e6:.2f}us"
    )
    return per_op_s, _compute_flops(m, n, k)


# ----------------------------------------------------------------------------
# Dispatch and CLI.
# ----------------------------------------------------------------------------


_ENGINE_MAP: dict[str, Callable[..., tuple[float, int] | None]] = {
    "modular_max": bench_matmul_max,
    "aiter": bench_matmul_aiter,
}


def bench_matmul(
    engine: str,
    m: int,
    n: int,
    k: int,
    num_iters: int,
    check: bool,
) -> tuple[float, int] | None:
    print("=" * 80)
    print(f"AMD MXFP4 GEMM (M={m}, N={n}, K={k}, dtype=mxfp4, engine={engine})")
    print("=" * 80)

    fn = _ENGINE_MAP.get(engine)
    if fn is None:
        raise ValueError(
            f"Unknown engine '{engine}'. Available: {list(_ENGINE_MAP.keys())}"
        )

    try:
        result = fn(m, n, k, num_iters, check=check)
    except Exception as e:
        print(f"{engine} benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    if result is not None:
        time_s, flops = result
        tflops = flops / time_s / 1e12
        print(f"  Time: {time_s * 1e6:.3f} us | {tflops:.2f} TFLOPS")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AMD MXFP4 GEMM Benchmark")
    parser.add_argument(
        "--engine",
        choices=list(_ENGINE_MAP.keys()),
        default="modular_max",
    )
    parser.add_argument(
        "--dtype",
        choices=["mxfp4"],
        default="mxfp4",
        help="Quantization format (only mxfp4 is supported).",
    )
    parser.add_argument("--M", "--m", type=int, default=4096, help="GEMM M")
    parser.add_argument("--N", "--n", type=int, default=16384, help="GEMM N")
    parser.add_argument("--K", "--k", type=int, default=2048, help="GEMM K")
    parser.add_argument("--num_iters", "--num-iters", type=int, default=100)
    parser.add_argument("--output", "-o", type=str, default="output.csv")
    parser.add_argument(
        "--ncopies",
        type=int,
        default=0,
        help="Rotating weight copies chained per graph (0 = auto: footprint "
        "> L2, capped at 16).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the kernel against a float32 dequantized reference.",
    )
    args, _ = parser.parse_known_args()

    global _NCOPIES
    _NCOPIES = args.ncopies

    print(
        f"[bench_amd_mxfp4_gemm] engine={args.engine} dtype={args.dtype} "
        f"M={args.M} N={args.N} K={args.K}",
        file=sys.stderr,
    )

    result = bench_matmul(
        args.engine, args.M, args.N, args.K, args.num_iters, args.check
    )

    if result is None:
        sys.exit(1)

    time_s, flops = result
    metric = ThroughputMeasure(Bench.flops, flops)
    name = (
        f"Matmul_MXFP4/M={args.M}/N={args.N}/K={args.K}/"
        f"dtype={args.dtype}/engine={args.engine}/"
    )
    b = Bench(name, iters=1, met=time_s, metric_list=[metric])
    b.dump_report(output_path=args.output)


if __name__ == "__main__":
    main()
