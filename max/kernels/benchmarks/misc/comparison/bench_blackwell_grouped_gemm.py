# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# ===----------------------------------------------------------------------=== #
# DeepGEMM grouped matmul benchmark (contiguous layout).
# Mimics bench.py scaffolding; uses naive float64 reference for correctness.
#
# Run via Bazel: ./bazelw run //max/kernels/benchmarks/misc/comparison:bench_grouped_gemm
# Usage example:
#   python $MODULAR_PATH/max/kernels/benchmarks/misc/comparison/setup_bench_env.py
#   source $MODULAR_PATH/.venv/bin/activate
#   # Uniform M per group (results in num_groups = 2, M = [64, 64]).
#   br //max/kernels/benchmarks/misc/comparison:bench_grouped_gemm -- --layout contiguous --shapes 2,64,512,512 --num-tests 5
#   # Varied M per group (results in num_groups = 2, M = [31, 97]).
#   br //max/kernels/benchmarks/misc/comparison:bench_grouped_gemm -- --layout contiguous --shapes 2,[31;97],512,512 --num-tests 5
# ===----------------------------------------------------------------------=== #

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import TypeAlias

import torch

# Local bench helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# DeepGEMM
import deep_gemm
from bench import bench_kineto_with_cupti_warmup
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import (
    get_mk_alignment_for_contiguous_layout,
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
)

Tensor: TypeAlias = torch.Tensor
FP8Pair: TypeAlias = tuple[Tensor, Tensor]
ContigInputs: TypeAlias = tuple[
    Tensor | FP8Pair,
    Tensor | FP8Pair,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    int,
]


@dataclass
class ShapeCfg:
    num_groups: int
    m_per_group: list[int]
    n: int
    k: int


def _parse_m_list(m_field: str) -> list[int]:
    """Parse m field: either scalar or bracketed list like [4;500]."""

    m_field = m_field.strip()
    if m_field.startswith("[") and m_field.endswith("]"):
        body = m_field[1:-1].strip()
        if not body:
            raise ValueError("Empty m_per_group list")
        return [int(x.strip()) for x in body.split(";")]
    return [int(m_field)]


def parse_shape(shape_str: str) -> ShapeCfg:
    """Parse a single CLI shape string into a config.

    Format: num_groups,m_per_group,n,k where m_per_group can be scalar or
    bracketed list, e.g. 2,[4;500],512,512.
    """

    parts = shape_str.split(",")
    if len(parts) != 4:
        raise ValueError(f"Shape must be 'num_groups,m,n,k', got {shape_str}")
    num_groups = int(parts[0])
    m_list = _parse_m_list(parts[1])
    n = int(parts[2])
    k = int(parts[3])
    if len(m_list) not in (1, num_groups):
        raise ValueError(
            "m_per_group list length must be 1 or num_groups; "
            f"got {len(m_list)} vs num_groups={num_groups}"
        )
    if len(m_list) == 1:
        m_list = [m_list[0]] * num_groups
    return ShapeCfg(num_groups=num_groups, m_per_group=m_list, n=n, k=k)


def align_up(x: int, align: int) -> int:
    """Round x up to the nearest multiple of align."""

    return (x + align - 1) // align * align


def generate_contiguous_inputs(
    cfg: ShapeCfg, dtype: torch.dtype
) -> ContigInputs:
    """Build contiguous-layout inputs plus padding indices."""
    align = get_mk_alignment_for_contiguous_layout()
    actual_ms = cfg.m_per_group
    aligned_ms = [align_up(m, align) for m in actual_ms]
    m_total = sum(aligned_ms)

    a_bf16 = torch.randn((m_total, cfg.k), device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn(
        (cfg.num_groups, cfg.n, cfg.k), device="cuda", dtype=torch.bfloat16
    )

    m_indices = torch.empty(m_total, device="cuda", dtype=torch.int32)
    start = 0
    for g, (m_real, m_aligned) in enumerate(
        zip(actual_ms, aligned_ms, strict=False)
    ):
        end_real = start + m_real
        end_aligned = start + m_aligned
        m_indices[start:end_real] = g
        m_indices[end_real:end_aligned] = -1
        a_bf16[end_real:end_aligned].zero_()
        start = end_aligned

    use_ue8m0 = get_arch_major() != 9
    if dtype == torch.float8_e4m3fn:
        a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=use_ue8m0)
        b_fp8 = (
            torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn),
            torch.empty(
                (
                    cfg.num_groups,
                    math.ceil(cfg.n / 128),
                    math.ceil(cfg.k / 128),
                ),
                device="cuda",
                dtype=torch.float,
            ),
        )
        for i in range(cfg.num_groups):
            b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(
                b_bf16[i], use_ue8m0=use_ue8m0
            )
        a_in, b_in = a_fp8, b_fp8
    else:
        a_in, b_in = a_bf16, b_bf16

    d_out = torch.empty((m_total, cfg.n), device="cuda", dtype=torch.bfloat16)
    return a_in, b_in, m_indices, d_out, a_bf16, b_bf16, sum(actual_ms)


def contiguous_reference(
    a_bf16: torch.Tensor, b_bf16: torch.Tensor, m_indices: torch.Tensor
) -> torch.Tensor:
    m_total, _ = a_bf16.shape
    num_groups, n, _ = b_bf16.shape
    out = torch.zeros((m_total, n), device="cuda", dtype=torch.float64)
    for g in range(num_groups):
        mask = m_indices == g
        if not mask.any():
            continue
        a_slice = a_bf16[mask].to(torch.float64)
        b_slice = b_bf16[g].to(torch.float64)
        out[mask] = a_slice @ b_slice.T
    return out


def run_case(cfg: ShapeCfg, args: argparse.Namespace) -> tuple[float, float]:
    if args.layout == "masked":
        raise NotImplementedError("masked layout is not supported")

    use_fp8 = args.dtype == "fp8"
    dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16

    a_in, b_in, m_idx, d_out, a_bf16, b_bf16, m_effective = (
        generate_contiguous_inputs(cfg, dtype)
    )

    def call() -> None:
        if use_fp8:
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_in, b_in, d_out, m_idx)
        else:
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                a_in, b_in, d_out, m_idx
            )

    ref = contiguous_reference(a_bf16, b_bf16, m_idx) if args.check else None

    # correctness
    if ref is not None:
        call()
        if use_fp8:
            rtol, atol = 5e-2, 5e-1
        else:
            rtol, atol = 1e-2, 1e-2
        torch.testing.assert_close(
            d_out.to(torch.float64), ref, rtol=rtol, atol=atol
        )

    # timing with CUPTI warmup for CUTLASS kernels
    # Use specific pattern to match only the actual GEMM kernel, not helper kernels
    # FP8 kernel: sm100_fp8_gemm_1d1d_impl, BF16 kernel: sm100_bf16_gemm_impl
    # Pattern "_gemm_" matches both but not "deep_gemm::transpose_and_pack..."
    kernel_name = "_gemm_"
    time_s = bench_kineto_with_cupti_warmup(
        call,
        kernel_names=kernel_name,
        num_tests=args.num_tests,
        suppress_kineto_output=True,
    )
    assert isinstance(time_s, float), "Expected single kernel timing"

    tflops = 2 * m_effective * cfg.n * cfg.k / time_s / 1e12
    return time_s, tflops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepGEMM grouped GEMM benchmark"
    )
    parser.add_argument(
        "--layout", choices=["contiguous", "masked"], default="contiguous"
    )
    parser.add_argument("--dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument(
        "--shape",
        default="2,128,512,512",
        help="Single shape as num_groups,m_per_group,n,k",
    )
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument(
        "--check", action="store_true", help="Run float64 reference check"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = parse_shape(args.shape)
    time_s, tflops = run_case(cfg, args)

    print("dtype,layout,num_groups,m_per_group,n,k,time_s,TFLOP/s")
    print(
        f"{args.dtype},{args.layout},{cfg.num_groups},{cfg.m_per_group},{cfg.n},{cfg.k},{time_s:.6f},{tflops:.3f}"
    )


if __name__ == "__main__":
    main()
