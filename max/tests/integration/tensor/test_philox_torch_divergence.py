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

"""Verify that MAX's Philox-backed GPU RNG matches PyTorch's CUDA Philox RNG.

PyTorch CUDA's `torch.randn` uses Philox 4x32-10 with a specific element-to-
counter mapping: each thread initializes its Philox state with
`curand_init(seed, thread_id, 0)` and consumes a single Philox step (4 raw
uint32) into 4 normals via cuRAND's Box-Muller. MAX's GPU kernel mirrors that
mapping, uniform conversion, and Box-Muller pairing, producing values that
match PyTorch CUDA within 1e-4 on B200.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import random

SEED = 42
N = 64

requires_gpu = pytest.mark.skipif(
    accelerator_count() == 0 or not torch.cuda.is_available(),
    reason="requires a CUDA GPU available to both MAX and PyTorch",
)


def _compare_max_and_torch(
    max_vals: list[float], torch_vals: list[float], label: str
) -> int:
    """Print a comparison table and return the number of close matches."""
    matches = sum(
        1
        for m, t in zip(max_vals, torch_vals, strict=False)
        if abs(m - t) < 1e-4
    )
    print(f"\n[{label}] {'Index':<6} {'MAX':>14} {'PyTorch':>14} {'Delta':>14}")
    print("-" * 60)
    for i in range(min(16, len(max_vals))):
        m, t = max_vals[i], torch_vals[i]
        print(f"{i:<6} {m:>14.8f} {t:>14.8f} {abs(m - t):>14.8f}")
    print(f"...\n[{label}] Matches within 1e-4: {matches}/{len(max_vals)}")
    return matches


@requires_gpu
def test_max_gaussian_deterministic_gpu() -> None:
    """MAX produces identical results on GPU when re-seeded with same value."""
    device = Accelerator()

    random.set_seed(SEED)
    t1 = random.gaussian([N], dtype=DType.float32, device=device)

    random.set_seed(SEED)
    t2 = random.gaussian([N], dtype=DType.float32, device=device)

    for i in range(N):
        assert t1[i].item() == t2[i].item(), (
            f"MAX GPU not deterministic at index {i}"
        )


@requires_gpu
def test_torch_randn_deterministic_gpu() -> None:
    """PyTorch produces identical results on GPU when re-seeded with same value."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    t1 = torch.randn(N, generator=gen, device="cuda")

    gen = torch.Generator(device="cuda").manual_seed(SEED)
    t2 = torch.randn(N, generator=gen, device="cuda")

    assert torch.equal(t1, t2), "PyTorch GPU not deterministic"


@requires_gpu
def test_max_and_torch_normals_converge_gpu() -> None:
    """MAX GPU randn matches PyTorch CUDA randn within 1e-4 on B200.

    The MAX GPU kernel mirrors cuRAND's element-to-counter mapping, uniform
    conversion, and Box-Muller pairing. Some ULP drift is expected because
    Mojo's `log` and `sqrt` lower to fast PTX intrinsics
    (`lg2.approx.f32 * ln2`, `sqrt.approx.ftz.f32`) while PyTorch's cuRAND is
    built without `-use_fast_math` and uses precise software variants.
    """
    random.set_seed(SEED)
    max_tensor = random.gaussian([N], dtype=DType.float32, device=Accelerator())

    gen = torch.Generator(device="cuda").manual_seed(SEED)
    torch_tensor = torch.randn(N, generator=gen, device="cuda")

    max_vals = [max_tensor[i].item() for i in range(N)]
    torch_vals = torch_tensor.cpu().tolist()

    matches = _compare_max_and_torch(max_vals, torch_vals, "GPU")
    assert matches >= N - 2, (
        f"Expected near-full alignment but only {matches}/{N} values matched. "
        "MAX and PyTorch GPU Philox alignment may have regressed."
    )
