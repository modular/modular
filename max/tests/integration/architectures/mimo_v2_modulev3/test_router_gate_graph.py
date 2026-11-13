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
"""The router gate becomes one fused GEMV only where that op can carry it.

Staged on virtual devices and never compiled. The op tiles the contraction in
``simd_width * num_threads`` elements with no tail guard, so the gate takes it
only for a BF16 activation whose K divides the 1024-element tile on the SM100
devices MiMo-V2 runs on. Every other shape and a host mesh keep the cast +
float32 matmul pair.
"""

from __future__ import annotations

from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import (
    MiMoV2Router,
)
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh, TensorLayout

MIXED_GEMV = "moe.router.gate.mixed.gemv"
EXPERTS = 8
TOP_K = 2
# The widest router-gate K tile: SM100's 4-element vector load over 256
# threads. Multiplying by it keeps the gate on; one element less does not.
TILED_HIDDEN = 1024
UNTILED_HIDDEN = TILED_HIDDEN - 1
ROWS = 4


def _gpu_mesh(n_devices: int = 1) -> DeviceMesh:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(n_devices)),
        (n_devices,),
        ("tp",),
    )


def _staged_router(hidden: int, mesh: DeviceMesh) -> str:
    with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
        router = MiMoV2Router(
            hidden_dim=hidden,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            norm_topk_prob=True,
            route_on_every_device=True,
        )
        staged, _registry = router.stage(
            TensorLayout(DType.bfloat16, [ROWS, hidden], mesh)
        )
    return str(staged)


def test_tiled_contraction_takes_the_fused_gemv() -> None:
    assert MIXED_GEMV in _staged_router(TILED_HIDDEN, _gpu_mesh())


def test_untiled_contraction_keeps_the_cast_and_matmul() -> None:
    mesh = _gpu_mesh()
    assert MIXED_GEMV not in _staged_router(UNTILED_HIDDEN, mesh)


def test_host_mesh_keeps_the_cast_and_matmul() -> None:
    """The fused kernel is GPU-only, so a CPU mesh runs the plain chain."""
    mesh = DeviceMesh((CPU(), CPU()), (2,), ("tp",))
    assert MIXED_GEMV not in _staged_router(TILED_HIDDEN, mesh)
