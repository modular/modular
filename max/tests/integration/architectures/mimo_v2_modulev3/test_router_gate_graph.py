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

Both admissible rows -- a replicated activation and a row-sharded one -- are
reached here through ``MiMoV2Router``: its tensor-parallel mode, which routes
on every device, and its data-parallel one, which routes each replica's own
rows. The fixtures' 256-wide hidden misses the tile, so only a 1024-wide gate
exercises them.
"""

from __future__ import annotations

import pytest
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
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Placement,
    Replicated,
    Sharded,
    TensorLayout,
)

MIXED_GEMV = "moe.router.gate.mixed.gemv"
EXPERTS = 8
TOP_K = 2
# The widest router-gate K tile: SM100's 4-element vector load over 256
# threads. Multiplying by it keeps the gate on; one element less does not.
TILED_HIDDEN = 1024
UNTILED_HIDDEN = TILED_HIDDEN - 1
ROWS = 4


def _stage_router_with_logits(
    hidden: int,
    mesh: DeviceMesh,
    placement: Placement,
    route_on_every_device: bool,
) -> tuple[str, Tensor]:
    """Stages the router on ``mesh`` with an activation at ``placement``.

    ``route_on_every_device`` is the model's, so a case covers the path its
    mode takes. Returns the staged MLIR and the gate logits: the shared
    wrapper's rule decides their placement, and ``forward`` returns only the
    expert picks.
    """
    captured: list[Tensor] = []

    class _LogitsKeepingRouter(MiMoV2Router):
        def _gate_logits(self, x: Tensor, weight: Tensor) -> Tensor:
            logits = super()._gate_logits(x, weight)
            captured.append(logits)
            return logits

    with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
        router = _LogitsKeepingRouter(
            hidden_dim=hidden,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            norm_topk_prob=True,
            route_on_every_device=route_on_every_device,
        )
        staged, _registry = router.stage(
            TensorLayout(
                DType.bfloat16,
                [ROWS, hidden],
                DeviceMapping(mesh, (placement,)),
            )
        )
    (logits,) = captured
    return str(staged), logits


def _gpu_mesh(n_devices: int = 1, axis: str = "tp") -> DeviceMesh:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(n_devices)),
        (n_devices,),
        (axis,),
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


@pytest.mark.parametrize(
    ("axis", "placement", "route_on_every_device"),
    [
        pytest.param("tp", Replicated(), True, id="tp-replicated"),
        pytest.param("dp", Sharded(0), False, id="dp-sharded"),
    ],
)
def test_both_admissible_rows_stage_the_fused_gemv(
    axis: str, placement: Placement, route_on_every_device: bool
) -> None:
    """The tensor- and data-parallel routers both reach the shared wrapper.

    A replicated activation is MiMo-V2's tensor-parallel mode, which sets
    ``route_on_every_device`` because its replicas hold identical rows, and
    a row-sharded one is its data-parallel mode, which does not. The shared
    rule admits both, so staging either one succeeds on the fused op, and
    each one's logits keep the activation's placement. What it refuses is
    ``max/tests/tests/nn/module_v3/common_layers/test_moe.py``'s rejection
    case.
    """
    mesh = _gpu_mesh(2, axis)
    mlir, logits = _stage_router_with_logits(
        TILED_HIDDEN, mesh, placement, route_on_every_device
    )

    assert MIXED_GEMV in mlir
    assert logits.mapping.mesh == mesh
    # The gate weight is replicated, so each device projects only its own
    # rows: the logits follow the activation, rows and all.
    assert logits.mapping.placements == (placement,)
    assert logits.shape[0] == ROWS
    assert logits.shape[1] == EXPERTS
