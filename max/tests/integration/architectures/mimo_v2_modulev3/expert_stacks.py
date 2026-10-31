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
"""Checks of the routed experts' per-device stacks, shared by the TP and EP
CPU tests.

The scales are E8M0, which the host backend refuses to compile, so random
uint8 codes stand in for them: the stacking only moves bytes.
"""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from max.graph import TensorValue
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import MiMoV2MoE
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMesh
from tiny_fixture import EXPERTS, HIDDEN, MOE_DIM

Array = npt.NDArray[np.uint8]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_random_experts(
    layer: MiMoV2MoE, monkeypatch: pytest.MonkeyPatch
) -> dict[str, tuple[Array, Array]]:
    """Fills every expert with random codes; returns them stacked per
    projection as ``(weights, scales)``.

    The scale interleave passes through, so the stacks come back exactly as
    the interleave would read them.
    """
    rng = np.random.default_rng(0)
    stacks: dict[str, tuple[list[Array], list[Array]]] = {
        name: ([], []) for name in PROJECTIONS
    }
    for expert in layer.experts:
        for name in PROJECTIONS:
            proj = getattr(expert, name)
            weight = rng.integers(
                0, 256, [int(dim) for dim in proj.weight.shape], dtype=np.uint8
            )
            scale = rng.integers(
                0,
                256,
                [int(dim) for dim in proj.weight_scale.shape],
                dtype=np.uint8,
            )
            proj.weight = Tensor.from_dlpack(weight)
            proj.weight_scale = Tensor.from_dlpack(scale)
            stacks[name][0].append(weight)
            stacks[name][1].append(scale)
    monkeypatch.setattr(
        layer.strategy,
        "prepare_weight_scales",
        lambda gate_up, down, device: (gate_up, down),
    )
    return {
        name: (np.stack(weights), np.stack(scales))
        for name, (weights, scales) in stacks.items()
    }


def realize(layer: MiMoV2MoE, mesh: DeviceMesh) -> list[list[Array]]:
    """Each device's ``(gate_up, gate_up_scales, down, down_scales)``."""
    with ensure_context():
        per_device = [
            [Tensor.from_graph_value(TensorValue(v)) for v in stacks]
            for stacks in layer._stacked_weights(mesh)
        ]
    return [[t.to_numpy() for t in stacks] for stacks in per_device]


def gate_up(gate: Array, up: Array) -> Array:
    """Each expert's gate rows, then its up rows."""
    return np.concatenate([gate, up], axis=1)


def check_staged_expert_stacks(
    text: str, n_devices: int, expert_parallel: bool
) -> None:
    """Checks that a device holds only its final expert weight stacks.

    A device temporary freed around the persistent stacks strands allocator
    granules and shrinks the KV cache.
    """
    # TODO(DISTINF-813): mo.shard_and_stack copies each device's slice
    # serially, far slower than the host stack; drop this once it is fixed.
    assert "mo.shard_and_stack" not in text
    local = EXPERTS // n_devices if expert_parallel else EXPERTS
    share = MOE_DIM if expert_parallel else MOE_DIM // n_devices
    # The scales go through device temporaries in the interleave by design,
    # so they are checked only for a gate-only or up-only stack.
    for rows in {MOE_DIM, share}:
        for width in (HIDDEN // 2, HIDDEN // 32):
            shape = rf"\[{local}, {rows}, {width}\], (ui8|f8e8m0fnu)"
            assert not re.search(shape, text), shape
    # The transferred gate/up share, the same share viewed per expert, and
    # the down share.
    final = {
        f"{2 * local}, {share}, {HIDDEN // 2}",
        f"{local}, {2 * share}, {HIDDEN // 2}",
        f"{local}, {HIDDEN}, {share // 2}",
    }
    on_device = set(
        re.findall(r"!mo\.tensor<\[([0-9, ]+)\], ui8, gpu:\d+>", text)
    )
    assert on_device, "no device-resident ui8 tensors in the staged graph"
    assert on_device <= final, on_device - final
