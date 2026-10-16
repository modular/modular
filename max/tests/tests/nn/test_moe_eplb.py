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
"""Tests applying an EPLB placement to MoE weights through the module tree."""

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.layer import LayerList, Module
from max.nn.moe import MoE, apply_eplb_placement

NUM_LOGICAL = 4


class _Block(Module):
    def __init__(self, layer_idx: int | None, num_experts: int) -> None:
        super().__init__()
        self.mlp = MoE(
            devices=[DeviceRef.CPU()],
            hidden_dim=8,
            num_experts=num_experts,
            num_experts_per_token=2,
            moe_dim=16,
            num_logical_experts=NUM_LOGICAL,
            dtype=DType.float32,
        )
        self.mlp.layer_idx = layer_idx

    def __call__(self, x):  # noqa: ANN001
        return x


class _Model(Module):
    def __init__(self, num_layers: int, num_experts: int = NUM_LOGICAL) -> None:
        super().__init__()
        self.layers = LayerList(
            [_Block(i, num_experts) for i in range(num_layers)]
        )

    def __call__(self, x):  # noqa: ANN001
        return x


def _logical_checkpoint(num_layers: int) -> dict[str, np.ndarray | Buffer]:
    """Builds a checkpoint whose expert tensors are tagged by (layer, expert).

    Non-expert weights (the bf16 gate) are zero buffers; only their identity
    matters to these tests.
    """
    reference = _Model(num_layers)
    state_dict: dict[str, np.ndarray | Buffer] = {}
    for name, weight in reference.raw_state_dict().items():
        state_dict[name] = Buffer.zeros(
            weight.shape.static_dims, dtype=weight.dtype
        )
    for layer in range(num_layers):
        block = reference.layers[layer]
        assert isinstance(block, _Block)
        for expert in range(NUM_LOGICAL):
            prefix = f"layers.{layer}.mlp.experts.{expert}."
            expert_module = block.mlp.experts[expert]
            assert isinstance(expert_module, Module)
            for name, weight in expert_module.raw_state_dict().items():
                state_dict[f"{prefix}{name}"] = np.full(
                    weight.shape.static_dims,
                    100 * layer + expert,
                    dtype=np.float32,
                )
    return state_dict


def _expert_tag(
    state_dict: dict[str, np.ndarray | Buffer], layer: int, slot: int
) -> float:
    value = state_dict[f"layers.{layer}.mlp.experts.{slot}.down_proj.weight"]
    assert isinstance(value, np.ndarray)
    return float(value[0, 0])


def test_identity_placement_is_a_no_op() -> None:
    model = _Model(num_layers=2)
    checkpoint = _logical_checkpoint(num_layers=2)
    phy2log = np.broadcast_to(np.arange(NUM_LOGICAL), (2, NUM_LOGICAL))

    remapped = apply_eplb_placement(model, checkpoint, phy2log)

    assert remapped.keys() == checkpoint.keys()
    for name, value in checkpoint.items():
        assert remapped[name] is value


def test_per_layer_placement_with_cycles() -> None:
    model = _Model(num_layers=2)
    checkpoint = _logical_checkpoint(num_layers=2)
    # Layer 0 is a 3-cycle, layer 1 a reversal: a single in-place pass would
    # read already-overwritten slots in both.
    phy2log = np.array([[1, 2, 0, 3], [3, 2, 1, 0]])

    remapped = apply_eplb_placement(model, checkpoint, phy2log)

    for layer in range(2):
        for slot in range(NUM_LOGICAL):
            assert _expert_tag(remapped, layer, slot) == (
                100 * layer + phy2log[layer, slot]
            )
    # Non-expert weights pass through untouched.
    gate = "layers.1.mlp.gate.gate_score.weight"
    assert remapped[gate] is checkpoint[gate]
    model.load_state_dict(remapped, strict=True)


def test_replicas_fill_extra_slots() -> None:
    num_phy = NUM_LOGICAL + 2
    model = _Model(num_layers=2, num_experts=num_phy)
    checkpoint = _logical_checkpoint(num_layers=2)
    phy2log = np.array([[0, 1, 2, 3, 0, 2], [3, 1, 2, 0, 3, 3]])

    remapped = apply_eplb_placement(model, checkpoint, phy2log)

    for layer in range(2):
        for slot in range(num_phy):
            assert _expert_tag(remapped, layer, slot) == (
                100 * layer + phy2log[layer, slot]
            )
    model.load_state_dict(remapped, strict=True)


def test_replica_slots_need_a_placement() -> None:
    model = _Model(num_layers=1, num_experts=NUM_LOGICAL + 2)
    with pytest.raises(ValueError, match="Missing required weights"):
        model.load_state_dict(_logical_checkpoint(num_layers=1), strict=True)


def test_missing_layer_idx_raises() -> None:
    model = _Model(num_layers=1)
    block = model.layers[0]
    assert isinstance(block, _Block)
    block.mlp.layer_idx = None
    with pytest.raises(ValueError, match="needs layer_idx"):
        apply_eplb_placement(
            model,
            _logical_checkpoint(num_layers=1),
            np.arange(NUM_LOGICAL)[None],
        )


def test_row_width_must_match_slots() -> None:
    model = _Model(num_layers=1)
    with pytest.raises(ValueError, match="expected 4"):
        apply_eplb_placement(
            model,
            _logical_checkpoint(num_layers=1),
            np.arange(NUM_LOGICAL + 1)[None],
        )


def test_logical_id_out_of_range_raises() -> None:
    model = _Model(num_layers=1)
    with pytest.raises(ValueError, match="outside"):
        apply_eplb_placement(
            model,
            _logical_checkpoint(num_layers=1),
            np.array([[0, 1, 2, NUM_LOGICAL]]),
        )
