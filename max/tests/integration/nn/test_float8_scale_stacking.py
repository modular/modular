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
"""Tests for float8_scale_stacking module.

Tests the can_use_fused_mlp function and its helper functions:
- `can_use_fused_mlp`: Checks if quantization scales allow fused MLP operations.
- `_concatable`: Checks if shapes can be concatenated along an axis.
- `_weight_scale_name`: Extracts weight prefix from scale names.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.graph.shape import Shape
from max.graph.weights.weights import WeightData
from max.nn.float8_scale_stacking import (
    _concatable,
    _weight_scale_name,
    can_use_fused_mlp,
)


def _create_weight_data(
    name: str,
    shape: tuple[int, ...],
    value: float | None = None,
) -> WeightData:
    """Creates a WeightData object for testing.

    Args:
        name: Name of the weight.
        shape: Shape of the weight tensor.
        value: If provided, fills the tensor with this scalar value.
               Used for scalar weights to test value comparison.

    Returns:
        WeightData with float32 dtype.
    """
    if value is not None:
        data = np.full(shape, value, dtype=np.float32)
    else:
        data = (
            np.random.randn(*shape).astype(np.float32)
            if shape
            else np.array(1.0, dtype=np.float32)
        )
    return WeightData(
        data=data,
        name=name,
        dtype=DType.float32,
        shape=Shape(shape),
    )


class TestWeightScaleName:
    """Tests for _weight_scale_name helper function."""

    def test_weight_scale_suffix(self) -> None:
        """Tests extraction of prefix from .weight_scale suffix."""
        assert (
            _weight_scale_name("layer.gate_proj.weight_scale")
            == "layer.gate_proj"
        )
        assert (
            _weight_scale_name("model.layers.0.q_proj.weight_scale")
            == "model.layers.0.q_proj"
        )

    def test_input_scale_suffix(self) -> None:
        """Tests extraction of prefix from .input_scale suffix."""
        assert (
            _weight_scale_name("layer.up_proj.input_scale") == "layer.up_proj"
        )
        assert _weight_scale_name("attn.k_proj.input_scale") == "attn.k_proj"

    def test_weight_scale_2_suffix(self) -> None:
        """Tests extraction of prefix from .weight_scale_2 suffix."""
        assert (
            _weight_scale_name("mlp.gate_proj.weight_scale_2")
            == "mlp.gate_proj"
        )
        assert (
            _weight_scale_name("layer.v_proj.weight_scale_2") == "layer.v_proj"
        )

    def test_non_scale_weights_return_none(self) -> None:
        """Tests that non-scale weight names return None."""
        assert _weight_scale_name("layer.weight") is None
        assert _weight_scale_name("model.bias") is None
        assert _weight_scale_name("embedding.weight") is None
        assert _weight_scale_name("gate_proj") is None

    def test_partial_match_returns_none(self) -> None:
        """Tests that partial suffix matches return None."""
        assert _weight_scale_name("layer.weight_scale_extra") is None
        assert _weight_scale_name("layer.input_scale_suffix") is None


class TestConcatable:
    """Tests for _concatable helper function."""

    def test_two_shapes_concatable_axis_0(self) -> None:
        """Tests concatenation check along axis 0 for two shapes."""
        assert _concatable(Shape((128, 256)), Shape((64, 256)), axis=0) is True
        assert (
            _concatable(Shape((10, 20, 30)), Shape((5, 20, 30)), axis=0) is True
        )

    def test_two_shapes_not_concatable(self) -> None:
        """Tests shapes that cannot be concatenated along axis 0."""
        assert (
            _concatable(Shape((128, 256)), Shape((128, 512)), axis=0) is False
        )
        assert (
            _concatable(Shape((10, 20, 30)), Shape((10, 25, 30)), axis=0)
            is False
        )

    def test_three_shapes_concatable(self) -> None:
        """Tests concatenation check for three shapes (QKV projections)."""
        q_shape = (512, 768)
        k_shape = (64, 768)
        v_shape = (64, 768)
        assert (
            _concatable(Shape(q_shape), Shape(k_shape), Shape(v_shape), axis=0)
            is True
        )

    def test_three_shapes_not_concatable(self) -> None:
        """Tests three shapes that cannot be concatenated."""
        q_shape = (512, 768)
        k_shape = (64, 512)  # Different second dimension
        v_shape = (64, 768)
        assert (
            _concatable(Shape(q_shape), Shape(k_shape), Shape(v_shape), axis=0)
            is False
        )

    def test_different_rank_shapes(self) -> None:
        """Tests shapes with different ranks cannot be concatenated."""
        assert _concatable(Shape((128, 256)), Shape((128,)), axis=0) is False
        assert (
            _concatable(Shape((10, 20)), Shape((10, 20, 30)), axis=0) is False
        )

    def test_single_shape(self) -> None:
        """Tests that a single shape is always concatable with itself."""
        assert _concatable(Shape((128, 256)), axis=0) is True

    def test_empty_shapes_raises(self) -> None:
        """Tests that no shapes raises ValueError."""
        with pytest.raises(ValueError, match="Must provide at least one shape"):
            _concatable(axis=0)

    def test_concat_axis_1(self) -> None:
        """Tests concatenation along axis 1."""
        assert _concatable(Shape((128, 256)), Shape((128, 512)), axis=1) is True
        assert _concatable(Shape((128, 256)), Shape((64, 512)), axis=1) is False


class TestCanUseFusedMLP:
    """Tests for can_use_fused_mlp function."""

    def test_compatible_gate_up_rowwise_scales(self) -> None:
        """Tests that compatible gate and up projection scales allow fusion."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (512, 1)
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale", (512, 1)
            ),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_missing_up_proj_allows_fusion(self) -> None:
        """Tests that missing up_proj (gate_proj alone) allows fusion."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (512, 1)
            ),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_incompatible_shapes_prevents_fusion(self) -> None:
        """Tests that incompatible shapes prevent fusion."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (512, 1)
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale",
                (512, 2),  # Different shape
            ),
        }
        assert can_use_fused_mlp(state_dict) is False

    def test_scalar_scales_equal_values_allows_fusion(self) -> None:
        """Tests that scalar scales with equal values allow fusion."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (), value=0.5
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale", (), value=0.5
            ),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_scalar_scales_different_values_prevents_fusion(self) -> None:
        """Tests that scalar scales with different values prevent fusion
        when no quant config (or non-tensor-wise config) is provided."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (), value=0.5
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale", (), value=0.7
            ),
        }
        assert can_use_fused_mlp(state_dict) is False

    def test_single_element_tensor_equal_values_allows_fusion(self) -> None:
        """Tests that single-element tensor scales with equal values allow fusion."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (1,), value=0.5
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale", (1,), value=0.5
            ),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_single_element_tensor_different_values_prevents_fusion(
        self,
    ) -> None:
        """Tests that single-element tensor scales with different values prevent
        fusion when no quant config (or non-tensor-wise config) is provided."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (1,), value=0.5
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale", (1,), value=0.7
            ),
        }
        assert can_use_fused_mlp(state_dict) is False

    def test_multiple_layers_all_compatible(self) -> None:
        """Tests fusion detection across multiple layers with compatible scales."""
        state_dict = {
            "layers.0.mlp.gate_proj.weight_scale": _create_weight_data(
                "layers.0.mlp.gate_proj.weight_scale", (512, 1)
            ),
            "layers.0.mlp.up_proj.weight_scale": _create_weight_data(
                "layers.0.mlp.up_proj.weight_scale", (512, 1)
            ),
            "layers.1.mlp.gate_proj.weight_scale": _create_weight_data(
                "layers.1.mlp.gate_proj.weight_scale", (512, 1)
            ),
            "layers.1.mlp.up_proj.weight_scale": _create_weight_data(
                "layers.1.mlp.up_proj.weight_scale", (512, 1)
            ),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_multiple_layers_one_incompatible(self) -> None:
        """Tests that one incompatible layer prevents fusion for all."""
        state_dict = {
            "layers.0.mlp.gate_proj.weight_scale": _create_weight_data(
                "layers.0.mlp.gate_proj.weight_scale", (512, 1)
            ),
            "layers.0.mlp.up_proj.weight_scale": _create_weight_data(
                "layers.0.mlp.up_proj.weight_scale", (512, 1)
            ),
            "layers.1.mlp.gate_proj.weight_scale": _create_weight_data(
                "layers.1.mlp.gate_proj.weight_scale", (512, 1)
            ),
            "layers.1.mlp.up_proj.weight_scale": _create_weight_data(
                "layers.1.mlp.up_proj.weight_scale",
                (512, 2),  # Incompatible
            ),
        }
        assert can_use_fused_mlp(state_dict) is False

    def test_empty_state_dict_allows_fusion(self) -> None:
        """Tests that empty state dict allows fusion (no constraints)."""
        assert can_use_fused_mlp({}) is True

    def test_non_scale_weights_ignored(self) -> None:
        """Tests that non-scale weights are ignored."""
        state_dict = {
            "layer.weight": _create_weight_data("layer.weight", (512, 256)),
            "layer.bias": _create_weight_data("layer.bias", (512,)),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_input_scale_suffix_compatible(self) -> None:
        """Tests that input_scale suffix weights are checked for compatibility."""
        state_dict = {
            "mlp.gate_proj.input_scale": _create_weight_data(
                "mlp.gate_proj.input_scale", (1, 256)
            ),
            "mlp.up_proj.input_scale": _create_weight_data(
                "mlp.up_proj.input_scale", (1, 256)
            ),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_weight_scale_2_suffix_compatible(self) -> None:
        """Tests that weight_scale_2 suffix weights are checked for compatibility."""
        state_dict = {
            "mlp.gate_proj.weight_scale_2": _create_weight_data(
                "mlp.gate_proj.weight_scale_2", (512, 1)
            ),
            "mlp.up_proj.weight_scale_2": _create_weight_data(
                "mlp.up_proj.weight_scale_2", (512, 1)
            ),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_non_gate_proj_scales_ignored(self) -> None:
        """Tests that non-gate_proj scales (like lm_head) are ignored."""
        state_dict = {
            "lm_head.weight": _create_weight_data("lm_head.weight", (512, 1)),
        }
        assert can_use_fused_mlp(state_dict) is True

    def test_tensor_wise_allows_different_scalar_scales(self) -> None:
        """Tests that tensor_wise=True allows fusion even when gate and up
        scalar scales differ."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (), value=0.5
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale", (), value=0.7
            ),
        }
        assert can_use_fused_mlp(state_dict, tensor_wise=True) is True

    def test_non_tensor_wise_blocks_different_scalar_scales(self) -> None:
        """Tests that tensor_wise=False blocks fusion when gate and up
        scalar scales differ."""
        state_dict = {
            "mlp.gate_proj.weight_scale": _create_weight_data(
                "mlp.gate_proj.weight_scale", (), value=0.5
            ),
            "mlp.up_proj.weight_scale": _create_weight_data(
                "mlp.up_proj.weight_scale", (), value=0.7
            ),
        }
        assert can_use_fused_mlp(state_dict, tensor_wise=False) is False
