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

from collections.abc import Mapping

from max.graph.shape import Shape
from max.graph.weights.weights import WeightData


def can_use_fused_mlp(state_dict: Mapping[str, WeightData]) -> bool:
    """Checks if the quantization scales can be used with fused MLP operations.

    Returns:
        Whether or not MLP can be fused.
    """
    for weight_name, weight_data in state_dict.items():
        weight_is_scalar = len(weight_data.shape) == 0 or (
            len(weight_data.shape) == 1 and weight_data.shape[0] == 1
        )
        weight_prefix = _weight_scale_name(weight_name)
        if weight_prefix is None:
            continue

        # MLP gate/up projection weights.
        if weight_prefix.endswith("gate_proj"):
            up_weight_name = weight_name.replace("gate_proj", "up_proj")
            up_weight = state_dict.get(up_weight_name)
            if up_weight is None:
                continue

            if not _concatable(weight_data.shape, up_weight.shape, axis=0):
                return False

            # We can only "stack" scalar values if they are equivalent.
            if weight_is_scalar:
                gate_value = weight_data.to_buffer().item()
                up_value = up_weight.to_buffer().item()
                if gate_value != up_value:
                    return False

    return True


def _concatable(*shapes: Shape, axis: int = 0) -> bool:
    """Checks if the two shapes are concat-able along the given axis."""
    if not shapes:
        raise ValueError("Must provide at least one shape.")

    first_shape = shapes[0]
    for shape in shapes[1:]:
        if len(first_shape) != len(shape):
            return False
        # Check all dimensions except the concatenation axis.
        for i in range(len(first_shape)):
            if i == axis:
                continue
            if first_shape[i] != shape[i]:
                return False

    return True


def _weight_scale_name(key: str) -> str | None:
    """Checks if the given key is a scale."""
    if key.endswith(".weight_scale"):
        return key[: -len(".weight_scale")]

    if key.endswith(".input_scale"):
        return key[: -len(".input_scale")]

    if key.endswith(".weight_scale_2"):
        return key[: -len(".weight_scale_2")]

    return None
