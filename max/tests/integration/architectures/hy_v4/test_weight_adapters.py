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
"""Checks the Hy4 weight adapter's routed-expert storage."""

from collections.abc import Callable
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.hy_v4.weight_adapters import (
    convert_mtp_state_dict,
    convert_safetensor_state_dict,
)

# Two experts, I = 3 (gate rows 0-2, up rows 3-5), H = 64.
_EXPERTS, _INTER, _HIDDEN = 2, 3, 64


def _weight(bits: np.ndarray, dtype: DType, name: str) -> NonCallableMock:
    buffer = Buffer.from_numpy(bits)
    if dtype != DType.uint8:
        buffer = buffer.view(dtype)
    weights = NonCallableMock()
    weights.data.return_value = WeightData(
        buffer, name, dtype, Shape(bits.shape)
    )
    return weights


def _experts(prefix: str) -> tuple[dict[str, NonCallableMock], np.ndarray]:
    # Each row holds its own index, so the output shows the row order.
    rows = np.arange(2 * _INTER, dtype=np.uint8)
    gate_up = np.broadcast_to(
        rows[None, :, None], (_EXPERTS, 2 * _INTER, _HIDDEN)
    ).copy()
    scales = gate_up[:, :, : _HIDDEN // 32].copy()
    down = np.arange(_EXPERTS * _HIDDEN * _INTER, dtype=np.uint8).reshape(
        _EXPERTS, _HIDDEN, _INTER
    )
    name = f"{prefix}mlp.experts."
    state = {
        name + "gate_up_proj": _weight(
            gate_up, DType.float8_e4m3fn, name + "gate_up_proj"
        ),
        name + "gate_up_proj_scale": _weight(
            scales, DType.uint8, name + "gate_up_proj_scale"
        ),
        name + "down_proj": _weight(
            down, DType.float8_e4m3fn, name + "down_proj"
        ),
    }
    return state, down


def _bits(data: WeightData) -> np.ndarray:
    return np.from_dlpack(data.to_buffer().view(DType.uint8))


@pytest.mark.parametrize(
    "convert, prefix, out",
    [
        (
            convert_safetensor_state_dict,
            "model.layers.0.",
            "language_model.layers.0.",
        ),
        (convert_mtp_state_dict, "model.mtp_layers.0.", "draft."),
    ],
)
def test_routed_gate_up_passes_through_uncopied(
    convert: Callable[..., dict[str, WeightData]], prefix: str, out: str
) -> None:
    state, down = _experts(prefix)
    config = NonCallableMock(indexer_types=["full"])
    converted = convert(state, config)

    # Checkpoint order, gate rows then up rows; the graph interleaves them.
    order = list(range(2 * _INTER))
    for leaf, key, dtype in (
        ("weight", "gate_up_proj", DType.float8_e4m3fn),
        ("weight_scale", "gate_up_proj_scale", DType.float8_e8m0fnu),
    ):
        data = converted[f"{out}mlp.experts.gate_up_proj.{leaf}"]
        assert data.dtype == dtype
        bits = _bits(data)
        assert (bits[:, :, 0] == np.array([order] * _EXPERTS)).all()
        assert (bits == bits[:, :, :1]).all()
        # A host copy of the 6.4 GB per-layer tensor stalls loading.
        source = state[f"{prefix}mlp.experts.{key}"].data()
        assert np.shares_memory(bits, _bits(source))
    np.testing.assert_array_equal(
        _bits(converted[f"{out}mlp.experts.down_proj.weight"]), down
    )
