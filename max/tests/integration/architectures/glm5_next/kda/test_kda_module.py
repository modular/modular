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
"""A KDA layer's conv weights split heads within each of q, k and v.

The conv's channels are q, k and v in that order, and each device convolves
the channels of its own heads of all three. So each conv weight is split by
head on its own, and a device's conv channels are its q rows, then its k
rows, then its v rows. A contiguous split of the concatenated channels has
**the same per-device shape** -- on the checkpoint's 24576 channels at TP8,
both give 3072 rows -- and differs only in values, on every device but the
first. So this checks values, not shapes, and asserts that a contiguous
split would have been wrong.

The meshes are simulated CPUs, and the layer is 16 heads of 8: placing the
weights needs neither a GPU nor the checkpoint's widths.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.glm5_next.layers.kda import (
    KimiDeltaAttention,
    layer_rows,
)
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh

HIDDEN = 32
HEADS = 16
HEAD_DIM = 8
KERNEL = 4
QKV_DIM = HEADS * HEAD_DIM
CONV_DIM = 3 * QKV_DIM


def _layer(num_devices: int) -> KimiDeltaAttention:
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        linear_num_heads=HEADS,
        linear_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=KERNEL,
        linear_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    )
    mesh = DeviceMesh((CPU(),) * num_devices, (num_devices,), ("tp",))
    with default_device(mesh), default_dtype(DType.float32):
        return KimiDeltaAttention(cast(Glm5NextConfig, config))


def _conv_weight() -> np.ndarray:
    """A conv weight whose every channel is distinguishable from every other,
    in the checkpoint's ``[channels, 1, kernel]`` layout."""
    return (
        np.arange(CONV_DIM * KERNEL, dtype=np.float32).reshape(
            CONV_DIM, 1, KERNEL
        )
        / 1000.0
    )


@pytest.mark.parametrize("num_devices", [4, 8])
def test_conv_shards_split_heads_within_each_projection(
    num_devices: int,
) -> None:
    """Device ``i``'s conv rows are its own heads of q, then of k, then of v."""
    layer = _layer(num_devices)
    full = _conv_weight()
    weights = {
        name: np.zeros([int(d) for d in weight.shape], np.float32)
        for name, weight in layer.parameters
    }
    for part, block in zip("qkv", np.split(full, 3, axis=0), strict=True):
        weights[f"{part}_conv1d"] = block
    layer.load_state_dict(weights)
    # Concatenated per device, as the layer's per-device core does.
    actual = [
        np.concatenate(
            [
                conv.local_shards[i].to_numpy()
                for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)
            ],
            axis=0,
        )
        for i in range(num_devices)
    ]

    width = QKV_DIM // num_devices
    for i, shard_weight in enumerate(actual):
        expected = np.concatenate(
            [full[block * QKV_DIM + i * width :][:width] for block in range(3)],
            axis=0,
        )
        np.testing.assert_array_equal(
            shard_weight,
            expected,
            err_msg=(
                f"TP{num_devices} device {i}'s conv shard is not its own"
                " heads of q, k and v"
            ),
        )
    # The trap itself: a contiguous split of `conv_dim` has the same
    # per-device shape and disagrees on all but the first device, which is
    # why a shape check cannot stand in for this one.
    disagreeing = sum(
        not np.array_equal(actual[i], full[i * 3 * width :][: 3 * width])
        for i in range(num_devices)
    )
    assert disagreeing >= num_devices - 1


def test_static_layer_rows_match_the_dynamic_slice() -> None:
    """Each KDA layer's static row is the row its dynamic slice takes."""
    num_layers, batch = 5, 3
    table = np.arange(num_layers * batch, dtype=np.int32).reshape(
        num_layers, batch
    )
    rows = Tensor.constant(table, device=CPU())
    for position in range(num_layers):
        layer = Tensor.constant(position, dtype=DType.int64, device=CPU())
        dynamic = layer_rows(rows, layer).to_numpy()
        static = layer_rows(rows, layer, position).to_numpy()
        np.testing.assert_array_equal(dynamic, table[position])
        np.testing.assert_array_equal(static, dynamic)
