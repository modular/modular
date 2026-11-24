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

The conv weights keep the checkpoint's dtype, which differs between the two
published checkpoints.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next.layers.kda import (
    KimiDeltaAttention,
    layer_rows,
)
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.realization_context import ensure_context
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
        linear_conv_dtype=DType.float32,
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


NUM_LAYERS = 5
_RESULT = "-> !mo.tensor<[batch], si32>"


def _layer_rows_mlir(position: int | None) -> str:
    """``layer_rows`` of a ``[NUM_LAYERS, batch]`` table, traced, not compiled."""
    with (
        Graph(
            "layer_rows",
            input_types=[
                TensorType(
                    DType.int32, [NUM_LAYERS, "batch"], device=DeviceRef.CPU()
                )
            ],
        ) as graph,
        ensure_context(),
    ):
        rows = Tensor.from_graph_value(graph.inputs[0].tensor)
        layer = F.constant(
            NUM_LAYERS - 1 if position is None else position,
            DType.int64,
            device=CPU(),
        )
        graph.output(TensorValue(layer_rows(rows, layer, position)))
    return str(graph)


@pytest.mark.parametrize("position", [0, NUM_LAYERS - 1])
def test_static_layer_rows_take_a_static_slice(position: int) -> None:
    """A static position takes its row by a static slice, which the compiler
    passes as a view, and has the dynamic slice's ``[batch]`` result."""
    static = _layer_rows_mlir(position)
    assert (
        f"starts = #mosh<ape[{position}, 0]>" in static
        and f"stops = #mosh<ape[{position + 1}, batch]>" in static
    ), static
    assert "mo.slice(" not in static.replace("rmo.slice(", ""), static
    assert _RESULT in static

    dynamic = _layer_rows_mlir(None)
    assert "mo.slice(%arg0" in dynamic, dynamic
    assert _RESULT in dynamic


@pytest.mark.parametrize("stored", [DType.float32, DType.bfloat16])
def test_conv_weights_keep_the_checkpoint_dtype(stored: DType) -> None:
    """NVIDIA's NVFP4 export stores the conv weights in float32, the FP8
    checkpoint in bfloat16. The layer declares them as stored, whatever the
    model's default dtype."""
    devices = [DeviceRef.CPU()]
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=HIDDEN,
        num_layers=1,
        page_size=128,
        devices=devices,
        num_q_heads=HEADS,
    )
    config = Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=devices,
        max_seq_len=128,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        linear_num_heads=HEADS,
        linear_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=KERNEL,
    )
    state_dict = {
        f"language_model.layers.0.self_attn.{p}_conv1d": SimpleNamespace(
            dtype=stored
        )
        for p in "qkv"
    }
    config.finalize(cast(Any, None), state_dict, cast(Any, None))
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
        layer = KimiDeltaAttention(config)
    convs = (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)
    assert {conv.dtype for conv in convs} == {stored}
