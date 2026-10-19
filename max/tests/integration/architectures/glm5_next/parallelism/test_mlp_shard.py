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
"""A sharded ``Glm5NextMLP`` still clamps before the activation.

The model never runs the dense MLP or the shared expert unsharded: both get a
sharding strategy at every device count and run through ``shard()``. A shard
that falls back to the stock ``MLP`` applies ``silu`` and then clamps, which
agrees with GLM below ``swiglu_limit`` and diverges above it, so only inputs
past the limit can tell the shards apart.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.pipelines.architectures.glm5_next.layers.mlp import Glm5NextMLP

_LIMIT = 10.0
_DIM = 8

# The +/-12 and +/-20 entries are the only ones that separate the orderings.
_X = np.array(
    [[-20.0, -3.0, 0.0, 3.0, 9.9, 10.0, 12.0, 20.0]], dtype=np.float32
)


def _weights() -> dict[str, np.ndarray]:
    # Identity gate and down, negated up: the pre-activation gate is `x` and
    # the up half is `-x`, so `_X` straddles the limit on both halves.
    eye = np.eye(_DIM, dtype=np.float32)
    return {
        "gate_proj.weight": eye,
        "up_proj.weight": -eye,
        "down_proj.weight": eye,
    }


def _glm_reference(x: np.ndarray) -> np.ndarray:
    """``silu(min(gate, limit)) * clamp(up, -limit, limit)``."""
    gate = torch.clamp(torch.from_numpy(x), max=_LIMIT)
    up = torch.clamp(torch.from_numpy(-x), min=-_LIMIT, max=_LIMIT)
    return (torch.nn.functional.silu(gate) * up).numpy()


@pytest.mark.parametrize(
    "strategy",
    [ShardingStrategy.replicate(1), ShardingStrategy.tensor_parallel(2)],
    ids=["replicate1", "tp2"],
)
def test_shards_clamp_before_the_activation(
    strategy: ShardingStrategy,
) -> None:
    mlp = Glm5NextMLP(
        dtype=DType.float32,
        quantization_encoding=None,
        hidden_dim=_DIM,
        feed_forward_length=_DIM,
        devices=[DeviceRef.CPU()],
        swiglu_limit=_LIMIT,
    )
    # Names the weights; the shards' weights are views of these.
    mlp.load_state_dict(_weights())
    mlp.sharding_strategy = strategy
    shards = mlp.shard([DeviceRef.CPU()] * strategy.num_devices)
    for shard in shards:
        assert isinstance(shard, Glm5NextMLP)

    spec = TensorType(DType.float32, shape=_X.shape, device=DeviceRef.CPU())
    with Graph("glm5_next_mlp_shard", input_types=[spec]) as graph:
        (x,) = graph.inputs
        # Tensor-parallel shards each hold a slice of the feed-forward axis
        # and return a partial sum; the allreduce is a sum.
        out = shards[0](x.tensor)
        for shard in shards[1:]:
            out = out + shard(x.tensor)
        graph.output(out)

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=mlp.state_dict())
    got = model.execute(Buffer.from_dlpack(_X))[0]
    np.testing.assert_allclose(
        np.from_dlpack(got), _glm_reference(_X), rtol=1e-6, atol=1e-6
    )
