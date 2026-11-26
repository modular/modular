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
"""A distributed ``Glm5NextMLP`` still clamps before the activation, and the
tensor-parallel MoE sums its shards before rounding.

The model never runs the dense MLP or the shared expert on one device's
whole batch: the decoder layer runs a dense MLP on each device's share of
the rows, and the tensor-parallel MoE splits its shared expert's columns. An
MLP that applies ``silu`` and then clamps agrees with GLM below
``swiglu_limit`` and diverges above it, so only inputs past the limit can
tell the two apart. The two-device mesh is two simulated CPUs.

Under the EP all-reduce exchange, the expert-parallel MoE splits its shared
expert's columns too, so each of the four ranks runs a quarter of it rather
than rank 0 all of it. That test traces the layer on four stand-in GPUs and
does not compile it.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from max import functional as F
from max.driver import CPU, Device
from max.dtype import DType
from max.graph import DeviceRef, Graph
from max.nn.common_layers.quant_linear import tensor_parallel_mlp
from max.nn.ep import EPBuffers
from max.nn.ep.init import payload_buffers, payload_types
from max.nn.ep_config import EPConfig
from max.pipelines.architectures.glm5_next.layers.mlp import (
    Glm5NextExpertParallelMoE,
    Glm5NextMLP,
    Glm5NextTensorParallelMoE,
    build_mlp,
)
from max.pipelines.architectures.glm5_next.model_config import Glm5NextConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
)

_LIMIT = 10.0
_DIM = 8

# The +/-12 and +/-20 entries are the only ones that separate the orderings.
# Two rows, so each device of the row-split case gets one.
_X = np.array(
    [
        [-20.0, -3.0, 0.0, 3.0, 9.9, 10.0, 12.0, 20.0],
        [20.0, 12.0, 10.0, 9.9, 3.0, 0.0, -3.0, -20.0],
    ],
    dtype=np.float32,
)


def _mesh() -> DeviceMesh:
    return DeviceMesh((CPU(), CPU()), (2,), ("tp",))


def _mlp() -> Glm5NextMLP:
    # Identity gate and down, negated up: the pre-activation gate is `x` and
    # the up half is `-x`, so `_X` straddles the limit on both halves.
    eye = np.eye(_DIM, dtype=np.float32)
    mlp = Glm5NextMLP(
        hidden_dim=_DIM, feed_forward_length=_DIM, swiglu_limit=_LIMIT
    )
    mlp.load_state_dict(
        {
            "gate_proj.weight": eye,
            "up_proj.weight": -eye,
            "down_proj.weight": eye,
        }
    )
    return mlp


def _glm_reference(x: np.ndarray) -> np.ndarray:
    """``silu(min(gate, limit)) * clamp(up, -limit, limit)``."""
    gate = torch.clamp(torch.from_numpy(x), max=_LIMIT)
    up = torch.clamp(torch.from_numpy(-x), min=-_LIMIT, max=_LIMIT)
    return (torch.nn.functional.silu(gate) * up).numpy()


def test_row_shards_clamp_before_the_activation() -> None:
    """The dense layers' form: replicated weights over a split of the rows."""
    mesh = _mesh()
    with default_device(mesh), default_dtype(DType.float32):
        mlp = _mlp()
        x = Tensor.from_dlpack(_X).to(DeviceMapping(mesh, (Sharded(0),)))
        y = mlp(x).to(DeviceMapping.replicated(mesh))
    assert y.placements == (Replicated(),)
    np.testing.assert_allclose(
        y.local_shards[0].to_numpy(), _glm_reference(_X), rtol=1e-6, atol=1e-6
    )


def test_column_shards_clamp_before_the_activation() -> None:
    """The shared expert's form: each device holds half the feed-forward axis
    and returns a partial sum."""
    mesh = _mesh()
    with default_device(mesh), default_dtype(DType.float32):
        mlp = tensor_parallel_mlp(_mlp())
        x = Tensor.from_dlpack(_X).to(DeviceMapping.replicated(mesh))
        partial = mlp(x)
        assert partial.placements == (Partial(),)
        y = partial.to(DeviceMapping.replicated(mesh))
    np.testing.assert_allclose(
        y.local_shards[0].to_numpy(), _glm_reference(_X), rtol=1e-6, atol=1e-6
    )


def test_tensor_parallel_moe_returns_the_reduced_sum() -> None:
    """Its float32 combine is all-reduced before the cast that rounds it, so
    the MoE returns the whole sum, replicated, in the input's dtype."""
    mesh = _mesh()
    with F.lazy(), default_dtype(DType.bfloat16):
        with default_device(mesh):
            moe = Glm5NextTensorParallelMoE(
                hidden_dim=256,
                num_experts=4,
                num_experts_per_token=2,
                moe_dim=128,
                has_shared_experts=True,
                shared_experts_dim=128,
            )
        x = Tensor.zeros([4, 256], device=DeviceMapping.replicated(mesh))
        y = moe(x)
    assert y.placements == (Replicated(),)
    assert y.dtype == DType.bfloat16


_EP_DEVICES = 4
_EP_HIDDEN = 64
_EP_MOE_DIM = 128


def _ep_moe_config() -> Glm5NextConfig:
    """The fields ``build_mlp`` reads, for one tiny bf16 MoE layer."""
    fields = SimpleNamespace(
        quant_scheme=None,
        swiglu_limit=_LIMIT,
        first_k_dense_replace=0,
        hidden_size=_EP_HIDDEN,
        intermediate_size=_EP_MOE_DIM,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=_EP_MOE_DIM,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        correction_bias_dtype=DType.float32,
        n_shared_experts=1,
    )
    return cast(Glm5NextConfig, fields)


def _allreduce_ep_config() -> EPConfig:
    return EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=_EP_HIDDEN,
        top_k=2,
        n_experts=8,
        max_tokens_per_rank=64,
        n_gpus_per_node=_EP_DEVICES,
        n_nodes=1,
        use_allreduce=True,
    )


def _fake_gpu(id: int) -> MagicMock:
    """A stand-in accelerator, so the lazy trace takes the GPU EP path."""
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = "gpu"
    return fake


def _placeholder_ep_buffers(config: EPConfig, mesh: DeviceMesh) -> EPBuffers:
    devices = {
        DeviceRef.CPU(): CPU(),
        **{DeviceRef.GPU(i): d for i, d in enumerate(mesh.devices)},
    }
    payload = {
        name: [
            Tensor.zeros(t.shape, dtype=t.dtype, device=devices[t.device])
            for t in types
        ]
        for name, types in payload_types(config).items()
    }
    return payload_buffers(payload, mesh)


def test_allreduce_ep_moe_runs_a_shared_expert_slice_on_every_rank() -> None:
    """Each of the four ranks holds a quarter of the shared expert's
    feed-forward axis and runs it, so the caller's all-reduce adds the whole
    shared expert once, rather than rank 0 running all of it."""
    devices = [_fake_gpu(i) for i in range(_EP_DEVICES)]
    # The same object for each id, so device equality is identity.
    with (
        patch("max.graph.type.Accelerator", side_effect=devices.__getitem__),
        F.lazy(),
    ):
        mesh = DeviceMesh(tuple(devices), (_EP_DEVICES,), ("tp",))
        config = _allreduce_ep_config()
        ep = _placeholder_ep_buffers(config, mesh)
        with default_device(mesh), default_dtype(DType.bfloat16):
            moe = build_mlp(_ep_moe_config(), 0, config)
            x = Tensor.zeros(
                [4, _EP_HIDDEN], device=DeviceMapping.replicated(mesh)
            )
            out = moe(x, ep=ep)
            ir = str(Graph.current._mlir_op)

        assert isinstance(moe, Glm5NextExpertParallelMoE)
        assert moe.shared_experts is not None
        gate = moe.shared_experts.gate_proj.weight
        assert isinstance(gate, Tensor)
        assert [int(s.shape[0]) for s in gate.local_shards] == [
            _EP_MOE_DIM // _EP_DEVICES
        ] * _EP_DEVICES
    assert list(out.shape) == [4, _EP_HIDDEN]
    # The shared expert's side-stream region takes one input a rank.
    region = re.search(r'"mo\.sequence"\(([^)]*)\) <\{streamId = 1', ir)
    assert region is not None
    assert region.group(1).count("%") == _EP_DEVICES
