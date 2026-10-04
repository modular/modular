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
"""Checks the fused sigmoid-GEMV router against a float32 reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.functional import transfer_to
from max.experimental.functional.spmd_ops import tensor_to_layout
from max.experimental.nn.common_layers.functional_kernels import (
    _moe_sigmoid_gemv_router_rule,
    moe_sigmoid_gemv_router,
)
from max.experimental.sharding import (
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    _moe_sigmoid_gemv_router,
    _moe_sigmoid_gemv_router_unsupported,
)
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    _runs_fused_router,
)

NUM_EXPERTS = 128
TOP_K = 6
HIDDEN = 2688
SCALE = 2.5


@pytest.mark.parametrize("tokens", [1, 64, 700])
def test_fused_router_matches_reference(tokens: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((tokens, HIDDEN)).astype(np.float32)
    weight = rng.uniform(-0.05, 0.05, (NUM_EXPERTS, HIDDEN)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, NUM_EXPERTS).astype(np.float32)
    x_bf16 = torch.from_numpy(x).to(torch.bfloat16)

    device = Accelerator()
    dev = DeviceRef.GPU()
    with Graph(
        "fused_router",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", HIDDEN], dev),
            TensorType(DType.float32, [NUM_EXPERTS, HIDDEN], dev),
            TensorType(DType.float32, [NUM_EXPERTS], dev),
        ],
    ) as graph:
        gx, gweight, gbias = (v.tensor for v in graph.inputs)
        graph.output(
            *_moe_sigmoid_gemv_router(
                gx,
                gweight,
                gbias,
                TOP_K,
                norm_weights=True,
                routed_scaling_factor=SCALE,
            )
        )

    model = InferenceSession(devices=[device]).load(graph)
    indices, weights = model.execute(
        Buffer.from_dlpack(x_bf16).to(device),
        Buffer.from_numpy(weight).to(device),
        Buffer.from_numpy(bias).to(device),
    )
    assert isinstance(indices, Buffer) and isinstance(weights, Buffer)
    got_idx = indices.to_numpy()
    got_w = weights.to_numpy()

    want_idx, want_w = _reference_route(x_bf16.float().numpy(), weight, bias)

    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_allclose(got_w, want_w, rtol=1e-4, atol=1e-6)


def _reference_route(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = 1 / (1 + np.exp(-(x @ weight.T)))
    idx = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :TOP_K]
    picked = np.take_along_axis(scores, idx, axis=-1)
    return idx, picked / picked.sum(-1, keepdims=True) * SCALE


@pytest.mark.parametrize("warp_size", [32, 64])
def test_unsupported_shapes(warp_size: int) -> None:
    # Nemotron-3.5-Lightning's router runs on both warp widths.
    assert (
        _moe_sigmoid_gemv_router_unsupported(
            NUM_EXPERTS, TOP_K, HIDDEN, warp_size
        )
        is None
    )
    # The tiny test config (4 experts, top 2) has fewer experts than a warp.
    assert _moe_sigmoid_gemv_router_unsupported(4, 2, 64, warp_size)
    assert _moe_sigmoid_gemv_router_unsupported(
        NUM_EXPERTS, warp_size + 1, HIDDEN, warp_size
    )
    assert _moe_sigmoid_gemv_router_unsupported(NUM_EXPERTS, TOP_K, 6, 64)
    # A float32 gate with hidden 32768 stages 2048-wide slices for 32 rows,
    # about 256 KiB of shared memory.
    assert _moe_sigmoid_gemv_router_unsupported(
        NUM_EXPERTS, TOP_K, 32768, warp_size
    )


def test_gate_falls_back_for_unsupported_shapes() -> None:
    specs = [DeviceSpec.accelerator()]
    assert _runs_fused_router(specs, NUM_EXPERTS, TOP_K, HIDDEN)
    assert not _runs_fused_router(
        specs, num_experts=4, num_experts_per_tok=2, hidden_size=64
    )


def test_functional_router_replicates_sharded_inputs() -> None:
    hidden = 256
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, hidden)).astype(np.float32)
    weight = rng.uniform(-0.05, 0.05, (NUM_EXPERTS, hidden)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, NUM_EXPERTS).astype(np.float32)

    # Two mesh devices on one GPU, sharded along the gate's contraction axis.
    gpu = Accelerator(0)
    mesh = DeviceMesh(devices=(gpu, gpu), mesh_shape=(2,), axis_names=("tp",))
    contraction = DeviceMapping(mesh, (Sharded(1),))
    gx = transfer_to(Tensor(x), contraction)
    gweight = transfer_to(Tensor(weight), contraction)
    gbias = transfer_to(Tensor(bias), DeviceMapping(mesh, (Sharded(0),)))

    action_set = _moe_sigmoid_gemv_router_rule(
        tensor_to_layout(gx),
        tensor_to_layout(gweight),
        tensor_to_layout(gbias),
    )
    assert action_set.axis_assignments == (
        AxisAssignment((Replicated(),) * 3, Replicated()),
    )

    indices, weights = moe_sigmoid_gemv_router(
        gx, gweight, gbias, TOP_K, True, SCALE
    )
    assert indices.placements == (Replicated(),)
    assert weights.placements == (Replicated(),)

    want_idx, want_w = _reference_route(x, weight, bias)
    np.testing.assert_array_equal(indices.to_numpy(), want_idx)
    np.testing.assert_allclose(weights.to_numpy(), want_w, rtol=1e-4, atol=1e-6)
