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
"""Verifies TP=2 matches TP=1 for the dense MLP and latent MoE block.

Builds each layer at TP=1 and TP=2 with identical weights and checks that
the TP=2 output matches TP=1.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor, default_device
from max.pipelines.architectures.kimi_k3_modulev3.layers.mlp import (
    KimiK3DenseMLP,
    tensor_parallel_dense_mlp,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
    tensor_parallel_latent_moe,
)


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


def _tp2_mesh() -> DeviceMesh:
    return DeviceMesh(
        devices=(Accelerator(0), Accelerator(1)),
        mesh_shape=(2,),
        axis_names=("tp",),
    )


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(DType.float32, 1e-4), (DType.bfloat16, 2e-2)],
)
def test_dense_mlp_tp2_matches_tp1(dtype: DType, tolerance: float) -> None:
    hidden, inter = 32, 64
    beta, linear_beta = 1.5, 1.5
    rng = np.random.default_rng(0)
    gate_up_np = (rng.standard_normal((2 * inter, hidden)) * 0.02).astype(
        np.float32
    )
    down_np = (rng.standard_normal((hidden, inter)) * 0.02).astype(np.float32)
    x_np = rng.standard_normal((5, hidden)).astype(np.float32)

    with default_device(Accelerator(0)):
        layer1 = KimiK3DenseMLP(
            hidden_size=hidden,
            intermediate_size=inter,
            beta=beta,
            linear_beta=linear_beta,
        )
        layer1.gate_up_proj.weight = Tensor(gate_up_np).cast(dtype)
        layer1.down_proj.weight = Tensor(down_np).cast(dtype)
        x1 = Tensor(x_np).cast(dtype)
        out1 = layer1(x1).cast(DType.float32).to_numpy()

    mesh = _tp2_mesh()
    with default_device(mesh):
        layer2 = KimiK3DenseMLP(
            hidden_size=hidden,
            intermediate_size=inter,
            beta=beta,
            linear_beta=linear_beta,
        )
        layer2.gate_up_proj.weight = Tensor(gate_up_np).cast(dtype)
        layer2.down_proj.weight = Tensor(down_np).cast(dtype)
        layer2 = tensor_parallel_dense_mlp(layer2, mesh)
        x2 = Tensor(x_np).cast(dtype)
        partial2 = layer2(x2)
        out2 = F.allreduce_sum(partial2).cast(DType.float32).to_numpy()

    err = _rel_error(out2, out1)
    assert err < tolerance, f"dense MLP TP=2 vs TP=1 diverges: rel_error={err}"


def test_dense_mlp_rejects_width_not_divisible_by_tp() -> None:
    mesh = _tp2_mesh()
    with default_device(mesh):
        layer = KimiK3DenseMLP(
            hidden_size=8, intermediate_size=5, beta=1.5, linear_beta=None
        )
        weight = layer.gate_up_proj.weight
        with pytest.raises(ValueError, match="not divisible"):
            tensor_parallel_dense_mlp(layer, mesh)
        assert layer.gate_up_proj.weight is weight


@pytest.mark.compile_on_gpu(
    "mo.moe.single.group.router is staged as an undeclared custom op the interpreter can't bind"
)
def test_latent_moe_tp2_matches_tp1() -> None:
    hidden = 32
    latent = 24
    moe_inter = 16
    # Kernel-valid: the MoE router kernel needs num_threads == num_experts
    # to be a whole number of warps.
    num_experts = 64
    top_k = 4
    shared_inter = moe_inter
    beta, linear_beta = 1.5, 1.5
    eps = 1e-5
    seq_len = 6

    rng = np.random.default_rng(1)

    def arr(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    gate_weight_np = arr(num_experts, hidden)
    bias_np = np.zeros(num_experts, dtype=np.float32)
    down_proj_np = arr(latent, hidden)
    up_proj_np = arr(hidden, latent)
    norm_np = np.ones(latent, dtype=np.float32)
    experts_gate_up_np = arr(num_experts, 2 * moe_inter, latent)
    experts_down_np = arr(num_experts, latent, moe_inter)
    shared_gate_up_np = arr(2 * shared_inter, hidden)
    shared_down_np = arr(hidden, shared_inter)
    x_np = rng.standard_normal((seq_len, hidden)).astype(np.float32)

    def build() -> KimiK3LatentMoE:
        layer = KimiK3LatentMoE(
            hidden_size=hidden,
            latent_size=latent,
            moe_intermediate_size=moe_inter,
            shared_intermediate_size=shared_inter,
            num_experts=num_experts,
            num_experts_per_token=top_k,
            beta=beta,
            linear_beta=linear_beta,
            rms_norm_eps=eps,
            routed_scaling_factor=1.0,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
        )
        layer.gate.gate_score.weight = Tensor(
            gate_weight_np, dtype=DType.float32
        )
        layer.gate.e_score_correction_bias = Tensor(
            bias_np, dtype=DType.float32
        )
        layer.routed_expert_down_proj.weight = Tensor(
            down_proj_np, dtype=DType.float32
        )
        layer.routed_expert_up_proj.weight = Tensor(
            up_proj_np, dtype=DType.float32
        )
        layer.routed_expert_norm.weight = Tensor(norm_np, dtype=DType.float32)
        layer.experts_gate_up_proj = Tensor(
            experts_gate_up_np, dtype=DType.float32
        )
        layer.experts_down_proj = Tensor(experts_down_np, dtype=DType.float32)
        layer.shared_experts.gate_up_proj.weight = Tensor(
            shared_gate_up_np, dtype=DType.float32
        )
        layer.shared_experts.down_proj.weight = Tensor(
            shared_down_np, dtype=DType.float32
        )
        return layer

    with default_device(Accelerator(0)):
        layer1 = build()
        x1 = Tensor(x_np, dtype=DType.float32)
        out1 = layer1(x1).to_numpy()

    mesh = _tp2_mesh()
    with default_device(mesh):
        layer2 = tensor_parallel_latent_moe(build(), mesh)
        x2 = Tensor(x_np, dtype=DType.float32)
        # This layer reduces internally, so the caller does not allreduce.
        out2 = layer2(x2).to_numpy()

    err = _rel_error(out2, out1)
    assert err < 1e-4, f"latent MoE TP=2 vs TP=1 diverges: rel_error={err}"
