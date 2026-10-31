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
"""Verifies expert-parallel output matches the single-device reference for
the latent MoE block.

Needs 2 real GPUs -- see the BUILD target's ``gpu-count`` exec property.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.architectures.kimi_k3_modulev3.layers import ep
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
    expert_parallel_config,
    expert_parallel_latent_moe,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


@pytest.mark.compile_on_gpu(
    "mo.moe.single.group.router is staged as an undeclared custom op the interpreter can't bind"
)
def test_latent_moe_ep2_matches_single_device() -> None:
    hidden = 32
    # Kernel-valid: EP combine's hid_dim must divide by WARP_SIZE *
    # simd_width_of[combine_dtype] -- thousands of elements on this
    # hardware, not the handful a shrunk synthetic shape would use
    # elsewhere in this port's tests.
    latent = 2048
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
        # The gate stays float32: a bf16 rounding here could flip a
        # close top-k pick, diverging the two sides for a reason that has
        # nothing to do with the exchange this test actually checks.
        layer.gate.gate_score.weight = Tensor(
            gate_weight_np, dtype=DType.float32
        )
        layer.gate.e_score_correction_bias = Tensor(
            bias_np, dtype=DType.float32
        )
        layer.routed_expert_down_proj.weight = Tensor(
            down_proj_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.routed_expert_up_proj.weight = Tensor(
            up_proj_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.routed_expert_norm.weight = Tensor(
            norm_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.experts_gate_up_proj = Tensor(
            experts_gate_up_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.experts_down_proj = Tensor(
            experts_down_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.shared_experts.gate_up_proj.weight = Tensor(
            shared_gate_up_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        layer.shared_experts.down_proj.weight = Tensor(
            shared_down_np, dtype=DType.float32
        ).cast(DType.bfloat16)
        return layer

    with default_device(Accelerator(0)):
        layer1 = build()
        x1 = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        out1 = layer1(x1).cast(DType.float32).to_numpy()

    devices = [Accelerator(0), Accelerator(1)]
    mesh = DeviceMesh(
        devices=tuple(devices), mesh_shape=(2,), axis_names=("ep",)
    )
    session = InferenceSession(devices=devices)
    with default_device(mesh):
        layer2 = build()
        layer2 = expert_parallel_latent_moe(layer2, mesh)
        config = expert_parallel_config(layer2, mesh, max_tokens_per_rank=64)
        buffers = ep.allocate_ep_buffers(config, mesh, session)
        x2 = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        out2 = (
            layer2.expert_parallel_forward(x2, config, buffers)
            .cast(DType.float32)
            .to_numpy()
        )

    err = _rel_error(out2, out1)
    assert err < 1e-2, f"latent MoE EP=2 vs single-device diverges: {err}"
