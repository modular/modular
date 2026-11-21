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
"""Mixture of Experts (MoE) modules for DeepseekV3.2, in the ModuleV3 API.

DeepSeek changed MoE datatypes in the V3 to V3.2 upgrade: accumulations run in
float32. The reference implementations differ only in the casts:

- V3:   ``self.w2(F.silu(self.w1(x)) * self.w3(x))``
- V3.2: ``self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))``

The float32 classes below are the V3 modules with those accumulation dtypes
set; see ``QuantizedMoE.gate_up_accum_dtype`` and ``combine_accum_dtype``. The
expert-parallel and local-experts classes keep the V3 dtypes, as the NVFP4
deployments that use them do.
"""

from __future__ import annotations

from typing import Any, ClassVar

from max import functional as F
from max.dtype import DType
from max.nn.common_layers.quant_moe import (
    ExpertParallelMoE,
    ExpertStacks,
    QuantizedMoE,
    TensorParallelMoE,
    _local_expert_matmul,
)
from max.nn.common_layers.quant_tensor import NVFP4Activation
from max.nn.ep import EPBuffers
from max.nn.ep_config import EPConfig
from max.nn.kernels import (
    grouped_quantize_dynamic_block_scaled,
    moe_create_indices,
    moe_finalize,
)
from max.tensor import Tensor


class DeepseekV3_2MoE(QuantizedMoE):
    """Single-device (or replicated) V3.2 MoE."""

    gate_up_accum_dtype: ClassVar[DType | None] = DType.float32
    combine_accum_dtype: ClassVar[DType | None] = DType.float32


class DeepseekV3_2TensorParallelMoE(TensorParallelMoE):
    """Tensor-parallel V3.2 MoE."""

    gate_up_accum_dtype: ClassVar[DType | None] = DType.float32
    combine_accum_dtype: ClassVar[DType | None] = DType.float32


class DeepseekV3_2ExpertParallelMoE(ExpertParallelMoE):
    """Expert-parallel V3.2 MoE.

    Neither upcast applies. V2's EP deployment runs the shared expert-parallel
    path, which takes the fused NVFP4 SwiGLU with no float32 cast, and the EP
    combine already accumulates in the dispatch dtype. A float32 gate/up
    accumulation here would rule out the fused SwiGLU and with it the
    graph compiler's MegaFFN fusion, which decode latency depends on.
    """


class DeepseekV3_2LocalExpertsMoE(ExpertParallelMoE):
    """V3.2 MoE on one device that holds every routed NVFP4 expert.

    A pipeline stage exchanges no tokens, so this runs the expert-parallel
    path's local expert kernels directly. Like that path it takes neither
    float32 upcast, which would rule out the fused SwiGLU. It does not use
    :meth:`ExpertParallelMoE.local_expert_partial_sums`, which takes packed
    routing rows and returns a partial sum for an all-reduce.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            # A stand-in: nothing is exchanged, so no dispatch, combine or
            # buffer reads it. The base class reads only `fused_shared_expert`
            # (off, so the shared expert runs apart) and `n_nodes`;
            # `max_tokens_per_rank` is unused.
            ep_config=EPConfig(
                dispatch_dtype=DType.bfloat16,
                combine_dtype=DType.bfloat16,
                hidden_size=hidden_dim,
                top_k=num_experts_per_token,
                n_experts=num_experts,
                max_tokens_per_rank=1,
                n_gpus_per_node=1,
                n_nodes=1,
            ),
            **kwargs,
        )
        # `forward` reads the first device's stacks and indexes every expert.
        if self.mesh.num_devices != 1:
            raise ValueError(
                "DeepseekV3_2LocalExpertsMoE holds every expert on one"
                f" device, not on a mesh of {self.mesh.num_devices}"
            )

    def forward(
        self,
        x: Tensor,
        layer_idx: Tensor | None = None,
        stacks: ExpertStacks | None = None,
        *,
        ep: EPBuffers | None = None,
    ) -> Tensor:
        """Returns the routed and shared experts' output for ``x``.

        ``layer_idx`` and ``ep`` are unused: nothing is exchanged.
        """
        del layer_idx, ep
        gate_up, down, global_scales, _, dispatch_scales, expert_scales = (
            stacks or self.expert_stacks()
        )
        assert global_scales is not None and dispatch_scales is not None
        # The shared expert reads only ``x``, so it runs on a side stream
        # under the routed experts, as in the expert-parallel deployments.
        shared = (
            self._compute_shared_experts(x, [0])[0]
            if self.shared_experts is not None
            else None
        )
        top_k = self.num_experts_per_token
        topk_idx, topk_weight = self.gate(x)
        _, start, restore, expert_ids, usage, scales_offset = (
            moe_create_indices(
                topk_idx.reshape([-1]).cast(DType.int32),
                self.num_experts,
                needs_scales_offset=True,
            )
        )
        # Every expert shares the one input scale, so each token row is
        # quantized once and its bytes go to its slots' sorted rows: the
        # sorted copy of ``x`` never materializes.
        data, scales = grouped_quantize_dynamic_block_scaled(
            x,
            row_offsets=start,
            scales_offsets=scales_offset,
            expert_ids=expert_ids,
            sf_tensor=dispatch_scales[0],
            restore=restore,
            top_k=top_k,
        )
        tokens = NVFP4Activation(
            data=data,
            scales=scales,
            input_scale=global_scales[0],
            scales_offset=scales_offset,
        )
        rows = _local_expert_matmul(
            tokens,
            gate_up[0],
            down[0],
            start,
            expert_ids,
            usage,
            quant_config=self.quant_config,
            scales_offset=scales_offset,
            estimated_total_m=(F.shape_to_tensor(x.shape)[0] * top_k).cast(
                DType.uint32
            ),
            accum_dtype=self.gate_up_accum_dtype,
            swiglu_limit=self.swiglu_limit,
            scales=expert_scales[0] if expert_scales else None,
        )
        out = moe_finalize(rows, restore, topk_weight, x.dtype)
        if shared is not None:
            out = out + shared
        return out
