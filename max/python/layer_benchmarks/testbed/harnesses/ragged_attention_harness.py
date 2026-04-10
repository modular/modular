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

"""Abstract base class for single-GPU ragged attention harnesses.

Provides shared build_and_compile, prepare_inputs, cleanup_inputs,
cuda_graph_eligible, and postprocess_torch_output for all single-GPU
attention harnesses. Subclasses only need to implement build_graph,
torch_reference_layer, and prepare_torch_inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import RequestID, TextGenerationContext, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn.kv_cache import KVCacheParams
from max.pipelines.core import TextContext
from typing_extensions import TypeVar

from testbed.harness import CompiledLayerBundle, LayerTestHarness


@dataclass
class AttentionStaticParams:
    """Base static params shared by all ragged attention harnesses.

    Every attention harness requires these fields. Model-specific harnesses
    extend this with extra fields (e.g. sliding_window_size, qk_norm_eps).
    """

    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float = 500000.0
    total_num_pages: int = 8192


@dataclass
class AttentionDynamicParams:
    """Per-shape parameters for ragged attention harnesses."""

    batch_size: int
    seq_len: int
    ctx_len: int = 0


AttentionStaticParamsT = TypeVar(
    "AttentionStaticParamsT",
    bound=AttentionStaticParams,
    default=AttentionStaticParams,
)


# Base mapping from HuggingFace parameter names to harness weight names.
# Subclasses can extend this with model-specific entries (e.g. q_norm, k_norm).
HF_TO_HARNESS_BASE: dict[str, str] = {
    "q_proj.weight": "qkv_proj.q.weight",
    "k_proj.weight": "qkv_proj.k.weight",
    "v_proj.weight": "qkv_proj.v.weight",
    "o_proj.weight": "o_proj.weight",
}


class RaggedAttentionHarness(
    LayerTestHarness[
        AttentionStaticParamsT,
        AttentionDynamicParams,
        list[TextGenerationContext],
    ]
):
    """ABC for single-GPU ragged attention harnesses.

    Subclasses must implement:
        - name (property)
        - build_graph()
        - torch_reference_layer()
        - prepare_torch_inputs()

    Shared behavior provided by this class:
        - build_and_compile: creates KVCacheParams, compiles graph, sets up
          PagedKVCacheManager
        - prepare_inputs: allocates KV claims, creates input tensors
        - cleanup_inputs: releases KV claims
        - cuda_graph_eligible: seq_len == 1
        - postprocess_torch_output: output[0].squeeze(0)
    """

    @staticmethod
    def dynamic_params_type() -> type:
        return AttentionDynamicParams

    def __init__(
        self,
        static_params: AttentionStaticParamsT,
        session: InferenceSession,
        device: Accelerator,
    ) -> None:
        super().__init__(static_params, session, device)
        self._kv_params = KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=static_params.n_kv_heads,
            head_dim=static_params.head_dim,
            num_layers=1,
            devices=[DeviceRef.GPU()],
        )
        self._kv_manager = PagedKVCacheManager(
            params=self._kv_params,
            total_num_pages=static_params.total_num_pages,
            session=session,
            max_batch_size=128,
        )

    def build_and_compile(self) -> CompiledLayerBundle:
        graph, weights_registry = self.build_graph()
        compiled = self.session.load(graph, weights_registry=weights_registry)

        return CompiledLayerBundle(
            compiled_model=compiled,
            device=self.device,
            session=self.session,
        )

    def prepare_inputs(
        self,
        bundle: CompiledLayerBundle,
        dynamic_params: AttentionDynamicParams,
    ) -> tuple[list[Buffer], list[TextGenerationContext]]:
        device = bundle.device
        total_len = dynamic_params.ctx_len + dynamic_params.seq_len

        batch: list[TextGenerationContext] = []
        for _ in range(dynamic_params.batch_size):
            ctx = TextContext(
                request_id=RequestID(),
                max_length=max(total_len, self.static_params.max_seq_len),
                tokens=TokenBuffer(np.empty(total_len, dtype=np.int64)),
            )
            self._kv_manager.claim(ctx.request_id, replica_idx=0)
            self._kv_manager.alloc(ctx, replica_idx=0)
            if dynamic_params.ctx_len > 0:
                ctx.tokens.skip_processing(dynamic_params.ctx_len)
            batch.append(ctx)

        kv_runtime = self._kv_manager.runtime_inputs(
            cast(list[list[TextGenerationContext]], [batch])
        ).inputs[0]
        assert kv_runtime.attention_dispatch_metadata is not None

        total_tokens = dynamic_params.batch_size * dynamic_params.seq_len
        torch_input = torch.randn(
            total_tokens, self.static_params.hidden_size, dtype=torch.bfloat16
        )
        input_tensor = Buffer.from_dlpack(torch_input).to(device)
        row_offsets = Buffer.from_numpy(
            np.array(
                [
                    i * dynamic_params.seq_len
                    for i in range(dynamic_params.batch_size + 1)
                ],
                dtype=np.uint32,
            )
        ).to(device)

        execute_args: list[Buffer] = [
            input_tensor,
            row_offsets,
            kv_runtime.blocks.to(device),
            kv_runtime.cache_lengths.to(device),
            kv_runtime.lookup_table.to(device),
            kv_runtime.max_lengths,
            kv_runtime.attention_dispatch_metadata,
        ]

        return execute_args, batch

    def cleanup_inputs(
        self,
        bundle: CompiledLayerBundle,
        context: list[TextGenerationContext],
    ) -> None:
        for ctx in context:
            self._kv_manager.release(ctx.request_id, replica_idx=0)

    def cuda_graph_eligible(
        self, dynamic_params: AttentionDynamicParams
    ) -> bool:
        return dynamic_params.seq_len == 1

    def postprocess_torch_output(self, output: object) -> torch.Tensor:
        assert isinstance(output, tuple)
        return output[0].squeeze(0)
