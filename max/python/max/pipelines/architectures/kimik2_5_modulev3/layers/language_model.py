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
"""Implements the Kimi K2.5 language model (ModuleV3).

The Kimi K2.5 text tower is the multi-GPU DeepseekV3 ModuleV3 model with a
multimodal-embedding merge inserted after token embedding. The heavy lifting
(MLA, MoE, EP, YaRN RoPE) is reused from ``deepseekV3_modulev3`` by composing a
:class:`DeepseekV3TextModel`; only the image-embedding merge and the two extra
graph inputs are added here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorValue
from max.nn import Module, as_subgraph
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    Replicated,
    Sharded,
    auto_reshard,
)

from ...deepseekV3_modulev3.deepseekV3 import (
    DeepseekV3TextModel,
    gather_last_tokens,
    split_replicated_batch,
)
from ..model_config import KimiK2_5Config


class KimiK2_5Decoder(Module[[DecoderInputs], DecoderOutputs]):
    """The Kimi K2.5 text tower, registered for the ``DECODER`` role.

    Composes a :class:`DeepseekV3TextModel` and reimplements its decode loop
    with a multimodal-embedding merge inserted after ``embed_tokens``. It
    runs tensor-parallel attention over a ``tp`` mesh, or one attention
    replica per device over a ``dp`` mesh, with its experts tensor-parallel
    or, under expert parallelism, spread across the devices.
    """

    graph_name: ClassVar[str] = "KimiK2_5MoEDecoder"
    supports_data_parallel: ClassVar[bool] = True
    consumes_vision: ClassVar[bool] = True

    def __init__(self, config: KimiK2_5Config) -> None:
        super().__init__()
        llm = config.llm_config
        if llm.mesh is None:
            raise ValueError("Kimi K2.5 needs a mesh: apply the runtime first")
        self.ep_batch_manager = (
            None if llm.ep_config is None else EPBatchManager(llm.ep_config)
        )
        # Quantized checkpoints keep norms, biases and embeddings in bf16.
        dtype = DType.bfloat16 if llm.quant_config is not None else llm.dtype
        with default_dtype(dtype):
            self.language_model = DeepseekV3TextModel(
                llm, self.ep_batch_manager
            )
        # The checkpoint names the text model's weights from its root.
        self.language_model.name_transparent = True
        self.config = llm
        self.kv_params = llm.kv_params

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Runs the language graph over the batch and its image embeddings."""
        mesh = self.config.mesh
        assert mesh is not None
        assert inputs.vision is not None
        assert inputs.batch_context_lengths is not None

        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        if self.config.data_parallel_degree > 1:
            batch_context_length = Tensor.from_shard_values(
                [TensorValue(shard) for shard in inputs.batch_context_lengths],
                DeviceMapping(mesh, (Replicated(),) * mesh.ndim),
            )
            assert inputs.host_input_row_offsets is not None
            data_parallel_splits = inputs.data_parallel_splits
            input_row_offsets_i64 = F.cast(
                inputs.host_input_row_offsets, DType.int64
            )
        else:
            batch_context_length = inputs.batch_context_lengths[0]

        kv_inputs = iter(x._graph_value for x in tree.leaves(inputs.kv))
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(kv_inputs),
            leaf=KVCacheInputsPerDevice,
        )
        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            assert inputs.ep is not None
            comm_buffers = self.ep_batch_manager.comm_buffers(
                [x._graph_value for x in tree.leaves(inputs.ep)]
            )

        kv_collection = PagedCacheValues.from_upstream(
            kv_collections,
            DeviceMapping(mesh, (Replicated(),) * mesh.ndim),
        )
        logits = self._run_text_model(
            inputs.tokens,
            kv_collection,
            inputs.input_row_offsets,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            batch_context_length,
            data_parallel_splits,
            input_row_offsets_i64,
            comm_buffers,
        )
        return DecoderOutputs(next_token_logits=logits)

    @auto_reshard(mode="raise")
    def _run_text_model(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        batch_context_length: Tensor,
        data_parallel_splits: Tensor | None,
        input_row_offsets_i64: Tensor | None,
        comm_buffers: EPCommBuffers | None,
    ) -> Tensor:
        lm = self.language_model
        mesh = lm.mesh
        if mesh is not None:
            tokens = F.distributed_broadcast(tokens, mesh)
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)

        h = lm.embed_tokens(tokens)
        # Scatter the vision embeddings into the (replicated) text stream at the
        # image-token positions, before the data-parallel batch split so the
        # scatter indices still address the full-batch row space.
        h = F_merge_multimodal_embeddings(
            h, image_embeddings, image_token_indices
        )

        # Under data parallelism, shard the replicated batch across the ``dp``
        # mesh axis; each replica then runs attention on its own rows.
        if self.config.data_parallel_degree > 1:
            assert data_parallel_splits is not None
            assert input_row_offsets_i64 is not None
            assert mesh is not None
            batch_placement = tuple(
                Sharded(0) if name == "dp" else Replicated()
                for name in mesh.axis_names
            )
            h, input_row_offsets = split_replicated_batch(
                h,
                input_row_offsets,
                input_row_offsets_i64,
                data_parallel_splits,
                DeviceMapping(mesh, batch_placement),
            )

        freqs_cis = F.cast(lm.rope.freqs_cis, h.dtype)
        if mesh is not None:
            freqs_cis = freqs_cis.to(mesh)
        else:
            freqs_cis = freqs_cis.to(h.device)

        # The MLA prefill plan depends only on the row offsets and KV cache, so
        # it is identical across layers. Compute it once.
        mla_prefill_metadata = None
        first_attn = lm.layers[0].self_attn
        if first_attn.graph_mode in ("prefill", "auto"):
            mla_prefill_metadata = first_attn.create_mla_prefill_metadata(
                input_row_offsets, kv_collection
            )
            # Host-substitute the per-layer D2H buffer_length copies with the
            # CPU batch_context_length so the graph stays capturable.
            mla_prefill_metadata.buffer_lengths = batch_context_length

        # The MoE blocks share one subgraph, as in ``DeepseekV3TextModel``.
        for idx, layer in enumerate(lm.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            call: Callable[..., Tensor] = layer
            if isinstance(layer.mlp, QuantizedMoE):
                call = as_subgraph(layer, name="moe_block")
            h = call(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
                freqs_cis,
                mla_prefill_metadata,
                comm_buffers,
            )

        last_token_h = gather_last_tokens(h, input_row_offsets)
        if self.config.data_parallel_degree > 1:
            last_token_h = F.allgather(last_token_h)
        last_logits = lm.lm_head(lm.norm(last_token_h))
        if mesh is not None:
            last_logits = last_logits.to(mesh.devices[0])
        return F.cast(last_logits, DType.float32)
