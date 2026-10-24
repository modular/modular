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
"""Implements the Kimi LM."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.graph import Module as GraphModule
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import DecoderOutputs
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from ...deepseekV3.deepseekV3 import (
    DeepseekV3,
)


class KimiK2_5MoEDecoder(DeepseekV3):
    subgraph_layer_prefix: str = "language_model.layers"

    graph_kv_params: KVCacheParamInterface | None = None
    """The KV cache parameters the traced graph's KV inputs follow.

    Defaults to the config's; the pipeline model sets its own, whose cache
    dtype can differ.
    """

    def __call__(  # type: ignore[override]
        self,
        tokens: TensorValue,
        image_embeddings: list[TensorValue],
        image_token_indices: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        eplb_counter_buffers_per_layer: list[list[BufferValue]] | None = None,
    ) -> DecoderOutputs:
        h = self.embed_tokens(tokens, signal_buffers)

        # Merge image embeddings into text embeddings
        h = [
            merge_multimodal_embeddings(
                inputs_embeds=h_device,
                multimodal_embeddings=image_embeddings_device,
                image_token_indices=image_token_indices_device,
            )
            for h_device, image_embeddings_device, image_token_indices_device in zip(
                h,
                image_embeddings,
                image_token_indices,
                strict=True,
            )
        ]

        input_row_offsets_per_dev = list(
            ops.distributed_broadcast(input_row_offsets, signal_buffers)
        )
        return self._process_hidden_states(
            h,
            signal_buffers,
            kv_collections,
            return_n_logits,
            input_row_offsets_per_dev,
            host_input_row_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,
            eplb_counter_buffers_per_layer,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
    ) -> tuple[TensorType | BufferType, ...]:
        all_input_types: tuple[TensorType | BufferType, ...] = (
            super().input_types(kv_params)
        )

        image_embeddings_types = [
            TensorType(
                DType.bfloat16,
                shape=[
                    "vision_merged_seq_len",
                    self.config.hidden_size,
                ],
                device=DeviceRef.from_device(device),
            )
            for device in self.config.devices
        ]

        image_token_indices_types = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.config.devices
        ]
        tokens_type, remaining_input_types = (
            all_input_types[0],
            all_input_types[1:],
        )
        return (
            tokens_type,
            *image_embeddings_types,
            *image_token_indices_types,
        ) + remaining_input_types

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph, whose inputs are :meth:`input_types` for
        :attr:`graph_kv_params`.

        Adapts the flat graph inputs to :meth:`__call__`: splits off the
        per-device vision-merge inputs and signal buffers, and unflattens the
        KV cache. The EP inputs, then the per-layer EPLB counters when
        profiling, are the trailing inputs.

        Args:
            *input_types: The graph's input types.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        n = len(self.config.devices)
        kv_params = self.graph_kv_params or self.config.kv_params
        num_eplb_layers = (
            self.config.num_hidden_layers
            if self.config.eplb_profile_enabled
            else 0
        )

        def forward(*inputs: Value[Any]) -> DecoderOutputs:
            tokens, all_inputs = inputs[0], inputs[1:]
            image_embeddings, all_inputs = all_inputs[:n], all_inputs[n:]
            image_token_indices, all_inputs = all_inputs[:n], all_inputs[n:]
            (
                devices_input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *variadic_args,
            ) = all_inputs

            variadic_args_iter = iter(variadic_args)
            signal_buffers = [next(variadic_args_iter).buffer for _ in range(n)]
            kv_caches_per_dev = tree.leaves(
                kv_params.unflatten_kv_inputs(variadic_args_iter),
                leaf=KVCacheInputsPerDevice,
            )
            batch_context_lengths = [
                next(variadic_args_iter).tensor for _ in range(n)
            ]

            remaining = list(variadic_args_iter)
            if num_eplb_layers:
                flat_bufs = [
                    v.buffer for v in remaining[-num_eplb_layers * n :]
                ]
                eplb_counter_buffers_per_layer: (
                    list[list[BufferValue]] | None
                ) = [
                    flat_bufs[layer_idx * n : (layer_idx + 1) * n]
                    for layer_idx in range(num_eplb_layers)
                ]
                ep_model_inputs = remaining[: -num_eplb_layers * n]
            else:
                eplb_counter_buffers_per_layer = None
                ep_model_inputs = remaining

            return self(
                tokens=tokens.tensor,
                image_embeddings=[v.tensor for v in image_embeddings],
                image_token_indices=[v.tensor for v in image_token_indices],
                signal_buffers=signal_buffers,
                kv_collections=kv_caches_per_dev,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=devices_input_row_offsets.tensor,
                host_input_row_offsets=host_input_row_offsets.tensor,
                data_parallel_splits=data_parallel_splits.tensor,
                batch_context_lengths=batch_context_lengths,
                ep_inputs=ep_model_inputs,
                eplb_counter_buffers_per_layer=eplb_counter_buffers_per_layer,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
