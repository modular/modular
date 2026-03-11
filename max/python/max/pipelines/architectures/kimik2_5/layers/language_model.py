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

from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.nn.attention.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.nn.comm import Signals
from max.nn.data_parallelism import split_batch_replicated
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import (
    forward_sharded_layers,
)
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)

from ...deepseekV3.deepseekV3 import (
    DeepseekV3,
    DeepseekV3DecoderLayer,
    _unpack_kv_collections,
)


class KimiDecoder(DeepseekV3):
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
    ) -> tuple[TensorValue, ...]:
        if not host_input_row_offsets.device == DeviceRef.CPU():
            raise ValueError("input_row_offsets must be located on CPU")
        if not data_parallel_splits.device == DeviceRef.CPU():
            raise ValueError("data_parallel_splits must be located on CPU")

        devices = self.config.devices
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

        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        # Keep this as explicit per-device `.to()` copies.
        # Broadcasting graph-time constants can hang when chained after
        # runtime-dependent collectives (GEX-3200).
        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        if not input_row_offsets.device == devices[0]:
            raise ValueError(
                f"input_row_offsets must be located on {devices[0]}"
            )
        input_row_offsets_ = ops.distributed_broadcast(
            input_row_offsets, signal_buffers
        )

        if len(devices) > 1:
            # Split batch across devices for data-parallel attention.
            h, input_row_offsets_ = split_batch_replicated(
                devices,
                h,
                input_row_offsets_,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )

        # Create MLA prefill metadata if not in decode mode
        if self.config.graph_mode != "decode":
            mla_prefill_metadata = self.layers[
                0
            ].self_attn.create_mla_prefill_metadata(  # type: ignore
                input_row_offsets_, kv_collections
            )

            # replace each device's buffer_lengths with the batch context length
            assert len(mla_prefill_metadata) == len(batch_context_lengths)
            for i in range(len(batch_context_lengths)):
                mla_prefill_metadata[i].buffer_lengths = batch_context_lengths[
                    i
                ]

        # Flatten MLAPrefillMetadata to list of TensorValues for subgraph calls
        mla_prefill_metadata_flat: list[TensorValue] = []
        for metadata in mla_prefill_metadata:
            mla_prefill_metadata_flat.extend(
                [
                    metadata.buffer_row_offsets,
                    metadata.cache_offsets,
                    metadata.buffer_lengths,
                ]
            )

        # Unpack KV collections once for use throughout the method
        kv_blocks, cache_lengths, lookup_tables, max_lengths, kv_scales = (
            _unpack_kv_collections(kv_collections)
        )

        # Extract dispatch metadata from KV collections (already on GPU
        # for MLA, on CPU for MHA — placed by the KV cache manager).
        mla_decode_scalar_args: list[TensorValue] | None = None
        if kv_collections[0].dispatch_metadata is not None:
            mla_decode_scalar_args = [
                kv.dispatch_metadata.tensor
                for kv in kv_collections
                if kv.dispatch_metadata is not None
            ]

        subgraph_input_types: list[Type[Any] | list[Type[Any]]] = [
            TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
            [hidden.type for hidden in h],
            [signal_buffer.type for signal_buffer in signal_buffers],
            [block.type for block in kv_blocks],
            [length.type for length in cache_lengths],
            [table.type for table in lookup_tables],
            [length.type for length in max_lengths],
            [scale.type for scale in kv_scales],
            [freq.type for freq in freqs_cis],
            [val.type for val in mla_prefill_metadata_flat],
            [offset.type for offset in input_row_offsets_],
        ]

        if mla_decode_scalar_args is not None:
            subgraph_input_types.append(
                [m.type for m in mla_decode_scalar_args]
            )

        if self.ep_manager is not None:
            subgraph_input_types.append(list(self.ep_manager.input_types()))

        subgraphs = []
        for group_idx, layer_group in enumerate(self.subgraph_layer_groups):
            assert len(layer_group) > 0, (
                "Subgraph layer groups must contain at least one layer"
            )
            subgraph_layer = self.layers[layer_group[0]]
            assert isinstance(subgraph_layer, DeepseekV3DecoderLayer), (
                "Subgraph layer must be a DeepseekV3DecoderLayer"
            )
            subgraphs.append(
                subgraph_layer.build_subgraph(
                    f"dist_transformer_block_{group_idx}",
                    subgraph_input_types,
                    f"layers.{layer_group[0]}.",
                )
            )

        for idx, layer in enumerate(self.layers):
            has_subgraph = False
            for group_idx, layer_group in enumerate(self.subgraph_layer_groups):
                if idx in layer_group:
                    has_subgraph = True
                    h = [
                        x.tensor
                        for x in ops.call(
                            subgraphs[group_idx],
                            ops.constant(
                                idx, DType.uint32, device=DeviceRef.CPU()
                            ),
                            *h,
                            *signal_buffers,
                            *kv_blocks,
                            *cache_lengths,
                            *lookup_tables,
                            *max_lengths,
                            *kv_scales,
                            *freqs_cis,
                            *mla_prefill_metadata_flat,
                            *input_row_offsets_,
                            *(
                                mla_decode_scalar_args
                                if mla_decode_scalar_args is not None
                                else ()
                            ),
                            *(ep_inputs if ep_inputs is not None else ()),
                            prefix=f"layers.{idx}.",
                        )
                    ]
                    break
            if not has_subgraph:
                h = layer(
                    ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                    h,
                    signal_buffers,
                    kv_blocks,
                    cache_lengths,
                    lookup_tables,
                    max_lengths,
                    kv_scales,
                    freqs_cis=freqs_cis,
                    mla_prefill_metadata_flat=mla_prefill_metadata_flat,
                    input_row_offsets=input_row_offsets_,
                    mla_decode_scalar_args=mla_decode_scalar_args,
                    ep_inputs=ep_inputs,
                )
                assert isinstance(h, list)

        if self.config.data_parallel_degree > 1:
            last_token_per_dev: list[TensorValue] = []
            for dev_idx in range(len(devices)):
                h0 = h[dev_idx]
                last_token_indices = input_row_offsets_[dev_idx][1:] - 1
                last_token_h = ops.gather(h0, last_token_indices, axis=0)
                last_token_per_dev.append(last_token_h)
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        else:
            last_token_distributed = [
                ops.gather(h_i, offsets_i[1:] - 1, axis=0)
                for h_i, offsets_i in zip(h, input_row_offsets_, strict=True)
            ]

        # Apply norm to each shard
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            if self.config.data_parallel_degree > 1:
                # Data-parallel case: gather variable tokens per device, then allgather
                # Create the range once on device 0 (range inputs must be on CPU)
                return_n_logits_range = ops.range(
                    start=return_n_logits[0],
                    stop=0,
                    step=-1,
                    out_dim="return_n_logits_range",
                    dtype=DType.int64,
                    device=devices[0],
                )
                variable_tokens_per_dev: list[TensorValue] = []
                for dev_idx in range(len(devices)):
                    h0 = h[dev_idx]
                    dev_return_n_logits_range = return_n_logits_range.to(
                        devices[dev_idx]
                    )
                    # Compute indices for last return_n_logits tokens per
                    # sequence on this device
                    dev_offsets = (
                        ops.unsqueeze(input_row_offsets_[dev_idx][1:], -1)
                        - dev_return_n_logits_range
                    )
                    indices = ops.reshape(dev_offsets, shape=(-1,))
                    variable_h = ops.gather(h0, indices, axis=0)
                    variable_tokens_per_dev.append(variable_h)

                variable_tokens_distributed = ops.allgather(
                    variable_tokens_per_dev, signal_buffers
                )

                norm_variable_tokens = forward_sharded_layers(
                    self.norm_shards, variable_tokens_distributed
                )
                logits = ops.cast(
                    self.lm_head(norm_variable_tokens, signal_buffers)[0],
                    DType.float32,
                )

                offsets = ops.range(
                    0,
                    TensorValue(logits.shape[0]) + return_n_logits[0],
                    return_n_logits[0],
                    out_dim="logit_offsets",
                    dtype=DType.int64,
                    device=devices[0],
                )
            else:
                # Non-EP case: keep existing single-device implementation
                return_n_logits_range = ops.range(
                    start=return_n_logits[0],
                    stop=0,
                    step=-1,
                    out_dim="return_n_logits_range",
                    dtype=DType.int64,
                    device=devices[0],
                )
                last_offsets = (
                    ops.unsqueeze(input_row_offsets_[0][1:], -1)
                    - return_n_logits_range
                )
                last_indices = ops.reshape(last_offsets, shape=(-1,))
                logits = ops.gather(
                    ops.cast(
                        self.lm_head(
                            forward_sharded_layers(self.norm_shards, h),
                            signal_buffers,
                        )[0],
                        DType.float32,
                    ),
                    last_indices,
                    axis=0,
                )
                offsets = ops.range(
                    0,
                    TensorValue(last_indices.shape[0]) + return_n_logits[0],
                    return_n_logits[0],
                    out_dim="logit_offsets",
                    dtype=DType.int64,
                    device=devices[0],
                )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(
                self.lm_head(
                    forward_sharded_layers(self.norm_shards, h),
                    signal_buffers,
                )[0],
                DType.float32,
            )
            offsets = input_row_offsets_[0]

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        ret_val: tuple[TensorValue, ...] = (last_logits,)
        if logits is not None and offsets is not None:
            ret_val += (logits, offsets)

        if self.return_hidden_states == ReturnHiddenStates.LAST:
            if self.config.data_parallel_degree > 1:
                ret_val += tuple(last_token_per_dev)
            else:
                ret_val += tuple(last_token_distributed)
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            norm_h = forward_sharded_layers(self.norm_shards, h)
            ret_val += tuple(norm_h)

        return ret_val

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        # TODO: Move input symbol computation from the manager classes.
        # It should be possible to compute the input symbols from the model
        # config.
        device_ref = self.config.devices[0]

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
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

        device_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        )

        # Add host input row offsets type, this is used to split the
        # concatenated DP inputs.
        host_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=[self.config.data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        )

        signals = Signals(devices=self.config.devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            *image_embeddings_types,
            *image_token_indices_types,
            device_input_row_offsets_type,
            host_input_row_offsets_type,
            return_n_logits_type,
            data_parallel_splits_type,
        ]
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(kv_params.get_symbolic_inputs().flatten())

        # Add batch context lengths
        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(self.config.devices))]
        )

        if self.ep_manager is not None:
            all_input_types.extend(self.ep_manager.input_types())
        return tuple(all_input_types)
