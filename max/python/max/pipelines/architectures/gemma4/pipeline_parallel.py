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

"""Text-only TP1 Gemma4 pipeline stage graphs."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TypedDict

from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.weights import WeightData
from max.nn.comm.allreduce import Signals
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates

from .gemma4 import Gemma4TextModel
from .model_config import Gemma4ForConditionalGenerationConfig


class StageCacheIndex(TypedDict):
    global_layer_id: int
    cache_class: int
    local_layer_index: int


def stage_cache_indices(
    layer_types: list[str], start: int, end: int
) -> list[StageCacheIndex]:
    """Returns stage-local indices for global decoder layers."""
    if not 0 <= start < end <= len(layer_types):
        raise ValueError("Invalid pipeline layer range")
    counts = [0, 0]
    mapping: list[StageCacheIndex] = []
    for global_id in range(start, end):
        kind = layer_types[global_id]
        if kind not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported attention type: {kind}")
        cache_class = int(kind == "full_attention")
        mapping.append(
            {
                "global_layer_id": global_id,
                "cache_class": cache_class,
                "local_layer_index": counts[cache_class],
            }
        )
        counts[cache_class] += 1
    return mapping


def stage_config(
    config: Gemma4ForConditionalGenerationConfig, start: int, end: int
) -> Gemma4ForConditionalGenerationConfig:
    """Copies configuration with KV allocations restricted to the stage."""
    text = config.text_config
    if config.unquantized_dtype != DType.bfloat16 or config.dtype not in (
        DType.bfloat16,
        DType.uint8,
    ):
        raise ValueError("Pipeline stages require BF16 or NVFP4 weights")
    if config.dtype == DType.uint8 and (
        text.quant_config is None or not text.quant_config.is_nvfp4
    ):
        raise ValueError(
            "Packed pipeline weights require NVFP4 quantization metadata"
        )
    if len(config.devices) != 1:
        raise ValueError("Gemma4 pipeline export currently requires TP1")
    if (
        text.enable_moe_block
        or text.hidden_size_per_layer_input
        or text.num_kv_shared_layers
        or text.target_layer_ids
        or text.return_hidden_states != ReturnHiddenStates.NONE
    ):
        raise ValueError(
            "Pipeline stages require dense text with no shared KV or auxiliary"
            " hidden state"
        )
    if len(text.layer_types) != text.num_hidden_layers:
        raise ValueError("Attention pattern must cover every global layer")
    mapping = stage_cache_indices(text.layer_types, start, end)
    children: dict[str, KVCacheParams] = {}
    for kind, cache_class in (("sliding_attention", 0), ("full_attention", 1)):
        params = config.kv_params.children[kind]
        assert isinstance(params, KVCacheParams)
        # Scale tensors would add stage graph inputs that the PP KV handoff
        # does not carry, so FP8 is accepted only as plain storage.
        if (
            params.dtype not in (DType.bfloat16, DType.float8_e4m3fn)
            or params.kvcache_quant_config is not None
            or params.enable_prefix_caching
            or params.data_parallel_degree != 1
            or params.speculative_method is not None
        ):
            raise ValueError(
                "Pipeline stages require unscaled BF16 or FP8 KV with data"
                " parallel degree 1 and without prefix caching or speculative"
                " decoding"
            )
        count = sum(row["cache_class"] == cache_class for row in mapping)
        if not count:
            raise ValueError(
                "Each initial Gemma4 stage must contain both cache classes"
            )
        children[kind] = dataclasses.replace(params, num_layers=count)
    return dataclasses.replace(
        config,
        vision_config=None,
        kv_params=MultiKVCacheParams.from_params(children),
    )


def stage_weights(
    weights: Mapping[str, WeightData],
    *,
    start: int,
    end: int,
    num_layers: int,
    tied_embeddings: bool,
) -> dict[str, WeightData]:
    """Selects global-named adapted weights without repacking quantized data."""
    selected = {}
    for name, value in weights.items():
        if name.startswith("layers."):
            layer_id = int(name.split(".", 2)[1])
            if start <= layer_id < end:
                selected[name] = value
        elif name.startswith("embed_tokens."):
            if start == 0 or (end == num_layers and tied_embeddings):
                selected[name] = value
        elif name.startswith(("norm.", "lm_head.")) and end == num_layers:
            selected[name] = value
    return selected


def build_stage_graph(
    config: Gemma4ForConditionalGenerationConfig,
    weights: Mapping[str, WeightData],
    *,
    start: int,
    end: int,
    capture_layer: int | None = None,
) -> tuple[Graph, dict[str, DLPackArray]]:
    """Builds one stage with an all-token BF16 hidden-state boundary.

    The first three slots match Mach's ordinary input preparation. The
    return-n-logits compatibility slot is read only by the tail.
    """
    config = stage_config(config, start, end)
    device = config.devices[0]
    head = start == 0
    tail = end == config.text_config.num_hidden_layers
    payload = TensorType(
        DType.int64 if head else DType.bfloat16,
        ["total_seq_len"]
        if head
        else [
            "total_seq_len",
            config.text_config.hidden_size,
        ],
        device=device,
    )
    input_types = [
        payload,
        TensorType(DType.int64, ["return_n_logits"], device=device.CPU()),
        TensorType(DType.uint32, ["input_row_offsets_len"], device=device),
        *Signals(config.devices).input_types(),
        *config.kv_params.flattened_kv_inputs(),
    ]
    with Graph(f"gemma4_stage_{start}_{end}", input_types=input_types) as graph:
        model = Gemma4TextModel(config, layer_range=(start, end))
        selected = stage_weights(
            weights,
            start=start,
            end=end,
            num_layers=config.text_config.num_hidden_layers,
            tied_embeddings=config.tie_word_embeddings,
        )
        # Some checkpoints include a redundant lm_head alias for the tied matrix.
        required = model.raw_state_dict()
        selected = {
            name: value for name, value in selected.items() if name in required
        }
        # The generic loader trusts uint8 checkpoint extents and replaces the
        # declaration, which can silently truncate an NVFP4 projection.
        for name, value in selected.items():
            expected = required[name]
            if expected.dtype != DType.uint8:
                continue
            expected_shape = tuple(expected.shape.static_dims)
            metadata_shape = tuple(value.shape.static_dims)
            actual_shape = tuple(Buffer.from_dlpack(value.data).shape)
            if (
                metadata_shape != expected_shape
                or actual_shape != expected_shape
            ):
                raise ValueError(
                    f"Packed NVFP4 weight '{name}' has invalid shape"
                    f" (expected={expected_shape}, metadata={metadata_shape},"
                    f" actual={actual_shape})"
                )
        model.load_state_dict(selected, weight_alignment=1, strict=True)
        payload_value, return_n_logits, row_offsets, signal, *cache_inputs = (
            graph.inputs
        )
        signals = [signal.buffer]
        sliding, full = config.kv_params.unflatten_basic_kv_tree(
            iter(cache_inputs)
        )
        caches = {"sliding_attention": sliding, "full_attention": full}
        hidden = (
            model.embed_tokens(payload_value.tensor, signals)
            if head
            else [payload_value.tensor]
        )
        captured = None
        for local_id, layer in enumerate(model.layers):
            hidden = layer(
                ops.constant(
                    model.global_layer_ids[local_id],
                    DType.uint32,
                    device=device,
                ),
                hidden,
                signals,
                caches[model._layer_kv_key[local_id]],
                input_row_offsets=[row_offsets.tensor],
            )
            if model.global_layer_ids[local_id] == capture_layer:
                captured = hidden[0]
        if capture_layer is not None and captured is None:
            raise ValueError("Capture layer is outside this stage")
        outputs = (
            model._postprocess_logits(
                hidden, [row_offsets.tensor], return_n_logits.tensor, signals
            )
            if tail
            else (hidden[0].cast(DType.bfloat16),)
        )
        graph.output(*outputs, *((captured,) if captured is not None else ()))
    return graph, model.state_dict()
