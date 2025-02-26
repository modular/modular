# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import KVCacheManager, KVCacheParams
from max.pipelines.nn import Linear

from ..llava.llava import (
    LlavaConditionalGeneration,
    LlavaConditionalGenerationTextOnly,
    LlavaVisionEncoder,
)
from ..llava.llava_projector import LlavaMultiModalConnector
from ..vision_encoder.graph import _vision_encoder
from .mistral_graph import _transformer


def _linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    """Unlike the vision encoder's version, this linear layer has a bias.
    This linear layer is used by the LlavaMultiModalConnector
    """
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None),
        bias=weights.bias.allocate(dtype, [in_features], None),
    )


def _multi_modal_projector(
    dtype: DType,
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
) -> LlavaMultiModalConnector:
    """Connects the vision encoder to the text decoder.
    This MLP projects the patch embeddings to the text-encoder's embeddings space.
    Input shape:
    Output shape:
    """
    return LlavaMultiModalConnector(
        _linear(
            dtype,
            pipeline_config.huggingface_config.text_config.hidden_size,
            pipeline_config.huggingface_config.vision_config.hidden_size,
            weights.linear_1,
        ),
        _linear(
            dtype,
            pipeline_config.huggingface_config.text_config.hidden_size,
            pipeline_config.huggingface_config.text_config.hidden_size,
            weights.linear_2,
        ),
    )


def _llava_vision_encoder_and_projector(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
) -> LlavaVisionEncoder:
    # TODO(AIPIPE-273): Once we have mo.if, use this version of Llava rather than creating 2 graphs
    vision_encoder = _vision_encoder(graph, pipeline_config, weights)
    multi_modal_projector = _multi_modal_projector(
        pipeline_config.dtype, pipeline_config, weights.multi_modal_projector
    )
    return LlavaVisionEncoder(
        vision_encoder=vision_encoder,
        multi_modal_projector=multi_modal_projector,
    )


def _llava_decoder(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> LlavaConditionalGenerationTextOnly:
    # Weights of pixtral decoder have the same names and shapes as weights of mistral.
    language_model = _transformer(
        graph=graph,
        params=pipeline_config,
        weights=weights,
        max_seq_len=max_seq_len,
        kv_params=kv_params,
    )

    return LlavaConditionalGenerationTextOnly(
        language_model=language_model,
        vocab_size=pipeline_config.huggingface_config.text_config.vocab_size,
        image_token_index=pipeline_config.huggingface_config.image_token_index,
    )


def _llava(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> LlavaConditionalGeneration:
    # TODO: Once we have mo.if, use this version of Llava rather than creating 2 graphs
    vision_encoder = _vision_encoder(graph, pipeline_config, weights)
    multi_modal_projector = _multi_modal_projector(
        dtype=pipeline_config.dtype,
        pipeline_config=pipeline_config,
        weights=weights.multi_modal_projector,
    )
    # Weights of pixtral have the same names and shapes as weights of mistral.
    language_model = _transformer(
        graph=graph,
        params=pipeline_config,
        weights=weights,
        max_seq_len=max_seq_len,
        kv_params=kv_params,
    )

    return LlavaConditionalGeneration(
        vision_encoder=vision_encoder,
        multi_modal_projector=multi_modal_projector,
        language_model=language_model,
        vocab_size=pipeline_config.huggingface_config.text_config.vocab_size,
        image_token_index=pipeline_config.huggingface_config.image_token_index,
        vision_feature_layer=pipeline_config.huggingface_config.vision_feature_layer,
        vision_feature_select_strategy=pipeline_config.huggingface_config.vision_feature_select_strategy,
        image_seq_length=pipeline_config.huggingface_config.image_seq_length,
    )


def _build_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    max_seq_len: int,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    # TODO: Make this work for multiple devices. Now getting the types for device [0]
    kv_cache_types = kv_manager.input_symbols()[0]

    input_ids_type = TensorType(
        DType.int64,
        shape=["total_seq_len"],
    )
    # TODO: should be changed to add "batch_size", "n_images" dims when working with multiple images
    pixel_values_type = TensorType(
        DType.float32,
        shape=["image_height", "image_width", "num_channels"],
    )

    attention_mask_type = TensorType(
        DType.float32,
        shape=["batch_size", 1, "num_patches", "num_patches"],
    )

    # Type of start and end position of each batch in the combined total_seq_len dimension.
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"]
    )

    # Initialize Graph.
    with Graph(
        "pixtral",
        input_types=[
            input_ids_type,
            pixel_values_type,
            attention_mask_type,
            input_row_offsets_type,
            *kv_cache_types,
        ],
    ) as graph:
        model = _llava(
            graph=graph,
            pipeline_config=pipeline_config,
            weights=weights,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )
        (
            input_ids,
            pixel_values,
            attention_mask,
            input_row_offsets,
            *kv_cache_inputs,
        ) = graph.inputs
        outputs = model(
            input_ids=input_ids.tensor,
            pixel_values=pixel_values.tensor,
            attention_mask=attention_mask.tensor,
            kv_cache_inputs=kv_cache_inputs,  # type: ignore
            input_row_offsets=input_row_offsets,
        )
        graph.output(*outputs)
        return graph


def _build_vision_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
) -> Graph:
    # Graph input types.
    pixel_values_type = TensorType(
        DType.float32,
        shape=["image_height", "image_width", "num_channels"],
    )

    attention_mask_type = TensorType(
        DType.float32,
        shape=["batch_size", 1, "num_patches", "num_patches"],
    )

    # Initialize Graph.
    with Graph(
        "pixtral_vision_encoder",
        input_types=[
            pixel_values_type,
            attention_mask_type,
        ],
    ) as graph:
        model = _llava_vision_encoder_and_projector(
            graph, pipeline_config, weights
        )
        (
            pixel_values,
            attention_mask,
        ) = graph.inputs
        outputs = model(
            pixel_values=pixel_values.tensor,
            attention_mask=attention_mask.tensor,
        )
        graph.output(outputs)
        return graph


def _build_text_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    max_seq_len: int,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    # TODO: Make this work for multiple devices. Now getting the types for device [0]
    kv_cache_types = kv_manager.input_symbols()[0]

    input_ids_type = TensorType(
        DType.int64,
        shape=["total_seq_len"],
    )

    # Type of start and end position of each batch in the combined total_seq_len dimension.
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"]
    )

    # num_images, num_patches_in_image, language_model_hidden_dim
    image_embeddings_type = TensorType(
        pipeline_config.dtype,
        shape=[
            # TODO(bduke): fix algebraic dim creation outside of graph contexts.
            "num_images",
            "num_patches_in_image",
            pipeline_config.huggingface_config.text_config.hidden_size,
        ],
    )

    # Initialize Graph.
    with Graph(
        "pixtral",
        input_types=[
            input_ids_type,
            image_embeddings_type,
            input_row_offsets_type,
            *kv_cache_types,
        ],
    ) as graph:
        model = _llava_decoder(
            graph=graph,
            pipeline_config=pipeline_config,
            weights=weights,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )
        (
            input_ids,
            image_embeds,
            input_row_offsets,
            *kv_cache_inputs,
        ) = graph.inputs
        outputs = model(
            input_ids=input_ids.tensor,
            image_embeds=image_embeds.tensor,
            kv_cache_inputs=kv_cache_inputs,  # type: ignore
            input_row_offsets=input_row_offsets,
        )
        graph.output(*outputs)
        return graph
