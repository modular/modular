# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import math
from typing import Tuple, Type, TypeVar

import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight, ops
from max.pipelines.architectures.pixtral.llava.llava import (
    LlavaConditionalGeneration,
)
from max.pipelines.architectures.pixtral.llava.llava_decoder import (
    Transformer as LLavaTransformer,
)
from max.pipelines.architectures.pixtral.llava.llava_projector import (
    LlavaMultiModalConnector,
)
from max.pipelines.architectures.pixtral.vision_encoder.attention import (
    Attention,
)
from max.pipelines.architectures.pixtral.vision_encoder.rotary_embedding_2d import (
    RotaryEmbedding2D,
)
from max.pipelines.architectures.pixtral.vision_encoder.transformer import (
    MLP,
    Transformer,
    TransformerBlock,
)
from max.pipelines.architectures.pixtral.vision_encoder.vision_encoder import (
    VisionEncoder,
)
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)
from max.pipelines.nn import MLP as nnMLP
from max.pipelines.nn import (
    AttentionWithRope,
    Conv2D,
    Embedding,
    Linear,
    OptimizedRotaryEmbedding,
    RMSNorm,
)
from max.pipelines.nn import TransformerBlock as nnTransformerBlock
from transformers import (
    AutoProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
    MistralConfig,
    PixtralProcessor,
    PixtralVisionConfig,
)
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

ACCURACY_RTOL = 1e-1
ACCURACY_ATOL = 1e-1
VOCAB_SIZE = 131072


@pytest.fixture
def pytorch_pixtral_processor() -> PixtralProcessor:
    model_id = "mistral-community/pixtral-12b"
    model_name = "mistralai/Pixtral-12B-2409"
    # returns a dict with the following keys: input_ids, attention_mask, pixel_values
    # input_ids and attention_mask map to tensors of the same size = (num_prompts, sequence_length)
    # pixel_values maps to a list of lists of tensors. length of the list is (num_prompts, num_images, tensor([num_channels, height, width]))
    processor = AutoProcessor.from_pretrained(model_id)
    return processor


@pytest.fixture
def pytorch_pixtral() -> LlavaForConditionalGeneration:
    model_id = "mistral-community/pixtral-12b"
    model_name = "mistralai/Pixtral-12B-2409"
    model = LlavaForConditionalGeneration.from_pretrained(model_id)
    return model


@pytest.fixture
def img_urls_and_prompt():
    IMG_URLS = [
        "https://picsum.photos/id/237/400/300",
        "https://picsum.photos/id/231/200/300",
        "https://picsum.photos/id/27/500/500",
        "https://picsum.photos/id/17/150/600",
    ]
    PROMPT = "<s>[INST]Describe the images.\n[IMG][IMG][IMG][IMG][/INST]"
    IMG_URLS = [
        "https://picsum.photos/id/237/400/300",
    ]
    PROMPT = "<s>[INST]Describe the images.\n[IMG][/INST]"
    return IMG_URLS, PROMPT


@pytest.fixture
def pytorch_connector():
    text_config = MistralConfig(
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        head_dim=128,
        hidden_act="silu",
        hidden_size=5120,
        initializer_range=0.02,
        intermediate_size=14336,
        max_position_embeddings=1024000,
        model_type="mistral",
        num_attention_heads=32,
        num_hidden_layers=40,
        num_key_value_heads=8,
        rms_norm_eps=1e-05,
        rope_theta=1000000000.0,
        sliding_window=None,
        tie_word_embeddings=False,
        vocab_size=131072,
    )
    vision_config = PixtralVisionConfig()
    return LlavaMultiModalProjector(
        LlavaConfig(
            vision_config,
            text_config,
            vision_feature_layer=-1,
            image_token_index=10,
            vision_feature_select_strategy="full",
            image_seq_length=1,
        )
    )


@pytest.fixture
def img_dtype():
    return torch.float32


@pytest.fixture
def size():
    return (1, 160, 1024)


@pytest.fixture
def img_features(size: Tuple, img_dtype: torch.dtype):
    # Gnerate imgs of shape (batch_size, seq_len=num_patches, hidden_size).
    return torch.randint(low=0, high=1, size=size).to(img_dtype)


@pytest.fixture
def graph_api_connector(pytorch_connector: LlavaMultiModalProjector):
    weights_registry: dict = {}
    weights_registry["linear_1"] = pytorch_connector.linear_1.weight.data
    weights_registry["linear_2"] = pytorch_connector.linear_2.weight.data

    linear_1 = Linear(
        Weight(
            name="linear_1",
            dtype=DType.from_numpy(weights_registry["linear_1"].numpy().dtype),
            shape=weights_registry["linear_1"].shape,
        )
    )
    linear_2 = Linear(
        Weight(
            name="linear_2",
            dtype=DType.from_numpy(weights_registry["linear_2"].numpy().dtype),
            shape=weights_registry["linear_2"].shape,
        )
    )
    connector = LlavaMultiModalConnector(linear_1, linear_2)
    return connector, weights_registry


def vision_encoder_given_pytorch_vision_encoder(pytorch_model, config):
    ########################### Weights ####################################
    # Collect all the weights into the weights registry.
    weights_registry: dict = {}

    def linear(name: str, array) -> Linear:
        """Creates a Linear layer backed by a weight."""
        weights_registry[name] = array
        return Linear(
            Weight(
                name=name,
                dtype=DType.from_numpy(array.numpy().dtype),
                shape=array.shape,
            )
        )

    rms_norm_weight = np.ones(config.hidden_size)
    filters = pytorch_model.patch_conv.weight.data
    filters = torch.permute(filters, (2, 3, 1, 0))

    mlp_gate_weights = [
        pytorch_model.transformer.layers[i].feed_forward.gate_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    mlp_up_weights = [
        pytorch_model.transformer.layers[i].feed_forward.up_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    mlp_down_weights = [
        pytorch_model.transformer.layers[i].feed_forward.down_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]

    attention_k_proj_weights = [
        pytorch_model.transformer.layers[i].attention.k_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    attention_v_proj_weights = [
        pytorch_model.transformer.layers[i].attention.v_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    attention_q_proj_weights = [
        pytorch_model.transformer.layers[i].attention.q_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    attention_o_proj_weights = [
        pytorch_model.transformer.layers[i].attention.o_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]

    ###################### Graph-API VisionEncoder #########################

    graph_patch_conv = Conv2D(
        filters.numpy(), stride=(config.patch_size, config.patch_size)
    )
    graph_ln_pre = RMSNorm(weight=rms_norm_weight, eps=1e-5)
    graph_rope = RotaryEmbedding2D(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_patches_per_side=config.image_size // config.patch_size,
    )
    attention_layers = []
    for i in range(config.num_hidden_layers):
        gate_proj = linear(
            name=f"vision.mlp_gate_weights_{i}", array=mlp_gate_weights[i]
        )
        down_proj = linear(
            name=f"vision.mlp_down_weights_{i}", array=mlp_down_weights[i]
        )
        up_proj = linear(
            name=f"vision.mlp_up_weights_{i}", array=mlp_up_weights[i]
        )
        mlp = MLP(gate_proj, down_proj, up_proj)

        wq = linear(
            name=f"vision.attention_q_proj_weights_{i}",
            array=attention_q_proj_weights[i],
        )
        wk = linear(
            name=f"vision.attention_k_proj_weights_{i}",
            array=attention_k_proj_weights[i],
        )
        wv = linear(
            name=f"vision.attention_v_proj_weights_{i}",
            array=attention_v_proj_weights[i],
        )
        wo = linear(
            name=f"vision.attention_o_proj_weights_{i}",
            array=attention_o_proj_weights[i],
        )

        attention = Attention(
            n_heads=config.num_attention_heads,
            dim=config.hidden_size,
            head_dim=config.hidden_size // config.num_attention_heads,
            dropout=config.attention_dropout,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
        )
        attention_norm = RMSNorm(weight=np.ones(config.hidden_size), eps=1e-5)
        mlp_norm = RMSNorm(weight=np.ones(config.hidden_size), eps=1e-5)
        attention_layers.append(
            TransformerBlock(attention, mlp, attention_norm, mlp_norm)
        )

    graph_transformer = Transformer(
        config.num_attention_heads, attention_layers, DType.float32
    )

    graph_encoder = VisionEncoder(
        patch_conv=graph_patch_conv,
        layer_norm=graph_ln_pre,
        patch_positional_embedding=graph_rope,
        transformer=graph_transformer,
        dtype=DType.float32,
        patch_size=config.patch_size,
        max_image_size=config.image_size,
    )
    return graph_encoder, weights_registry


T = TypeVar("T")


def missing_value(t: Type[T]) -> T:
    # TODO: Remove this function and replace the call site with a real value
    raise NotImplementedError


def mistral_given_pytorch_mistral(pytorch_model, config):
    # refer to:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L682
    # TODO: init these values for Mistral test
    kv_params = missing_value(KVCacheParams)  # passed to AttentionWithRope
    kv_collection_constructor = missing_value(
        FetchContinuousBatchingKVCacheCollection
    )  # passed to LLavaTransformer

    #######################Init weights with pytorch weights ###################
    weights_registry: dict = {}

    mlp_gate_weights = [
        pytorch_model.model.layers[i].mlp.gate_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    mlp_up_weights = [
        pytorch_model.model.layers[i].mlp.up_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]
    mlp_down_weights = [
        pytorch_model.model.layers[i].mlp.down_proj.weight.data
        for i in range(config.num_hidden_layers)
    ]

    ############ Define Graph-API layers with weights with pytorch #############
    def linear(name: str, array) -> Linear:
        """Creates a Linear layer backed by a weight."""
        weights_registry[name] = array
        return Linear(
            Weight(
                name=name,
                dtype=DType.from_numpy(array.numpy().dtype),
                shape=array.shape,
            )
        )

    def _weight(name: str, array) -> Weight:
        weights_registry[name] = array
        return Weight(
            name=name,
            dtype=DType.from_numpy(array.numpy().dtype),
            shape=array.shape,
        )

    def attention(
        kv_params, rope: OptimizedRotaryEmbedding, layer_idx: int
    ) -> AttentionWithRope:
        wq = ops.transpose(
            _weight(
                f"text.wq_weights_{layer_idx}",
                pytorch_model.model.layers[
                    layer_idx
                ].self_attn.q_proj.weight.data,
            ),
            0,
            1,
        )
        wk = ops.transpose(
            _weight(
                f"text.wk_weights_{layer_idx}",
                pytorch_model.model.layers[
                    layer_idx
                ].self_attn.k_proj.weight.data,
            ),
            0,
            1,
        )
        wv = ops.transpose(
            _weight(
                f"text.wv_weights_{layer_idx}",
                pytorch_model.model.layers[
                    layer_idx
                ].self_attn.v_proj.weight.data,
            ),
            0,
            1,
        )
        wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

        return AttentionWithRope(
            n_heads=config.num_attention_heads,
            kv_params=kv_params,
            wqkv=wqkv,
            wo=linear(
                name=f"text.wo_weights_{i}",
                array=pytorch_model.model.layers[
                    layer_idx
                ].self_attn.o_proj.weight.data,
            ),
            rope=rope,
            layer_idx=ops.constant(layer_idx, DType.uint32),
            scale=math.sqrt(1 / config.head_dim),
        )

    rope = OptimizedRotaryEmbedding(
        dim=config.num_attention_heads * config.head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_length,
    )

    transformer_layers = []
    for i in range(config.num_hidden_layers):
        gate_proj = linear(
            name=f"text.mlp_gate_weights_{i}", array=mlp_gate_weights[i]
        )
        down_proj = linear(
            name=f"text.mlp_down_weights_{i}", array=mlp_down_weights[i]
        )
        up_proj = linear(
            name=f"text.mlp_up_weights_{i}", array=mlp_up_weights[i]
        )

        layer = nnTransformerBlock(
            attention=attention(kv_params, rope, i),
            mlp=nnMLP(gate_proj, down_proj, up_proj),
            attention_norm=RMSNorm(
                pytorch_model.model.layers[
                    i
                ].post_attention_layernorm.weight.data,
                config.rms_norm_eps,
            ),
            mlp_norm=RMSNorm(
                pytorch_model.model.layers[i].input_layernorm.weight.data,
                config.rms_norm_eps,
            ),
        )
        transformer_layers.append(layer)

    norm_layer = RMSNorm(
        pytorch_model.model.norm.weight.data, config.rms_norm_eps
    )
    embedding_layer = Embedding(
        _weight(
            "text.embed_tokens", pytorch_model.model.embed_tokens.weight.data
        )
    )
    output_linear = linear(
        "text.output_linear", pytorch_model.lm_head.weight.data
    )

    model = LLavaTransformer(
        config.hidden_size,
        config.num_attention_heads,
        transformer_layers,
        norm_layer,
        output_linear,
        embedding_layer,
        kv_params,
        kv_collection_constructor,
    )
    return model, weights_registry


@pytest.fixture
def graph_api_pixtral(
    pytorch_pixtral, graph_api_connector: Tuple[LlavaMultiModalConnector, dict]
):
    weights_registry: dict = {}
    # Create a vision encoder with the weights of pytorch_pixtral.vision_tower
    vision_encoder, encoder_weights = (
        vision_encoder_given_pytorch_vision_encoder(
            pytorch_pixtral.vision_tower, pytorch_pixtral.config.vision_config
        )
    )
    weights_registry.update(encoder_weights)
    connector, connector_weights = graph_api_connector
    weights_registry.update(connector_weights)
    print(
        "##################### Pytorch Mistral Config #######################"
    )
    print(pytorch_pixtral.config.text_config)
    print("##################### Pytorch Mistral Model #######################")
    print(pytorch_pixtral.language_model)
    mistral, mistral_weights = mistral_given_pytorch_mistral(
        pytorch_pixtral.language_model, pytorch_pixtral.config.text_config
    )
    weights_registry.update(mistral_weights)
    pixtral = LlavaConditionalGeneration(
        vision_encoder,
        connector,
        mistral,
        VOCAB_SIZE,
        image_token_index=10,
        vision_feature_layer=-1,
        vision_feature_select_strategy="full",
        image_seq_length=1,
    )
    return pixtral, weights_registry


def test_connector(
    img_features: torch.Tensor,
    pytorch_connector: LlavaMultiModalProjector,
    graph_api_connector: Tuple[LlavaMultiModalConnector, dict],
    size: Tuple,
):
    weights_registry: dict = {}
    connector = graph_api_connector[0]
    weights_registry.update(graph_api_connector[1])

    session = InferenceSession()
    graph = Graph(
        "Llava_MLP",
        connector,
        input_types=(TensorType(DType.float32, size),),
    )

    compiled = session.load(graph, weights_registry=weights_registry)

    output_tensor = compiled.execute(img_features)[0]
    assert isinstance(output_tensor, Tensor)
    output = output_tensor.to_numpy()
    pytorch_output = pytorch_connector(img_features).detach().numpy()

    np.testing.assert_allclose(
        output,
        pytorch_output,
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def test_processor(img_urls_and_prompt: Tuple, pytorch_pixtral_processor):
    IMG_URLS, PROMPT = img_urls_and_prompt
    pytorch_inputs = pytorch_pixtral_processor(
        text=PROMPT, images=IMG_URLS, return_tensors="pt"
    )
    pytorch_input_ids = pytorch_inputs["input_ids"].detach().numpy()
    pytorch_attention_mask = pytorch_inputs["attention_mask"].detach().numpy()
    pytorch_pixel_values = pytorch_inputs["pixel_values"]


@pytest.mark.skip(reason="no way of currently testing this")
def test_pixtral(
    img_urls_and_prompt: Tuple,
    pytorch_pixtral_processor: PixtralProcessor,
    pytorch_pixtral: LlavaForConditionalGeneration,
    graph_api_pixtral: Tuple[LlavaConditionalGeneration, dict],
):
    IMG_URLS, PROMPT = img_urls_and_prompt
    pytorch_inputs = pytorch_pixtral_processor(
        text=PROMPT, images=IMG_URLS, return_tensors="pt"
    )
    pytorch_input_ids = pytorch_inputs["input_ids"]
    pytorch_pixel_values = pytorch_inputs["pixel_values"][0]
    pytorch_attention_mask = pytorch_inputs["attention_mask"]

    with torch.no_grad():
        pytorch_outputs = pytorch_pixtral(
            input_ids=pytorch_input_ids,
            pixel_values=pytorch_pixel_values,
            attention_mask=pytorch_attention_mask,
        )
        # output.logits shape = (1, 2242, 131072)

    # pixel_values_tesnor = torch.stack(pytorch_pixel_values)
    # print("pixel_values_tesnor.shape = ", pixel_values_tesnor.shape)

    # TODO: figure out input to LlavaConditionalGeneration

    weights_registry: dict = {}
    pixtral = graph_api_pixtral[0]
    weights_registry.update(graph_api_pixtral[1])

    # pytorch_input_ids.shape torch.Size([1, 2242]). for one image: [1, 502]
    # pytorch_pixel_values .shape= [
    # torch.Size([304, 400, 3]), torch.Size([304, 208, 3]),
    # torch.Size([512, 512, 3]), torch.Size([608, 160, 3])]
    graph_api_pixel_values = [
        torch.permute(img, (1, 2, 0)) for img in pytorch_pixel_values
    ]

    print(
        pytorch_input_ids.shape, [img.shape for img in graph_api_pixel_values]
    )

    # # TODO: Convert pytorch_pixel_values to a tensor rather than lists
    # session = InferenceSession()
    # graph = Graph(
    #     "Llava",
    #     pixtral,
    #     input_types=(
    #         TensorType(DType.int64, pytorch_input_ids.shape),
    #         TensorType(DType.float32, graph_api_pixel_values[0].shape),
    #     ),
    # )

    # compiled = session.load(graph, weights_registry=weights_registry)

    # output = compiled.execute(pytorch_input_ids, graph_api_pixel_values[0])[
    #     0
    # ].to_numpy()

    # np.testing.assert_allclose(
    #     output,
    #     pytorch_outputs.logits,
    #     equal_nan=True,
    #     rtol=ACCURACY_RTOL,
    #     atol=ACCURACY_ATOL,
    # )
