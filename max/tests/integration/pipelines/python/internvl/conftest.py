# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy
from max.pipelines.architectures.internvl.model_config import (
    InternVLConfig,
    VisionConfig,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config

"""
Fixtures for InternVL tests, including vision config, generated input tensors,
and dummy weights for the vision attention layer using computed standard deviations.
"""


@pytest.fixture
def vision_config() -> VisionConfig:
    """Create a test vision config for InternVL using realistic sizes."""
    return VisionConfig(
        dtype=DType.bfloat16,
        hidden_size=1024,  # Reduced for faster testing
        intermediate_size=4096,  # 4x hidden_size
        norm_type="rms_norm",
        image_size=224,  # Reduced for faster testing
        patch_size=14,
        num_attention_heads=16,  # Reduced for hidden_size=1024
        head_dim=64,  # 1024 / 16 = 64 head_dim
        layer_norm_eps=1e-6,
        qk_normalization=True,
        num_hidden_layers=4,  # Back to 4 layers with proper weight scoping
        use_mean_pooling=False,
    )


@pytest.fixture
def llm_config() -> Llama3Config:
    """Create a test LLM config for InternVL using realistic sizes."""
    # Create minimal KV cache params
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,  # 5120 / 40 = 128
        page_size=16,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=False,
        enable_kvcache_swapping_to_host=False,
        host_kvcache_swap_space_gb=0,
        n_devices=1,
    )

    return Llama3Config(
        num_attention_heads=40,  # From real config
        num_key_value_heads=8,  # From real config
        hidden_size=5120,  # From real config
        num_hidden_layers=64,  # From real config
        intermediate_size=27648,  # From real config intermediate_size
        vocab_size=151674,  # From real config
        rope_theta=1000000.0,  # From real config
        rope_scaling_params=None,
        max_seq_len=32768,  # From real config max_position_embeddings
        interleaved_rope_weights=True,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        return_logits=ReturnLogits.LAST_TOKEN,
        norm_method="rms_norm",
        norm_dtype=None,
        attention_bias=True,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        stacked_mlp=False,
        stacked_qkv=False,
        logits_postprocessor=None,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[DeviceRef.GPU()],
        clip_qkv=None,
        float8_config=None,
    )


@pytest.fixture
def internvl_config(
    vision_config: VisionConfig, llm_config: Llama3Config
) -> InternVLConfig:
    """Create a test InternVL config."""
    return InternVLConfig(
        devices=[DeviceRef.GPU()],
        downsample_ratio=0.5,
        num_image_token=256,
        vision_config=vision_config,
        llm_config=llm_config,
    )


@pytest.fixture
def vision_input_tensor(vision_config: VisionConfig) -> torch.Tensor:
    """Create a vision input tensor (patches from image)."""
    torch.manual_seed(42)
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    batch_size = 1
    return torch.randn(
        batch_size,
        num_patches,  # sequence length is number of patches
        vision_config.hidden_size,
        dtype=torch.bfloat16,
    ).to("cuda")


@pytest.fixture
def vision_attention_weights(
    vision_config: VisionConfig,
) -> dict[str, torch.Tensor]:
    """Create attention weights for InternVL vision attention using computed standard deviations.

    These STD values are estimated based on typical vision transformer patterns and
    should be replaced with actual computed values from real InternVL checkpoints
    when available.
    """
    torch.manual_seed(42)

    # Computed from InternVL vision model analysis (estimated values)
    # ==================================================
    # INTERNVL VISION ATTENTION LAYER WEIGHT STANDARD DEVIATIONS
    # ==================================================
    # QKV_PROJ_STD: 0.0289  # For stacked QKV projection
    # O_PROJ_STD: 0.0282    # Output projection
    # Q_NORM_STD: 0.1854    # Q normalization (smaller than text models)
    # K_NORM_STD: 0.1967    # K normalization (smaller than text models)
    # ==================================================

    QKV_PROJ_STD = 0.0289
    O_PROJ_STD = 0.0282
    Q_NORM_STD = 0.1854
    K_NORM_STD = 0.1967

    hidden_size = vision_config.hidden_size

    # Since we use stacked_qkv=True, we need qkv_proj.weight
    qkv_weight = (
        torch.randn(
            3 * hidden_size,  # Q, K, V stacked
            hidden_size,
            dtype=torch.bfloat16,
        )
        * QKV_PROJ_STD
    )

    o_proj_weight = (
        torch.randn(
            hidden_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        * O_PROJ_STD
    )

    # QK normalization weights (RMSNorm uses only weight, no bias)
    q_norm_weight = (
        torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
        )
        * Q_NORM_STD
    )

    k_norm_weight = (
        torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
        )
        * K_NORM_STD
    )

    return {
        "qkv_proj.weight": qkv_weight,
        "o_proj.weight": o_proj_weight,
        "q_norm.weight": q_norm_weight,
        "k_norm.weight": k_norm_weight,
    }


@pytest.fixture
def vision_encoder_layer_weights(
    vision_config: VisionConfig,
) -> dict[str, torch.Tensor]:
    """Create encoder layer weights for InternVisionEncoderLayer using computed standard deviations.

    These STD values are estimated based on typical vision transformer patterns and
    should be replaced with actual computed values from real InternVL checkpoints
    when available.
    """
    torch.manual_seed(42)

    # Computed from InternVL vision model analysis (estimated values)
    # ==================================================
    # INTERNVL VISION ENCODER LAYER WEIGHT STANDARD DEVIATIONS
    # ==================================================
    # QKV_PROJ_STD: 0.0289  # For stacked QKV projection
    # O_PROJ_STD: 0.0282    # Output projection
    # Q_NORM_STD: 0.1854    # Q normalization
    # K_NORM_STD: 0.1967    # K normalization
    # GATE_PROJ_STD: 0.0245 # MLP gate projection
    # DOWN_PROJ_STD: 0.0176 # MLP down projection
    # NORM_STD: 0.2156      # Layer normalization weights
    # LAYER_SCALE_STD: 0.1  # Layer scale parameters (ls1, ls2)
    # ==================================================

    QKV_PROJ_STD = 0.0289
    O_PROJ_STD = 0.0282
    Q_NORM_STD = 0.1854
    K_NORM_STD = 0.1967
    GATE_PROJ_STD = 0.0245
    DOWN_PROJ_STD = 0.0176
    NORM_STD = 0.2156
    LAYER_SCALE_STD = 0.1

    hidden_size = vision_config.hidden_size
    intermediate_size = vision_config.intermediate_size

    # Attention weights (same as vision_attention_weights)
    qkv_weight = (
        torch.randn(
            3 * hidden_size,  # Q, K, V stacked
            hidden_size,
            dtype=torch.bfloat16,
        )
        * QKV_PROJ_STD
    )

    o_proj_weight = (
        torch.randn(
            hidden_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        * O_PROJ_STD
    )

    q_norm_weight = (
        torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
        )
        * Q_NORM_STD
    )

    k_norm_weight = (
        torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
        )
        * K_NORM_STD
    )

    # MLP weights
    gate_proj_weight = (
        torch.randn(
            intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        * GATE_PROJ_STD
    )

    down_proj_weight = (
        torch.randn(
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
        )
        * DOWN_PROJ_STD
    )

    # MLP bias weights
    gate_proj_bias = torch.randn(
        intermediate_size,
        dtype=torch.bfloat16,
    ) * (GATE_PROJ_STD * 0.1)  # Smaller scale for biases

    down_proj_bias = torch.randn(
        hidden_size,
        dtype=torch.bfloat16,
    ) * (DOWN_PROJ_STD * 0.1)  # Smaller scale for biases

    # Layer normalization weights
    norm1_weight = (
        torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
        )
        * NORM_STD
    )

    norm2_weight = (
        torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
        )
        * NORM_STD
    )

    # Layer scale parameters (initialized close to 1.0 with small variation)
    ls1 = (
        torch.ones(hidden_size, dtype=torch.bfloat16)
        + torch.randn(hidden_size, dtype=torch.bfloat16) * LAYER_SCALE_STD
    )
    ls2 = (
        torch.ones(hidden_size, dtype=torch.bfloat16)
        + torch.randn(hidden_size, dtype=torch.bfloat16) * LAYER_SCALE_STD
    )

    return {
        # Attention weights (prefixed with attn.)
        "attn.qkv_proj.weight": qkv_weight,
        "attn.o_proj.weight": o_proj_weight,
        "attn.q_norm.weight": q_norm_weight,
        "attn.k_norm.weight": k_norm_weight,
        # MLP weights
        "mlp.gate_proj.weight": gate_proj_weight,
        "mlp.down_proj.weight": down_proj_weight,
        # MLP bias weights
        "mlp.gate_proj.bias": gate_proj_bias,
        "mlp.down_proj.bias": down_proj_bias,
        # Layer normalization weights
        "norm1.weight": norm1_weight,
        "norm2.weight": norm2_weight,
        # Layer scale parameters
        "ls1": ls1,
        "ls2": ls2,
    }


@pytest.fixture
def vision_pixel_values(vision_config: VisionConfig) -> torch.Tensor:
    """Create pixel values tensor for vision model input (BHWC format)."""
    torch.manual_seed(42)
    batch_size = 1
    channels = 3
    height = vision_config.image_size
    width = vision_config.image_size
    return torch.randn(
        batch_size,
        height,
        width,
        channels,
        dtype=torch.bfloat16,
    ).to("cuda")


@pytest.fixture
def embeddings_weights(vision_config: VisionConfig) -> dict[str, torch.Tensor]:
    """Create embeddings weights for InternVisionEmbeddings.

    These STD values are estimated based on typical vision transformer patterns and
    should be replaced with actual computed values from real InternVL checkpoints
    when available.
    """
    torch.manual_seed(42)

    # Computed from InternVL vision model analysis (estimated values)
    # ==================================================
    # INTERNVL VISION EMBEDDINGS WEIGHT STANDARD DEVIATIONS
    # ==================================================
    # CLS_TOKEN_STD: 0.02  # Class token initialization
    # PATCH_CONV_STD: 0.02  # Patch embedding convolution
    # POS_EMBED_STD: 0.02  # Position embeddings
    # ==================================================

    CLS_TOKEN_STD = 0.02
    PATCH_CONV_STD = 0.02
    POS_EMBED_STD = 0.02

    hidden_size = vision_config.hidden_size

    # Class token embedding
    class_embedding = (
        torch.randn(1, 1, hidden_size, dtype=torch.bfloat16) * CLS_TOKEN_STD
    )

    # Patch embedding convolution weights and bias
    # Note: Conv2D weights in PyTorch are (out_channels, in_channels, kernel_h, kernel_w)
    patch_embedding_weight = (
        torch.randn(
            hidden_size,  # out_channels
            3,  # in_channels (RGB)
            vision_config.patch_size,
            vision_config.patch_size,
            dtype=torch.bfloat16,
        )
        * PATCH_CONV_STD
    )
    patch_embedding_bias = torch.randn(hidden_size, dtype=torch.bfloat16) * (
        PATCH_CONV_STD * 0.1
    )  # Smaller scale for biases

    # Position embeddings
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    num_positions = num_patches + 1  # +1 for class token
    position_embedding = (
        torch.randn(1, num_positions, hidden_size, dtype=torch.bfloat16)
        * POS_EMBED_STD
    )

    return {
        "patch_embedding.weight": patch_embedding_weight,
        "patch_embedding.bias": patch_embedding_bias,
        "class_embedding": class_embedding,
        "position_embedding": position_embedding,
    }


@pytest.fixture
def vision_model_weights(
    vision_config: VisionConfig,
) -> dict[str, torch.Tensor]:
    """Create complete vision model weights including embeddings and all encoder layers.

    These STD values are estimated based on typical vision transformer patterns and
    should be replaced with actual computed values from real InternVL checkpoints
    when available.
    """
    torch.manual_seed(42)

    # Computed from InternVL vision model analysis (estimated values)
    # ==================================================
    # INTERNVL VISION MODEL WEIGHT STANDARD DEVIATIONS
    # ==================================================
    # Embeddings
    CLS_TOKEN_STD = 0.02  # Class token initialization
    PATCH_CONV_STD = 0.02  # Patch embedding convolution
    POS_EMBED_STD = 0.02  # Position embeddings
    # Encoder layers (same as encoder layer weights)
    QKV_PROJ_STD = 0.0289  # For stacked QKV projection
    O_PROJ_STD = 0.0282  # Output projection
    Q_NORM_STD = 0.1854  # Q normalization
    K_NORM_STD = 0.1967  # K normalization
    GATE_PROJ_STD = 0.0245  # MLP gate projection (fc1)
    DOWN_PROJ_STD = 0.0176  # MLP down projection (fc2)
    NORM_STD = 0.2156  # Layer normalization weights
    LAYER_SCALE_STD = 0.1  # Layer scale parameters (ls1, ls2)
    # ==================================================

    hidden_size = vision_config.hidden_size
    intermediate_size = vision_config.intermediate_size
    num_layers = vision_config.num_hidden_layers

    weights = {}

    # ===== EMBEDDINGS WEIGHTS =====
    # Class token embedding
    weights["embeddings.class_embedding"] = (
        torch.randn(1, 1, hidden_size, dtype=torch.bfloat16) * CLS_TOKEN_STD
    )

    # Patch embedding convolution weights and bias
    weights["embeddings.patch_embedding.filter"] = (
        torch.randn(
            hidden_size,
            3,  # RGB channels
            vision_config.patch_size,
            vision_config.patch_size,
            dtype=torch.bfloat16,
        )
        * PATCH_CONV_STD
    )
    weights["embeddings.patch_embedding.bias"] = torch.randn(
        hidden_size, dtype=torch.bfloat16
    ) * (PATCH_CONV_STD * 0.1)

    # Position embeddings
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    num_positions = num_patches + 1  # +1 for class token
    weights["embeddings.position_embedding"] = (
        torch.randn(1, num_positions, hidden_size, dtype=torch.bfloat16)
        * POS_EMBED_STD
    )

    # ===== ENCODER LAYER WEIGHTS =====
    for layer_idx in range(num_layers):
        # Attention weights (same as vision_encoder_layer_weights)
        qkv_weight = (
            torch.randn(
                3 * hidden_size,  # Q, K, V stacked
                hidden_size,
                dtype=torch.bfloat16,
            )
            * QKV_PROJ_STD
        )

        o_proj_weight = (
            torch.randn(
                hidden_size,
                hidden_size,
                dtype=torch.bfloat16,
            )
            * O_PROJ_STD
        )

        q_norm_weight = (
            torch.randn(
                hidden_size,
                dtype=torch.bfloat16,
            )
            * Q_NORM_STD
        )

        k_norm_weight = (
            torch.randn(
                hidden_size,
                dtype=torch.bfloat16,
            )
            * K_NORM_STD
        )

        # MLP weights (fc1/fc2 naming for simple MLP)
        fc1_weight = (
            torch.randn(
                intermediate_size,
                hidden_size,
                dtype=torch.bfloat16,
            )
            * GATE_PROJ_STD
        )

        fc2_weight = (
            torch.randn(
                hidden_size,
                intermediate_size,
                dtype=torch.bfloat16,
            )
            * DOWN_PROJ_STD
        )

        # MLP bias weights
        fc1_bias = torch.randn(
            intermediate_size,
            dtype=torch.bfloat16,
        ) * (GATE_PROJ_STD * 0.1)  # Smaller scale for biases

        fc2_bias = torch.randn(
            hidden_size,
            dtype=torch.bfloat16,
        ) * (DOWN_PROJ_STD * 0.1)  # Smaller scale for biases

        # Layer normalization weights
        norm1_weight = (
            torch.randn(
                hidden_size,
                dtype=torch.bfloat16,
            )
            * NORM_STD
        )

        norm2_weight = (
            torch.randn(
                hidden_size,
                dtype=torch.bfloat16,
            )
            * NORM_STD
        )

        # Layer scale parameters (initialized close to 1.0 with small variation)
        ls1 = (
            torch.ones(hidden_size, dtype=torch.bfloat16)
            + torch.randn(hidden_size, dtype=torch.bfloat16) * LAYER_SCALE_STD
        )
        ls2 = (
            torch.ones(hidden_size, dtype=torch.bfloat16)
            + torch.randn(hidden_size, dtype=torch.bfloat16) * LAYER_SCALE_STD
        )

        # Add all encoder layer weights with proper naming
        layer_prefix = f"encoder_layers.{layer_idx}"
        weights.update(
            {
                # Attention weights (prefixed with attn.)
                f"{layer_prefix}.attn.qkv_proj.weight": qkv_weight,
                f"{layer_prefix}.attn.o_proj.weight": o_proj_weight,
                f"{layer_prefix}.attn.q_norm.weight": q_norm_weight,
                f"{layer_prefix}.attn.k_norm.weight": k_norm_weight,
                # MLP weights (fc1/fc2 naming for simple MLP)
                f"{layer_prefix}.mlp.fc1.weight": fc1_weight,
                f"{layer_prefix}.mlp.fc2.weight": fc2_weight,
                # MLP bias weights
                f"{layer_prefix}.mlp.fc1.bias": fc1_bias,
                f"{layer_prefix}.mlp.fc2.bias": fc2_bias,
                # Layer normalization weights
                f"{layer_prefix}.norm1.weight": norm1_weight,
                f"{layer_prefix}.norm2.weight": norm2_weight,
                # Layer scale parameters
                f"{layer_prefix}.ls1": ls1,
                f"{layer_prefix}.ls2": ls2,
            }
        )

    return weights
