# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core pipeline parallel algorithm tests + KV cache construction validation."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import ReturnLogits
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from max.pipelines.architectures.llama3.model import PipelineStageKVParams
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.llama3.pipeline_parallel_llama3 import (
    PipelineParallelLlama3,
)
from transformers.models.llama.configuration_llama import LlamaConfig


@pytest.fixture
def mock_hf_config():
    """Create a minimal HuggingFace config for testing."""
    return LlamaConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=32,
        intermediate_size=11008,
        vocab_size=32000,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=2048,
    )


@pytest.fixture
def mock_llama3_config(mock_hf_config: LlamaConfig) -> Llama3Config:
    """Create a mock Llama3Config for testing."""
    devices = [DeviceRef("gpu", 0), DeviceRef("gpu", 1)]

    return Llama3Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=32,
        rope_theta=10000.0,
        rope_scaling_params=None,
        max_seq_len=2048,
        intermediate_size=11008,
        interleaved_rope_weights=False,
        vocab_size=32000,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=32,
            head_dim=128,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            page_size=16,
            n_devices=1,
            pipeline_parallel_degree=2,
            total_num_layers=32,
        ),
        return_logits=ReturnLogits.LAST_TOKEN,
        norm_method="rms_norm",
        norm_dtype=None,
        attention_bias=False,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        stacked_mlp=False,
        stacked_qkv=False,
        logits_postprocessor=None,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=devices,
        clip_qkv=None,
        float8_config=None,
        pipeline_parallel_degree=2,
        tensor_parallel_degree=1,
    )


# =============================================================================
# Essential Algorithm Tests
# =============================================================================


def test_compute_stage_assignments() -> None:
    """Test stage assignment algorithm - CRITICAL for PP to work."""
    # Test even split
    assignments = PipelineParallelLlama3._compute_stage_assignments(32, 2)
    expected = [(0, 16), (16, 32)]
    assert assignments == expected

    # Test uneven split
    assignments = PipelineParallelLlama3._compute_stage_assignments(33, 2)
    expected = [(0, 17), (17, 33)]
    assert assignments == expected

    # Test edge case: more stages than layers
    assignments = PipelineParallelLlama3._compute_stage_assignments(2, 4)
    expected = [(0, 1), (1, 2)]
    assert assignments == expected

    # Test single stage
    assignments = PipelineParallelLlama3._compute_stage_assignments(32, 1)
    expected = [(0, 32)]
    assert assignments == expected


def test_get_stage_for_layer(mock_llama3_config: Llama3Config):
    """Test O(1) layer-to-stage lookup - CRITICAL for PP to work."""
    model = PipelineParallelLlama3(mock_llama3_config)

    # Test layer-to-stage mapping
    assert model._get_stage_for_layer(0, 32) == 0  # First layer
    assert model._get_stage_for_layer(15, 32) == 0  # Last layer of stage 0
    assert model._get_stage_for_layer(16, 32) == 1  # First layer of stage 1
    assert model._get_stage_for_layer(31, 32) == 1  # Last layer


# =============================================================================
# Essential KV Cache Construction Tests
# =============================================================================


def test_continuous_kv_cache_construction(mock_llama3_config: Llama3Config):
    """Test KV cache collections created correctly for continuous batching strategy."""
    # Create stage parameters for continuous batching
    base_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=32,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
        page_size=16,
        n_devices=1,
        pipeline_parallel_degree=2,
        total_num_layers=32,
    )

    stage_params = PipelineStageKVParams(
        stage_id=0,
        start_layer=0,
        end_layer=16,
        num_layers_in_stage=16,
        base_params=base_params,
        device=DeviceRef("gpu", 0),
    )

    # Test stage-specific KV params creation
    stage_kv_params = stage_params.create_stage_params()
    assert stage_kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS
    assert stage_kv_params.n_devices == 1  # Single device per stage
    assert stage_kv_params.stage_id == 0
    assert stage_kv_params.total_num_layers == 32

    # Test continuous batching collection creation
    kv_collection = FetchContinuousBatchingKVCacheCollection(
        stage_kv_params, num_layers=16
    )
    assert kv_collection is not None


def test_paged_kv_cache_construction(mock_llama3_config: Llama3Config):
    """Test KV cache collections created correctly for paged strategy."""
    # Create stage parameters for paged strategy
    base_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=32,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=16,
        n_devices=1,
        pipeline_parallel_degree=2,
        total_num_layers=32,
    )

    stage_params = PipelineStageKVParams(
        stage_id=1,
        start_layer=16,
        end_layer=32,
        num_layers_in_stage=16,
        base_params=base_params,
        device=DeviceRef("gpu", 1),
    )

    # Test stage-specific KV params creation
    stage_kv_params = stage_params.create_stage_params()
    assert stage_kv_params.cache_strategy == KVCacheStrategy.PAGED
    assert stage_kv_params.n_devices == 1  # Single device per stage
    assert stage_kv_params.stage_id == 1
    assert stage_kv_params.total_num_layers == 32

    # Test paged collection creation
    kv_collection = FetchPagedKVCacheCollection(stage_kv_params, num_layers=16)
    assert kv_collection is not None


def test_stage_kv_params_creation(mock_llama3_config: Llama3Config):
    """Test stage-specific KV parameters are created correctly for both strategies."""
    strategies = [KVCacheStrategy.CONTINUOUS, KVCacheStrategy.PAGED]

    for strategy in strategies:
        base_params = KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=32,
            head_dim=128,
            cache_strategy=strategy,
            page_size=16,
            n_devices=1,
            pipeline_parallel_degree=2,
            total_num_layers=32,
        )

        # Test stage 0 parameters
        stage_0_params = PipelineStageKVParams(
            stage_id=0,
            start_layer=0,
            end_layer=16,
            num_layers_in_stage=16,
            base_params=base_params,
            device=DeviceRef("gpu", 0),
        )

        stage_0_kv_params = stage_0_params.create_stage_params()

        # Verify stage-specific parameters
        assert stage_0_kv_params.cache_strategy == strategy
        assert stage_0_kv_params.stage_id == 0
        assert stage_0_kv_params.n_devices == 1
        assert stage_0_kv_params.pipeline_parallel_degree == 2
        assert stage_0_kv_params.total_num_layers == 32

        # Verify inheritance from base params
        assert stage_0_kv_params.dtype == base_params.dtype
        assert stage_0_kv_params.n_kv_heads == base_params.n_kv_heads
        assert stage_0_kv_params.head_dim == base_params.head_dim


# =============================================================================
# Essential Model Creation Tests
# =============================================================================


def test_model_creation_continuous_strategy(mock_llama3_config: Llama3Config):
    """Test PP model creation with continuous batching KV cache."""
    # Set continuous batching strategy
    mock_llama3_config.kv_params.cache_strategy = KVCacheStrategy.CONTINUOUS

    # Test model creation
    model = PipelineParallelLlama3(mock_llama3_config)

    # Verify model properties
    assert model.pp_degree == 2
    assert len(model.stage_assignments) == 2
    assert model.stage_assignments == [(0, 16), (16, 32)]

    # Verify stage assignments are computed correctly
    assert model._get_stage_for_layer(0, 32) == 0
    assert model._get_stage_for_layer(16, 32) == 1


def test_model_creation_paged_strategy(mock_llama3_config: Llama3Config):
    """Test PP model creation with paged KV cache."""
    # Set paged strategy
    mock_llama3_config.kv_params.cache_strategy = KVCacheStrategy.PAGED

    # Test model creation
    model = PipelineParallelLlama3(mock_llama3_config)

    # Verify model properties
    assert model.pp_degree == 2
    assert len(model.stage_assignments) == 2
    assert model.stage_assignments == [(0, 16), (16, 32)]

    # Verify stage assignments are computed correctly
    assert model._get_stage_for_layer(0, 32) == 0
    assert model._get_stage_for_layer(16, 32) == 1


def test_config_validation(mock_llama3_config: Llama3Config):
    """Test PP config validation prevents invalid configurations."""
    # Test valid configuration
    mock_llama3_config.pipeline_parallel_degree = 2
    mock_llama3_config.tensor_parallel_degree = 1
    model = PipelineParallelLlama3(mock_llama3_config)  # Should not raise
    assert model.pp_degree == 2

    # Test pipeline parallel degree must be > 1 - should raise ValueError
    mock_llama3_config.pipeline_parallel_degree = 1
    with pytest.raises(
        ValueError, match="Pipeline parallel degree must be > 1"
    ):
        PipelineParallelLlama3(mock_llama3_config)
