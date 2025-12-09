# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from unittest.mock import MagicMock

from max.pipelines.architectures.deepseekV3 import deepseekV3_arch
from max.pipelines.lib import PipelineRole, SupportedEncoding

MAX_SEND_TOKENS_PER_RANK = 128
NUM_RANKS = 8


def mock_pipeline_config(pipeline_role: PipelineRole) -> MagicMock:
    pipeline_config = MagicMock()

    # Model config attributes
    pipeline_config.model_config = MagicMock()
    pipeline_config.model_config.quantization_encoding = (
        SupportedEncoding.float8_e4m3fn
    )
    pipeline_config.model_config.data_parallel_degree = NUM_RANKS
    pipeline_config.model_config.device_specs = [
        MagicMock() for _ in range(NUM_RANKS)
    ]

    # Pipeline config attributes
    pipeline_config.pipeline_role = pipeline_role
    pipeline_config.max_length = 1024 * 1024  # ~million tokens
    pipeline_config.max_batch_context_length = None
    pipeline_config.ep_size = NUM_RANKS
    pipeline_config.prefill_chunk_size = MAX_SEND_TOKENS_PER_RANK

    return pipeline_config


def mock_huggingface_config() -> MagicMock:
    huggingface_config = MagicMock()

    # HuggingFace config attributes
    huggingface_config.num_attention_heads = 128
    huggingface_config.qk_nope_head_dim = 128
    huggingface_config.n_routed_experts = 256
    huggingface_config.moe_intermediate_size = 2048
    huggingface_config.hidden_size = 7168

    return huggingface_config


def test_deepseekv3_memory_estimation() -> None:
    deepseek_model = deepseekV3_arch.pipeline_model
    pipeline_config = mock_pipeline_config(PipelineRole.DecodeOnly)
    huggingface_config = mock_huggingface_config()

    memory_estimated = deepseek_model.estimate_activation_memory(
        pipeline_config, huggingface_config
    )

    max_recv_tokens_per_rank = (
        MAX_SEND_TOKENS_PER_RANK * huggingface_config.n_routed_experts
    )
    moe_min_memory = (
        max_recv_tokens_per_rank * huggingface_config.moe_intermediate_size * 1
    )  # Float8
    moe_min_memory += (
        max_recv_tokens_per_rank * huggingface_config.hidden_size * 2
    )  # BFloat16
    moe_min_memory *= NUM_RANKS

    assert memory_estimated > moe_min_memory


def test_deepseekv3_memory_estimation_exact() -> None:
    deepseek_model = deepseekV3_arch.pipeline_model
    huggingface_config = mock_huggingface_config()

    # For DecodeOnly, we only need to consider moe_activation_memory
    pipeline_config = mock_pipeline_config(PipelineRole.DecodeOnly)
    mem = deepseek_model.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 6442450944

    # For PrefillAndDecode, we also need to consider mla_activation_memory
    pipeline_config = mock_pipeline_config(PipelineRole.PrefillAndDecode)
    mem = deepseek_model.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 549755813888
