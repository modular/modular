# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max.driver import DeviceSpec, Tensor
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineEngine,
    SupportedEncoding,
)
from max.pipelines.core import TextContext
from max.pipelines.lib.speculative_decoding import (
    SpeculativeDecodingTextGenerationPipeline,
)
from test_common.mocks import mock_estimate_memory_footprint
from test_common.pipeline_model_dummy import DUMMY_ARCH
from test_common.registry import prepare_registry


@pytest.fixture
def setup_speculative_decoding_pipeline(num_steps: int = 10):
    """Fixture to set up a speculative decoding pipeline with common configuration."""
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    pipeline_config = PipelineConfig(
        model_path=model_name,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.accelerator()],
        draft_model_path=model_name,
        max_batch_size=4,
        max_num_steps=num_steps,
        max_length=1024,
    )
    pipeline_config.model_config.kv_cache_config.cache_strategy = (
        KVCacheStrategy.PAGED
    )
    pipeline_config.model_config.kv_cache_config.kv_cache_page_size = 128
    pipeline_config.model_config.kv_cache_config.device_memory_utilization = 0.3

    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, SpeculativeDecodingTextGenerationPipeline)

    # Create contexts for two test prompts
    req_id1 = "1"
    tokens1 = np.array([1, 450, 6593, 310, 2834, 338], dtype=np.int64)

    context1 = TextContext(
        cache_seq_id=1,
        prompt=tokens1.tolist(),
        tokens=tokens1,
        max_length=1024,
    )

    req_id2 = "2"
    tokens2 = np.array(
        [
            1,
            11644,
            2113,
            278,
            3186,
            3652,
            297,
            29871,
            29906,
            29900,
            29906,
            29900,
        ],
        dtype=np.int64,
    )
    context2 = TextContext(
        cache_seq_id=2,
        prompt=tokens2.tolist(),
        tokens=tokens2,
        max_length=1024,
    )

    pipeline_request = {req_id1: context1, req_id2: context2}
    context_batch = [context1, context2]

    return {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "pipeline": pipeline,
        "context1": context1,
        "context2": context2,
        "req_id1": req_id1,
        "req_id2": req_id2,
        "pipeline_request": pipeline_request,
        "context_batch": context_batch,
        "num_steps": num_steps,
    }


@prepare_registry
@mock_estimate_memory_footprint
@pytest.mark.skip(
    reason="TODO(AITLIB-339): This test is flaky due to bad huggingface cache hydration"
)
def test_config__validate_device_and_encoding_combinations(
    smollm_135m_local_path,
    llama_3_1_8b_instruct_local_path,
):
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations
    config = PipelineConfig(
        model_path=smollm_135m_local_path,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.cpu()],
        draft_model_path=smollm_135m_local_path,
    )

    with pytest.raises(ValueError):
        # Invalid device/encoding combinations
        config = PipelineConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding=SupportedEncoding.float32,
            device_specs=[DeviceSpec.cpu()],
            draft_model_path=smollm_135m_local_path,
            engine=PipelineEngine.HUGGINGFACE,
        )


@pytest.mark.skip(reason="TODO(AITLIB-363): Division by zero error.")
def test_config__validate_target_and_draft_architecture(
    exaone_2_4b_local_path,
    smollm_135m_local_path,
    deepseek_r1_distill_llama_8b_local_path,
):
    with pytest.raises(ValueError):
        # Test that when the target & draft architectures are different
        # we raise an error.
        config = PipelineConfig(
            model_path=exaone_2_4b_local_path,
            quantization_encoding=SupportedEncoding.q4_k,
            device_specs=[DeviceSpec.cpu()],
            draft_model_path=smollm_135m_local_path,
        )

    with pytest.raises(ValueError):
        # Test that the target & draft architectures are the same,
        # but the tokenizers are different
        config = PipelineConfig(
            model_path=deepseek_r1_distill_llama_8b_local_path,
            quantization_encoding=SupportedEncoding.q4_k,
            weight_path=[
                Path(
                    "lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
                )
            ],
            draft_model_path=smollm_135m_local_path,
        )


def test_config__validate_huggingface_engine(llama_3_1_8b_instruct_local_path):
    """Test that speculative decoding is not supported with HuggingFace engine."""
    with pytest.raises(
        ValueError,
        match="Speculative Decoding not supported with the HuggingFace Engine",
    ):
        PipelineConfig(
            model_path=llama_3_1_8b_instruct_local_path,
            quantization_encoding=SupportedEncoding.bfloat16,
            device_specs=[DeviceSpec.accelerator()],
            draft_model_path=llama_3_1_8b_instruct_local_path,
            engine=PipelineEngine.HUGGINGFACE,
        )


def test_speculative_decoding_no_rejection(
    setup_speculative_decoding_pipeline,
):
    data: dict[str, Any] = setup_speculative_decoding_pipeline
    pipeline: SpeculativeDecodingTextGenerationPipeline = data["pipeline"]
    context_batch: list[TextContext] = data["context_batch"]
    context1: TextContext = data["context1"]
    context2: TextContext = data["context2"]
    num_steps: int = data["num_steps"]

    assert context1.start_idx == 0
    assert context2.start_idx == 0

    num_steps, draft_tokens, draft_logits, model_inputs = (
        pipeline.generate_draft_tokens(context_batch, num_steps)
    )

    # Merge draft tokens with target tokens
    merged_tokens, merged_offsets = pipeline._ragged_token_merger(
        model_inputs.tokens,  # type: ignore
        model_inputs.input_row_offsets,  # type: ignore
        draft_tokens,
    )

    assert isinstance(merged_tokens, Tensor)
    assert isinstance(merged_offsets, Tensor)
    # Verify draft tokens with target model
    first_rejected_tokens, sampled_target_tokens = (
        pipeline.verify_draft_tokens_with_target_model(
            context_batch,
            num_steps,
            draft_tokens,
            draft_logits,
            merged_tokens,
            merged_offsets,
        )
    )

    # If the draft and target models are the same then no tokens are rejected.
    assert np.all(first_rejected_tokens.to_numpy() == num_steps)

    pipeline.update_contexts(
        context_batch,
        first_rejected_tokens.to_numpy(),
        sampled_target_tokens.to_numpy(),
        draft_tokens.to_numpy(),
        num_steps,
    )

    context1, context2 = context_batch

    # subtract 1 because all draft tokens are accepted, next draft input includes the token generated from the target model
    assert context1.start_idx == (len(context1.prompt_tokens) + num_steps - 1)
    assert context2.start_idx == (len(context2.prompt_tokens) + num_steps - 1)

    assert np.all(context1.generated_tokens[:-1] == draft_tokens.to_numpy()[0])
    assert np.all(context2.generated_tokens[:-1] == draft_tokens.to_numpy()[1])


def test_speculative_decoding_partial_rejection(
    setup_speculative_decoding_pipeline,
):
    data: dict[str, Any] = setup_speculative_decoding_pipeline
    pipeline: SpeculativeDecodingTextGenerationPipeline = data["pipeline"]
    context_batch: list[TextContext] = data["context_batch"]
    context1: TextContext = data["context1"]
    context2: TextContext = data["context2"]
    num_steps: int = data["num_steps"]

    assert context1.start_idx == 0
    assert context2.start_idx == 0

    num_steps, draft_tokens, draft_logits, model_inputs = (
        pipeline.generate_draft_tokens(context_batch, num_steps)
    )

    # Merge draft tokens with target tokens
    merged_tokens, merged_offsets = pipeline._ragged_token_merger(
        model_inputs.tokens,  # type: ignore
        model_inputs.input_row_offsets,  # type: ignore
        draft_tokens,
    )

    # For the first sequence we'll manually change the tokens and logits so that only part of that sequence is accepted

    draft_logits_host = np.copy(draft_logits.to_numpy())
    draft_logits_host[0, num_steps // 2 :] = float("inf")
    draft_logits = Tensor.from_numpy(draft_logits_host).to(draft_logits.device)

    assert isinstance(merged_tokens, Tensor)
    assert isinstance(merged_offsets, Tensor)
    # Verify draft tokens with target model
    first_rejected_tokens, sampled_target_tokens = (
        pipeline.verify_draft_tokens_with_target_model(
            context_batch,
            num_steps,
            draft_tokens,
            draft_logits,
            merged_tokens,
            merged_offsets,
        )
    )
    first_rejected_tokens_host = first_rejected_tokens.to_numpy()
    assert first_rejected_tokens_host[0] == num_steps // 2
    assert first_rejected_tokens_host[1] == num_steps

    draft_tokens_host = draft_tokens.to_numpy()

    pipeline.update_contexts(
        context_batch,
        first_rejected_tokens_host,
        sampled_target_tokens.to_numpy(),
        draft_tokens_host,
        num_steps,
    )

    context1, context2 = context_batch

    assert context1.start_idx == (len(context1.prompt_tokens) + num_steps // 2)
    # subtract 1 because all draft tokens are accepted, next draft input includes the token generated from the target model
    assert context2.start_idx == (len(context2.prompt_tokens) + num_steps - 1)

    assert np.all(
        context1.generated_tokens[:-1] == draft_tokens_host[0, : num_steps // 2]
    )
    assert np.all(context2.generated_tokens[:-1] == draft_tokens_host[1])


def test_speculative_decoding_multiple_token_without_rejection(
    setup_speculative_decoding_pipeline,
):
    data: dict[str, Any] = setup_speculative_decoding_pipeline
    pipeline: SpeculativeDecodingTextGenerationPipeline = data["pipeline"]
    context1: TextContext = data["context1"]
    context2: TextContext = data["context2"]
    pipeline_request: dict[str, TextContext] = data["pipeline_request"]
    num_steps: int = data["num_steps"]

    context1_len = context1.current_length
    context2_len = context2.current_length
    for _ in range(5):
        pipeline.next_token(pipeline_request, num_steps)

        # num_steps generated from draft and +1 from the target
        assert context1.current_length == context1_len + (num_steps + 1)
        assert context2.current_length == context2_len + (num_steps + 1)

        context1_len = context1.current_length
        context2_len = context2.current_length


def test_speculative_decoding_context_update(
    setup_speculative_decoding_pipeline,
):
    data: dict[str, Any] = setup_speculative_decoding_pipeline
    pipeline: SpeculativeDecodingTextGenerationPipeline = data["pipeline"]
    context1: TextContext = data["context1"]
    context2: TextContext = data["context2"]
    req_id1: str = data["req_id1"]
    req_id2: str = data["req_id2"]
    pipeline_request: dict[str, TextContext] = data["pipeline_request"]
    context_batch: list[TextContext] = data["context_batch"]
    num_steps: int = data["num_steps"]

    draft_tokens = np.array(
        [
            [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
        ]
    )
    assert draft_tokens.shape == (2, num_steps)

    reject_token1 = 4
    reject_token2 = 7
    first_rejected_tokens = np.array(
        [[reject_token1], [reject_token2]], dtype=np.int32
    )
    sampled_target_tokens = np.array([[88], [99]], dtype=np.int32)

    # The index bump hack from generate_draft_tokens()
    for i, context in enumerate(context_batch):
        context.bump_token_indices(active_idx=num_steps, end_idx=num_steps)

    pipeline.update_contexts(
        context_batch,
        first_rejected_tokens,
        sampled_target_tokens,
        draft_tokens,
        num_steps,
    )

    # length of prompt + length of non rejected draft tokens + 1 for the new token
    assert (
        context1.current_length
        == len(context1.prompt_tokens) + reject_token1 + 1
    )
    assert (
        context2.current_length
        == len(context2.prompt_tokens) + reject_token2 + 1
    )

    assert context1._draft_offset == 0
    assert context2._draft_offset == 0

    assert context1.start_idx == len(context1.prompt_tokens) + reject_token1
    assert context2.start_idx == len(context2.prompt_tokens) + reject_token2

    assert np.all(
        context1.tokens
        == np.concatenate(
            (
                context1.prompt_tokens,
                draft_tokens[0, :reject_token1],
                sampled_target_tokens[0],
            )
        )
    )

    assert np.all(
        context2.tokens
        == np.concatenate(
            (
                context2.prompt_tokens,
                draft_tokens[1, :reject_token2],
                sampled_target_tokens[1],
            )
        )
    )

    response = pipeline.build_response(
        pipeline_request, list(pipeline_request.values())
    )
    assert len(response) == 2
    assert len(response[req_id1].tokens) == reject_token1 + 1
    assert len(response[req_id2].tokens) == reject_token2 + 1
    response_tokens1 = np.array(
        [t.next_token for t in response[req_id1].tokens]
    )
    response_tokens2 = np.array(
        [t.next_token for t in response[req_id2].tokens]
    )

    assert np.all(
        response_tokens1
        == np.concatenate(
            (draft_tokens[0, :reject_token1], sampled_target_tokens[0])
        )
    )
    assert np.all(
        response_tokens2
        == np.concatenate(
            (draft_tokens[1, :reject_token2], sampled_target_tokens[1])
        )
    )
