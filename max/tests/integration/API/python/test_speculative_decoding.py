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
from max.interfaces import TextGenerationInputs
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, SupportedEncoding
from max.pipelines.core import TextContext
from max.pipelines.lib.speculative_decoding import (
    SpeculativeDecodingTextGenerationPipeline,
)
from test_common.mocks import mock_estimate_memory_footprint
from test_common.pipeline_model_dummy import DUMMY_ARCH
from test_common.registry import prepare_registry


@pytest.fixture(scope="session")
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
        tokens=tokens1,
        max_length=1024,
    )
    context1.assign_to_cache(1)

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
        tokens=tokens2,
        max_length=1024,
    )
    context2.assign_to_cache(2)

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
    smollm_135m_local_path,  # noqa: ANN001
    llama_3_1_8b_instruct_local_path,  # noqa: ANN001
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_ARCH)

    # Valid device/encoding combinations
    config = PipelineConfig(
        model_path=smollm_135m_local_path,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.cpu()],
        draft_model_path=smollm_135m_local_path,
    )


@pytest.mark.skip(reason="TODO(AITLIB-363): Division by zero error.")
def test_config__validate_target_and_draft_architecture(
    exaone_2_4b_local_path,  # noqa: ANN001
    smollm_135m_local_path,  # noqa: ANN001
    deepseek_r1_distill_llama_8b_local_path,  # noqa: ANN001
) -> None:
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


@pytest.mark.skip("TODO: Re-enable Speculative Decoding Tests")
def test_speculative_decoding_no_rejection(
    setup_speculative_decoding_pipeline,  # noqa: ANN001
) -> None:
    data: dict[str, Any] = setup_speculative_decoding_pipeline
    pipeline: SpeculativeDecodingTextGenerationPipeline = data["pipeline"]
    context_batch: list[TextContext] = data["context_batch"]
    context1: TextContext = data["context1"]
    context2: TextContext = data["context2"]
    num_steps: int = data["num_steps"]

    assert context1.start_idx == 0
    assert context2.start_idx == 0

    # Generate draft tokens.
    draft_inputs, draft_num_steps = pipeline.prepare_batch(
        pipeline._draft_model,
        context_batch,
        num_steps,
        return_n_logits=1,
        is_draft=True,
    )

    num_steps, draft_tokens, draft_logits, model_inputs, all_draft_logits = (
        pipeline.generate_draft_tokens(
            context_batch, draft_num_steps, draft_inputs
        )
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
    first_rejected_tokens, recovered_tokens, bonus_tokens = (
        pipeline.verify_draft_tokens_with_target_model(
            draft_inputs,
            context_batch,
            num_steps,
            draft_tokens,
            draft_logits,
            merged_tokens,
            merged_offsets,
            all_draft_logits,
        )
    )

    # If the draft and target models are the same then no tokens are rejected.
    assert np.all(first_rejected_tokens.to_numpy() == num_steps)

    pipeline.update_contexts(
        context_batch=context_batch,
        first_rejected_tokens=first_rejected_tokens.to_numpy(),
        recovered_tokens=recovered_tokens.to_numpy(),
        bonus_tokens=bonus_tokens.to_numpy(),
        draft_tokens=draft_tokens.to_numpy(),
        num_draft_tokens_generated=num_steps,
    )

    context1, context2 = context_batch

    # subtract 1 because all draft tokens are accepted, next draft input includes the token generated from the target model
    assert context1.start_idx == (len(context1.prompt_tokens) + num_steps - 1)
    assert context2.start_idx == (len(context2.prompt_tokens) + num_steps - 1)

    assert np.all(context1.generated_tokens[:-1] == draft_tokens.to_numpy()[0])
    assert np.all(context2.generated_tokens[:-1] == draft_tokens.to_numpy()[1])


@pytest.mark.skip("TODO: E2EOPT-403 Re-enable Speculative Decoding Tests")
def test_speculative_decoding_partial_rejection(
    setup_speculative_decoding_pipeline,  # noqa: ANN001
) -> None:
    data: dict[str, Any] = setup_speculative_decoding_pipeline
    pipeline: SpeculativeDecodingTextGenerationPipeline = data["pipeline"]
    context_batch: list[TextContext] = data["context_batch"]
    context1: TextContext = data["context1"]
    context2: TextContext = data["context2"]
    num_steps: int = data["num_steps"]

    assert context1.start_idx == 0
    assert context2.start_idx == 0

    # Generate draft tokens.
    draft_inputs, draft_num_steps = pipeline.prepare_batch(
        pipeline._draft_model,
        context_batch,
        num_steps,
        return_n_logits=1,
        is_draft=True,
    )
    num_steps, draft_tokens, draft_logits, model_inputs, all_draft_logits = (
        pipeline.generate_draft_tokens(context_batch, num_steps, draft_inputs)
    )

    # Merge draft tokens with target tokens
    merged_tokens, merged_offsets = pipeline._ragged_token_merger(
        model_inputs.tokens,  # type: ignore
        model_inputs.input_row_offsets,  # type: ignore
        draft_tokens,
    )

    # For the first sequence we'll manually change the tokens and logits so that only part of that sequence is accepted

    draft_logits_host = np.copy(draft_logits.to_numpy())
    draft_logits_host[0, num_steps // 2 :] = 10000.0
    draft_logits = Tensor.from_numpy(draft_logits_host).to(draft_logits.device)

    # Permute to [batch, num_steps, vocab] and set large logit values for half the tokens in the first batch.
    # Then permute back to the expected shape
    all_draft_logits_host = np.permute_dims(
        np.copy(all_draft_logits.to_numpy()), [1, 0, 2]
    )
    batch_size, steps, _ = all_draft_logits_host.shape
    batch_indices = np.arange(batch_size - 1)[:, np.newaxis]
    step_start = steps // 2
    step_indices = np.arange(step_start, steps)[np.newaxis, :]
    token_values = draft_tokens.to_numpy()[:, step_start:]
    all_draft_logits_host[batch_indices, step_indices, token_values] = 10000.0
    all_draft_logits = Tensor.from_numpy(
        np.permute_dims(all_draft_logits_host, [1, 0, 2])
    ).to(all_draft_logits.device)

    assert isinstance(merged_tokens, Tensor)
    assert isinstance(merged_offsets, Tensor)
    # Verify draft tokens with target model
    first_rejected_tokens, recovered_tokens, bonus_tokens = (
        pipeline.verify_draft_tokens_with_target_model(
            draft_inputs,
            context_batch,
            num_steps,
            draft_tokens,
            draft_logits,
            merged_tokens,
            merged_offsets,
            all_draft_logits,
        )
    )
    first_rejected_tokens_host = first_rejected_tokens.to_numpy()
    assert first_rejected_tokens_host[0] == num_steps // 2
    assert first_rejected_tokens_host[1] == num_steps

    draft_tokens_host = draft_tokens.to_numpy()

    pipeline.update_contexts(
        context_batch=context_batch,
        first_rejected_tokens=first_rejected_tokens_host,
        recovered_tokens=recovered_tokens.to_numpy(),
        bonus_tokens=bonus_tokens.to_numpy(),
        draft_tokens=draft_tokens_host,
        num_draft_tokens_generated=num_steps,
    )

    context1, context2 = context_batch

    # subtract 1 because recovered token has not been processed by either model
    assert context1.start_idx == (
        len(context1.prompt_tokens) + (num_steps // 2) - 1
    )
    # subtract 1 because all draft tokens are accepted, next draft input includes the token generated from the target model
    assert context2.start_idx == (len(context2.prompt_tokens) + num_steps - 1)

    assert np.all(
        context1.generated_tokens[:-1] == draft_tokens_host[0, : num_steps // 2]
    )
    assert np.all(context2.generated_tokens[:-1] == draft_tokens_host[1])


@pytest.mark.skip("TODO: E2EOPT-403 Re-enable Speculative Decoding Tests")
def test_speculative_decoding_multiple_token_without_rejection(
    setup_speculative_decoding_pipeline,  # noqa: ANN001
) -> None:
    data: dict[str, Any] = setup_speculative_decoding_pipeline
    pipeline: SpeculativeDecodingTextGenerationPipeline = data["pipeline"]
    context1: TextContext = data["context1"]
    context2: TextContext = data["context2"]
    pipeline_request: dict[str, TextContext] = data["pipeline_request"]
    num_steps: int = data["num_steps"]

    context1_len = context1.current_length
    context2_len = context2.current_length
    for _ in range(5):
        inputs = TextGenerationInputs(
            batch=pipeline_request, num_steps=num_steps
        )
        pipeline.next_token(inputs)

        # num_steps generated from draft and +1 from the target
        assert context1.current_length == context1_len + (num_steps + 1)
        assert context2.current_length == context2_len + (num_steps + 1)

        context1_len = context1.current_length
        context2_len = context2.current_length


@pytest.mark.skip("TODO: E2EOPT-403 Re-enable Speculative Decoding Tests")
def test_speculative_decoding_context_update(
    setup_speculative_decoding_pipeline,  # noqa: ANN001
) -> None:
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

    recovered_tokens = np.array(
        [
            [88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
            [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        ],
        dtype=np.int32,
    )
    bonus_tokens = np.array(
        [
            [1001],
            [2001],
        ],
        dtype=np.int32,
    )

    reject_token1_idx = 4
    reject_token2_idx = num_steps
    first_rejected_tokens = np.array(
        [[reject_token1_idx], [reject_token2_idx]], dtype=np.int32
    )

    # The index bump hack from generate_draft_tokens()
    for i, context in enumerate(context_batch):  # noqa: B007
        context.bump_token_indices(active_idx=num_steps, end_idx=num_steps)

    pipeline.update_contexts(
        context_batch,
        first_rejected_tokens,
        recovered_tokens,
        bonus_tokens,
        draft_tokens,
        num_draft_tokens_generated=num_steps,
    )

    # length of prompt + length of non rejected draft tokens + 1 for the new token
    assert (
        context1.current_length
        == len(context1.prompt_tokens) + reject_token1_idx + 1
    )
    assert (
        context2.current_length
        == len(context2.prompt_tokens) + reject_token2_idx + 1
    )

    assert context1._draft_offset == 0
    assert context2._draft_offset == 0

    # subtract 1 because the recovered token has not been processed by the draft
    # or dtarget model
    assert (
        context1.start_idx
        == len(context1.prompt_tokens) + reject_token1_idx - 1
    )

    # subtract 1 because the bonus token has not been run through the draft model
    assert (
        context2.start_idx
        == len(context2.prompt_tokens) + reject_token2_idx - 1
    )

    assert np.all(
        context1.all_tokens
        == np.concatenate(
            (
                context1.prompt_tokens,
                draft_tokens[0, :reject_token1_idx],
                recovered_tokens[0, reject_token1_idx][np.newaxis],
            )
        )
    )

    assert np.all(
        context2.all_tokens
        == np.concatenate(
            (
                context2.prompt_tokens,
                draft_tokens[1, :reject_token2_idx],
                bonus_tokens[1],
            )
        )
    )

    response = pipeline.build_response(
        pipeline_request, list(pipeline_request.values())
    )
    assert len(response) == 2
    assert len(response[req_id1].tokens) == reject_token1_idx + 1
    assert len(response[req_id2].tokens) == reject_token2_idx + 1
    response_tokens1 = response[req_id1].tokens
    response_tokens2 = response[req_id2].tokens

    assert np.all(
        response_tokens1
        == np.concatenate(
            (
                draft_tokens[0, :reject_token1_idx],
                recovered_tokens[0, reject_token1_idx][np.newaxis],
            )
        )
    )
    assert np.all(
        response_tokens2
        == np.concatenate(
            (draft_tokens[1, :reject_token2_idx], bonus_tokens[1])
        )
    )
