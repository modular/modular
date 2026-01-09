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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from max.driver import DeviceSpec
from max.interfaces import (
    PipelineTokenizer,
    RequestID,
    SamplingParams,
    TokenBuffer,
)
from max.nn.kv_cache import KVCacheStrategy, RaggedKVCacheInputs
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, SupportedEncoding
from max.pipelines.core import TextContext
from max.pipelines.lib.speculative_config import (
    SpeculativeMethod,
)
from max.pipelines.lib.speculative_decoding import (
    StandaloneSpeculativeDecodingPipeline,
)
from test_common.pipeline_model_dummy import DUMMY_GEMMA_ARCH, DUMMY_LLAMA_ARCH
from test_common.registry import prepare_registry


@dataclass
class SpeculativeDecodingSetup:
    model_name: str
    tokenizer: PipelineTokenizer  # type: ignore
    pipeline: StandaloneSpeculativeDecodingPipeline
    context1: TextContext
    context2: TextContext
    req_id1: RequestID
    req_id2: RequestID
    pipeline_request: dict[RequestID, TextContext]
    context_batch: list[TextContext]
    num_steps: int


@pytest.fixture(scope="function")
def setup_speculative_decoding_pipeline(num_steps: int = 10):  # noqa: ANN201
    """Fixture to set up a speculative decoding pipeline with common configuration."""
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    pipeline_config = PipelineConfig(
        model_path=model_name,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.accelerator()],
        draft_model_path=model_name,
        speculative_method=SpeculativeMethod.STANDALONE,
        num_speculative_tokens=10,
        max_batch_size=4,
        max_num_steps=num_steps,
        max_length=1024,
    )
    pipeline_config.model.kv_cache_config.cache_strategy = KVCacheStrategy.PAGED
    pipeline_config.model.kv_cache_config.kv_cache_page_size = 128
    pipeline_config.model.kv_cache_config.device_memory_utilization = 0.3

    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, StandaloneSpeculativeDecodingPipeline)

    # Create contexts for two test prompts
    req_id1 = RequestID()
    tokens1 = np.array([1, 450, 6593, 310, 2834, 338], dtype=np.int64)

    context1 = TextContext(
        request_id=req_id1,
        tokens=TokenBuffer(tokens1),
        max_length=1024,
        sampling_params=SamplingParams(top_k=1),
    )

    req_id2 = RequestID()
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
        request_id=req_id2,
        tokens=TokenBuffer(tokens2),
        max_length=1024,
        sampling_params=SamplingParams(top_k=1),
    )
    pipeline_request = {req_id1: context1, req_id2: context2}
    context_batch = [context1, context2]

    return SpeculativeDecodingSetup(
        model_name=model_name,
        tokenizer=tokenizer,
        pipeline=pipeline,
        context1=context1,
        context2=context2,
        req_id1=req_id1,
        req_id2=req_id2,
        pipeline_request=pipeline_request,
        context_batch=context_batch,
        num_steps=num_steps,
    )


@pytest.mark.skip("TODO: Re-enable Speculative Decoding Tests")
@prepare_registry
def test_config__validate_device_and_encoding_combinations(
    smollm_135m_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    # Valid device/encoding combinations
    config = PipelineConfig(
        model_path=smollm_135m_local_path,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.cpu()],
        draft_model_path=smollm_135m_local_path,
    )


@pytest.mark.skip("TODO: Re-enable Speculative Decoding Tests")
@prepare_registry
def test_config__validate_target_and_draft_architecture(
    smollm_135m_local_path: str,
    deepseek_r1_distill_llama_8b_local_path: str,
    lmstudio_deepseek_r1_distill_llama_8b_local_path: str,
    gemma_3_1b_it_local_path: str,
) -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
    PIPELINE_REGISTRY.register(DUMMY_GEMMA_ARCH)

    with pytest.raises(
        ValueError,
        match=r"architecture for the draft_model \(.*\) does not match the architecture retrieved for the target model \(.*\)",
    ):
        # Test that when the target & draft architectures are different
        # we raise an error.
        config = PipelineConfig(
            model_path=smollm_135m_local_path,
            device_specs=[DeviceSpec.accelerator()],
            draft_model_path=gemma_3_1b_it_local_path,
            draft_device_specs=[DeviceSpec.accelerator()],
        )

    with pytest.raises(
        ValueError,
        match=r"tokenizer for draft_model \(.*\) does not match the vocabulary of the tokenizer for the target model",
    ):
        # Test that the target & draft architectures are the same,
        # but the tokenizers are different
        config = PipelineConfig(
            model_path=deepseek_r1_distill_llama_8b_local_path,
            quantization_encoding=SupportedEncoding.q6_k,
            device_specs=[DeviceSpec.accelerator()],
            weight_path=[
                Path(
                    lmstudio_deepseek_r1_distill_llama_8b_local_path,
                    "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf",
                )
            ],
            draft_model_path=smollm_135m_local_path,
        )


def test_speculative_decoding_context_update(
    setup_speculative_decoding_pipeline: SpeculativeDecodingSetup,
) -> None:
    pipeline: StandaloneSpeculativeDecodingPipeline = (
        setup_speculative_decoding_pipeline.pipeline
    )
    context1: TextContext = setup_speculative_decoding_pipeline.context1
    context2: TextContext = setup_speculative_decoding_pipeline.context2
    req_id1: RequestID = context1.request_id
    req_id2: RequestID = context2.request_id
    pipeline_request: dict[RequestID, TextContext] = (
        setup_speculative_decoding_pipeline.pipeline_request
    )
    context_batch: list[TextContext] = (
        setup_speculative_decoding_pipeline.context_batch
    )
    num_steps: int = setup_speculative_decoding_pipeline.num_steps

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
        dtype=np.int64,
    )
    bonus_tokens = np.array(
        [
            [1001],
            [2001],
        ],
        dtype=np.int64,
    )

    reject_token1_idx = 4
    reject_token2_idx = num_steps
    first_rejected_tokens = np.array(
        [[reject_token1_idx], [reject_token2_idx]], dtype=np.int64
    )

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
        len(context1.tokens)
        == len(context1.tokens.prompt) + reject_token1_idx + 1
    )
    assert (
        len(context2.tokens)
        == len(context2.tokens.prompt) + reject_token2_idx + 1
    )

    assert context1._draft_offset == 0
    assert context2._draft_offset == 0

    # subtract 1 because the recovered token has not been processed by the draft
    # or dtarget model
    assert (
        context1.tokens.processed_length
        == len(context1.tokens.prompt) + reject_token1_idx - 1
    )

    # subtract 1 because the bonus token has not been run through the draft model
    assert (
        context2.tokens.processed_length
        == len(context2.tokens.prompt) + reject_token2_idx - 1
    )

    assert np.all(
        context1.tokens.all
        == np.concatenate(
            (
                context1.tokens.prompt,
                draft_tokens[0, :reject_token1_idx],
                recovered_tokens[0, reject_token1_idx][np.newaxis],
            )
        )
    )

    assert np.all(
        context2.tokens.all
        == np.concatenate(
            (
                context2.tokens.prompt,
                draft_tokens[1, :reject_token2_idx],
                bonus_tokens[1],
            )
        )
    )

    response = pipeline.build_response(
        context_batch=list(pipeline_request.values())
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


def test_draft_model_encoding_selection() -> None:
    """Test that draft model encoding is correctly selected from config or fallback."""
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    # Test 1: When draft_model_config.quantization_encoding is specified explicitly
    pipeline_config = PipelineConfig(
        model_path=model_name,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.accelerator()],
        draft_model_path=model_name,
        speculative_method=SpeculativeMethod.STANDALONE,
        num_speculative_tokens=10,
        max_batch_size=4,
        max_num_steps=5,
        max_length=1024,
    )
    pipeline_config.model.kv_cache_config.cache_strategy = KVCacheStrategy.PAGED
    pipeline_config.model.kv_cache_config.kv_cache_page_size = 128
    pipeline_config.model.kv_cache_config.device_memory_utilization = 0.3

    # Set draft model quantization encoding explicitly
    assert pipeline_config.draft_model_config is not None
    pipeline_config.draft_model_config.quantization_encoding = (
        SupportedEncoding.float32
    )

    _, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, StandaloneSpeculativeDecodingPipeline)

    # Test 2: When draft_model_config.quantization_encoding is None (fallback to first supported)
    # This test verifies that the fallback mechanism works when no explicit encoding is set
    pipeline_config2 = PipelineConfig(
        model_path=model_name,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.accelerator()],
        draft_model_path=model_name,
        speculative_method=SpeculativeMethod.STANDALONE,
        num_speculative_tokens=10,
        max_batch_size=4,
        max_num_steps=5,
        max_length=1024,
    )
    pipeline_config2.model.kv_cache_config.cache_strategy = (
        KVCacheStrategy.PAGED
    )
    pipeline_config2.model.kv_cache_config.kv_cache_page_size = 128
    pipeline_config2.model.kv_cache_config.device_memory_utilization = 0.3

    # Ensure draft model quantization encoding is None to test fallback
    assert pipeline_config2.draft_model_config is not None
    pipeline_config2.draft_model_config.quantization_encoding = None

    # The pipeline should still be created successfully, falling back to the first supported encoding
    _, pipeline2 = PIPELINE_REGISTRY.retrieve(pipeline_config2)
    assert isinstance(pipeline2, StandaloneSpeculativeDecodingPipeline)


def test_kv_cache_claiming_protocol() -> None:
    """Test that claim is called before fetch in prepare_batch."""
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    pipeline_config = PipelineConfig(
        model_path=model_name,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.accelerator()],
        draft_model_path=model_name,
        speculative_method=SpeculativeMethod.STANDALONE,
        num_speculative_tokens=10,
        max_batch_size=4,
        max_num_steps=5,
        max_length=1024,
    )
    pipeline_config.model.kv_cache_config.cache_strategy = KVCacheStrategy.PAGED
    pipeline_config.model.kv_cache_config.kv_cache_page_size = 128
    pipeline_config.model.kv_cache_config.device_memory_utilization = 0.3

    _tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)
    assert isinstance(pipeline, StandaloneSpeculativeDecodingPipeline)

    # Create a test context
    tokens = np.array([1, 450, 6593], dtype=np.int64)
    context = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(tokens),
        max_length=1024,
        sampling_params=SamplingParams(top_k=1),
    )
    batch = [context]

    # Mock the KV cache manager to track method calls
    mock_kv_manager = Mock()
    mock_kv_manager.contains.return_value = False  # Simulate new request
    mock_kv_manager.get_runtime_inputs.return_value = []

    # Track call order
    call_order = []

    def track_claim(request_id: RequestID) -> None:
        call_order.append(("claim", request_id))

    def track_get_runtime_inputs(
        batch: dict[RequestID, TextContext], num_steps: int
    ) -> Sequence[RaggedKVCacheInputs]:
        call_order.append(
            ("get_runtime_inputs", [ctx.request_id for ctx in batch], num_steps)  # type: ignore
        )
        return []

    mock_kv_manager.claim.side_effect = track_claim
    mock_kv_manager.get_runtime_inputs.side_effect = track_get_runtime_inputs

    # Replace the KV manager in both models
    with patch.object(pipeline._draft_model, "kv_manager", mock_kv_manager):
        with patch.object(
            pipeline._target_model, "kv_manager", mock_kv_manager
        ):
            # Call prepare_batch for draft model
            pipeline.prepare_batch(
                pipeline._draft_model,
                batch,
                num_steps=3,
                return_n_logits=1,
                is_draft=True,
            )

            # Verify that claim was called before get_runtime_inputs
            assert len(call_order) >= 2, (
                f"Expected at least 2 calls, got {len(call_order)}: {call_order}"
            )

            # Check that claim was called first
            first_call = call_order[0]
            assert first_call[0] == "claim", (
                f"First call should be claim, got {first_call[0]}"
            )
            assert first_call[1] == context.request_id, (
                f"claim should be called with request_id {context.request_id}, got {first_call[1]}"
            )

            # Check that get_runtime_inputs was called after claim
            get_runtime_inputs_calls = [
                call for call in call_order if call[0] == "get_runtime_inputs"
            ]
            assert len(get_runtime_inputs_calls) > 0, (
                "get_runtime_inputs should have been called"
            )

            # Verify contains was called to check if request was already claimed
            mock_kv_manager.contains.assert_called_with(context.request_id)
