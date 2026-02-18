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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import Buffer, DeviceSpec
from max.interfaces import (
    PipelineTokenizer,
    RequestID,
    SamplingParams,
    TextGenerationInputs,
    TokenBuffer,
)
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig, SupportedEncoding
from max.pipelines.core import TextContext
from max.pipelines.lib.speculative_config import SpeculativeMethod
from max.pipelines.lib.speculative_decoding import (
    StandaloneSpeculativeDecodingPipeline,
)


@dataclass
class SpeculativeDecodingSetup:
    model_name: str
    tokenizer: PipelineTokenizer
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
        speculative_method=SpeculativeMethod.STANDALONE,
        num_speculative_tokens=10,
        quantization_encoding=SupportedEncoding.float32,
        device_specs=[DeviceSpec.accelerator()],
        draft_model_path=model_name,
        draft_device_specs=[DeviceSpec.accelerator()],
        max_batch_size=4,
        max_num_steps=num_steps,
        max_length=1024,
        cache_strategy="paged",
        kv_cache_page_size=128,
        device_memory_utilization=0.3,
    )

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

    target_kv_manager = pipeline.kv_managers[-1]
    target_kv_manager.claim(req_id1, replica_idx=0)
    target_kv_manager.claim(req_id2, replica_idx=0)
    target_kv_manager.alloc(context1, replica_idx=0, num_steps=num_steps)
    target_kv_manager.alloc(context2, replica_idx=0, num_steps=num_steps)

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


def test_speculative_decoding_no_rejection(
    setup_speculative_decoding_pipeline: SpeculativeDecodingSetup,
) -> None:
    pipeline = setup_speculative_decoding_pipeline.pipeline
    context_batch = setup_speculative_decoding_pipeline.context_batch
    context1 = setup_speculative_decoding_pipeline.context1
    context2 = setup_speculative_decoding_pipeline.context2
    num_steps = setup_speculative_decoding_pipeline.num_steps

    assert context1.tokens.processed_length == 0
    assert context2.tokens.processed_length == 0

    # Generate draft tokens.
    draft_inputs, draft_num_steps = pipeline.prepare_batch(
        pipeline._draft_model,
        context_batch,
        [context_batch],
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
        model_inputs.tokens,  # type: ignore[attr-defined]
        model_inputs.input_row_offsets,  # type: ignore[attr-defined]
        draft_tokens,
    )

    assert isinstance(merged_tokens, Buffer)
    assert isinstance(merged_offsets, Buffer)
    # Verify draft tokens with target model
    first_rejected_tokens, recovered_tokens, bonus_tokens = (
        pipeline.verify_draft_tokens_with_target_model(
            draft_inputs,
            context_batch,
            [context_batch],
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
    assert context1.tokens.processed_length == (
        len(context1.tokens.prompt) + num_steps - 1
    )
    assert context2.tokens.processed_length == (
        len(context2.tokens.prompt) + num_steps - 1
    )

    assert np.all(context1.tokens.generated[:-1] == draft_tokens.to_numpy()[0])
    assert np.all(context2.tokens.generated[:-1] == draft_tokens.to_numpy()[1])


def test_speculative_decoding_partial_rejection(
    setup_speculative_decoding_pipeline: SpeculativeDecodingSetup,
) -> None:
    pipeline = setup_speculative_decoding_pipeline.pipeline
    context_batch = setup_speculative_decoding_pipeline.context_batch
    context1 = setup_speculative_decoding_pipeline.context1
    context2 = setup_speculative_decoding_pipeline.context2
    num_steps = setup_speculative_decoding_pipeline.num_steps

    assert context1.tokens.processed_length == 0
    assert context2.tokens.processed_length == 0

    # Generate draft tokens.
    draft_inputs, _draft_num_steps = pipeline.prepare_batch(
        pipeline._draft_model,
        context_batch,
        [context_batch],
        num_steps,
        return_n_logits=1,
        is_draft=True,
    )
    num_steps, draft_tokens, draft_logits, model_inputs, all_draft_logits = (
        pipeline.generate_draft_tokens(context_batch, num_steps, draft_inputs)
    )

    # Merge draft tokens with target tokens
    merged_tokens, merged_offsets = pipeline._ragged_token_merger(
        model_inputs.tokens,  # type: ignore[attr-defined]
        model_inputs.input_row_offsets,  # type: ignore[attr-defined]
        draft_tokens,
    )

    # For the first sequence we'll manually change the tokens and logits so that only part of that sequence is accepted

    draft_logits_host = np.copy(draft_logits.to_numpy())
    draft_logits_host[0, num_steps // 2 :] = 10000.0
    draft_logits = Buffer.from_numpy(draft_logits_host).to(draft_logits.device)

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
    all_draft_logits = Buffer.from_numpy(
        np.permute_dims(all_draft_logits_host, [1, 0, 2])
    ).to(all_draft_logits.device)

    assert isinstance(merged_tokens, Buffer)
    assert isinstance(merged_offsets, Buffer)
    # Verify draft tokens with target model
    first_rejected_tokens, recovered_tokens, bonus_tokens = (
        pipeline.verify_draft_tokens_with_target_model(
            draft_inputs,
            context_batch,
            [context_batch],
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
    assert context1.tokens.processed_length == (
        len(context1.tokens.prompt) + (num_steps // 2) - 1
    )
    # subtract 1 because all draft tokens are accepted, next draft input includes the token generated from the target model
    assert context2.tokens.processed_length == (
        len(context2.tokens.prompt) + num_steps - 1
    )

    assert np.all(
        context1.tokens.generated[:-1] == draft_tokens_host[0, : num_steps // 2]
    )
    assert np.all(context2.tokens.generated[:-1] == draft_tokens_host[1])


def test_speculative_decoding_multiple_token_without_rejection(
    setup_speculative_decoding_pipeline: SpeculativeDecodingSetup,
) -> None:
    pipeline = setup_speculative_decoding_pipeline.pipeline
    context1 = setup_speculative_decoding_pipeline.context1
    context2 = setup_speculative_decoding_pipeline.context2
    pipeline_request = setup_speculative_decoding_pipeline.pipeline_request
    num_steps = setup_speculative_decoding_pipeline.num_steps

    context1_len = len(context1.tokens)
    context2_len = len(context2.tokens)
    for _ in range(5):
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[list(pipeline_request.values())], num_steps=num_steps
        )
        pipeline.execute(inputs)

        # num_steps generated from draft and +1 from the target
        assert len(context1.tokens) == context1_len + (num_steps + 1)
        assert len(context2.tokens) == context2_len + (num_steps + 1)

        context1_len = len(context1.tokens)
        context2_len = len(context2.tokens)
