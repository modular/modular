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

from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession
from max.interfaces import TextGenerationInputs, TokenBuffer
from max.pipelines.core import TextContext
from max.pipelines.core.context import FUTURE_TOKEN
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    AsyncBatch,
    ScatterFutureTokenProcessor,
)

OOB_IDX = np.iinfo(np.int32).min


def _create_context(
    active_tokens: list[int], chunk: int | None = None
) -> TextContext:
    context = TextContext(
        max_length=1024,
        tokens=TokenBuffer(np.array(active_tokens, dtype=np.int64)),
    )
    if chunk is not None:
        context.tokens.chunk(chunk)
    return context


def _to_ragged_inputs(contexts: list[list[TextContext]]) -> list[int]:
    flat_contexts = [ctx for batch in contexts for ctx in batch]
    return np.concatenate(
        [context.tokens.active for context in flat_contexts]
    ).tolist()


def _to_cpu(arr: list[int] | list[list[int]]) -> Buffer:
    return Buffer.from_dlpack(np.array(arr, dtype=np.int64))


def _to_gpu(arr: list[int] | list[list[int]]) -> Buffer:
    return _to_cpu(arr).to(Accelerator())


def _create_async_batch(
    batches: list[list[TextContext]],
) -> AsyncBatch[TextContext]:
    # generated_tokens is of shape (num_reqs, 1)
    generated_tokens: list[list[int]] = []
    for batch in batches:
        for ctx in batch:
            generated_tokens.append([ctx.tokens.active[-1] + 1])
            ctx.update_with_future_token()
    return AsyncBatch(
        inputs=TextGenerationInputs(batches=batches, num_steps=1),
        generated_tokens_device=_to_gpu(generated_tokens),
        generated_tokens_host=_to_cpu(generated_tokens),
        copy_event=MagicMock(),
    )


@pytest.fixture(scope="module")
def scatter_future_token_processor() -> ScatterFutureTokenProcessor:
    return ScatterFutureTokenProcessor(
        InferenceSession(devices=[Accelerator()])
    )


def _scatter(
    scatter_future_token_processor: ScatterFutureTokenProcessor,
    prev_batch: AsyncBatch[TextContext],
    curr_batch: list[list[TextContext]],
    ragged_input_tokens: Buffer,
) -> tuple[list[int], list[int]]:
    curr_inputs = TextGenerationInputs(batches=curr_batch, num_steps=1)
    future_tok_indices = (
        scatter_future_token_processor._compute_scatter_future_tok_indices(
            prev_batch=prev_batch,
            inputs=curr_inputs,
            ragged_input_tokens=ragged_input_tokens,
        )
    )
    new_ragged_input_tokens = (
        scatter_future_token_processor.scatter_future_tokens(
            prev_batch=prev_batch,
            inputs=curr_inputs,
            ragged_input_tokens=ragged_input_tokens,
        )
    )
    return (
        future_tok_indices.to_numpy().tolist(),
        new_ragged_input_tokens.to_numpy().tolist(),
    )


def test_basic(
    scatter_future_token_processor: ScatterFutureTokenProcessor,
) -> None:
    req_a = _create_context([11, 12, 13])
    req_b = _create_context([21, 22, 23, 24])
    req_c = _create_context([31, 32])
    req_d = _create_context([41, 42])
    req_e = _create_context([51, 52])

    # replica1 has 2 reqs, replica2 has 0 reqs, replica3 has 1 reqs, replica4 has 0 reqs
    prev_batch = [[req_a, req_b], [], [req_c], []]
    prev_ragged_inputs = _to_ragged_inputs(prev_batch)
    assert prev_ragged_inputs == [11, 12, 13, 21, 22, 23, 24, 31, 32]

    async_batch = _create_async_batch(batches=prev_batch)
    generated_tokens = (
        async_batch.generated_tokens_host.to_numpy().flatten().tolist()
    )
    assert generated_tokens == [14, 25, 33]

    tests: list[
        tuple[list[list[TextContext]], list[int], list[int], list[int]]
    ] = [
        (
            # test case where there is partial overlap between prev and curr batch
            [[], [req_b], [req_c, req_d], []],
            [FUTURE_TOKEN, FUTURE_TOKEN, 41, 42],
            [OOB_IDX, 0, 1],
            [25, 33, 41, 42],
        ),
        (
            # test case where reqs in curr batch are not in same order as prev batch
            [[req_e], [req_b, req_a], [req_d], []],
            [51, 52, FUTURE_TOKEN, FUTURE_TOKEN, 41, 42],
            [3, 2, OOB_IDX],
            [51, 52, 25, 14, 41, 42],
        ),
        (
            # test case where prev and curr batch have no overlapping reqs
            [[], [], [], [req_e, req_d]],
            [51, 52, 41, 42],
            [OOB_IDX, OOB_IDX, OOB_IDX],
            [51, 52, 41, 42],  # no change
        ),
    ]
    for (
        curr_batch,
        expected_curr_ragged_inputs,
        expected_future_tok_indices,
        expected_new_ragged_inputs,
    ) in tests:
        curr_ragged_inputs = _to_ragged_inputs(curr_batch)
        assert curr_ragged_inputs == expected_curr_ragged_inputs
        future_tok_indices, new_ragged_inputs = _scatter(
            scatter_future_token_processor=scatter_future_token_processor,
            prev_batch=async_batch,
            curr_batch=curr_batch,
            ragged_input_tokens=_to_gpu(curr_ragged_inputs),
        )
        assert future_tok_indices == expected_future_tok_indices
        assert new_ragged_inputs == expected_new_ragged_inputs


def test_chunked_prefill(
    scatter_future_token_processor: ScatterFutureTokenProcessor,
) -> None:
    req_a = _create_context([11, 12, 13])
    req_b = _create_context([21, 22, 23, 24, 25], chunk=3)
    assert req_b.tokens.active.tolist() == [21, 22, 23]
    req_c = _create_context([31, 32, 33, 34, 35, 36, 37, 38], chunk=4)
    assert req_c.tokens.active.tolist() == [31, 32, 33, 34]
    req_d = _create_context([41, 42, 43])
    req_e = _create_context([51, 52, 53])

    prev_batch = [req_a, req_b, req_c]
    prev_ragged_inputs = _to_ragged_inputs([prev_batch])
    assert prev_ragged_inputs == [11, 12, 13, 21, 22, 23, 31, 32, 33, 34]

    async_batch = _create_async_batch(batches=[prev_batch])
    generated_tokens = (
        async_batch.generated_tokens_host.to_numpy().flatten().tolist()
    )
    assert generated_tokens == [14, 24, 35]

    # Make sure that we correctly handle chunked requests
    # Note that the sampled tokens for chunked requests are dropped/ignored
    curr_batch = [req_d, req_b, req_e, req_a, req_c]
    curr_ragged_inputs = _to_ragged_inputs([curr_batch])
    # fmt: off
    assert curr_ragged_inputs == [41, 42, 43, 24, 25, 51, 52, 53, FUTURE_TOKEN, 35, 36, 37, 38]
    future_tok_indices, new_ragged_inputs = _scatter(
        scatter_future_token_processor=scatter_future_token_processor,
        prev_batch=async_batch,
        curr_batch=[curr_batch],
        ragged_input_tokens=_to_gpu(curr_ragged_inputs),
    )
    assert future_tok_indices == [8, OOB_IDX, OOB_IDX]
    assert new_ragged_inputs == [41, 42, 43, 24, 25, 51, 52, 53, 14, 35, 36, 37, 38]
    # fmt: on
