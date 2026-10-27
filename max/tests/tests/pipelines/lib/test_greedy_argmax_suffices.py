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
"""A model may skip the sampler only when argmax is what the sampler picks."""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.context import SamplingParams, TextContext, TokenBuffer
from max.pipelines.sampling import SamplingConfig, greedy_argmax_suffices


def _context(
    sampling_params: SamplingParams | None = None,
    *,
    log_probabilities: int = 0,
    label_token_ids: list[int] | None = None,
) -> TextContext:
    """A context that samples greedily unless given other parameters."""
    return TextContext(
        max_length=128,
        tokens=TokenBuffer(np.arange(4, dtype=np.int64)),
        sampling_params=sampling_params or SamplingParams(top_k=1),
        log_probabilities=log_probabilities,
        label_token_ids=label_token_ids,
    )


def _suffices(
    *contexts: TextContext,
    config: SamplingConfig | None = None,
    constrained_decoding: bool = False,
) -> bool:
    return greedy_argmax_suffices(
        list(contexts),
        config or SamplingConfig(),
        constrained_decoding=constrained_decoding,
    )


def test_greedy_batch_suffices() -> None:
    assert _suffices(_context(), _context(SamplingParams(temperature=0)))


def test_constrained_decoding_needs_sampler() -> None:
    assert not _suffices(_context(), constrained_decoding=True)


@pytest.mark.parametrize(
    "context",
    [
        # The default top_k=-1 samples the full distribution.
        _context(SamplingParams()),
        _context(SamplingParams(top_k=50)),
        _context(
            SamplingParams(top_k=1, logits_processors=[lambda inputs: None])
        ),
        _context(log_probabilities=3),
        # Label scoring reads the full logits of the step.
        _context(label_token_ids=[7, 9]),
    ],
    ids=[
        "full-distribution",
        "top-k",
        "logits-processor",
        "log-probabilities",
        "label-scoring",
    ],
)
def test_non_greedy_context_needs_sampler(context: TextContext) -> None:
    assert not _suffices(_context(), context)


def test_penalties_need_sampler_only_when_enabled() -> None:
    ctx = _context(SamplingParams(top_k=1, repetition_penalty=1.2))
    assert _suffices(ctx)
    assert not _suffices(ctx, config=SamplingConfig(enable_penalties=True))


def test_min_tokens_need_sampler_only_when_enabled() -> None:
    ctx = _context(SamplingParams(top_k=1, min_new_tokens=4))
    assert _suffices(ctx)
    assert not _suffices(ctx, config=SamplingConfig(enable_min_tokens=True))
