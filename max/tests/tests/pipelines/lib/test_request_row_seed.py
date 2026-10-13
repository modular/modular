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
"""A pinned ``sampling_params.seed`` has to reproduce.

The sampler keys its RNG on :func:`request_row_seed`, so anything that reaches
that key and is not the seed breaks the reproducibility a client asks for by
pinning one. The request id is the obvious candidate, because the server mints
a fresh one per HTTP request -- two identical requests would then draw
different tokens, and the same request replayed would never come back.

Deliberately the cheapest statement of it: no graph, no GPU, no sampling, so
it fails in seconds and for one reason.
"""

from __future__ import annotations

import numpy as np
from max.pipelines.context import SamplingParams, TextContext, TokenBuffer
from max.pipelines.modeling.types import RequestID
from max.pipelines.sampling import request_row_seed


def _context(seed: int, prompt_len: int = 8) -> TextContext:
    """A context with its own fresh request id, as the server would mint it."""
    return TextContext(
        request_id=RequestID(),
        max_length=128,
        tokens=TokenBuffer(np.arange(prompt_len, dtype=np.int64)),
        sampling_params=SamplingParams(seed=seed),
    )


def test_same_seed_reproduces_across_requests() -> None:
    """Two requests pinned to one seed get one key, despite distinct ids."""
    first, second = _context(seed=1234), _context(seed=1234)

    assert first.request_id.value != second.request_id.value, (
        "the fixture must hand the two contexts different ids, or this test "
        "cannot see the failure it exists to catch"
    )
    assert request_row_seed(first) == request_row_seed(second), (
        "a pinned seed no longer reproduces: something outside "
        "sampling_params.seed and the token count is reaching the RNG key"
    )


def test_different_seeds_stay_apart() -> None:
    """The control: without this, returning a constant would pass above."""
    assert request_row_seed(_context(seed=1234)) != request_row_seed(
        _context(seed=5678)
    )


def test_the_key_advances_as_tokens_are_generated() -> None:
    """Consecutive steps of one request must not draw the same randomness."""
    context = _context(seed=1234)
    before = request_row_seed(context)

    context.update_with_future_token()
    context.realize_future_token(new_token=99, log_probabilities=None)

    assert request_row_seed(context) != before
