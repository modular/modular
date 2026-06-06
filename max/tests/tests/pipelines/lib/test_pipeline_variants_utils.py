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
"""Tests for pipeline_variants/utils.py."""

import numpy as np
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TokenBuffer,
)
from max.pipelines.lib.pipeline_variants.utils import build_response
from max.pipelines.modeling.types import RequestID


def create_text_context(prompt_len: int, max_length: int) -> TextContext:
    """Create a TextContext for testing."""
    tokens = np.arange(prompt_len, dtype=np.int64)
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(tokens),
    )


def advance_to_processed(ctx: TextContext) -> None:
    """Advance context so prompt tokens are marked as processed.

    After this call, processed_length equals the original token count.
    """
    ctx.update_with_future_token()
    ctx.realize_future_token(new_token=99, log_probabilities=None)


class TestBuildResponse:
    """Tests for build_response function."""

    def test_marks_maximum_length_when_at_limit(self) -> None:
        """Context is marked MAXIMUM_LENGTH when at the boundary."""
        max_seq_len = 100
        # Create context with 99 tokens
        ctx = create_text_context(prompt_len=99, max_length=max_seq_len)

        # Advance to processed state: processed_length = 99
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 99

        # current_length = 99 + 1 = 100
        # With max_growth_per_step=1: 100 + 1 = 101 > 100 → MAXIMUM_LENGTH
        build_response([ctx], max_seq_len=max_seq_len, max_growth_per_step=1)

        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH

    def test_does_not_mark_when_below_limit(self) -> None:
        """Context is not marked when there's room for growth."""
        max_seq_len = 100
        ctx = create_text_context(prompt_len=50, max_length=max_seq_len)

        # Advance to processed state: processed_length = 50
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 50

        # current_length = 50 + 1 = 51
        # With max_growth_per_step=1: 51 + 1 = 52 <= 100 → not done
        build_response([ctx], max_seq_len=max_seq_len, max_growth_per_step=1)

        assert ctx.status != GenerationStatus.MAXIMUM_LENGTH

    def test_max_growth_per_step_for_speculative_decoding(self) -> None:
        """Larger max_growth_per_step triggers earlier termination.

        This is the core logic for speculative decoding: with 3 spec tokens,
        max_growth_per_step = 4, so we stop earlier to prevent KV cache overflow.
        """
        max_seq_len = 100
        max_growth = 4  # e.g., 3 speculative tokens + 1 bonus

        # At length 96, after advance: processed_length = 96
        # current_length = 96 + 1 = 97
        # With max_growth_per_step=4: 97 + 4 = 101 > 100 → MAXIMUM_LENGTH
        ctx_near_limit = create_text_context(
            prompt_len=96, max_length=max_seq_len
        )
        advance_to_processed(ctx_near_limit)
        assert ctx_near_limit.tokens.processed_length == 96

        build_response(
            [ctx_near_limit],
            max_seq_len=max_seq_len,
            max_growth_per_step=max_growth,
        )
        assert ctx_near_limit.status == GenerationStatus.MAXIMUM_LENGTH

        # With default max_growth_per_step=1: 97 + 1 = 98 <= 100 → not done
        ctx_with_default = create_text_context(
            prompt_len=96, max_length=max_seq_len
        )
        advance_to_processed(ctx_with_default)

        build_response(
            [ctx_with_default], max_seq_len=max_seq_len, max_growth_per_step=1
        )
        assert ctx_with_default.status != GenerationStatus.MAXIMUM_LENGTH

    def test_respects_per_request_max_length(self) -> None:
        """Per-request max_length is respected when lower than global."""
        global_max_seq_len = 100
        per_request_max = 50

        # Create context with per-request limit of 50
        ctx = create_text_context(prompt_len=49, max_length=per_request_max)
        advance_to_processed(ctx)
        assert ctx.tokens.processed_length == 49

        # current_length = 49 + 1 = 50
        # With max_growth_per_step=1: 50 + 1 = 51 > 50 → MAXIMUM_LENGTH
        build_response(
            [ctx], max_seq_len=global_max_seq_len, max_growth_per_step=1
        )

        assert ctx.status == GenerationStatus.MAXIMUM_LENGTH
