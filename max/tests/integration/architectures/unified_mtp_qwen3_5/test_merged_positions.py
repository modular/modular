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
"""Tests M-RoPE positions over the merged ``[real, draft_1..draft_K]`` window."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.architectures.qwen3_5.batch_processor import (
    context_position_rows,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_rope_index,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.batch_processor import (
    merged_position_rows,
)
from max.pipelines.context import ImageMetadata, TextContext, TokenBuffer
from max.pipelines.kv_cache.paged_kv_cache.cache_manager import (
    prompt_tokens_for_context,
)

NUM_DRAFTS = 3


ROPE_DELTA = -5


IMAGE_TOKEN_ID = 99


def _text_context(num_tokens: int) -> TextContext:
    return TextContext(
        max_length=num_tokens + 64,
        tokens=TokenBuffer(np.arange(num_tokens, dtype=np.int64) + 1),
    )


def _image_context(num_tokens: int) -> Qwen3VLTextAndVisionContext:
    """Returns a request with an image and a fixed rope delta."""
    tokens = np.arange(num_tokens, dtype=np.int64) + 1
    # The context checks that the image span holds image tokens.
    tokens[1:3] = IMAGE_TOKEN_ID
    return Qwen3VLTextAndVisionContext(
        max_length=num_tokens + 64,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(
                start_idx=1,
                end_idx=3,
                pixel_values=np.zeros(0, dtype=np.float32),
            )
        ],
        vision_token_ids=[IMAGE_TOKEN_ID],
        spatial_merge_size=2,
        rope_delta=ROPE_DELTA,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=98,
        vision_start_token_id=97,
        vision_end_token_id=96,
        image_token_indices=np.array([1, 2], dtype=np.int32),
        decoder_position_ids=np.tile(
            np.arange(num_tokens, dtype=np.int64), (3, 1)
        ),
        vision_data=None,
    )


def _decode(ctx: TextContext, num_drafts: int) -> TextContext:
    """Advances past the prompt and arms the request with ``num_drafts``."""
    ctx.update(new_token=1)
    ctx.spec_decoding_state.draft_tokens_to_verify = [7] * num_drafts
    return ctx


def _row(
    positions: npt.NDArray[np.int64], start: int, width: int
) -> npt.NDArray[np.int64]:
    return positions[:, start : start + width]


def test_a_prefill_row_is_exactly_its_real_positions() -> None:
    """Checks a prefill's merged positions are its real positions."""
    contexts = [_text_context(6), _text_context(3)]

    merged = merged_position_rows(contexts)

    np.testing.assert_array_equal(
        merged,
        np.concatenate([context_position_rows(c) for c in contexts], axis=1),
    )
    assert merged.shape == (3, 9)


@pytest.mark.parametrize("num_drafts", [0, 1, NUM_DRAFTS])
def test_each_row_is_as_wide_as_the_cache_sizes_its_query(
    num_drafts: int,
) -> None:
    """Checks each row's width matches ``prompt_tokens_for_context``."""
    contexts = [
        _decode(_text_context(6), num_drafts),
        _decode(_text_context(4), num_drafts),
    ]

    merged = merged_position_rows(contexts)

    widths = [prompt_tokens_for_context(c) for c in contexts]
    assert merged.shape[1] == sum(widths)
    assert widths == [1 + num_drafts] * 2


def test_each_draft_continues_its_own_rows_ramp() -> None:
    """Checks each request's drafts continue from its own last position."""
    short, long = (
        _decode(_text_context(4), NUM_DRAFTS),
        _decode(_text_context(9), NUM_DRAFTS),
    )

    merged = merged_position_rows([short, long])

    width = 1 + NUM_DRAFTS
    for index in range(2):
        row = _row(merged, index * width, width)
        first = int(row[0, 0])
        expected = np.tile(np.arange(width, dtype=np.int64) + first, (3, 1))
        np.testing.assert_array_equal(row, expected)
    assert int(merged[0, 0]) != int(merged[0, width])


def test_a_draft_continues_past_the_rope_delta() -> None:
    """Checks draft positions include the request's rope delta."""
    ctx = _decode(_image_context(8), NUM_DRAFTS)

    merged = merged_position_rows([ctx])

    real = int(context_position_rows(ctx)[0, 0])
    assert real == ctx.tokens.processed_length + ROPE_DELTA
    np.testing.assert_array_equal(
        merged[0],
        np.arange(1 + NUM_DRAFTS, dtype=np.int64) + real,
    )
    naive = np.arange(1 + NUM_DRAFTS) + ctx.tokens.processed_length
    assert not np.array_equal(merged[0], naive)


def test_a_text_neighbour_keeps_its_own_positions() -> None:
    """Checks one request's rope delta does not affect its neighbor."""
    image = _decode(_image_context(8), NUM_DRAFTS)
    text = _decode(_text_context(8), NUM_DRAFTS)

    merged = merged_position_rows([image, text])

    width = 1 + NUM_DRAFTS
    assert int(_row(merged, 0, width)[0, 0]) == 8 + ROPE_DELTA
    assert int(_row(merged, width, width)[0, 0]) == 8


def test_a_text_only_prompt_has_no_rope_delta() -> None:
    """Checks a text-only prompt's rope delta is zero."""
    tokens = np.arange(11, dtype=np.int64) + 1

    _, deltas = get_rope_index(
        spatial_merge_size=2,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=98,
        vision_start_token_id=97,
        input_ids=tokens.reshape(1, -1),
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=np.ones((1, tokens.size), dtype=np.int64),
    )

    assert int(deltas.item()) == 0


def test_an_image_prompt_is_rejectable_on_arrival() -> None:
    """Checks an image prompt reports its images before any forward."""
    ctx = _image_context(8)

    assert ctx.images
    assert ctx.needs_vision_encoding
    assert ctx.rope_delta != 0
