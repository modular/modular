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
"""Checks that Qwen3-VL serves a prompt in chunks, without compiling."""

from __future__ import annotations

import numpy as np
from max.pipelines.architectures.qwen3vl_moe import (
    qwen3vl_arch,
    qwen3vl_moe_arch,
)
from max.pipelines.architectures.qwen3vl_moe.batch_processor import (
    context_position_rows,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.context import ImageMetadata, TokenBuffer
from max.pipelines.modeling.types import RequestID

_IMAGE_TOKEN = 9


def test_neither_checkpoint_turns_chunked_prefill_off() -> None:
    """A full-context prefill overflows the TP allreduce signal buffer."""
    for arch in (qwen3vl_arch, qwen3vl_moe_arch):
        assert "enable_chunked_prefill" not in arch.required_arguments


def test_a_chunk_that_splits_an_image_takes_its_slice_of_the_positions() -> (
    None
):
    tokens = np.array(
        [1, 2, *[_IMAGE_TOKEN] * 4, 3, 4, 5, 6, 7, 8], dtype=np.int64
    )
    positions = np.stack(
        [
            np.array([0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 8, 9]),
            np.array([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9]),
            np.array([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9]),
        ]
    ).astype(np.int64)
    ctx = Qwen3VLTextAndVisionContext(
        request_id=RequestID(),
        max_length=32,
        tokens=TokenBuffer(tokens),
        vision_token_ids=[_IMAGE_TOKEN],
        images=[
            ImageMetadata(start_idx=2, end_idx=6, pixel_values=np.zeros((4, 1)))
        ],
        spatial_merge_size=2,
        rope_delta=-2,
        image_token_id=_IMAGE_TOKEN,
        video_token_id=0,
        vision_start_token_id=2,
        vision_end_token_id=3,
        image_token_indices=np.arange(2, 6, dtype=np.int32),
        decoder_position_ids=positions,
        vision_data=None,
    )
    ctx.tokens.chunk(4)
    np.testing.assert_array_equal(context_position_rows(ctx), positions[:, :4])

    ctx.tokens.advance_chunk()
    ctx.tokens.chunk(4)
    assert ctx.needs_vision_encoding
    np.testing.assert_array_equal(context_position_rows(ctx), positions[:, 4:8])
