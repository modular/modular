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
"""GLM-5.3-Flash image and video placeholder expansion.

The vision-to-language interface is nothing but positions: the tower's rows
are scattered onto the placeholder tokens, in order. So the gate is that each
image's span holds exactly as many placeholders as the tower emits rows for it,
sits inside its own ``<|begin_of_image|>``/``<|end_of_image|>`` pair, and leaves
every other token where the chat template put it. A miscount is not a shape
error downstream: the scatter would write some rows onto text, or leave
placeholders holding their token embedding. A video's frames use the same
placeholder, one run per temporal patch, each followed by its timestamp.

No checkpoint is loaded. The tokenizer is built around a stub delegate and the
token ids are the checkpoint's, from ``tokenizer.json``.
"""

from __future__ import annotations

import asyncio
import io
import re
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from max.pipelines.architectures.glm5_next.context import (
    Glm5NextTextAndVisionContext,
)
from max.pipelines.architectures.glm5_next.image_processor import (
    Glm5NextImageProcessor,
)
from max.pipelines.architectures.glm5_next.tokenizer import (
    Glm5NextTokenizer,
    _Video,
)
from max.pipelines.architectures.glm5_next.video_processor import (
    Glm5NextVideoProcessor,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import VisionPreprocessCache
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from PIL import Image

_BEGIN_OF_IMAGE = 154830
_END_OF_IMAGE = 154831
_BEGIN_OF_VIDEO = 154832
_END_OF_VIDEO = 154833
_IMAGE = 154854
_VIDEO = 154855
_EOS = 154820
_MERGE_SIZE = 2

_ADDED_TOKENS = {
    "<|begin_of_image|>": _BEGIN_OF_IMAGE,
    "<|end_of_image|>": _END_OF_IMAGE,
    "<|begin_of_video|>": _BEGIN_OF_VIDEO,
    "<|end_of_video|>": _END_OF_VIDEO,
    "<|image|>": _IMAGE,
    "<|video|>": _VIDEO,
}
_ADDED_TOKEN_PATTERN = re.compile(
    "(" + "|".join(re.escape(token) for token in _ADDED_TOKENS) + ")"
)

# A rendered user turn around one video, as the chat template emits it.
_VIDEO_PROMPT = "<|user|>why?<|begin_of_video|><|video|><|end_of_video|>"

# Small enough to keep preprocessing fast, large enough that the two images
# below land on different, non-square grids.
_MAX_IMAGE_TOKENS = 64

_IMAGE_SIZES = [(100, 60), (230, 410)]


def _encode(text: str, add_special_tokens: bool = False) -> list[int]:
    """Splits on the added tokens first, as a HuggingFace tokenizer does.

    Text between them maps one character to one id.
    """
    ids: list[int] = []
    for piece in _ADDED_TOKEN_PATTERN.split(text):
        if piece in _ADDED_TOKENS:
            ids.append(_ADDED_TOKENS[piece])
        else:
            ids.extend(ord(char) for char in piece)
    return ids


def _tokenizer() -> Glm5NextTokenizer:
    tokenizer = Glm5NextTokenizer.__new__(Glm5NextTokenizer)
    tokenizer.image_token_id = _IMAGE
    tokenizer._image_start_token_id = _BEGIN_OF_IMAGE
    tokenizer._image_end_token_id = _END_OF_IMAGE
    tokenizer._video_token_id = _VIDEO
    tokenizer._video_start_token_id = _BEGIN_OF_VIDEO
    tokenizer._video_end_token_id = _END_OF_VIDEO
    tokenizer.image_processor = Glm5NextImageProcessor(
        max_image_tokens=_MAX_IMAGE_TOKENS
    )
    tokenizer.video_processor = Glm5NextVideoProcessor()
    tokenizer._preprocess_cache = VisionPreprocessCache(0)
    tokenizer._video_preprocess_cache = VisionPreprocessCache(0)
    tokenizer._hash_images = True
    tokenizer._hash_videos = True
    tokenizer.max_length = 1 << 20
    tokenizer._eos_token_ids = {_EOS}
    tokenizer.delegate = MagicMock()
    tokenizer.delegate.__len__.return_value = 154880
    tokenizer.delegate.encode.side_effect = _encode
    return tokenizer


def _png(width: int, height: int, seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(
        0, 255, (height, width, 3), dtype=np.uint8
    )
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _prompt(num_images: int) -> np.ndarray:
    """``text, [begin, <|image|>, end, text] * num_images``, as the template renders."""
    ids = [11]
    for index in range(num_images):
        ids += [_BEGIN_OF_IMAGE, _IMAGE, _END_OF_IMAGE, 20 + index]
    return np.array(ids, dtype=np.int64)


def test_each_span_holds_one_placeholder_per_tower_row() -> None:
    tokenizer = _tokenizer()
    images = [_png(w, h, seed) for seed, (w, h) in enumerate(_IMAGE_SIZES)]
    prompt = _prompt(len(images))

    expanded, metadata, grids = tokenizer.expand_image_placeholders(
        prompt, images, [None] * len(images)
    )

    assert len(metadata) == len(images)
    assert grids.shape == (len(images), 3)
    for image, grid in zip(metadata, grids, strict=True):
        # The tower's spatial merge folds `merge**2` patches into one row.
        tower_rows = image.pixel_values.shape[0] // _MERGE_SIZE**2
        assert tower_rows == int(np.prod(grid)) // _MERGE_SIZE**2
        assert image.end_idx - image.start_idx == tower_rows
        assert np.all(expanded[image.start_idx : image.end_idx] == _IMAGE)
        assert expanded[image.start_idx - 1] == _BEGIN_OF_IMAGE
        assert expanded[image.end_idx] == _END_OF_IMAGE

    # The two grids differ, so a span sized from the wrong image would fail
    # the count above rather than pass by coincidence.
    assert tuple(grids[0]) != tuple(grids[1])

    # Every other token survives, in order.
    np.testing.assert_array_equal(
        expanded[expanded != _IMAGE], prompt[prompt != _IMAGE]
    )
    assert int(np.sum(expanded == _IMAGE)) == sum(
        image.end_idx - image.start_idx for image in metadata
    )


def test_text_only_request_keeps_literal_placeholders() -> None:
    """A prompt quoting ``<|image|>`` or ``<|video|>`` without media is text.

    It still gets a vision context: the vision seam asserts one for every
    request of a vision-capable architecture.
    """
    tokenizer = _tokenizer()
    prompt = np.array(
        [11, _IMAGE, 12, _BEGIN_OF_VIDEO, _IMAGE, _END_OF_VIDEO, _VIDEO, 13],
        dtype=np.int64,
    )
    request = TextGenerationRequest(
        request_id=RequestID(),
        model_name="zai-org/GLM-5.3-Flash",
        messages=[TextGenerationRequestMessage(role="user", content="hi")],
    )

    with patch.object(
        tokenizer,
        "_generate_prompt_and_token_ids",
        AsyncMock(return_value=("", prompt)),
    ):
        context = asyncio.run(tokenizer.new_context(request))

    assert isinstance(context, Glm5NextTextAndVisionContext)
    assert context.images == []
    assert not context.needs_vision_encoding
    np.testing.assert_array_equal(context.tokens.all, prompt)


def test_placeholder_count_must_match_images() -> None:
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="2 image placeholders"):
        tokenizer.expand_image_placeholders(
            _prompt(2), [_png(64, 64, 0)], [None]
        )


def test_prompt_too_long_is_rejected_before_preprocessing() -> None:
    """The length check runs before any image is patchified.

    Tiny images are upscaled to the token floor, so patchifying first would let
    many of them take gigabytes before the check.
    """
    tokenizer = _tokenizer()
    tokenizer.max_length = 100
    num_images = 8
    images = [_png(1, 1, seed) for seed in range(num_images)]

    with (
        patch.object(
            tokenizer.image_processor,
            "preprocess",
            side_effect=AssertionError("preprocessed before the check"),
        ),
        pytest.raises(PromptTooLongError),
    ):
        tokenizer.expand_image_placeholders(
            _prompt(num_images), images, [None] * num_images
        )


def test_video_expands_to_one_entry_per_temporal_patch() -> None:
    """``replace_video_token``'s frame runs, each a ``grid_t = 1`` entry.

    Grid-only: the expansion never reads pixels.
    """
    tokenizer = _tokenizer()
    grid = (2, 4, 6)
    rows = grid[1] * grid[2]
    video = _Video(
        patches=np.zeros((grid[0] * rows, 0), dtype=np.float32),
        grid=grid,
        indices=[0, 5, 15, 20],
        fps=10.0,
        content_hash=None,
    )

    expanded, metadata, grids = tokenizer.expand_image_placeholders(
        np.array(_encode(_VIDEO_PROMPT)), [], [], [video]
    )

    # A timestamp per temporal patch, from its first frame: 0 and 15 at 10 fps.
    frame = (
        "<|begin_of_image|>"
        + "<|image|>" * (rows // _MERGE_SIZE**2)
        + "<|end_of_image|>"
    )
    expected = _VIDEO_PROMPT.replace(
        "<|video|>", f"{frame}0.0 seconds{frame}1.5 seconds"
    )
    np.testing.assert_array_equal(expanded, _encode(expected))
    assert grids.tolist() == [[1, 4, 6], [1, 4, 6]]
    for entry in metadata:
        assert np.all(expanded[entry.start_idx : entry.end_idx] == _IMAGE)
        assert expanded[entry.start_idx - 1] == _BEGIN_OF_IMAGE
        assert expanded[entry.end_idx] == _END_OF_IMAGE


def test_video_placeholder_count_must_match_videos() -> None:
    tokenizer = _tokenizer()
    video = _Video(
        patches=np.zeros((4, 0), dtype=np.float32),
        grid=(1, 2, 2),
        indices=[0, 0],
        fps=1.0,
        content_hash=None,
    )
    with pytest.raises(ValueError, match="0 video placeholders"):
        tokenizer.expand_image_placeholders(_prompt(0), [], [], [video])


def test_repeated_image_is_preprocessed_once() -> None:
    """A resent image reuses its patches rather than resizing them again."""
    tokenizer = _tokenizer()
    tokenizer._preprocess_cache = VisionPreprocessCache(1 << 30)
    image = _png(*_IMAGE_SIZES[0], seed=0)

    with patch.object(
        tokenizer.image_processor,
        "preprocess",
        wraps=tokenizer.image_processor.preprocess,
    ) as preprocess:
        _, first, _ = tokenizer.expand_image_placeholders(
            _prompt(1), [image], [7]
        )
        _, second, _ = tokenizer.expand_image_placeholders(
            _prompt(1), [image], [7]
        )

    assert preprocess.call_count == 1
    assert second[0].pixel_values is first[0].pixel_values


def test_repeated_clip_is_decoded_once_per_budget() -> None:
    """A resent clip reuses its patches unless its decode budget differs.

    The budget can lower the frame cap, so it keys the sample too.
    """
    tokenizer = _tokenizer()
    tokenizer._video_preprocess_cache = VisionPreprocessCache(1 << 30)
    frames = [np.zeros((56, 56, 3), dtype=np.uint8)] * 2

    with patch.object(
        tokenizer.video_processor,
        "decode",
        return_value=(frames, [0, 1], 1.0),
    ) as decode:
        first = tokenizer._preprocess_videos([b"clip"], None)
        second = tokenizer._preprocess_videos([b"clip"], None)
        tokenizer._preprocess_videos([b"clip"], 1 << 20)

    assert decode.call_count == 2
    assert second[0] is first[0]
    assert first[0].content_hash is not None
