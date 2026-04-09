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

"""Tests for Gemma4Tokenizer — structural checks with mocked dependencies."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, NonCallableMock

import numpy as np
import pytest
from max.interfaces import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    VideoContentPart,
)
from max.pipelines.architectures.gemma4.tokenizer import Gemma4Tokenizer
from max.pipelines.architectures.gemma4.video_processor import VideoMetadata
from max.pipelines.lib import KVCacheConfig
from PIL import Image
from pytest_mock import MockerFixture

# Fake token IDs matching what the mock HF config exposes.
IMAGE_TOKEN_ID = 262144
BOI_TOKEN_ID = 255999
EOI_TOKEN_ID = 256000
VIDEO_TOKEN_ID = 256001  # assigned by add_special_tokens
EOS_TOKEN_ID = 1


def _create_mock_hf_config() -> NonCallableMock:
    """Create a mock HuggingFace config with Gemma4 vision attributes."""
    cfg = NonCallableMock()
    cfg.image_token_id = IMAGE_TOKEN_ID
    cfg.boi_token_id = BOI_TOKEN_ID
    cfg.eoi_token_id = EOI_TOKEN_ID
    cfg.eos_token_id = [EOS_TOKEN_ID, 106]
    return cfg


@pytest.fixture
def mock_pipeline_config() -> MagicMock:
    """Create a mock PipelineConfig for Gemma4 tests."""
    hf_config = _create_mock_hf_config()

    kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    kv_cache_config.enable_prefix_caching = False

    model_config = MagicMock()
    model_config.huggingface_config = hf_config
    model_config.kv_cache = kv_cache_config

    runtime_config = MagicMock()
    runtime_config.max_vision_cache_entries = 0

    pipeline_config = MagicMock()
    pipeline_config.model = model_config
    pipeline_config.runtime = runtime_config
    return pipeline_config


def _make_mock_delegate() -> MagicMock:
    """Build a mock AutoTokenizer delegate with Gemma4 special tokens."""
    delegate = MagicMock()
    delegate.eos_token_id = EOS_TOKEN_ID
    delegate.model_max_length = 4096
    delegate.image_token = "<image_soft_token>"
    delegate.boi_token = "<start_of_image>"
    delegate.eoi_token = "<end_of_image>"
    delegate.chat_template = None

    # convert_tokens_to_ids is called for <|video|>
    delegate.convert_tokens_to_ids.return_value = VIDEO_TOKEN_ID

    # decode fallback — not exercised when token attrs are set
    delegate.decode.side_effect = lambda ids: "".join(f"[tok_{t}]" for t in ids)

    return delegate


def _patch_tokenizer_deps(mocker: MockerFixture, delegate: MagicMock) -> None:
    """Patch external dependencies of Gemma4Tokenizer.__init__."""
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.AutoTokenizer"
        ".from_pretrained",
        return_value=delegate,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.load_processor_config",
        return_value={},
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=MagicMock(
            return_value=([], [], []),
            pooling_kernel_size=3,
        ),
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4VideoProcessor",
        return_value=MagicMock(
            return_value=([], [], [], []),
            pooling_kernel_size=3,
        ),
    )


def _make_image_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="red").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_smoke(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Text-only request: context is created, no image/video metadata."""
    delegate = _make_mock_delegate()
    text_tokens = np.array([2, 100, 200, 300, 3], dtype=np.int64)
    delegate.return_value = {"input_ids": [text_tokens.tolist()]}
    delegate.apply_chat_template.return_value = "Hello world"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello world")
        ],
        request_id=RequestID("test-text"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context is not None
    assert len(context.images) == 0
    assert len(context.video_token_ranges) == 0
    # All token_type_ids should be 0 (text)
    assert np.all(context.mm_token_type_ids == 0)


@pytest.mark.asyncio
async def test_prompt_too_long(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Prompt exceeding max_length raises ValueError."""
    delegate = _make_mock_delegate()
    delegate.model_max_length = 5
    long_tokens = list(range(10))
    delegate.return_value = {"input_ids": [long_tokens]}
    delegate.apply_chat_template.return_value = "a very long prompt"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user", content="a very long prompt"
            )
        ],
        request_id=RequestID("test-long"),
        model_name="test-model",
    )

    with pytest.raises(
        ValueError,
        match="encoded_prompt is greater than the max_length",
    ):
        await tokenizer.new_context(request)


@pytest.mark.asyncio
async def test_image_tokens_inserted(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Image request: image tokens appear in input_ids with correct
    token_type_ids and image metadata."""
    delegate = _make_mock_delegate()
    num_soft_tokens = 4

    # Simulate tokenized output after placeholder expansion:
    #   [BOS, ..text.., BOI, IMG, IMG, IMG, IMG, EOI, ..text.., EOS]
    input_ids = np.array(
        [2, 100, BOI_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_soft_tokens
        + [EOI_TOKEN_ID, 200, 3],
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}
    # One placeholder per image; the tokenizer expands it to BOI+IMG*N+EOI
    delegate.apply_chat_template.return_value = (
        "Describe <image_soft_token> this."
    )
    _patch_tokenizer_deps(mocker, delegate)

    # Mock image processor to return one image with matching soft token count
    fake_pixels = np.zeros((num_soft_tokens * 9, 768), dtype=np.float32)
    fake_pos_ids = np.zeros((num_soft_tokens * 9, 2), dtype=np.int32)
    img_processor_mock = MagicMock(
        return_value=([fake_pixels], [fake_pos_ids], [num_soft_tokens]),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=img_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="Describe this."),
                ],
            )
        ],
        images=[_make_image_bytes()],
        request_id=RequestID("test-img"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    tokens = context.tokens.all
    # Image tokens are present in the token sequence
    img_mask = tokens == IMAGE_TOKEN_ID
    assert img_mask.sum() == num_soft_tokens

    # token_type_ids mark image tokens as 1
    assert np.all(context.mm_token_type_ids[img_mask] == 1)
    # Non-image tokens remain 0
    assert np.all(context.mm_token_type_ids[~img_mask] == 0)

    # Image metadata is populated
    assert len(context.images) == 1
    meta = context.images[0]
    assert meta.pixel_values.shape == fake_pixels.shape
    assert meta.start_idx < meta.end_idx


@pytest.mark.asyncio
async def test_image_placement_multi_turn(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """In a multi-turn conversation the image placeholder lands after earlier
    turns, not at the beginning."""
    delegate = _make_mock_delegate()

    # Chat template output with the image placeholder in the last user turn
    templated = (
        "<start_of_turn>user\nHello<end_of_turn>\n"
        "<start_of_turn>model\nHi!<end_of_turn>\n"
        "<start_of_turn>user\n\n\n<image_soft_token>\n\n"
        "What is this?<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    delegate.apply_chat_template.return_value = templated

    num_soft_tokens = 4

    # Build token IDs that mirror the expanded text structure
    pre_image = [2, 10, 11, 12, 13, 14]
    post_image = [15, 16, 17, 3]
    input_ids = np.array(
        pre_image
        + [BOI_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_soft_tokens
        + [EOI_TOKEN_ID]
        + post_image,
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}

    _patch_tokenizer_deps(mocker, delegate)

    fake_pixels = np.zeros((num_soft_tokens * 9, 768), dtype=np.float32)
    fake_pos_ids = np.zeros((num_soft_tokens * 9, 2), dtype=np.int32)
    img_processor_mock = MagicMock(
        return_value=([fake_pixels], [fake_pos_ids], [num_soft_tokens]),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=img_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello"),
            TextGenerationRequestMessage(role="assistant", content="Hi!"),
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="What is this?"),
                ],
            ),
        ],
        images=[_make_image_bytes()],
        request_id=RequestID("test-multi"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    tokens = context.tokens.all
    img_positions = np.where(tokens == IMAGE_TOKEN_ID)[0]
    assert len(img_positions) == num_soft_tokens
    # Image tokens should appear after the preamble, not at position 0
    assert img_positions[0] > 0

    # Verify the image block is contiguous
    assert np.all(np.diff(img_positions) == 1)


@pytest.mark.asyncio
async def test_video_tokens_inserted(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Video request: video tokens appear in input_ids with token_type_ids=2
    and video metadata is populated."""
    delegate = _make_mock_delegate()

    num_video_soft_tokens = 2
    num_frames = 2

    # Simulate tokenized output with video tokens
    input_ids = np.array(
        [2, 100]
        + [BOI_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * num_video_soft_tokens
        + [EOI_TOKEN_ID]
        + [BOI_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * num_video_soft_tokens
        + [EOI_TOKEN_ID]
        + [200, 3],
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}
    delegate.apply_chat_template.return_value = "Describe <|video|> this."

    _patch_tokenizer_deps(mocker, delegate)

    # Mock video processor to return frame data

    fake_frame_pv = np.zeros((num_frames, 36, 768), dtype=np.float32)
    fake_frame_pos = np.zeros((num_frames, 36, 2), dtype=np.int32)
    video_meta = VideoMetadata(
        timestamps=[0.0, 1.0],
    )

    video_processor_mock = MagicMock(
        return_value=(
            [fake_frame_pv],
            [fake_frame_pos],
            [num_video_soft_tokens],
            [video_meta],
        ),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4VideoProcessor",
        return_value=video_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    VideoContentPart(),
                    TextContentPart(text="Describe this."),
                ],
            )
        ],
        videos=[_make_image_bytes()],
        request_id=RequestID("test-vid"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    tokens = context.tokens.all
    vid_mask = tokens == VIDEO_TOKEN_ID
    assert vid_mask.sum() == num_video_soft_tokens * num_frames

    # token_type_ids mark video tokens as 2
    assert np.all(context.mm_token_type_ids[vid_mask] == 2)

    # Video token ranges are populated
    assert len(context.video_token_ranges) > 0
    for start, end in context.video_token_ranges:
        assert all(tokens[start:end] == VIDEO_TOKEN_ID)

    # No image metadata for a video-only request
    assert len(context.images) == 0


@pytest.mark.asyncio
async def test_text_only_no_images_or_videos(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Pure text request has empty pixel_position_ids and video lists."""
    delegate = _make_mock_delegate()
    delegate.return_value = {"input_ids": [[2, 50, 51, 52, 3]]}
    delegate.apply_chat_template.return_value = "Just text"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Just text")
        ],
        request_id=RequestID("test-no-mm"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert len(context.pixel_position_ids) == 0
    assert len(context.video_frame_patches) == 0
    assert len(context.video_frame_pos_ids) == 0
    assert len(context.video_token_ranges) == 0
