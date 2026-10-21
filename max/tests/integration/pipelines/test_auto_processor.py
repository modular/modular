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
"""Tests for AutoProcessor's conversation conversion and decoding."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest
from max.pipelines import AutoProcessor, AutoTokenizer
from max.pipelines.lib import generate_local_model_path
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    VideoContentPart,
)
from PIL import Image

REPO_ID = "modularai/SmolLM-135M-Instruct-FP32"
QUESTION = "What is 2 + 2?"


@pytest.fixture(scope="module")
def model_path() -> str:
    # A cached snapshot skips the Hub probe, which CI runs without network.
    try:
        return generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        return REPO_ID


@pytest.fixture(scope="module")
def processor(model_path: str) -> AutoProcessor:
    return AutoProcessor(model_path)


def _png(color: str) -> Image.Image:
    return Image.new("RGB", (4, 3), color)


def _png_bytes(color: str) -> bytes:
    buffer = io.BytesIO()
    _png(color).save(buffer, format="PNG")
    return buffer.getvalue()


def _text_turn(text: str) -> list[dict[str, object]]:
    return [{"role": "user", "content": text}]


def _image_turn(*images: object) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": QUESTION},
            ],
        }
    ]


def test_string_content_stays_a_string(processor: AutoProcessor) -> None:
    out = processor(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": QUESTION},
        ]
    )
    assert out == {
        "messages": [
            [
                TextGenerationRequestMessage(
                    role="system", content="Be brief."
                ),
                TextGenerationRequestMessage(role="user", content=QUESTION),
            ]
        ],
        "images": [[]],
    }


def test_image_parts_become_placeholders_in_order(
    processor: AutoProcessor,
) -> None:
    out = processor(_image_turn(_png_bytes("red"), _png_bytes("blue")))
    assert out["messages"] == [
        [
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    ImageContentPart(),
                    TextContentPart(text=QUESTION),
                ],
            )
        ]
    ]
    assert out["images"] == [[_png_bytes("red"), _png_bytes("blue")]]


def test_pil_images_are_encoded_as_png(processor: AutoProcessor) -> None:
    image = _png("red")
    [[encoded]] = processor(_image_turn(image))["images"]
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.format == "PNG"
        assert list(decoded.getdata()) == list(image.getdata())


def test_cmyk_images_are_encoded_as_rgb(processor: AutoProcessor) -> None:
    image = Image.new("CMYK", (4, 3), (0, 255, 255, 0))
    [[encoded]] = processor(_image_turn(image))["images"]
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.mode == "RGB"
        assert list(decoded.getdata()) == list(image.convert("RGB").getdata())


@pytest.mark.parametrize("key", ["image", "path"])
@pytest.mark.parametrize("as_path", [str, Path])
def test_local_paths_are_read(
    processor: AutoProcessor,
    tmp_path: Path,
    key: str,
    as_path: Callable[[Path], object],
) -> None:
    path = tmp_path / "red.png"
    path.write_bytes(_png_bytes("red"))
    out = processor(
        [{"role": "user", "content": [{"type": "image", key: as_path(path)}]}]
    )
    assert out["images"] == [[_png_bytes("red")]]


def test_alpha_images_keep_their_alpha(processor: AutoProcessor) -> None:
    image = Image.new("PA", (4, 3))
    [[encoded]] = processor(_image_turn(image))["images"]
    with Image.open(io.BytesIO(encoded)) as decoded:
        assert decoded.mode == "RGBA"


def test_file_urls_are_read_as_local_paths(
    processor: AutoProcessor, tmp_path: Path
) -> None:
    path = tmp_path / "red.png"
    path.write_bytes(_png_bytes("red"))
    out = processor(_image_turn(f"file://{path}"))
    assert out["images"] == [[_png_bytes("red")]]


def test_a_none_image_falls_back_on_path(
    processor: AutoProcessor, tmp_path: Path
) -> None:
    path = tmp_path / "red.png"
    path.write_bytes(_png_bytes("red"))
    part = {"type": "image", "image": None, "path": str(path)}
    out = processor([{"role": "user", "content": [part]}])
    assert out["images"] == [[_png_bytes("red")]]


def test_image_parts_keep_sizing_hints(processor: AutoProcessor) -> None:
    part = {
        "type": "image",
        "image": _png_bytes("red"),
        "detail": "low",
        "max_long_side_pixel": 512,
    }
    [[message]] = processor([{"role": "user", "content": [part]}])["messages"]
    assert message.content == [
        ImageContentPart(detail="low", max_long_side_pixel=512)
    ]


def test_a_list_of_conversations_keeps_row_order(
    processor: AutoProcessor,
) -> None:
    out = processor(
        [
            _text_turn(QUESTION),
            _image_turn(_png_bytes("red")),
            _image_turn(_png_bytes("green"), _png_bytes("blue")),
        ]
    )
    assert out.keys() == {"messages", "images"}
    assert [len(row) for row in out["messages"]] == [1, 1, 1]
    assert out["messages"][0][0].content == QUESTION
    assert out["images"] == [
        [],
        [_png_bytes("red")],
        [_png_bytes("green"), _png_bytes("blue")],
    ]


def test_one_conversation_is_a_batch_of_one(processor: AutoProcessor) -> None:
    conversation = _image_turn(_png_bytes("red"))
    assert processor(conversation) == processor([conversation])


def test_rows_build_valid_requests(processor: AutoProcessor) -> None:
    out = processor(
        [
            _image_turn(_png_bytes("red"), _png_bytes("blue")),
            _text_turn(QUESTION),
        ]
    )
    for messages, images in zip(out["messages"], out["images"], strict=True):
        request = TextGenerationRequest(
            request_id=RequestID(),
            model_name=REPO_ID,
            messages=messages,
            images=images,
        )
        assert request.number_of_images == len(images)


def test_video_parts_pass_through(processor: AutoProcessor) -> None:
    out = processor(
        [{"role": "user", "content": [{"type": "video", "video": "a.mp4"}]}]
    )
    assert out["messages"] == [
        [
            TextGenerationRequestMessage(
                role="user", content=[VideoContentPart()]
            )
        ]
    ]
    assert out["images"] == [[]]


@pytest.mark.parametrize(
    ("conversations", "error", "match"),
    [
        ([], ValueError, "at least one message"),
        ("hello", TypeError, "a conversation or a list"),
        (b"hello", TypeError, "a conversation or a list"),
        (5, TypeError, "a conversation or a list"),
        ([[{"role": "user", "content": QUESTION}], []], ValueError, "Row 1"),
        ([[{"role": "user", "content": QUESTION}], "hi"], TypeError, "row 1"),
        ([{"role": "user", "content": QUESTION}, "hi"], TypeError, "message"),
        ([{"role": "user", "content": ["hi"]}], TypeError, "content part"),
        (
            [{"role": "user", "content": [{"type": "image"}]}],
            ValueError,
            'no "image"',
        ),
        (_image_turn(42), TypeError, "image of type int"),
        *(
            (_image_turn(source), ValueError, "URL or base64")
            for source in (
                "https://example.com/cat.png",
                "HTTP://example.com/cat.png",
                "data:image/png;base64,iVBORw0KGgo=",
            )
        ),
        *(
            (
                [{"role": "user", "content": [{"type": "image", key: value}]}],
                ValueError,
                "URL or base64",
            )
            for key, value in (
                ("url", "https://example.com/cat.png"),
                ("base64", "iVBORw0KGgo="),
            )
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        }
                    ],
                }
            ],
            ValueError,
            "URL or base64",
        ),
        ([{"role": "robot", "content": QUESTION}], ValueError, "role"),
        (
            [{"role": "user", "content": [{"type": "audio"}]}],
            ValueError,
            "audio",
        ),
    ],
)
def test_rejects_malformed_conversations(
    processor: AutoProcessor,
    conversations: object,
    error: type[Exception],
    match: str,
) -> None:
    # Typed loosely so the call can pass what the signature rejects.
    call: Callable[..., object] = processor
    with pytest.raises(error, match=match):
        call(conversations)


def test_missing_image_path_raises(
    processor: AutoProcessor, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        processor(_image_turn(str(tmp_path / "missing.png")))


def test_decodes_like_auto_tokenizer(
    processor: AutoProcessor, model_path: str
) -> None:
    tokenizer = AutoTokenizer(model_path)
    rows = tokenizer([QUESTION, "<|im_start|>Hi"])["input_ids"]
    assert processor.decode(rows[0]) == tokenizer.decode(rows[0])
    assert processor.batch_decode(
        rows, skip_special_tokens=True
    ) == tokenizer.batch_decode(rows, skip_special_tokens=True)
