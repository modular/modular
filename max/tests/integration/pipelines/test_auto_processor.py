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

import base64
import gzip
import http.server
import io
import socket
import textwrap
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from max.pipelines import AutoProcessor, AutoTokenizer
from max.pipelines.lib import auto_processor, generate_local_model_path
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


_REDIRECTS = {"/redirect": "/red.png", "/to-file": "file:///etc/hosts"}


class _ImageHandler(http.server.BaseHTTPRequestHandler):
    """Serves a red PNG under its names, plus the bodies the fetch rejects."""

    def do_GET(self) -> None:
        path = urllib.parse.unquote(self.path)
        if path in _REDIRECTS:
            self.send_response(302)
            self.send_header("Location", _REDIRECTS[path])
            self.end_headers()
            return
        if path == "/big.bin":
            body = bytes(1024 * 1024 + 1)
        elif path == "/gzip.png":
            body = gzip.compress(_png_bytes("red"))
        elif path in (
            "/red.png",
            "/slow.png",
            "/drip.png",
            "/truncated.png",
            "/red é.png",
        ):
            body = _png_bytes("red")
        else:
            self.send_error(404)
            return
        if path == "/slow.png":
            time.sleep(1)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        if path == "/gzip.png":
            self.send_header("Content-Encoding", "gzip")
        # A truncated body promises more bytes than it sends.
        extra = 100 if path == "/truncated.png" else 0
        self.send_header("Content-Length", str(len(body) + extra))
        self.end_headers()
        try:
            if path == "/drip.png":
                # Each piece beats the read timeout, but the whole body
                # takes longer than the test's fetch timeout.
                for start in range(0, len(body), 16):
                    self.wfile.write(body[start : start + 16])
                    time.sleep(0.1)
            else:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def image_server() -> Iterator[str]:
    """Yields the base URL of a local image server, so no test needs the net."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


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


def test_pil_images_come_out_as_copies(processor: AutoProcessor) -> None:
    rgb = _png("red")
    cmyk = Image.new("CMYK", (4, 3), (0, 255, 255, 0))
    [[first, second]] = processor(_image_turn(rgb, cmyk))["images"]
    for kept, image in ((first, rgb), (second, cmyk)):
        assert isinstance(kept, Image.Image)
        assert kept is not image
        assert kept.mode == image.mode
        assert list(kept.getdata()) == list(image.getdata())


def test_pil_images_outlive_close(processor: AutoProcessor) -> None:
    image = _png("red")
    out = processor(_image_turn(image))
    image.close()
    [[kept]] = out["images"]
    assert isinstance(kept, Image.Image)
    assert list(kept.getdata()) == list(_png("red").getdata())


def test_pil_images_outlive_their_file(
    processor: AutoProcessor, tmp_path: Path
) -> None:
    path = tmp_path / "red.png"
    path.write_bytes(_png_bytes("red"))
    with Image.open(path) as image:
        out = processor(_image_turn(image))
    # generate() encodes the image after the caller's block closes its file.
    [[kept]] = out["images"]
    assert isinstance(kept, Image.Image)
    assert list(kept.getdata()) == list(_png("red").getdata())


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
        encoded = [image for image in images if isinstance(image, bytes)]
        assert encoded == images
        request = TextGenerationRequest(
            request_id=RequestID(),
            model_name=REPO_ID,
            messages=messages,
            images=encoded,
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
        (
            [{"role": "user", "content": [{"type": "image", "base64": 42}]}],
            TypeError,
            "base64 data of type int",
        ),
        (
            [{"role": "user", "content": [{"type": "image_url"}]}],
            ValueError,
            'no "url" string',
        ),
        (_image_turn("data:image/png;base64"), ValueError, "no payload"),
        (_image_turn("data:image/png;base64,@@@@"), ValueError, "not decode"),
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


@pytest.mark.parametrize(
    "part",
    [
        lambda url: {"type": "image", "image": url},
        lambda url: {"type": "image", "url": url},
        lambda url: {"type": "image", "path": url},
        lambda url: {"type": "image", "image": url.upper()[:4] + url[4:]},
        lambda url: {"type": "image_url", "image_url": {"url": url}},
        lambda url: {"type": "image_url", "image_url": url},
        lambda url: {
            "type": "image",
            "url": url.replace("/red.png", "/redirect"),
        },
        lambda url: {"type": "image", "url": url.replace("red", "red é")},
    ],
)
def test_http_images_are_fetched(
    processor: AutoProcessor,
    image_server: str,
    part: Callable[[str], dict[str, object]],
) -> None:
    out = processor(
        [{"role": "user", "content": [part(f"{image_server}/red.png")]}]
    )
    assert out["images"] == [[_png_bytes("red")]]
    assert out["messages"] == [
        [
            TextGenerationRequestMessage(
                role="user", content=[ImageContentPart()]
            )
        ]
    ]


def test_image_url_parts_keep_their_detail(
    processor: AutoProcessor, image_server: str
) -> None:
    part = {
        "type": "image_url",
        "image_url": {"url": f"{image_server}/red.png", "detail": "low"},
    }
    [[message]] = processor([{"role": "user", "content": [part]}])["messages"]
    assert message.content == [ImageContentPart(detail="low")]


@pytest.mark.parametrize("path", ["missing.png", "truncated.png", "to-file"])
def test_a_url_that_fails_raises_os_error(
    processor: AutoProcessor, image_server: str, path: str
) -> None:
    with pytest.raises(OSError, match=r"Row 0 .* could not be fetched"):
        processor(_image_turn(f"{image_server}/{path}"))


def test_a_closed_port_raises_os_error(processor: AutoProcessor) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(OSError, match=r"Row 0 .* could not be fetched"):
        processor(_image_turn(f"http://127.0.0.1:{port}/red.png"))


def test_a_url_over_the_size_limit_raises(
    processor: AutoProcessor,
    image_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_processor, "_MAX_FETCH_MIB", 1)
    with pytest.raises(ValueError, match="over 1 MiB"):
        processor(_image_turn(f"{image_server}/big.bin"))


def test_a_slow_url_times_out(
    processor: AutoProcessor,
    image_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_processor, "_FETCH_TIMEOUT_SECONDS", 0.1)
    with pytest.raises(OSError, match="could not be fetched") as error:
        processor(_image_turn(f"{image_server}/slow.png"))
    assert isinstance(error.value.__cause__, httpx.TimeoutException)


def test_a_url_that_trickles_bytes_times_out(
    processor: AutoProcessor,
    image_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_processor, "_FETCH_TIMEOUT_SECONDS", 0.25)
    with pytest.raises(OSError, match=r"took over 0\.25 seconds"):
        processor(_image_turn(f"{image_server}/drip.png"))


def test_a_compressed_url_raises_os_error(
    processor: AutoProcessor, image_server: str
) -> None:
    with pytest.raises(OSError, match="Content-Encoding 'gzip'"):
        processor(_image_turn(f"{image_server}/gzip.png"))


_RED_BASE64 = base64.b64encode(_png_bytes("red")).decode()


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        (
            {"type": "image", "image": f"data:image/png;base64,{_RED_BASE64}"},
            _png_bytes("red"),
        ),
        (
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_RED_BASE64}"},
            },
            _png_bytes("red"),
        ),
        ({"type": "image", "base64": _RED_BASE64}, _png_bytes("red")),
        (
            {"type": "image", "base64": f"data:image/png;base64,{_RED_BASE64}"},
            _png_bytes("red"),
        ),
        (
            {"type": "image", "base64": "\n".join(textwrap.wrap(_RED_BASE64))},
            _png_bytes("red"),
        ),
        # The URL-safe alphabet with its padding stripped, for "+/8=".
        ({"type": "image", "base64": "-_8"}, b"\xfb\xff"),
    ],
)
def test_base64_images_decode_locally(
    processor: AutoProcessor, part: dict[str, object], expected: bytes
) -> None:
    out = processor([{"role": "user", "content": [part]}])
    assert out["images"] == [[expected]]


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
