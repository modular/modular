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

"""Turns chat conversations with inline images into AutoModel inputs."""

from __future__ import annotations

import base64
import binascii
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import httpx
from max.pipelines.modeling.types import (
    ImageContentPart,
    TextGenerationRequestMessage,
)
from PIL import Image

from .auto_model import AutoTokenizer

__all__ = ["AutoProcessor", "ProcessorOutput"]

_IMAGE_HINTS = ("detail", "max_long_side_pixel")

# The timeout bounds each image's whole fetch, so a slow host cannot hang a
# batch, and the cap bounds the memory each image takes.
_FETCH_CONNECT_TIMEOUT_SECONDS = 10.0
_FETCH_TIMEOUT_SECONDS = 30.0
_MAX_FETCH_MIB = 100
_MAX_FETCH_REDIRECTS = 5
# Some hosts reject a library's default User-Agent with HTTP 403.
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    # A compressed body could expand far past the cap in one decode step.
    "Accept-Encoding": "identity",
}


class ProcessorOutput(TypedDict):
    """The inputs :class:`~max.pipelines.AutoProcessor` returns for a batch.

    Pass them to :meth:`~max.pipelines.AutoModel.generate` as keyword
    arguments.
    """

    messages: list[list[TextGenerationRequestMessage]]
    """One conversation per row, with a placeholder part for each image."""

    images: list[list[bytes | Image.Image]]
    """Each row's images as encoded bytes or PIL images, in the order of its
    image parts."""


class AutoProcessor:
    """Prepares chat conversations with images for :class:`AutoModel`.

    Calling it returns the ``messages`` and ``images`` that
    :meth:`AutoModel.generate` takes, not Hugging Face's ``input_ids`` and
    ``pixel_values``. The model's own tokenizer applies the chat template and
    expands each image at generate time, as serve does. Decoding uses the
    model's Hugging Face tokenizer.

    .. Skipped: the example downloads a model.
    .. skip: next

    .. code-block:: python

        from PIL import Image

        processor = AutoProcessor("google/gemma-3-4b-it")
        model = AutoModel("google/gemma-3-4b-it", max_batch_size=2)
        conversations = [
            [{"role": "user", "content": [
                {"type": "image", "image": Image.open("cat.png")},
                {"type": "text", "text": "Describe this image."},
            ]}],
            [{"role": "user", "content": "What is 2 + 2?"}],
        ]
        outputs = model.generate(**processor(conversations), max_new_tokens=32)
        new_tokens = [
            sequence[length:]
            for sequence, length in zip(
                outputs.sequences, outputs.prompt_lengths
            )
        ]
        print(processor.batch_decode(new_tokens, skip_special_tokens=True))

    Args:
        model_path: A Hugging Face repo ID or a local model directory.
        revision: The repo revision to load.
        trust_remote_code: Whether to run tokenizer code from the repo.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self._tokenizer = AutoTokenizer(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def __call__(
        self,
        conversations: Sequence[Mapping[str, object]]
        | Sequence[Sequence[Mapping[str, object]]],
    ) -> ProcessorOutput:
        """Returns ``messages`` and ``images`` with one row per conversation.

        ``conversations`` is one conversation or a list of them, in Hugging
        Face message format. An image part holds a PIL image, bytes, a local
        path, an http(s) URL or a ``data:`` URI under ``"image"``, ``"path"``
        or ``"url"``, or base64 under ``"base64"``. OpenAI ``image_url``
        parts work too. An image part keeps its ``detail`` and
        ``max_long_side_pixel`` hints. Each row's images come out in the
        order of its image parts, a PIL image as a copy and anything else
        as encoded bytes.

        URLs are fetched as given, with no host checks, with a 30 second
        timeout and a 100 MiB cap per image.

        Raises:
            TypeError: If a conversation, message, content part or image has
                the wrong type.
            ValueError: If a conversation is empty, a message is invalid, an
                image part holds no image, a fetched image is over the size
                limit, or base64 data does not decode.
            OSError: If an image path cannot be read or a URL cannot be
                fetched.
        """
        return _model_inputs(conversations)

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        """Returns the text for ``token_ids``."""
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        sequences: Iterable[Sequence[int]],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Returns the text for each token-id list in ``sequences``."""
        return self._tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )


def _model_inputs(conversations: object) -> ProcessorOutput:
    messages: list[list[TextGenerationRequestMessage]] = []
    images: list[list[bytes | Image.Image]] = []
    for row, conversation in enumerate(_conversation_rows(conversations)):
        row_images: list[bytes | Image.Image] = []
        messages.append(
            [
                _convert_message(message, row, row_images)
                for message in conversation
            ]
        )
        images.append(row_images)
    return {"messages": messages, "images": images}


def _convert_message(
    message: object, row: int, images: list[bytes | Image.Image]
) -> TextGenerationRequestMessage:
    if not isinstance(message, Mapping):
        raise TypeError(
            f"Row {row} has a message of type {type(message).__name__}. "
            "Pass each message as a dict."
        )
    fields = dict(message)
    content = fields.get("content")
    if isinstance(content, Sequence) and not isinstance(content, str):
        fields["content"] = [
            _convert_part(part, row, images) for part in content
        ]
    return TextGenerationRequestMessage.model_validate(fields)


def _convert_part(
    part: object, row: int, images: list[bytes | Image.Image]
) -> object:
    if not isinstance(part, Mapping):
        raise TypeError(
            f"Row {row} has a content part of type {type(part).__name__}. "
            "Pass each content part as a dict."
        )
    kind = part.get("type")
    if kind == "image":
        images.append(_image(part, row))
        return ImageContentPart.model_validate(
            {key: part[key] for key in _IMAGE_HINTS if key in part}
        )
    if kind == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, Mapping):
            fields, url = image_url, image_url.get("url")
        else:
            fields, url = {}, image_url
        if not isinstance(url, str):
            raise ValueError(
                f'Row {row} has an image_url part with no "url" string.'
            )
        images.append(_read_source(url, row))
        return ImageContentPart.model_validate(
            {key: fields[key] for key in _IMAGE_HINTS if key in fields}
        )
    # The message model checks text and video parts and rejects other types.
    return dict(part)


def _conversation_rows(
    conversations: object,
) -> list[Sequence[object]]:
    if isinstance(conversations, (str, bytes)) or not isinstance(
        conversations, Sequence
    ):
        raise TypeError(
            "conversations must be a conversation or a list of conversations."
        )
    if not conversations:
        raise ValueError("conversations must hold at least one message.")
    if isinstance(conversations[0], Mapping):
        return [conversations]
    rows: list[Sequence[object]] = []
    for row, conversation in enumerate(conversations):
        if isinstance(conversation, (str, bytes)) or not isinstance(
            conversation, Sequence
        ):
            raise TypeError(
                "conversations must be a conversation or a list of "
                f"conversations, and row {row} is a "
                f"{type(conversation).__name__}."
            )
        if not conversation:
            raise ValueError(f"Row {row} holds no messages.")
        rows.append(conversation)
    return rows


def _image(part: Mapping[object, object], row: int) -> bytes | Image.Image:
    image = part.get("image")
    for key in ("path", "url"):
        if image is None:
            image = part.get(key)
    if image is None:
        encoded = part.get("base64")
        if encoded is None:
            raise ValueError(
                f'Row {row} has an image part with no "image", "path", "url" '
                'or "base64" entry.'
            )
        if not isinstance(encoded, str):
            raise TypeError(
                f"Row {row} has base64 data of type {type(encoded).__name__}. "
                "Pass it as a string."
            )
        if encoded.lower().startswith("data:"):
            return _read_source(encoded, row)
        return _decode_base64(encoded, row)
    if isinstance(image, Image.Image):
        # The copy holds its own pixels, so a caller may close or edit the
        # image before generate() encodes it.
        return image.copy()
    if isinstance(image, bytes):
        return image
    if isinstance(image, str):
        return _read_source(image, row)
    if isinstance(image, os.PathLike):
        return Path(image).read_bytes()
    raise TypeError(
        f"Row {row} has an image of type {type(image).__name__}. Pass a PIL "
        "image, encoded bytes, a local path or a URL."
    )


def _read_source(source: str, row: int) -> bytes:
    lowered = source.lower()
    if lowered.startswith(("http://", "https://")):
        return _fetch(source, row)
    if lowered.startswith("data:"):
        _, comma, payload = source.partition(",")
        if not comma:
            raise ValueError(f"Row {row} has a data URI with no payload.")
        return _decode_base64(payload, row)
    if lowered.startswith("file://"):
        source = source[len("file://") :]
    return Path(source).read_bytes()


def _fetch(url: str, row: int) -> bytes:
    max_bytes = _MAX_FETCH_MIB * 1024 * 1024
    deadline = time.monotonic() + _FETCH_TIMEOUT_SECONDS
    body = bytearray()
    try:
        with (
            httpx.Client(
                headers=_FETCH_HEADERS,
                timeout=httpx.Timeout(
                    _FETCH_TIMEOUT_SECONDS,
                    connect=_FETCH_CONNECT_TIMEOUT_SECONDS,
                ),
                follow_redirects=True,
                max_redirects=_MAX_FETCH_REDIRECTS,
            ) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            encoding = response.headers.get("Content-Encoding", "")
            if encoding.strip().lower() not in ("", "identity"):
                raise OSError(
                    f"Row {row} has an image URL served with Content-Encoding "
                    f"{encoding!r}, which AutoProcessor does not decode: {url}"
                )
            # Without a chunk size, each network read comes back as it
            # arrives, so the deadline is checked even while bytes trickle in.
            for chunk in response.iter_raw():
                body += chunk
                if len(body) > max_bytes:
                    raise ValueError(
                        f"Row {row} has an image URL whose body is over "
                        f"{_MAX_FETCH_MIB} MiB: {url}"
                    )
                if time.monotonic() > deadline:
                    raise OSError(
                        f"Row {row} has an image URL that took over "
                        f"{_FETCH_TIMEOUT_SECONDS:g} seconds to fetch: {url}"
                    )
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        # httpx raises for a body cut short of its Content-Length too, so a
        # truncated image never comes back as partial bytes.
        raise OSError(
            f"Row {row} has an image URL that could not be fetched: {url}"
        ) from error
    return bytes(body)


def _decode_base64(encoded: str, row: int) -> bytes:
    # As max serve does, this accepts line breaks, stripped padding and the
    # URL-safe alphabet.
    encoded = "".join(encoded.split())
    encoded += "=" * (-len(encoded) % 4)
    url_safe = "-" in encoded or "_" in encoded
    try:
        return base64.b64decode(
            encoded, altchars=b"-_" if url_safe else None, validate=True
        )
    except binascii.Error as error:
        raise ValueError(
            f"Row {row} has base64 image data that does not decode."
        ) from error
