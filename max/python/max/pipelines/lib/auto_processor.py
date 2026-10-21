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

import io
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from max.pipelines.modeling.types import (
    ImageContentPart,
    TextGenerationRequestMessage,
)
from PIL import Image

from .auto_model import AutoTokenizer

__all__ = ["AutoProcessor", "ProcessorOutput"]

_IMAGE_HINTS = ("detail", "max_long_side_pixel")

# The modes Pillow writes as PNG. Other modes, such as CMYK, convert to RGBA
# when they carry alpha and to RGB otherwise.
_PNG_MODES = frozenset(
    {"1", "L", "LA", "I", "I;16", "I;16B", "P", "RGB", "RGBA"}
)


class ProcessorOutput(TypedDict):
    """The inputs :class:`~max.pipelines.AutoProcessor` returns for a batch.

    Pass them to :meth:`~max.pipelines.AutoModel.generate` as keyword
    arguments.
    """

    messages: list[list[TextGenerationRequestMessage]]
    """One conversation per row, with a placeholder part for each image."""

    images: list[list[bytes]]
    """Each row's images as encoded bytes, in the order of its image parts."""


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
        Face message format. An image part holds a PIL image, encoded bytes
        or a local path (plain or ``file://``) under ``"image"`` or
        ``"path"``, and keeps its ``detail`` and ``max_long_side_pixel``
        hints. Each row's images come out as encoded bytes in the order of
        its image parts.

        Raises:
            TypeError: If a conversation, message, content part or image has
                the wrong type.
            ValueError: If a conversation is empty, a message is invalid, or
                an image part holds no image or names a URL or base64 data.
            OSError: If an image path cannot be read.
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
    images: list[list[bytes]] = []
    for row, conversation in enumerate(_conversation_rows(conversations)):
        row_images: list[bytes] = []
        messages.append(
            [
                _convert_message(message, row, row_images)
                for message in conversation
            ]
        )
        images.append(row_images)
    return {"messages": messages, "images": images}


def _convert_message(
    message: object, row: int, images: list[bytes]
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


def _convert_part(part: object, row: int, images: list[bytes]) -> object:
    if not isinstance(part, Mapping):
        raise TypeError(
            f"Row {row} has a content part of type {type(part).__name__}. "
            "Pass each content part as a dict."
        )
    kind = part.get("type")
    if kind == "image":
        images.append(_image_bytes(part, row))
        return ImageContentPart.model_validate(
            {key: part[key] for key in _IMAGE_HINTS if key in part}
        )
    if kind == "image_url":
        raise _remote_image(row)
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


def _remote_image(row: int) -> ValueError:
    return ValueError(
        f"Row {row} has an image from a URL or base64 data. AutoProcessor "
        "reads PIL images, encoded bytes and local paths. Download or decode "
        "the image and pass its bytes."
    )


def _image_bytes(part: Mapping[object, object], row: int) -> bytes:
    image = part.get("image")
    if image is None:
        image = part.get("path")
    if isinstance(image, Image.Image):
        if image.mode not in _PNG_MODES:
            has_alpha = {"A", "a"} & set(image.getbands())
            image = image.convert("RGBA" if has_alpha else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    if isinstance(image, bytes):
        return image
    if isinstance(image, str) and image.lower().startswith("file://"):
        image = image[len("file://") :]
    if isinstance(image, str) and image.lower().startswith(
        ("http://", "https://", "data:")
    ):
        raise _remote_image(row)
    if isinstance(image, (str, os.PathLike)):
        return Path(image).read_bytes()
    if image is None and ("url" in part or "base64" in part):
        raise _remote_image(row)
    if image is None:
        raise ValueError(
            f'Row {row} has an image part with no "image" or "path" entry.'
        )
    raise TypeError(
        f"Row {row} has an image of type {type(image).__name__}. Pass a PIL "
        "image, encoded bytes or a local path."
    )
