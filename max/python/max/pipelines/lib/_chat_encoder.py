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

"""Loads the ``--tokenizer-impl`` encoder for a checkpoint's chat prompts.

HuggingFace still renders the chat template; the encoder only turns the
rendered prompt into token ids. So it is used only for a checkpoint whose
encoding it reproduces, which it claims by the digest of the
``tokenizer.json`` it encodes as.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from transformers import PreTrainedTokenizerBase
from transformers.utils import cached_file

logger = logging.getLogger("max.pipelines")

# The parts of a ``tokenizer.json`` an encode with ``add_special_tokens=False``
# reads. transformers can change them while loading, for example to add an
# extra special token that ``tokenizer_config.json`` declares.
_ENCODE_SECTIONS = ("added_tokens", "normalizer", "pre_tokenizer", "model")


@runtime_checkable
class ChatEncoder(Protocol):
    """An encoder ``--tokenizer-impl`` may name for rendered chat prompts.

    It is built with no arguments, and encodes text exactly as one
    ``tokenizer.json`` does with ``add_special_tokens=False``.
    """

    def tokenizer_json_sha256(self) -> str:
        """Returns the SHA-256 hex digest of the ``tokenizer.json`` it encodes as."""
        ...

    def encode(self, text: str) -> bytes:
        """Returns the token ids of ``text`` as little-endian u32s.

        Bytes rather than a list, so no Python int is built per id. May
        release the GIL, and is called from several threads at once.
        """
        ...


def _why_not_eligible(
    encoder: object,
    delegate: PreTrainedTokenizerBase,
    model_path: str,
    revision: str | None,
) -> str | None:
    """Returns why ``encoder`` cannot encode this checkpoint's prompts."""
    if not isinstance(encoder, ChatEncoder):
        return "it is not a chat encoder"
    try:
        path = cached_file(model_path, "tokenizer.json", revision=revision)
    except OSError:
        path = None
    if path is None:
        return "the checkpoint has no tokenizer.json"
    on_disk = Path(path).read_bytes()
    if hashlib.sha256(on_disk).hexdigest() != encoder.tokenizer_json_sha256():
        return "the checkpoint's tokenizer.json is not the one it encodes as"
    backend = getattr(delegate, "backend_tokenizer", None)
    if backend is None or delegate.split_special_tokens:
        return "the HuggingFace tokenizer does not encode as its tokenizer.json"
    loaded, saved = json.loads(backend.to_str()), json.loads(on_disk)
    if any(loaded.get(key) != saved.get(key) for key in _ENCODE_SECTIONS):
        return "the HuggingFace tokenizer does not encode as its tokenizer.json"
    return None


def load_chat_encoder(
    tokenizer_impl: str | None,
    delegate: PreTrainedTokenizerBase,
    model_path: str,
    revision: str | None,
) -> ChatEncoder | None:
    """Builds the ``--tokenizer-impl`` chat encoder, when it can serve.

    Returns ``None``, so chat prompts encode with the HuggingFace tokenizer,
    when the flag is unset, its class cannot be imported or built, or it does
    not encode exactly as this checkpoint's tokenizer.
    """
    if tokenizer_impl is None:
        return None
    try:
        module_path, sep, class_name = tokenizer_impl.partition(":")
        if not sep:
            raise ValueError("expected 'module.path:ClassName'")
        encoder = getattr(importlib.import_module(module_path), class_name)()
    except Exception:
        logger.warning(
            "Tokenizer %r unavailable; chat prompts encode with the "
            "HuggingFace tokenizer.",
            tokenizer_impl,
            exc_info=True,
        )
        return None
    try:
        reason = _why_not_eligible(encoder, delegate, model_path, revision)
    except Exception as e:
        reason = f"checking it raised {type(e).__name__}: {e}"
    if reason is not None:
        logger.warning(
            "Not using tokenizer %r because %s; chat prompts encode with the "
            "HuggingFace tokenizer.",
            tokenizer_impl,
            reason,
        )
        return None
    logger.info("Chat prompts encode with tokenizer %r.", tokenizer_impl)
    return encoder
