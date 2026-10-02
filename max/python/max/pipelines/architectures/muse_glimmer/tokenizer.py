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
"""Muse Glimmer tokenizer: the shared text-and-vision one on our own processor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from max.pipelines.lib import TextAndVisionTokenizer
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import (
    resolve_eos_token_ids,
    resolve_single_special_token,
)
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# The template opens every channel body with ``<|message|>`` and closes a
# non-final channel, the reasoning one included, with ``<|eom|>``. ``<|eom|>``
# must never stop generation: the answer channel follows it in the same turn.
_MESSAGE_TOKEN = "<|message|>"
_END_MESSAGE_TOKEN = "<|eom|>"

_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


def to_reasoning_strength(options: Mapping[str, Any]) -> dict[str, Any]:
    """Maps OpenAI's ``reasoning_effort`` onto the template's
    ``reasoning_strength`` and drops the thinking toggles it does not read.

    ``xhigh`` is only reachable as ``reasoning_strength`` in
    ``chat_template_kwargs``; any other effort falls back to the template's
    default (``high``).
    """
    options = dict(options)
    effort = options.pop("reasoning_effort", None)
    options.pop("enable_thinking", None)
    options.pop("thinking", None)
    if effort in _REASONING_EFFORTS and "reasoning_strength" not in options:
        options["reasoning_strength"] = effort
    return options


class MuseGlimmerProcessor:
    """Stands in for ``transformers``' ``MuseGlimmerProcessor``, which the
    installed version does not ship, in the shape
    :class:`~max.pipelines.lib.TextAndVisionTokenizer` expects. Text only for
    now."""

    def __init__(self, delegate: PreTrainedTokenizerBase) -> None:
        self.delegate = delegate

    def apply_chat_template(self, messages: Any, **options: Any) -> str:
        templated = self.delegate.apply_chat_template(messages, **options)
        assert isinstance(templated, str)
        return templated

    def __call__(
        self,
        *,
        text: str | Sequence[int],
        images: Sequence[Image.Image] | None = None,
        add_special_tokens: bool = True,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        if images:
            # TODO: run the Muse Glimmer image processor.
            raise NotImplementedError("Muse Glimmer image inputs")
        token_ids = (
            self.delegate.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(text, str)
            else list(text)
        )
        return {"input_ids": [token_ids]}


class MuseGlimmerTokenizer(TextAndVisionTokenizer):
    """Tokenizer for Muse Glimmer.

    Stops on the ``eos_token_id`` list in ``generation_config.json``, the only
    file that names ``<|eot|>``, and exposes the reasoning-delimiter ids the
    ``ReasoningPipelineTokenizer`` protocol asks for.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
        **unused_kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        if chat_template is not None:
            self.delegate.chat_template = chat_template
        self.max_length = max_length or self.delegate.model_max_length

        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )

        huggingface_config = pipeline_config.model.huggingface_config
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.processor = MuseGlimmerProcessor(self.delegate)
        self.vision_token_ids = [
            huggingface_config.image_token_id,
            huggingface_config.video_token_id,
        ]

        self._reasoning_start_token_id = resolve_single_special_token(
            self.delegate, _MESSAGE_TOKEN
        )
        self._reasoning_end_token_id = resolve_single_special_token(
            self.delegate, _END_MESSAGE_TOKEN
        )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the chat template with ``reasoning_effort`` translated."""
        return super().apply_chat_template(
            messages, tools, **to_reasoning_strength(chat_template_options)
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<|message|>``."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``<|eom|>``."""
        return self._reasoning_end_token_id
