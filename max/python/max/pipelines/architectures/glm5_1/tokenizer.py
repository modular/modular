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

"""GLM-4.5+ text tokenizer that exposes its reasoning-delimiter token ids."""

from __future__ import annotations

from typing import Any

from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import (
    TextTokenizer,
    resolve_single_special_token,
)

_THINK_START_TOKEN = "<think>"
_THINK_END_TOKEN = "</think>"


class GlmTokenizer(TextTokenizer):
    """Text tokenizer for GLM-4.5+ (GLM-5.1 / GLM-5.2).

    Identical to :class:`~max.pipelines.lib.tokenizer.TextTokenizer` but also
    implements
    :class:`~max.pipelines.modeling.types.ReasoningPipelineTokenizer` by
    resolving the ``<think>``/``</think>`` delimiter token IDs at construction.
    The overlap (speculative/MTP) text-generation pipeline requires these ids
    on the tokenizer when a ``reasoning_parser`` is configured.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        chat_template: str | None = None,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            enable_llama_whitespace_fix=enable_llama_whitespace_fix,
            chat_template=chat_template,
            **unused_kwargs,
        )
        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_START_TOKEN
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_END_TOKEN
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<think>`` (opens a GLM reasoning span)."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``</think>`` (closes a GLM reasoning span)."""
        return self._reasoning_end_token_id
