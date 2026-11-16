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
"""The Inkling2 tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from max.pipelines.lib import TextAndVisionTokenizer, resolve_eos_token_ids
from max.pipelines.lib.tokenizer import (
    ReasoningDelimitersMixin,
    resolve_single_special_token,
    resolve_token_id_bound,
)
from max.pipelines.modeling.types import TokenIds
from transformers import AutoTokenizer

from .model_config import InklingVisionConfig
from .processor import InklingProcessor, load_processor_config

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


# Opens the JSON payload of a tool call. The tool parser matches this exact
# string, so it has to survive detokenization.
TOOL_CALL_JSON_MARKER = "<|content_invoke_tool_json|>"


class InklingTokenizer(ReasoningDelimitersMixin, TextAndVisionTokenizer):
    """Tokenizer for Inkling, whose own ``AutoProcessor`` is not shipped."""

    reasoning_delimiters = ("<|content_thinking|>", "<|end_message|>")

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        huggingface_config = pipeline_config.model.huggingface_config
        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )
        # ``new_context`` bounds every prompt id by the embedding row count;
        # this __init__ replaces the base one, so it resolves the bound itself.
        self._token_id_bound = resolve_token_id_bound(
            self.delegate, pipeline_config
        )

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.processor = InklingProcessor(
            self.delegate,
            load_processor_config(model_path, revision),
            InklingVisionConfig.from_hf(huggingface_config.vision_config),
        )
        self.vision_token_ids = [self.processor.image_token_id]

        self._resolve_reasoning_delimiters(self.delegate)

        tool_call_json_id = resolve_single_special_token(
            self.delegate, TOOL_CALL_JSON_MARKER
        )
        self.skipped_special_token_ids: set[int] = set(
            self.delegate.all_special_ids
        ) - {tool_call_json_id}

    async def decode(self, encoded: TokenIds, **kwargs) -> str:
        """Decodes tokens, dropping every special id except the tool-call marker."""
        # Log-probability responses decode one token id (a plain int) and the
        # CLI passes a token list; normalize both to a rank-1 array.
        token_ids = np.atleast_1d(np.asarray(encoded))

        if not kwargs.get("skip_special_tokens", True):
            return await super().decode(token_ids, **kwargs)

        filtered_ids = [
            token_id
            for token_id in token_ids.tolist()
            if token_id not in self.skipped_special_token_ids
        ]
        return await super().decode(
            np.array(filtered_ids, dtype=np.int64),
            **{**kwargs, "skip_special_tokens": False},
        )
