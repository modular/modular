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
"""Nemotron-H tokenizer."""

from __future__ import annotations

from typing import Any

from max.pipelines.lib import TextTokenizer
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import resolve_single_special_token


class NemotronHTokenizer(TextTokenizer):
    """A text tokenizer that knows Nemotron's reasoning delimiters.

    The reasoning parser needs the ids of ``<think>`` and ``</think>``.
    """

    def __init__(
        self, model_path: str, pipeline_config: PipelineConfig, **kwargs: Any
    ) -> None:
        super().__init__(model_path, pipeline_config, **kwargs)
        self._reasoning_start_token_id = resolve_single_special_token(
            self.delegate, "<think>"
        )
        self._reasoning_end_token_id = resolve_single_special_token(
            self.delegate, "</think>"
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """The id of ``<think>``."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """The id of ``</think>``."""
        return self._reasoning_end_token_id
