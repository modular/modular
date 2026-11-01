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
"""The Phi-3 text tokenizer."""

from __future__ import annotations

from max.pipelines.lib import PipelineConfig, TextTokenizer


class Phi3Tokenizer(TextTokenizer):
    """A text tokenizer with the Llama whitespace fix on by default.

    Phi-3 checkpoints ship a Llama SentencePiece tokenizer, which drops the
    leading space of a token decoded on its own.
    """

    # The registry used to turn the fix on by matching the pipeline model's
    # class name, which a module registration no longer has.
    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = True,
        chat_template: str | None = None,
        **unused_kwargs: object,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            enable_llama_whitespace_fix=enable_llama_whitespace_fix,
            chat_template=chat_template,
        )
