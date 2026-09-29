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
"""MiMo-V2.6-Flash tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.pipelines.lib import TextTokenizer
from transformers import GenerationConfig

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


def generation_eos_token_ids(generation_config: GenerationConfig) -> set[int]:
    """Returns ``generation_config.json``'s ``eos_token_id`` as a set.

    The field may be one id, a list of ids, or unset.
    """
    eos = generation_config.eos_token_id
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)


class MiMoV2Tokenizer(TextTokenizer):
    """:class:`TextTokenizer` that also stops on ``generation_config.json``'s
    ``eos_token_id`` list; ``config.json`` and the tokenizer name only
    ``<|im_end|>``."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        **kwargs,
    ) -> None:
        super().__init__(model_path, pipeline_config, **kwargs)
        self._eos_token_ids.update(
            generation_eos_token_ids(pipeline_config.model.generation_config)
        )
