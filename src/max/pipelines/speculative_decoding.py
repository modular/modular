# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# mypy: disable-error-code="import-not-found"
"""Speculative Decoding Text Generation Pipeline"""

from typing import Type, TypeVar

from .config import PipelineConfig
from .context import InputContext
from .interfaces import TextGenerationResponse, TokenGenerator
from .pipeline import PipelineModel

T = TypeVar("T", bound=InputContext)


class SpeculativeDecodingTextGenerationPipeline(TokenGenerator[T]):
    """Generalized token generator pipeline with speculative decoding."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: Type[PipelineModel],
        eos_token_id: int,
    ) -> None:
        raise NotImplementedError(
            "init not yet implemented for SpeculativeDecodingTextGenerationPipeline"
        )

    def next_token(
        self, batch: dict[str, T], num_steps: int
    ) -> dict[str, TextGenerationResponse]:
        """Provided a batch, execute both the draft model for num_steps and the target model for num_steps + 1 tokens, accepting final tokens via rejection sampling, returning the variable list of token integers."""
        raise NotImplementedError(
            "next_token not yet implemented for SpeculativeDecodingTextGenerationPipeline"
        )

    def release(self, context: T) -> None:
        """Releases resources associated with this context.

        Args:
            context (TokenGeneratorContext): Finished context.

        """
        raise NotImplementedError(
            "release not yet implemented for SpeculativeDecodingTextGenerationPipeline"
        )
