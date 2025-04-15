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

"""Interfaces for embeddings generation pipeline behaviors."""

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

EmbeddingsGeneratorContext = TypeVar("EmbeddingsGeneratorContext")


@runtime_checkable
class EmbeddingsGenerator(Generic[EmbeddingsGeneratorContext], Protocol):
    """Interface for LLM embeddings-generator models."""

    def encode(
        self, batch: dict[str, EmbeddingsGeneratorContext]
    ) -> dict[str, Any]:
        """Computes embeddings for a batch of inputs.

        Args:
            batch (dict[str, EmbeddingsGeneratorContext]): Batch of contexts to generate
                embeddings for.

        Returns:
            dict[str, Any]: Dictionary mapping request IDs to their corresponding
                embeddings. Each embedding is typically a numpy array or tensor of
                floating point values.
        """
        ...
