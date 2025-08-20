# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces and response structures for embedding generation in the MAX API."""

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import msgspec
import numpy as np
import numpy.typing as npt


class EmbeddingsOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Response structure for embedding generation.

    Configuration:
        embeddings: The generated embeddings as a NumPy array.
    """

    embeddings: npt.NDArray[np.floating[Any]]
    """The generated embeddings as a NumPy array."""

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the embedding generation process is complete.

        Returns:
            bool: Always True, as embedding generation is a single-step operation.
        """
        return True


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
