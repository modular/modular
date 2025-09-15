# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces and response structures for embedding generation in the MAX API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import msgspec
import numpy as np
import numpy.typing as npt
from max.interfaces.context import InputContext
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import RequestID

EmbeddingsGenerationContextType = TypeVar(
    "EmbeddingsGenerationContextType", bound=InputContext
)


@dataclass(frozen=True)
class EmbeddingsGenerationInputs(
    PipelineInputs, Generic[EmbeddingsGenerationContextType]
):
    batches: list[dict[RequestID, EmbeddingsGenerationContextType]]

    @property
    def batch(self) -> dict[RequestID, EmbeddingsGenerationContextType]:
        """Returns merged batches."""
        return {k: v for batch in self.batches for k, v in batch.items()}


class EmbeddingsGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
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


def _check_embeddings_output_implements_pipeline_output(
    x: EmbeddingsGenerationOutput,
) -> PipelineOutput:
    return x
