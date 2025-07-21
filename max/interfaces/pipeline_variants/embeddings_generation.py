# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces and response structures for embedding generation in the MAX API."""

import msgspec
import numpy as np


class EmbeddingsOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Response structure for embedding generation.

    Configuration:
        embeddings: The generated embeddings as a NumPy array.
    """

    embeddings: np.ndarray

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the embedding generation process is complete.

        Returns:
            bool: Always True, as embedding generation is a single-step operation.
        """
        return True
