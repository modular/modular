# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces and response structures for embedding generation in the MAX API."""

import msgspec
import numpy as np


class EmbeddingsResponse(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Response structure for embedding generation.

    Configuration:
        embeddings: The generated embeddings as a NumPy array.
    """

    embeddings: np.ndarray
