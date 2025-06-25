# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import sys

import numpy as np

# get directory of current file
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_dir)

# Install mojo import hook
import max.mojo.importer

# Imports from 'mojo_module.mojo'
from mojo_module import mojo_block_hasher  # type: ignore


def block_hasher(
    tokens: np.ndarray, block_size: int, parent_hash: int
) -> list[int]:
    """Hash tokens into blocks for prefix caching.

    The token list is partitioned into blocks of size `block_size`. The tokens in
    each block are hashed together with the hash of the previous block.

    This calls into the `mojo_block_hasher` function defined in `mojo_module.mojo`.

    Args:
        tokens: A 1D numpy array of token IDs.
        block_size: The number of tokens per block. Must be greater than 0.
        parent_hash: The hash value of the parent block.

    Returns:
        A list of block hash values.
    """
    if tokens.ndim != 1:
        raise ValueError(
            f"tokens must be a 1D array, found {tokens.ndim}D array"
        )
    if block_size <= 0:
        raise ValueError(
            f"block_size must be greater than 0, found {block_size}"
        )
    # Cast the array to int64 as that is what the mojo block hasher expects.
    if tokens.dtype != np.int64:
        tokens = tokens.astype(np.int64)
    return mojo_block_hasher(tokens, block_size, parent_hash)
