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
import max._mojo.mojo_importer

# Imports from 'mojo_module.mojo'
from mojo_module import mojo_block_hasher  # type: ignore


def block_hasher(tokens: np.ndarray, block_size: int) -> list[int]:
    return mojo_block_hasher(tokens, block_size)
