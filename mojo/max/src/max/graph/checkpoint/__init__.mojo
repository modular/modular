# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""APIs to save and load checkpoints for MAX graphs."""

from .metadata import current_version, read_version, VersionInfo
from .save_load import load, save
from .tensor_dict import TensorDict
