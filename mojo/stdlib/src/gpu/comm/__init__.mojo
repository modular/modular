# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The `gpu.comm` package provides communication primitives for GPUs.

This package includes functions for sending and receiving data between GPUs,
as well as for synchronizing threads across GPUs.
"""

from .allgather import allgather
from .allreduce import allreduce
