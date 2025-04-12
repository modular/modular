# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides low-level programming constructs for working with GPUs.

These low level constructs allow you to write code that runs on the GPU with
traditional programming style--partitioning work across threads that are mapped
onto 1-, 2-, or 3-dimensional blocks. The thread blocks can subsequently be
grouped into a grid of thread blocks.

A _kernel_ is a function that runs on the GPU in parallel across many threads.
Currently, the
[`DeviceContext`](/mojo/stdlib/gpu/host/device_context/DeviceContext) struct
provides the interface for compiling and launching GPU kernels inside MAX
[custom operations](/max/custom-ops/).

The [`gpu.host`](/mojo/stdlib/gpu/host/) package includes APIs to manage
interaction between the _host_ (that is, the CPU) and _device_ (that is, the GPU
or accelerator).

See the [`gpu.id`](/mojo/stdlib/gpu/id#aliases) module for a list of aliases you
can use to access information about the grid and the current thread, including
block dimensions, block index in the grid and thread index.

The [`sync`](/mojo/stdlib/gpu/sync/) module provides functions for synchronizing
threads.

For an example of launching a GPU kernel from a MAX custom operation, see the
[vector addition example](https://github.com/modular/max/blob/main/examples/custom_ops/kernels/vector_addition.mojo)
in the MAX repo.
"""

from .cluster import *
from .globals import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE
from .grid_controls import *
from .id import *
from .memory import *
from .semaphore import Semaphore
from .sync import *
