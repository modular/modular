# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides a generic programming model for working with GPUs. This model
dispatches tasks to a 1-, 2-, or 3-dimensional array of threads on the GPU.
Threads are grouped into _thread blocks_, where all of the threads in a block
have a common pool of shared memory that they can use to optimize data transfer.

A _kernel_ is a function that runs on the GPU in parallel across a grid of
threads. Currently, the
[`DeviceContext`](/mojo/stdlib/gpu/host/device_context/DeviceContext) struct
provides the interface for compiling and launching GPU kernels.

See the [`gpu.id`](/mojo/stdlib/gpu/id#aliases) module for a list of aliases you
can use to access information about the grid and the current thread, including
block dimensions, block index in the grid and thread index.

The [`gpu.host`](/mojo/stdlib/gpu/host/) package includes APIs to manage
interaction between the _host_ (that is, the CPU) and _device_ (that is, the GPU
or accelerator).

The [`sync`](/mojo/stdlib/gpu/sync/) module provides functions for synchronizing
threads.

For an example of launching a GPU kernel from a MAX custom operation, see the
[vector addition example](https://github.com/modular/max/blob/main/examples/custom_ops/kernels/vector_addition.mojo)
in the MAX repo.
"""

from .globals import *
from .id import *
from .memory import *
from .semaphore import Semaphore
from .shuffle import *
from .sync import *
