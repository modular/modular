# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Grid Dependent Control primitives for NVIDIA Hopper (SM90+) GPUs.

This module provides low-level primitives for managing grid dependencies on NVIDIA 
Hopper architecture and newer GPUs. It enables efficient orchestration of multi-grid 
workloads by allowing grids to launch dependent grids and synchronize with them.

The module includes functions that map directly to CUDA grid dependency control 
instructions, providing fine-grained control over grid execution order:

- `launch_dependent_grids()`: Triggers execution of grids that depend on the 
  current grid
- `wait_on_dependent_grids()`: Blocks until all dependent grids complete execution

These primitives are essential for implementing complex GPU execution pipelines where 
multiple kernels need to execute in a specific order with minimal overhead. They 
eliminate the need for host-side synchronization when orchestrating dependent GPU work.
"""
from sys.info import _is_sm_9x


@always_inline("nodebug")
fn launch_dependent_grids():
    """Launches dependent grids that were previously configured to depend on the
    current grid.

    This function triggers the execution of dependent grids that have been configured
    with a dependency on the current grid. It maps directly to the CUDA grid
    dependency control instruction for launching dependent grids.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
        - Must be called by all threads in a thread block to avoid undefined behavior.
        - Typically used in multi-grid pipeline scenarios where one grid's completion
          should trigger the execution of other grids.
    """
    constrained[
        _is_sm_9x(),
        "grid dep control launch is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.griddepcontrol.launch.dependents`[_type=None]()


@always_inline("nodebug")
fn wait_on_dependent_grids():
    """Waits for all dependent grids launched by this grid to complete execution.

    This function blocks the calling grid until all dependent grids that were launched
    by this grid have completed their execution. It provides a synchronization point
    between parent and child grids in a multi-grid dependency chain.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
        - Must be called by all threads in a thread block to avoid undefined behavior.
        - Can be used to ensure dependent grid work is complete before proceeding
          with subsequent operations in the parent grid.
    """
    constrained[
        _is_sm_9x(),
        "grid dep control wait is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.griddepcontrol.wait`[_type=None]()
