# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides low-level Grid Dependent Control primitives for NVIDIA GPUs. 
These instructions are used for control execution of dependent grids.
"""
from sys.info import _is_sm_9x


@always_inline("nodebug")
fn launch_dependent_grids():
    constrained[
        _is_sm_9x(),
        "grid dep control launch is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.griddepcontrol.launch.dependents`[_type=None]()


@always_inline("nodebug")
fn wait_on_dependent_grids():
    constrained[
        _is_sm_9x(),
        "grid dep control wait is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.griddepcontrol.wait`[_type=None]()
