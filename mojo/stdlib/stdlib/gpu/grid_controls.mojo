# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from sys import env_get_int, has_nvidia_gpu_accelerator

from .host.info import DEFAULT_GPU, H100
from .host.launch_attribute import (
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)

alias _ENABLE_PDL_LAUNCH = _enable_pdl_launch()


@doc_private
@always_inline("nodebug")
fn _enable_pdl_launch() -> Bool:
    """Determines if programmatic dependency launch (PDL) is enabled.

    Checks if the current GPU supports PDL (Hopper SM90+ architecture) and if
    the feature is explicitly enabled via environment variable. Returns False
    for unsupported GPUs or when PDL is disabled.

    Returns:
        True if PDL is supported and enabled, False otherwise.
    """

    if not has_nvidia_gpu_accelerator():
        return False

    if DEFAULT_GPU < H100:
        return False

    if PDLLevel() == PDLLevel.OFF:
        return False

    return True


@doc_private
@always_inline("nodebug")
fn pdl_launch_attributes() -> List[LaunchAttribute]:
    """Returns launch attributes for programmatic dependency launch (PDL).

    This function configures launch attributes to enable programmatic stream
    serialization on supported GPUs. When PDL is enabled, it returns a list
    containing a single launch attribute that enables grid dependency control.

    Returns:
        A list of launch attributes. Contains the PDL attribute if enabled,
        otherwise returns an empty list.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
        - When disabled, returns an empty list for compatibility with older GPUs.
    """

    @parameter
    if _ENABLE_PDL_LAUNCH:
        return List[LaunchAttribute](
            LaunchAttribute(
                LaunchAttributeID.PROGRAMMATIC_STREAM_SERIALIZATION,
                LaunchAttributeValue(True),
            )
        )
    else:
        return List[LaunchAttribute]()


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

    @parameter
    if _ENABLE_PDL_LAUNCH:
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

    @parameter
    if _ENABLE_PDL_LAUNCH:
        __mlir_op.`nvvm.griddepcontrol.wait`[_type=None]()


@register_passable("trivial")
struct PDLLevel:
    """Programmatic Dependency Launch (PDL) level."""

    var _level: Int

    alias OFF = PDLLevel(0)
    alias OVERLAP_AT_END = PDLLevel(1)
    alias OVERLAP_AT_BEGINNING = PDLLevel(2)
    alias NO_WAIT_OVERLAP_AT_END = PDLLevel(3)

    @always_inline
    fn __init__(out self):
        """Initialize the PDL level to OFF."""
        self = PDLLevel(env_get_int["PDL_LEVEL", 0]())

    @always_inline
    fn __init__(out self, level: Int):
        """Initialize the PDL level.

        Args:
            level: The PDL level to initialize.
        """
        self._level = level

    @always_inline
    fn __eq__(self, other: PDLLevel) -> Bool:
        """Check if the PDL level is equal to another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is equal to the other PDL level, False otherwise.
        """
        return self._level == other._level

    @always_inline
    fn __eq__(self, other: Int) -> Bool:
        """Check if the PDL level is equal to another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is equal to the other PDL level, False otherwise.
        """
        return self._level == other

    @always_inline
    fn __ne__(self, other: PDLLevel) -> Bool:
        """Check if the PDL level is not equal to another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is not equal to the other PDL level, False otherwise.
        """
        return self._level != other._level

    @always_inline
    fn __gt__(self, other: PDLLevel) -> Bool:
        """Check if the PDL level is greater than another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is greater than the other PDL level, False otherwise.
        """
        return self._level > other._level

    @always_inline
    fn __ge__(self, other: PDLLevel) -> Bool:
        """Check if the PDL level is greater than or equal to another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is greater or equal to the other PDL level,
            False otherwise.
        """
        return self._level >= other._level


struct PDL:
    """Programmatic Dependency Launch (PDL) control structure.

    This struct provides a way to manage programmatic stream serialization on
    NVIDIA GPUs. It includes functions for launching dependent grids and waiting
    for them to complete.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
    """

    @always_inline
    fn __init__(out self):
        """Initialize the PDL control structure."""
        pass

    @always_inline
    fn __enter__(self):
        """Launch dependent grids that were previously configured to depend on the
        current grid."""
        wait_on_dependent_grids()

    @always_inline
    fn __exit__(self):
        """Wait for all dependent grids launched by this grid to complete execution.
        """
        launch_dependent_grids()
