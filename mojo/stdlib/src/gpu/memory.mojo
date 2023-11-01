# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs memory operations."""

from math import is_power_of_2
from sys.info import alignof, simdwidthof, sizeof, triple_is_nvidia_cuda

from memory import stack_allocation as _generic_stack_allocation
from memory.unsafe import DTypePointer


# ===----------------------------------------------------------------------===#
# Address Space
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct AddressSpace:
    var _value: Int

    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    alias GENERIC = AddressSpace(0)
    """Generic address space."""
    alias GLOBAL = AddressSpace(1)
    """Global address space."""
    alias CONSTANT = AddressSpace(2)
    """Constant address space."""
    alias SHARED = AddressSpace(3)
    """Shared address space."""
    alias PARAM = AddressSpace(4)
    """Param address space."""
    alias LOCAL = AddressSpace(5)
    """Local address space."""

    @always_inline("nodebug")
    fn __init__(value: Int) -> Self:
        return Self {_value: value}

    @always_inline("nodebug")
    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("nodebug")
    fn __eq__(self, other: AddressSpace) -> Bool:
        """The True if the two address spaces are equal and False otherwise.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() == other.value()
