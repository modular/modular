# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs."""

from DType import DType
from Intrinsics import llvm_intrinsic
from Memory import stack_allocation as _stack_allocation
from Pointer import DTypePointer
from SIMD import Int32


# ===----------------------------------------------------------------------===#
# Address Space
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct AddressSpace:
    var _value: Int

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

    fn __init__(value: Int) -> Self:
        return Self {_value: value}

    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value


# ===----------------------------------------------------------------------===#
# stack allocation
# ===----------------------------------------------------------------------===#


@always_inline
fn stack_allocation[
    count: Int, type: DType, alignment: Int, address_space: AddressSpace
]() -> DTypePointer[type]:
    return _stack_allocation[count, type, alignment, address_space.value()]()


@always_inline
fn stack_allocation[
    count: Int, type: DType, address_space: AddressSpace
]() -> DTypePointer[type]:
    return _stack_allocation[count, type, 1, address_space.value()]()


# ===----------------------------------------------------------------------===#
# ThreadIdx
# ===----------------------------------------------------------------------===#


struct ThreadIdx:
    """ThreadIdx provides static methods for getting the x/y/z coordinates of
    a thread within a block."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` coordinate of the thread within the block.

        Returns: The `x` coordinate within the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.tid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` coordinate of the thread within the block.

        Returns: The `y` coordinate within the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.tid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` coordinate of the thread within the block.

        Returns: The `z` coordinate within the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.tid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# BlockIdx
# ===----------------------------------------------------------------------===#


struct BlockIdx:
    """BlockIdx provides static methods for getting the x/y/z coordinates of
    a block within a grid."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` coordinate of the block within a grid.

        Returns: The `x` coordinate within the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ctaid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` coordinate of the block within a grid.

        Returns: The `y` coordinate within the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ctaid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` coordinate of the block within a grid.

        Returns: The `z` coordinate within the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ctaid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# BlockDim
# ===----------------------------------------------------------------------===#


struct BlockDim:
    """BlockDim provides static methods for getting the x/y/z dimension of a
    block."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` dimension of the block.

        Returns: The `x` dimension of the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ntid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` dimension of the block.

        Returns: The `y` dimension of the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ntid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` dimension of the block.

        Returns: The `z` dimension of the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ntid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# GridDim
# ===----------------------------------------------------------------------===#


struct GridDim:
    """GridDim provides static methods for getting the x/y/z dimension of a
    grid."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` dimension of the grid.

        Returns: The `x` dimension of the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.nctaid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` dimension of the grid.

        Returns: The `y` dimension of the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.nctaid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` dimension of the grid.

        Returns: The `z` dimension of the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.nctaid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# barrier
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn barrier():
    """Performs a synchronization barrier on block (equivelent to `__syncthreads`
    in CUDA).
    """
    llvm_intrinsic["llvm.nvvm.barrier0", NoneType]()
