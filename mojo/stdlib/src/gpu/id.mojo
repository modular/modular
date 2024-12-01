# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs id operations."""

from os import abort
from sys import is_nvidia_gpu, llvm_intrinsic

from gpu import WARP_SIZE
from math import fma

# ===----------------------------------------------------------------------===#
# ThreadIdx
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct _ThreadIdx:
    """ThreadIdx provides static methods for getting the x/y/z coordinates of
    a thread within a block."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StringLiteral:
        @parameter
        if is_nvidia_gpu():
            return "llvm.nvvm.read.ptx.sreg.tid." + dim
        else:
            return "llvm.amdgcn.workitem.id." + dim

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a thread within a block.

        Returns:
            The `x`, `y`, or `z` coordinates of a thread within a block.
        """
        constrained[
            dim in ("x", "y", "z"), "the accessor must be either x, y, or z"
        ]()
        alias intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            int(llvm_intrinsic[intrinsic_name, Int32, has_side_effect=False]())
        )


alias ThreadIdx = _ThreadIdx()


# ===----------------------------------------------------------------------===#
# BlockIdx
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct _BlockIdx:
    """BlockIdx provides static methods for getting the x/y/z coordinates of
    a block within a grid."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StringLiteral:
        @parameter
        if is_nvidia_gpu():
            return "llvm.nvvm.read.ptx.sreg.ctaid." + dim
        else:
            return "llvm.amdgcn.workgroup.id." + dim

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a block within a grid.

        Returns:
            The `x`, `y`, or `z` coordinates of a block within a grid.
        """
        constrained[
            dim in ("x", "y", "z"), "the accessor must be either x, y, or z"
        ]()
        alias intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            int(llvm_intrinsic[intrinsic_name, Int32, has_side_effect=False]())
        )


alias BlockIdx = _BlockIdx()

# ===----------------------------------------------------------------------===#
# BlockDim
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_gcn_idx[offset: Int]() -> UInt:
    var ptr = llvm_intrinsic[
        "llvm.amdgcn.implicitarg.ptr",
        UnsafePointer[Int16, address_space=4],
        has_side_effect=False,
    ]()
    return UInt(int(ptr.load[alignment=4](offset)))


@register_passable("trivial")
struct _BlockDim:
    """BlockDim provides static methods for getting the x/y/z dimension of a
    block."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the block.

        Returns:
            The `x`, `y`, or `z` dimension of the block.
        """
        constrained[
            dim in ("x", "y", "z"), "the accessor must be either x, y, or z"
        ]()

        @parameter
        if is_nvidia_gpu():
            alias intrinsic_name = "llvm.nvvm.read.ptx.sreg.ntid." + dim
            return UInt(
                int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
        else:

            @parameter
            fn _get_offset() -> Int:
                @parameter
                if dim == "x":
                    return 6
                elif dim == "y":
                    return 7
                else:
                    constrained[dim == "z"]()
                    return 8

            return _get_gcn_idx[_get_offset()]()


alias BlockDim = _BlockDim()

# ===----------------------------------------------------------------------===#
# GridDim
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct _GridDim:
    """GridDim provides static methods for getting the x/y/z dimension of a
    grid."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the grid.

        Returns:
            The `x`, `y`, or `z` dimension of the grid.
        """
        constrained[
            dim in ("x", "y", "z"), "the accessor must be either x, y, or z"
        ]()

        @parameter
        if is_nvidia_gpu():
            alias intrinsic_name = "llvm.nvvm.read.ptx.sreg.nctaid." + dim
            return UInt(
                int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
        else:

            @parameter
            fn _get_offset() -> Int:
                @parameter
                if dim == "x":
                    return 0
                elif dim == "y":
                    return 1
                else:
                    constrained[dim == "z"]()
                    return 2

            return _get_gcn_idx[_get_offset()]()


alias GridDim = _GridDim()

# ===----------------------------------------------------------------------===#
# GridDim
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct _GlobalIdx:
    """Global provides static methods for getting the x/y/z global offset of
    the kernel launch."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the program.

        Returns:
            The `x`, `y`, or `z` dimension of the program.
        """
        constrained[
            dim in ("x", "y", "z"), "the accessor must be either x, y, or z"
        ]()
        var thread_idx = ThreadIdx.__getattr__[dim]()
        var block_idx = BlockIdx.__getattr__[dim]()
        var block_dim = BlockDim.__getattr__[dim]()

        return fma(block_idx, block_dim, thread_idx)


alias GlobalIdx = _GlobalIdx()

# ===----------------------------------------------------------------------===#
# lane_id
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn lane_id() -> UInt:
    """Returns the lane ID of the current thread.

    Returns:
        The lane ID of the the current thread.
    """

    @parameter
    if is_nvidia_gpu():
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.laneid",
                    Int32,
                    has_side_effect=False,
                ]().cast[DType.uint32]()
            )
        )

    else:
        alias none = Scalar[DType.int32](-1)
        alias zero = Scalar[DType.int32](0)
        var t = llvm_intrinsic[
            "llvm.amdgcn.mbcnt.lo", Int32, has_side_effect=False
        ](none, zero)
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.amdgcn.mbcnt.hi", Int32, has_side_effect=False
                ](none, t).cast[DType.uint32]()
            )
        )


# ===----------------------------------------------------------------------===#
# sm_id
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn sm_id() -> UInt:
    """Returns the SM ID of the current thread.
    Returns:
        The SM ID of the the current thread.
    """

    @parameter
    if is_nvidia_gpu():
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.smid", Int32, has_side_effect=False
                ]().cast[DType.uint32]()
            )
        )
    else:
        constrained[False, "The sm_id function is not supported by AMD GPUs."]()
        return abort[Int]("function not available")
