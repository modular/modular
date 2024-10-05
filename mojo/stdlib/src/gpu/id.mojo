# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs id operations."""

from sys import llvm_intrinsic, triple_is_nvidia_cuda

# ===----------------------------------------------------------------------===#
# ThreadIdx
# ===----------------------------------------------------------------------===#


struct ThreadIdx:
    """ThreadIdx provides static methods for getting the x/y/z coordinates of
    a thread within a block."""

    @always_inline
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StringLiteral:
        if triple_is_nvidia_cuda():
            return "llvm.nvvm.read.ptx.sreg.tid." + dim
        else:
            return "llvm.amdgcn.workitem.id." + dim

    @staticmethod
    @always_inline("nodebug")
    fn x() -> UInt:
        """Gets the `x` coordinate of the thread within the block.

        Returns:
            The `x` coordinate within the block.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    Self._get_intrinsic_name["x"](),
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn y() -> UInt:
        """Gets the `y` coordinate of the thread within the block.

        Returns:
            The `y` coordinate within the block.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    Self._get_intrinsic_name["y"](),
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn z() -> UInt:
        """Gets the `z` coordinate of the thread within the block.

        Returns:
            The `z` coordinate within the block.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    Self._get_intrinsic_name["z"](),
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )


# ===----------------------------------------------------------------------===#
# BlockIdx
# ===----------------------------------------------------------------------===#


struct BlockIdx:
    """BlockIdx provides static methods for getting the x/y/z coordinates of
    a block within a grid."""

    @always_inline
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StringLiteral:
        if triple_is_nvidia_cuda():
            return "llvm.nvvm.read.ptx.sreg.ctaid." + dim
        else:
            return "llvm.amdgcn.workgroup.id." + dim

    @staticmethod
    @always_inline("nodebug")
    fn x() -> UInt:
        """Gets the `x` coordinate of the block within a grid.

        Returns:
            The `x` coordinate within the grid.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    Self._get_intrinsic_name["x"](),
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn y() -> UInt:
        """Gets the `y` coordinate of the block within a grid.

        Returns:
            The `y` coordinate within the grid.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    Self._get_intrinsic_name["y"](),
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn z() -> UInt:
        """Gets the `z` coordinate of the block within a grid.

        Returns:
            The `z` coordinate within the grid.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    Self._get_intrinsic_name["z"](),
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )


# ===----------------------------------------------------------------------===#
# BlockDim
# ===----------------------------------------------------------------------===#


struct BlockDim:
    """BlockDim provides static methods for getting the x/y/z dimension of a
    block."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> UInt:
        """Gets the `x` dimension of the block.

        Returns:
            The `x` dimension of the block.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.ntid.x",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn y() -> UInt:
        """Gets the `y` dimension of the block.

        Returns:
            The `y` dimension of the block.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.ntid.y",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn z() -> UInt:
        """Gets the `z` dimension of the block.

        Returns:
            The `z` dimension of the block.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.ntid.z",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )


# ===----------------------------------------------------------------------===#
# GridDim
# ===----------------------------------------------------------------------===#


struct GridDim:
    """GridDim provides static methods for getting the x/y/z dimension of a
    grid."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> UInt:
        """Gets the `x` dimension of the grid.

        Returns:
            The `x` dimension of the grid.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.nctaid.x",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn y() -> UInt:
        """Gets the `y` dimension of the grid.

        Returns:
            The `y` dimension of the grid.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.nctaid.y",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )

    @staticmethod
    @always_inline("nodebug")
    fn z() -> UInt:
        """Gets the `z` dimension of the grid.

        Returns:
            The `z` dimension of the grid.
        """
        return UInt(
            int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.nctaid.z",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )


# ===----------------------------------------------------------------------===#
# lane_id
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn lane_id() -> UInt:
    """Returns the lane ID of the current thread.

    Returns:
        The lane ID of the the current thread.
    """
    return UInt(
        int(
            llvm_intrinsic[
                "llvm.nvvm.read.ptx.sreg.laneid", Int32, has_side_effect=False
            ]().cast[DType.uint32]()
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
    return UInt(
        int(
            llvm_intrinsic[
                "llvm.nvvm.read.ptx.sreg.smid", Int32, has_side_effect=False
            ]().cast[DType.uint32]()
        )
    )
