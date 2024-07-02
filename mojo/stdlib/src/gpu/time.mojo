# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes utilities for getting time info on the GPU."""

from sys._assembly import inlined_assembly
from sys.intrinsics import llvm_intrinsic

# ===----------------------------------------------------------------------===#
# clock
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn clock() -> Int:
    """Returns a 32-bit unsigned cycle counter."""
    return int(llvm_intrinsic["llvm.nvvm.read.ptx.sreg.clock", Int32]())


@always_inline("nodebug")
fn clock64() -> Int:
    """Returns a 64-bit unsigned cycle counter."""
    return int(llvm_intrinsic["llvm.nvvm.read.ptx.sreg.clock64", Int64]())


# ===----------------------------------------------------------------------===#
# now
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn now() -> Int:
    """Returns a 64-bit global nanosecond timer.

    Returns:
        The current time in ns.
    """
    return int(
        inlined_assembly[
            "mov.u64  $0, %globaltimer;", UInt64, constraints="=l"
        ]()
    )


# ===----------------------------------------------------------------------===#
# time_function
# ===----------------------------------------------------------------------===#


@always_inline
@parameter
fn time_function[func: fn () -> None]() -> Int:
    """Measures the time spent in the function.

    Parameters:
        func: The function to time.

    Returns:
        The time elapsed in the function in ns.
    """

    var tic = now()
    func()
    var toc = now()
    return toc - tic


@always_inline
@parameter
fn time_function[func: fn () capturing -> None]() -> Int:
    """Measures the time spent in the function.

    Parameters:
        func: The function to time.

    Returns:
        The time elapsed in the function in ns.
    """

    var tic = now()
    func()
    var toc = now()
    return toc - tic
