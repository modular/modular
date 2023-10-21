# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

from .ptx_assembly import ptx_assembly
from .sys import is_sm_greater_equal


# ===----------------------------------------------------------------------===#
# warpgroup_reg
# ===----------------------------------------------------------------------===#


fn warpgroup_reg_alloc[count: Int]():
    """Provides a hint to the system to update the maximum number of per-thread
    registers owned by the executing warp to the value specified by the
    imm-reg-count operand.

    Used to request additional registers such that the absolute per-thread
    maximum register count is increased from its current value to imm-reg-count.
    """

    @parameter
    if is_sm_greater_equal[90]():
        ptx_assembly[
            "setmaxnreg.inc.sync.aligned.u32 $0", NoneType, constraints="i"
        ](UInt32(count))


fn warpgroup_reg_dealloc[count: Int]():
    """Provides a hint to the system to update the maximum number of per-thread
    registers owned by the executing warp to the value specified by the
    imm-reg-count operand.

    Used to release extra registers such that the absolute per-thread maximum
    register count is reduced from its current value to imm-reg-count.
    """

    @parameter
    if is_sm_greater_equal[90]():
        ptx_assembly[
            "setmaxnreg.dec.sync.aligned.u32 $0", NoneType, constraints="i"
        ](UInt32(count))
