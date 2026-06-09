# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not kgen -emit=llvm %s 2>&1 | FileCheck %s


# CHECK: invalid 'llvm.frame_pointer' value 'bogus'
# CHECK-SAME: expected "none", "non-leaf", "all", or "reserved"
@export
@__llvm_metadata(`llvm.frame_pointer`=__mlir_attr.`"bogus"`)
def fn_bad_frame_pointer() abi("Mojo"):
    pass


@export
def use() abi("Mojo"):
    fn_bad_frame_pointer()
