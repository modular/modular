# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# RUN: kgen -disable-prebuilt-packages -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc


# CHECK-LABEL: register_intrinsics
@export
fn register_intrinsics():
    # CHECK: setmaxnreg.inc.sync.aligned.u32
    warpgroup_reg_alloc[42]()
    # CHECK: setmaxnreg.dec.sync.aligned.u32
    warpgroup_reg_dealloc[42]()
