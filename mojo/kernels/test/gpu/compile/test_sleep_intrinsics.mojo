# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -disable-prebuilt-packages -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from gpu.intrinsics import sleep


# CHECK-LABEL: sleep_intrinsics
@export
fn sleep_intrinsics():
    @parameter
    if not triple_is_nvidia_cuda():
        return
    # CHECK: nanosleep.u32 100
    sleep[0.0000001]()
