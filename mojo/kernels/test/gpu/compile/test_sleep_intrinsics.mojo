# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from gpu.time import sleep


# CHECK-LABEL: sleep_intrinsics
@export
fn sleep_intrinsics():
    # CHECK: mov.b32 {{.*}}%r1, 100;
    # CHECK: nanosleep.u32 %r1
    sleep(0.0000001)
