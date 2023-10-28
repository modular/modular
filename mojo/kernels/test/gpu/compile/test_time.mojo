# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -disable-prebuilt-packages -emit-asm --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_90 --target-features="" %s | FileCheck %s

from gpu.time import *


# CHECK-LABEL: clock_functions
@export
fn clock_functions():
    # CHECK: mov.u32 {{.*}}, %clock;
    _ = clock()
    # CHECK: mov.u64 {{.*}}, %clock64;
    _ = clock64()
    # CHECK: mov.u64 {{.*}}, %globaltimer;
    _ = now()


# CHECK-LABEL: time_functions
@export
fn time_functions(some_value: Int) -> Int:
    var tmp = some_value

    @always_inline
    @parameter
    fn something():
        tmp += 1

    # CHECK: mov.u64 {{.*}}, %globaltimer;
    # CHECK: add.s64 {{.*}}, 1;
    # CHECK: mov.u64 {{.*}}, %globaltimer;
    _ = time_function[something]()

    return tmp
