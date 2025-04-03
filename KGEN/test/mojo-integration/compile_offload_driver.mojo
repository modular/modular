# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: NVIDIA-GPU
# UNSUPPORTED: asan, ubsan
# RUN: MODULAR_USE_DRIVER_CUBIN_COMPILER=1 kgen -emit -kgen-debug-only=object-compiler %s -o %t 2>&1 | FileCheck %s

from gpu.host import DeviceContext
from memory import UnsafePointer
from sys import argv, sizeof
from sys.info import _current_target
from gpu.host._compile import _get_gpu_target
from builtin.string_literal import get_string_literal_slice
from collections.string import StaticString


@value
@register_passable("trivial")
struct _Info:
    var kernel: __mlir_type.`!kgen.string`
    var num_captures: __mlir_type.index


@value
@register_passable("trivial")
struct Info:
    var kernel: StaticString
    var num_captures: Int


@always_inline
fn _compile_info[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    /,
    emission_kind: Int = 0,
    compile_options: StaticString = "nvptx-short-ptr=true",
    compile_target: __mlir_type.`!kgen.target` = _get_gpu_target(),
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=compile_target,
        emission_kind = index(emission_kind),
        emission_option = get_string_literal_slice[compile_options]().value,
        func=func,
        _type=_Info,
    ]()

    return Info(
        kernel=StringLiteral(info.kernel), num_captures=info.num_captures
    )


fn hello():
    pass


def main():
    t1 = _compile_info[hello, emission_kind=3]()
    print(t1.kernel)


# CHECK: Falling back to using the driver to compile PTX to CUBIN
