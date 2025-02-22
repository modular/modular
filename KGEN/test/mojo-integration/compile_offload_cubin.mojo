# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: NVIDIA-GPU
# UNSUPPORTED: asan, ubsan
# RUN: %mojo -O0 %s -o %t.cubin
# RUN: file /usr/local/cuda/bin/nvdisasm && (/usr/local/cuda/bin/nvdisasm %t.cubin | FileCheck %s)

from gpu.host import DeviceContext
from memory import UnsafePointer
from sys import argv, sizeof
from sys.info import _current_target
from gpu.host._compile import _get_gpu_target


@value
@register_passable("trivial")
struct _Info:
    var kernel: __mlir_type.`!kgen.string`
    var num_captures: __mlir_type.index


@value
@register_passable("trivial")
struct Info:
    var kernel: StringLiteral
    var num_captures: Int


@always_inline
fn _compile_info[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    /,
    emission_kind: Int = 0,
    compile_options: StringLiteral = "nvptx-short-ptr=true",
    compile_target: __mlir_type.`!kgen.target` = _get_gpu_target(),
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=compile_target,
        emission_kind = index(emission_kind),
        emission_option = compile_options.value,
        func=func,
        _type=_Info,
    ]()

    return Info(kernel=info.kernel, num_captures=info.num_captures)


fn hello():
    pass


def main():
    with DeviceContext() as ctx:
        t1 = _compile_info[hello, emission_kind=3]()
        idx = 0
        args = argv()
        for arg in argv():
            idx = idx + 1
            if arg == "-o":
                break

        # write the shared object binary to a file for checking
        with open(args[idx], "w") as f:
            f.write(t1.kernel)


# CHECK: nv.info
# CHECK: nv.callgraph
# CHECK: nv.constant
# CHECK: .text
