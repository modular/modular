# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: NVIDIA-GPU
# UNSUPPORTED: asan, ubsan
# RUN: MODULAR_USE_PTXAS=1 kgen -emit -kgen-debug-only=object-compiler %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-PTXAS
# RUN: MODULAR_USE_PTXAS=1 %mojo -O0 %s -o %t.cubin
# RUN: file /usr/local/cuda/bin/nvdisasm && (/usr/local/cuda/bin/nvdisasm %t.cubin | FileCheck %s --check-prefix=CHECK-CUBIN)

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


fn hello_world():
    pass


def main():
    with DeviceContext() as ctx:
        t1 = _compile_info[hello_world, emission_kind=3]()
        idx = 0
        args = argv()
        for arg in argv():
            idx = idx + 1
            if arg == "-o":
                break

        # write the shared object binary to a file for checking
        with open(args[idx], "w") as f:
            f.write(t1.kernel)


# CHECK-CUBIN: nv.info
# CHECK-CUBIN: nv.callgraph
# CHECK-CUBIN: nv.constant
# CHECK-PTXAS: Successfully compiled PTX to CUBIN via ptxas
