# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from testing import *
from gpu.host.info import H100
from gpu.host._compile import _compile_code_asm
from gpu.memory import *


def test_multimem_ld_reduce():
    print("== test_multimem_ld_reduce")
    assert_true(
        "multimem.ld_reduce.weak.gpu.global.max.v2.bf16 "
        in _compile_code_asm[
            multimem_ld_reduce[
                DType.bfloat16,
                count=2,
                reduction = ReduceOp.MAX,
                scope = Scope.GPU,
                consistency = Consistency.WEAK,
            ],
            target = H100.target(),
        ]()
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16 "
        in _compile_code_asm[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction = ReduceOp.ADD,
                scope = Scope.SYSTEM,
                consistency = Consistency.RELAXED,
            ],
            target = H100.target(),
        ]()
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 "
        in _compile_code_asm[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction = ReduceOp.ADD,
                scope = Scope.SYSTEM,
                consistency = Consistency.RELAXED,
                output_width=2,
            ],
            target = H100.target(),
        ]()
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.max.v4.bf16x2 "
        in _compile_code_asm[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction = ReduceOp.MAX,
                scope = Scope.SYSTEM,
                consistency = Consistency.RELAXED,
                output_width=2,
            ],
            target = H100.target(),
        ]()
    )


def main():
    test_multimem_ld_reduce()
