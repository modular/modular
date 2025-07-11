# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from gpu.host.compile import _compile_code_asm
from gpu.host.info import H100
from gpu.memory import *
from testing import *


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
                accum_type = DType.bfloat16,
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
                accum_type = DType.bfloat16,
                consistency = Consistency.RELAXED,
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
                accum_type = DType.bfloat16,
                consistency = Consistency.RELAXED,
            ],
            target = H100.target(),
        ]()
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16 "
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
                accum_type = DType.bfloat16,
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
                accum_type = DType.bfloat16,
                output_width=2,
            ],
            target = H100.target(),
        ]()
    )


def test_multimem_st():
    print("== test_multimem_st")

    assert_true(
        "multimem.st.weak.cta.global.v2.bf16x2 "
        in _compile_code_asm[
            multimem_st[
                DType.bfloat16,
                count=2,
                scope = Scope.BLOCK,
                consistency = Consistency.WEAK,
                width=2,
            ],
            target = H100.target(),
        ]()
    )

    assert_true(
        "multimem.st.relaxed.gpu.global.v4.f32 "
        in _compile_code_asm[
            multimem_st[
                DType.float32,
                count=4,
                scope = Scope.GPU,
                consistency = Consistency.RELAXED,
            ],
            target = H100.target(),
        ]()
    )

    assert_true(
        "multimem.st.release.sys.global.v4.f16x2 "
        in _compile_code_asm[
            multimem_st[
                DType.float16,
                count=4,
                scope = Scope.SYSTEM,
                consistency = Consistency.RELEASE,
                width=2,
            ],
            target = H100.target(),
        ]()
    )

    assert_true(
        "multimem.st.relaxed.cluster.global.v4.bf16 "
        in _compile_code_asm[
            multimem_st[
                DType.bfloat16,
                count=4,
                scope = Scope.CLUSTER,
                consistency = Consistency.RELAXED,
            ],
            target = H100.target(),
        ]()
    )


def main():
    test_multimem_ld_reduce()
    test_multimem_st()
