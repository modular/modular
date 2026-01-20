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
"""Test that LayoutTensor IndexList codegen matches variadic codegen.

This test verifies that using IndexList to index into a LayoutTensor produces
the same quality code as using variadic arguments. The key requirement is that
IndexList should not generate worse code than the variadic overload.

Before the fix (MSTDL-2166), IndexList generated inefficient code that loaded
strides from memory and computed offsets at runtime, while variadic args
could use compile-time constants.

After the fix, both should produce equivalent code.
"""

from compile import compile_info
from gpu.host import get_gpu_target
from layout import Layout, LayoutTensor
from testing import assert_true
from utils import Index


comptime layout_2d = Layout.row_major(10, 20)


fn test_store_codegen_equivalence() raises:
    """Test that store(IndexList) produces equivalent code to store(m, n)."""

    fn variadic_kernel(
        output: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
    ):
        output.store[1](2, 3, 1234)

    fn indexlist_kernel(
        output: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
    ):
        output.store[1](Index(2, 3), 1234)

    var variadic_asm = String(
        compile_info[
            variadic_kernel, emission_kind="asm", target = get_gpu_target()
        ]()
    )
    var indexlist_asm = String(
        compile_info[
            indexlist_kernel, emission_kind="asm", target = get_gpu_target()
        ]()
    )

    # Check that IndexList doesn't load strides from memory when variadic doesn't
    # ld.param.v2.b32 is the instruction used to load stride pairs from params
    var variadic_loads_strides = "ld.param.v2.b32" in variadic_asm
    var indexlist_loads_strides = "ld.param.v2.b32" in indexlist_asm

    # IndexList should not load strides if variadic doesn't
    if not variadic_loads_strides:
        assert_true(
            not indexlist_loads_strides,
            "IndexList loads strides from memory but variadic doesn't. "
            + "IndexList should produce equivalent or better code.\n\n"
            + "Variadic ASM:\n"
            + variadic_asm
            + "\n\nIndexList ASM:\n"
            + indexlist_asm,
        )


fn test_load_codegen_equivalence() raises:
    """Test that load(IndexList) produces equivalent code to load(m, n)."""

    fn variadic_kernel(
        input: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.load[1](2, 3)
        output.store[1](0, 0, val)

    fn indexlist_kernel(
        input: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.load[1](Index(2, 3))
        output.store[1](0, 0, val)

    var variadic_asm = String(
        compile_info[
            variadic_kernel, emission_kind="asm", target = get_gpu_target()
        ]()
    )
    var indexlist_asm = String(
        compile_info[
            indexlist_kernel, emission_kind="asm", target = get_gpu_target()
        ]()
    )

    # Check stride loading behavior
    var variadic_loads_strides = "ld.param.v2.b32" in variadic_asm
    var indexlist_loads_strides = "ld.param.v2.b32" in indexlist_asm

    if not variadic_loads_strides:
        assert_true(
            not indexlist_loads_strides,
            "IndexList loads strides from memory but variadic doesn't. "
            + "IndexList should produce equivalent or better code.\n\n"
            + "Variadic ASM:\n"
            + variadic_asm
            + "\n\nIndexList ASM:\n"
            + indexlist_asm,
        )


fn test_aligned_load_codegen_equivalence() raises:
    """Test that aligned_load(IndexList) produces equivalent code to aligned_load(m, n).
    """

    fn variadic_kernel(
        input: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.aligned_load[1](2, 3)
        output.store[1](0, 0, val)

    fn indexlist_kernel(
        input: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[DType.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.aligned_load[1](Index(2, 3))
        output.store[1](0, 0, val)

    var variadic_asm = String(
        compile_info[
            variadic_kernel, emission_kind="asm", target = get_gpu_target()
        ]()
    )
    var indexlist_asm = String(
        compile_info[
            indexlist_kernel, emission_kind="asm", target = get_gpu_target()
        ]()
    )

    # Check stride loading behavior
    var variadic_loads_strides = "ld.param.v2.b32" in variadic_asm
    var indexlist_loads_strides = "ld.param.v2.b32" in indexlist_asm

    if not variadic_loads_strides:
        assert_true(
            not indexlist_loads_strides,
            "IndexList loads strides from memory but variadic doesn't. "
            + "IndexList should produce equivalent or better code.\n\n"
            + "Variadic ASM:\n"
            + variadic_asm
            + "\n\nIndexList ASM:\n"
            + indexlist_asm,
        )


def main():
    test_store_codegen_equivalence()
    test_load_codegen_equivalence()
    test_aligned_load_codegen_equivalence()
