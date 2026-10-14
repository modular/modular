# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Check that TileTensor coordinate indexing does not add stride loads.

Static layouts should compile to constant offsets through both the subscript
and explicit coordinate APIs, including explicitly aligned loads.
"""

from std.compile import compile_info
from max.gpu.host import get_gpu_target
from layout import Coord, TileTensor, row_major
from std.testing import assert_true


comptime layout_2d = row_major[10, 20]()


def test_store_codegen_equivalence() raises:
    """Check coordinate and subscript stores for redundant stride loads."""

    def subscript_kernel(
        output: TileTensor[.int32, type_of(layout_2d), MutAnyOrigin],
    ):
        output[2, 3] = 1234

    def coordinate_kernel(
        output: TileTensor[.int32, type_of(layout_2d), MutAnyOrigin],
    ):
        output.store(Coord(2, 3), 1234)

    var subscript_asm = String(
        compile_info[
            subscript_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )
    var coordinate_asm = String(
        compile_info[
            coordinate_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )

    # Check that Coord doesn't load strides from memory when subscript doesn't
    # ld.param.v2.b32 is the instruction used to load stride pairs from params
    var subscript_loads_strides = "ld.param.v2.b32" in subscript_asm
    var coordinate_loads_strides = "ld.param.v2.b32" in coordinate_asm

    # Coord should not load strides if subscript doesn't
    if not subscript_loads_strides:
        assert_true(
            not coordinate_loads_strides,
            "Coord loads strides from memory but subscript doesn't. "
            + "Coord should produce equivalent or better code.\n\n"
            + "Subscript ASM:\n"
            + subscript_asm
            + "\n\nCoord ASM:\n"
            + coordinate_asm,
        )


def test_load_codegen_equivalence() raises:
    """Check coordinate and subscript loads for redundant stride loads."""

    def subscript_kernel(
        input: TileTensor[.int32, type_of(layout_2d), ImmutAnyOrigin],
        output: TileTensor[.int32, type_of(layout_2d), MutAnyOrigin],
    ):
        var val = input[2, 3]
        output[0, 0] = val

    def coordinate_kernel(
        input: TileTensor[.int32, type_of(layout_2d), ImmutAnyOrigin],
        output: TileTensor[.int32, type_of(layout_2d), MutAnyOrigin],
    ):
        var val = input.load(Coord(2, 3))
        output[0, 0] = val

    var subscript_asm = String(
        compile_info[
            subscript_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )
    var coordinate_asm = String(
        compile_info[
            coordinate_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )

    # Check stride loading behavior
    var subscript_loads_strides = "ld.param.v2.b32" in subscript_asm
    var coordinate_loads_strides = "ld.param.v2.b32" in coordinate_asm

    if not subscript_loads_strides:
        assert_true(
            not coordinate_loads_strides,
            "Coord loads strides from memory but subscript doesn't. "
            + "Coord should produce equivalent or better code.\n\n"
            + "Subscript ASM:\n"
            + subscript_asm
            + "\n\nCoord ASM:\n"
            + coordinate_asm,
        )


def test_aligned_load_codegen_equivalence() raises:
    """Check explicit alignment with tuple and coordinate loads."""

    def subscript_kernel(
        input: TileTensor[.int32, type_of(layout_2d), ImmutAnyOrigin],
        output: TileTensor[.int32, type_of(layout_2d), MutAnyOrigin],
    ):
        var val = input.load[alignment=4]((2, 3))
        output[0, 0] = val

    def coordinate_kernel(
        input: TileTensor[.int32, type_of(layout_2d), ImmutAnyOrigin],
        output: TileTensor[.int32, type_of(layout_2d), MutAnyOrigin],
    ):
        var val = input.load[alignment=4](Coord(2, 3))
        output[0, 0] = val

    var subscript_asm = String(
        compile_info[
            subscript_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )
    var coordinate_asm = String(
        compile_info[
            coordinate_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )

    # Check stride loading behavior
    var subscript_loads_strides = "ld.param.v2.b32" in subscript_asm
    var coordinate_loads_strides = "ld.param.v2.b32" in coordinate_asm

    if not subscript_loads_strides:
        assert_true(
            not coordinate_loads_strides,
            "Coord loads strides from memory but subscript doesn't. "
            + "Coord should produce equivalent or better code.\n\n"
            + "Subscript ASM:\n"
            + subscript_asm
            + "\n\nCoord ASM:\n"
            + coordinate_asm,
        )


def main() raises:
    test_store_codegen_equivalence()
    test_load_codegen_equivalence()
    test_aligned_load_codegen_equivalence()
