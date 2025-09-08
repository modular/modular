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

from utils.fast_div import FastDiv
from gpu.host import get_gpu_target
from layout.layout_tensor import LayoutTensor
from layout.layout import Layout, IntTuple
from gpu.host.compile import _compile_code
from testing import assert_true
from python import Python


def contains_fastdiv_div_sequence(asm: String) -> Bool:
    var re = Python.import_module("re")
    var fastdiv_pattern = String(
        r"ld\.global\.b32\s+[^;]+;\s*"
        r"mov\.b32\s+[^;]+;\s*"
        r"mul\.hi\.u32\s+[^;]+;\s*"
        r"sub\.s32\s+[^;]+;\s*"
        r"shr\.u32\s+[^;]+;\s*"
        r"add\.s32\s+[^;]+;\s*"
        r"shr\.u32\s+[^;]+;\s*"
        r"st\.global\.b32\s+[^;]+;"
    )
    var result = re.search(fastdiv_pattern, asm)
    return result is not None


def contains_power_of_2_sequence(asm: String) -> Bool:
    var re = Python.import_module("re")
    var shift_pattern = String(
        r"ld\.global\.b32\s+[^;]+;\s*"
        r"shr\.u32\s+[^;]+;\s*"
        r"st\.global\.b32\s+[^;]+;"
    )
    var shift_result = re.search(shift_pattern, asm)
    return shift_result is not None


fn fast_div_kernel[
    dtype: DType,
    layout: Layout,
    divisor: Int,
](input: LayoutTensor[dtype, layout, MutableAnyOrigin],):
    alias fast_div = FastDiv[dtype](divisor)
    var x = input[0]
    var result = rebind[Scalar[fast_div.uint_type]](x) / fast_div
    input[0] = result.cast[dtype]()


def main():
    alias dtype = DType.uint32
    alias layout = Layout(IntTuple(1))
    alias kernel_fast_div_4 = fast_div_kernel[dtype, layout, 4]
    alias kernel_fast_div_3 = fast_div_kernel[dtype, layout, 3]

    var asm = _compile_code[
        kernel_fast_div_4,
        target = get_gpu_target["sm_90"](),
    ]().asm
    assert_true(contains_power_of_2_sequence(asm))

    asm = _compile_code[
        kernel_fast_div_3,
        target = get_gpu_target["sm_90"](),
    ]().asm
    assert_true(contains_fastdiv_div_sequence(asm))
