# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

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

from compiler_internal import register
from tensor_internal import (
    InputTensor,
    OutputTensor,
    VariadicTensors,
    foreach,
)
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from layout import LayoutTensor, Layout
from gpu import global_idx, thread_idx, block_idx, AddressSpace, barrier
from math import ceildiv, sqrt
from algorithm.functional import vectorize


@register("grayscale")
struct Grayscale:
    @staticmethod
    fn execute[
        # The kind of device this is running on: "cpu" or "gpu"
        target: StaticString,
    ](
        img_out: OutputTensor[type = DType.uint8, rank=2],
        img_in: InputTensor[type = DType.uint8, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn color_to_grayscale[
            simd_width: Int
        ](idx: IndexList[img_out.rank]) -> SIMD[DType.uint8, simd_width]:
            var row = idx[0]
            var col = idx[1]

            var r_idx = IndexList[3](row, col, 0)
            var g_idx = IndexList[3](row, col, 1)
            var b_idx = IndexList[3](row, col, 2)

            var r_f32 = img_in.load[simd_width](r_idx).cast[DType.float32]()
            var g_f32 = img_in.load[simd_width](g_idx).cast[DType.float32]()
            var b_f32 = img_in.load[simd_width](b_idx).cast[DType.float32]()

            var gray_f32 = 0.21 * r_f32 + 0.71 * g_f32 + 0.07 * b_f32

            return min(gray_f32, 255).cast[DType.uint8]()

        foreach[color_to_grayscale, target=target, simd_width=1](img_out, ctx)


@register("myadd")
struct MyAdd:
    @staticmethod
    fn execute[
        type: DType, rank: Int, target: StaticString
    ](
        C: OutputTensor[type=type, rank=rank],
        A: InputTensor[type=type, rank=rank],
        B: InputTensor[type=type, rank=rank],
    ) raises:
        @parameter
        @always_inline
        fn doit[
            simd_width: Int
        ](idx: IndexList[C.rank]) -> SIMD[C.type, simd_width]:
            var a = A.load[simd_width](idx)
            var b = B.load[simd_width](idx)
            return a + b

        foreach[doit, target=target](C)
