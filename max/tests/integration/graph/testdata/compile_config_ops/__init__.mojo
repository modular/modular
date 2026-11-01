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

from std.sys import get_defined_int

import extensibility
from max.gpu.host import DeviceContext
from std.logger import Logger
from extensibility import foreach, OutputTensor, InputTensor, Tensor


from std.utils.coord import Coord
from std.utils.index import IndexList

comptime logger = Logger()


@extensibility.register("use_splitk_reduction_scheme")
struct UseSplitkReductionScheme:
    @staticmethod
    def execute(
        output: OutputTensor[dtype=.int32, rank=1, ...],
    ):
        comptime split_k_reduction_scheme = get_defined_int[
            "SPLITK_REDUCTION_SCHEME", 2
        ]()
        output[0] = Int32(split_k_reduction_scheme)


@extensibility.register("use_logger")
struct UseLogger:
    @staticmethod
    def execute(
        output: OutputTensor[dtype=.int32, rank=1, ...],
    ):
        logger.error("I'm a custom Mojo function!")
        output[0] = Int32(logger.level._value)


@extensibility.register("add_one_custom")
struct AddOneCustom:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        def add_one[width: Int](idx: Coord) {var} -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[target=target](add_one, output, ctx)


@extensibility.register_shape_function("add_one_custom")
def add_one_custom_shape(
    x: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    raise "NotImplemented"
