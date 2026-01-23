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

from sys import env_get_int

import compiler
from logger import Logger
from tensor import foreach, OutputTensor, InputTensor
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList

comptime logger = Logger()


@compiler.register("use_splitk_reduction_scheme")
struct UseSplitkReductionScheme:
    @staticmethod
    fn execute(
        output: OutputTensor[dtype = DType.int32, rank=1],
    ):
        comptime split_k_reduction_scheme = env_get_int[
            "SPLITK_REDUCTION_SCHEME", 2
        ]()
        output[0] = split_k_reduction_scheme


@compiler.register("use_logger")
struct UseLogger:
    @staticmethod
    fn execute(
        output: OutputTensor[dtype = DType.int32, rank=1],
    ):
        logger.error("I'm a custom Mojo function!")
        output[0] = logger.level._value


@compiler.register("add_one_custom")
struct AddOneCustom:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        fn add_one[width: Int](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[add_one, target=target](output, ctx)

    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"
