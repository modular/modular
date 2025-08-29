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

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    OutputTensor,
    foreach,
)
from utils.index import IndexList


@compiler.register("add_constant_custom")
struct AddConstantCustom[value: Int]:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor,
        x: InputTensor[dtype = outp.dtype, rank = outp.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn add_constant[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + value

        foreach[add_constant, target=target](outp, ctx)
