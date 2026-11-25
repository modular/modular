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

# DOC: develop/build-custom-ops.mdx

import extensibility

from extensibility import CPU, Input, Output, Tensor, TileTensorable

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


# The serial loop makes this a CPU-only kernel, so it registers for CPUs only.
@extensibility.register("keep_positive", CPU)
struct KeepPositive:
    @staticmethod
    def execute(output: Some[Output & Tensor], x: Some[Input & Tensor]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 1
        comptime assert type_of(x).rank == 1
        var output_tile = output.to_tile_tensor[dtype]()
        var x_tile = x.to_tile_tensor[dtype]().as_imm()

        var count = 0
        for i in range(Int(x_tile.dim[0]())):
            var value = x_tile.load[width=1](Coord(i))
            if value > 0:
                output_tile.store[width=1](Coord(count), value)
                count += 1


# The output length depends on the input's values, so no input dim can size
# it. The shape function counts the positive values on the host.
@extensibility.register_shape_function("keep_positive")
def keep_positive_shape(x: Some[TileTensorable]) raises -> IndexList[1]:
    var x_tile = x.to_tile_tensor()
    var length = coord_to_index_list(x.shape())[0]
    var count = 0
    for i in range(length):
        if x_tile[i] > 0:
            count += 1
    return IndexList[1](count)
