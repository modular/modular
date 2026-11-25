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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s
#
# `tiled_load` takes an atom whose engine is `Synchronous` or has a count
# route: it issues many times and its caller waits once. A grouped engine
# needs a commit the TileOp cannot place.

from std.sys import size_of

from layout import ImmTileTensor, MutTileTensor, TileTensor, row_major
from layout.atoms import LoadAtom, registers_map, vector_map
from layout.atoms.cp_async import CpAsyncEngine
from layout.layout import TensorLayout
from layout.owned_tensor import OwnedGPUTensor
from layout.tensor_engine import DefaultEngine, TensorEngine
from layout.tileops import tiled_load


@fieldwise_init
struct GroupedLoad(LoadAtom):
    comptime AsyncEngine = CpAsyncEngine
    comptime shape_values = ParameterList.of[4]().values
    comptime dtype = DType.float32
    comptime Engine = DefaultEngine[element_width=1]
    comptime mem_map = vector_map(4, size_of[DType.float32]())
    comptime reg_map = registers_map(4)

    def __call__[
        T: DType,
        LR: TensorLayout,
        ER: TensorEngine,
        LM: TensorLayout,
        EM: TensorEngine,
    ](
        self,
        dst: MutTileTensor[T, LR, Engine=ER, ...],
        src: ImmTileTensor[T, LM, Engine=EM, ...],
    ):
        pass


def main():
    var mem = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var regs = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    # CHECK: per-thread TileOps take an atom whose async engine is Synchronous or has a count route
    tiled_load(GroupedLoad(), TileTensor(regs), TileTensor(mem))
