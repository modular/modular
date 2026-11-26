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
# A data atom's `AsyncEngine` must be an async engine.

from std.sys import size_of

from layout import ImmTileTensor, MutTileTensor
from layout.atoms import LoadAtom, registers_map, vector_map
from layout.layout import TensorLayout
from layout.tensor_engine import DefaultEngine, TensorEngine


# CHECK: 'NotAnEngine' does not implement all requirements for 'DataAtom'
@fieldwise_init
struct NotAnEngine(LoadAtom):
    # CHECK: comptime member 'AsyncEngine' type 'AnyStruct[Int]' does not conform to trait's required type 'AsyncEngine'
    comptime AsyncEngine = Int
    comptime shape_values = ParameterList.of[1]().values
    comptime dtype = DType.float32
    comptime Engine = DefaultEngine[element_width=1]
    comptime mem_map = vector_map(1, size_of[DType.float32]())
    comptime reg_map = registers_map(1)

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
    _ = NotAnEngine()
