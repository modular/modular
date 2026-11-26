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
# A sealing variant's token is the only way to wait for what it covers, so
# it cannot be dropped: only its route's wait consumes it.

from layout import ImmTileTensor, MutTileTensor, TileTensor, row_major
from layout.atoms import HandleValue, SealingTransferAtom
from layout.layout import TensorLayout
from layout.owned_tensor import OwnedGPUTensor
from layout.tensor_engine import TensorEngine


struct Ticket(not Deinitable, HandleValue):
    var id: Int

    def __init__(out self, id: Int):
        self.id = id


@fieldwise_init
struct SealedCopy(SealingTransferAtom):
    comptime Token = Ticket

    def __call__[
        T: DType,
        LD: TensorLayout,
        ED: TensorEngine,
        LS: TensorLayout,
        ES: TensorEngine,
    ](
        self,
        dst: MutTileTensor[T, LD, Engine=ED, ...],
        src: ImmTileTensor[T, LS, Engine=ES, ...],
    ) -> Ticket:
        return Ticket(1)


def main():
    var src = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    var dst = OwnedGPUTensor[.float32, .GENERIC](row_major[1, 4]())
    # CHECK: type 'Ticket' does not conform to 'Deinitable' and must be explicitly destroyed
    _ = SealedCopy()(TileTensor(dst), TileTensor(src))
