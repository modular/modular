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

from gpu import thread_idx
from gpu.host import DeviceContext
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from utils import IndexList


trait BaseT(TrivialRegisterPassable):
    fn get_val(self, idx: Int) -> Float32:
        ...


@fieldwise_init
struct ImplT(BaseT):
    var values: LayoutTensor[DType.float32, Layout(UNKNOWN_VALUE), MutAnyOrigin]

    def __init__(
        out self,
        buf: LayoutTensor[mut=True, DType.float32, Layout(UNKNOWN_VALUE)],
    ):
        self.values = buf.as_any_origin()

    fn get_val(self, idx: Int) -> Float32:
        return self.values[idx][0]


def trait_repro_sub[t: BaseT](thing: t, ctx: DeviceContext, size: Int):
    @parameter
    @__copy_capture(thing)
    fn kernel_fn():
        var idx = Int(thread_idx.x)
        print(thing.get_val(idx) * 2)

    comptime kernel = kernel_fn
    ctx.enqueue_function_experimental[kernel](grid_dim=(1,), block_dim=(size))


def trait_repro(ctx: DeviceContext):
    comptime size = 5
    var managed_buf = ManagedLayoutTensor[DType.float32, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](size)),
        ctx,
    )
    var host_buf = managed_buf.tensor[update=False]()
    for i in range(size):
        host_buf[i] = Float32(i)

    var thing = ImplT(managed_buf.device_tensor())
    trait_repro_sub(thing, ctx, size)
    host_buf = managed_buf.tensor()

    for i in range(size):
        print(host_buf[i])


def main():
    with DeviceContext() as ctx:
        trait_repro(ctx)
