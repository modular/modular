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

from buffer import DimList, NDBuffer
from gpu import thread_idx
from gpu.host import DeviceContext
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from utils import IndexList


@register_passable("trivial")
trait BaseT:
    fn get_val(self, idx: Int) -> Float32:
        ...


@fieldwise_init
@register_passable("trivial")
struct ImplT(BaseT, ImplicitlyCopyable, Movable):
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
        print(Float32(thing.get_val(idx)) * 2)

    comptime kernel = kernel_fn
    ctx.enqueue_function_checked[kernel, kernel](
        grid_dim=(1,), block_dim=(size)
    )


def trait_repro(ctx: DeviceContext):
    comptime size = 5
    var stack = InlineArray[Float32, size](uninitialized=True)
    var host_buf = LayoutTensor[DType.float32, Layout(UNKNOWN_VALUE)](
        stack,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](size)),
    )
    for i in range(size):
        host_buf[i] = i

    var device_buf = ctx.enqueue_create_buffer[DType.float32](size)
    with device_buf.map_to_host() as mapped:
        for i in range(size):
            mapped[i] = host_buf[i][0]
    var device_nd = LayoutTensor[DType.float32, Layout(UNKNOWN_VALUE)](
        device_buf,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](size)),
    )
    var thing = ImplT(device_nd)
    trait_repro_sub(thing, ctx, size)
    with device_buf.map_to_host() as mapped:
        for i in range(size):
            host_buf[i] = mapped[i]
    ctx.synchronize()

    for i in range(size):
        print(host_buf[i])

    _ = device_buf^


def main():
    with DeviceContext() as ctx:
        trait_repro(ctx)
