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

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from layout import RowMajorLayout, TileTensor, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor


trait BaseT(TrivialRegisterPassable):
    def get_val(self, idx: Int) -> Float32:
        ...


@fieldwise_init
struct ImplT(BaseT):
    @__allow_legacy_any_origin_fields
    var values: TileTensor[.float32, RowMajorLayout[Int], MutAnyOrigin]

    def __init__(
        out self,
        buf: TileTensor[mut=True, .float32, RowMajorLayout[Int], _],
    ) raises:
        self.values = buf.as_unsafe_any_origin()

    def get_val(self, idx: Int) -> Float32:
        return self.values[idx]


def trait_repro_sub[t: BaseT](thing: t, ctx: DeviceContext, size: Int) raises:
    @__parameter
    @__copy_capture(thing)
    def kernel_fn():
        var idx = thread_idx.x
        print(thing.get_val(idx) * 2)

    comptime kernel = kernel_fn
    ctx.enqueue_function[kernel](grid_dim=(1,), block_dim=(size))


def trait_repro(ctx: DeviceContext) raises:
    comptime size = 5
    var managed_buf = HostDeviceTileTensor[.float32](row_major(size), ctx)
    var host_buf = managed_buf.host_tensor()
    for i in range(size):
        host_buf[i] = Float32(i)
    managed_buf.to_device()

    var thing = ImplT(managed_buf.device_tensor())
    trait_repro_sub(thing, ctx, size)
    managed_buf.to_host()

    for i in range(size):
        print(host_buf[i])


def main() raises:
    with DeviceContext() as ctx:
        trait_repro(ctx)
