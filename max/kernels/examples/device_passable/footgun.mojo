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
# A closure-argument footgun that typechecks: bit-copying a capture that needs
# the host-to-device rewrite. See
# max/docs/internal/max-serve/device-passable-closures.md.

from layout import TileTensor, row_major
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

comptime N = 8


# Launched with `host_args=`: `cb` is bit-copied in host layout.
def read_host_closure[
    CbType: ImplicitlyCopyable & RegisterPassable & def(Int) -> Float32
](dst: MutPointer[Float32, MutAnyOrigin], n: Int32, cb: CbType):
    var i = global_idx.x
    if i >= Int(n):
        return
    dst[unsafe_offset=i] = cb(i)


def report(label: String, buf: DeviceBuffer[.float32]) raises:
    var wrong = 0
    with buf.map_to_host() as h:
        for i in range(N):
            print("  ", label, " i=", i, " got=", h[i], " want=", Float32(i))
            if h[i] != Float32(i):
                wrong += 1
    print(label, ": ", wrong, "/", N, " elements wrong")


def make_source(ctx: DeviceContext) raises -> DeviceBuffer[.float32]:
    var src_buf = ctx.enqueue_create_buffer[.float32](N)
    with src_buf.map_to_host() as h:
        for i in range(N):
            h[i] = Float32(i)
    return src_buf


def footgun_not_rewrite_when_should(ctx: DeviceContext) raises:
    var src_buf = make_source(ctx)
    var out = ctx.enqueue_create_buffer[.float32](N)
    out.enqueue_fill(Float32(-1))

    var tile = TileTensor(src_buf.device_ptr(), row_major[N]())

    def read_fn(i: Int) {var tile} -> Float32:
        return tile[i][0]

    # WRONG: `tile`'s `DevicePointer` needed encoding; `host_args=` bit-copies
    # it.
    ctx.enqueue_function[read_host_closure[type_of(read_fn)]](
        out,
        Int32(N),
        host_args=(read_fn,),
        grid_dim=1,
        block_dim=N,
    )
    report("not_rewrite_when_should", out)
    _ = src_buf


def main() raises:
    with DeviceContext() as ctx:
        footgun_not_rewrite_when_should(ctx)
