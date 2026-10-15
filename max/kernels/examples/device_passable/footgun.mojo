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
# The two closure-argument footguns (both misuses typecheck).
# See max/docs/internal/max-serve/device-passable-closures.md.

from extensibility import InputFusion, InputTensor, StaticTensorSpec
from layout import TileTensor, row_major
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.builtin.device_passable import DevicePassable
from std.utils import IndexList

comptime N = 8
comptime spec = StaticTensorSpec[.float32, 1, ...].get_unknown()


# Input fusion computing `scale * i + bias`. No pointers, so its host bytes are
# already valid on the device.
@fieldwise_init
struct AffineRead(InputFusion):
    var scale: Float32
    var bias: Float32

    def load[
        dtype: DType,
        rank: Int,
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](
            (Float32(idx[0]) * self.scale + self.bias).cast[dtype]()
        )


# Launched with `host_arg=`: `cb` is bit-copied in host layout.
def read_host_closure[
    CbType: ImplicitlyCopyable & RegisterPassable & def(Int) -> Float32
](dst: MutPointer[Float32, MutAnyOrigin], n: Int32, cb: CbType):
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    dst[unsafe_offset=i] = cb(i)


# Launched with the default overload: `cb` is encoded to its `device_type`.
def read_encoded_closure[
    CbType: (def(Int) -> Float32) & DevicePassable & ImplicitlyCopyable
](dst: MutPointer[Float32, MutAnyOrigin], n: Int32, cb: CbType):
    var i = Int(global_idx.x)
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

    # WRONG: `tile`'s `DevicePointer` needed encoding; `host_arg=` bit-copies it.
    ctx.enqueue_function[read_host_closure[type_of(read_fn)]](
        out, Int32(N), host_arg=read_fn, grid_dim=1, block_dim=N
    )
    report("not_rewrite_when_should", out)
    _ = src_buf


def footgun_rewrite_when_should_not(ctx: DeviceContext) raises:
    var src_buf = make_source(ctx)
    var out = ctx.enqueue_create_buffer[.float32](N)
    out.enqueue_fill(Float32(-1))

    var slice = InputTensor[static_spec=spec](
        src_buf.unsafe_ptr(), IndexList[1](N)
    )._bind_to_fused_input(AffineRead(scale=1.0, bias=0.0))

    def read_fn(i: Int) {var slice} -> Float32:
        return slice._fused_load[width=1](IndexList[1](i))[0]

    # WRONG: `slice` was already device-valid; encoding rewrites it to
    # `LayoutTensor`, and the kernel reads `in_fusion` from those bytes.
    ctx.enqueue_function[read_encoded_closure[type_of(read_fn)]](
        out, Int32(N), read_fn, grid_dim=1, block_dim=N
    )
    report("rewrite_when_should_not", out)
    _ = src_buf


def main() raises:
    with DeviceContext() as ctx:
        footgun_not_rewrite_when_should(ctx)
        footgun_rewrite_when_should_not(ctx)
