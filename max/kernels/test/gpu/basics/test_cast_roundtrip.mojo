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

from gpu import *
from gpu.host import DeviceContext
from testing import assert_equal, assert_true

from utils.numerics import inf, nan, neg_inf, isnan


fn id(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = Float32(BFloat16(input[tid]))


@no_inline
fn run_vec_add(ctx: DeviceContext) raises:
    print("== run_vec_add")

    alias length = 1024

    var in_host = UnsafePointer[Float32].alloc(length)

    for i in range(length):
        in_host[i] = Float32(i)

    in_host[4] = nan[DType.float32]()
    in_host[5] = inf[DType.float32]()
    in_host[6] = neg_inf[DType.float32]()
    in_host[7] = -0.0

    var in_device = ctx.enqueue_create_buffer[DType.float32](length)
    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    in_device.enqueue_copy_from(in_host)

    var block_dim = 32

    ctx.enqueue_function[id](
        in_device,
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    var expected = List[Float32](
        0.0,
        1.0,
        2.0,
        3.0,
        nan[DType.float32](),
        inf[DType.float32](),
        neg_inf[DType.float32](),
        -0.0,
        8.0,
        9.0,
    )
    with out_device.map_to_host() as out_host:
        for i in range(10):
            print("at index", i, "the value is", out_host[i])
            if isnan(expected[i]):
                assert_true(isnan(out_host[i]))
            else:
                assert_equal(expected[i], out_host[i])

    _ = in_device

    in_host.free()


def main():
    with DeviceContext() as ctx:
        run_vec_add(ctx)
