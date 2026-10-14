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

from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from layout import Idx, TileTensor, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor

from std.utils.coord import Coord


def test_elementwise_print[
    c_type: DType,
](c01: TileTensor[c_type, ...], ctx: DeviceContext) raises:
    var M = Int(c01.dim[0]())
    var N = Int(c01.dim[1]()) // 2
    comptime simd_width = simd_width_of[
        c_type, target=get_gpu_target["sm_80"]()
    ]()

    @inline(.always)
    def binary[simd_width: Int, alignment: Int = 1](idx0: Coord) {var}:
        var m: Int = Int(idx0[0].value())
        var n: Int = Int(idx0[1].value())
        print("print thousands of messages: m=", m, " n=", n, sep="")

    print("about to call elementwise, M=", M, "N=", N)
    elementwise[simd_width, target="gpu"](binary, (M, N), ctx)
    print("called elementwise")
    # Avoid exiting in the middle of the call to the kernel that is printing the test messages.
    ctx.synchronize()
    print("finished elementwise")


def test_dual_matmul[
    N: Int = 512, K: Int = 512
](ctx: DeviceContext, M: Int = 512) raises:
    comptime dst_type = DType.float32
    var mat_c01 = HostDeviceTileTensor[dst_type](row_major(M, Idx[2 * N]), ctx)
    test_elementwise_print(
        mat_c01.device_tensor(),
        ctx,
    )
    print("returned from test_elementwise_print")


def main() raises:
    with DeviceContext() as ctx:
        test_dual_matmul(ctx)
