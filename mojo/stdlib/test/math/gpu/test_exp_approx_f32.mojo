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
from math.fast import exp_approx_f32
from math import exp as ref_exp
from algorithm.functional import elementwise
from buffer import NDBuffer
from gpu import *
from gpu.host import DeviceContext, get_gpu_target
from testing import *
from utils import Index, IndexList


@parameter
def run_exp_approx_test[simd_width: Int](ctx: DeviceContext):
    alias dtype = DType.float32
    alias length = 256

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)

    # Fill test data symmetrically around zero
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = 0.001 * (Scalar[dtype](i) - length // 2)

    var in_buffer = NDBuffer[dtype, 1](in_device.unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[dtype, 1](out_device.unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)
        var v = in_buffer.load[width=simd_width](idx)
        out_buffer.store[width=simd_width](idx, exp_approx_f32[simd_width](v))

    # Launch elementwise kernel on GPU (width is compile-time parameter)
    elementwise[func, simd_width, target="gpu"](IndexList[1](length), ctx)

    # Validate results
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var msg = String(
                "Mismatch at index ",
                i,
                " for SIMD width=",
                simd_width,
                " value=",
                in_host[i],
            )
            assert_almost_equal(
                out_host[i],
                Scalar[dtype](ref_exp(Scalar[dtype](in_host[i]))),
                msg=msg,
                atol=1e-07,
                # The relaxed relative tolerance (rtol=2e-03) is chosen based on the expected error bounds of the FA-4 exp approximation method.
                # This value ensures the test passes for all inputs while accounting for the approximation's maximum observed error.
                rtol=2e-03,
            )


def main():
    with DeviceContext() as ctx:
        run_exp_approx_test[1](ctx)
        run_exp_approx_test[2](ctx)
