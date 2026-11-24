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

# Verify that the simdgroup 8x8 matrix multiply-accumulate intrinsic produces
# correct results on Apple Metal GPU.

# REQUIRES: APPLE-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from std.sys import llvm_intrinsic

from max.gpu import lane_id
from max.gpu.host import DeviceContext
from std.memory import Pointer


# Per-lane SIMD width = 64 / 32 lanes = 2 elements per lane
# (matching the per-lane fragment size for an 8x8 simdgroup matrix)
comptime _FRAG = 2


@__name(t"_probe_mm_kernel")
def _probe_mm_kernel(
    out_ptr: Pointer[Float32, MutAnyOrigin],
):
    var a = SIMD[.float32, _FRAG](1.0)
    var b = SIMD[.float32, _FRAG](1.0)
    var c = SIMD[.float32, _FRAG](0.0)

    var d = llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[.float32, _FRAG],
    ](a, b, c)

    if Int(lane_id()) == 0:
        out_ptr[unsafe_offset=0] = d[0]
        out_ptr[unsafe_offset=1] = d[1]


def main() raises:
    var ctx = DeviceContext()
    var out_dev = ctx.enqueue_create_buffer[.float32](2)
    ctx.enqueue_function[_probe_mm_kernel](
        out_dev.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(32, 1, 1),
    )
    var out_host = List[Float32](length=2, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=out_host.unsafe_ptr(), src_buf=out_dev)
    ctx.synchronize()

    # CHECK: [probe] lane-0 d[0..2] = 8.0 8.0
    print("[probe] lane-0 d[0..2] =", out_host[0], out_host[1])
