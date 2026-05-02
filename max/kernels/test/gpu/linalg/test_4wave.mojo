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
"""AMD 4-wave matmul correctness test (FP8 only).

Compile-time configuration:
  mojo -D M=128 -D N=4096 -D K=4096 test_4wave.mojo

Defaults to a small smoke shape (M=128, N=128, K=128).
"""

from std.sys import get_defined_int, get_defined_string

from layout import Idx, Coord, TileTensor, row_major
from std.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from std.testing import assert_equal
from std.random import random_float64
from linalg.matmul.gpu.amd.amd_4wave_matmul import (
    amd_4wave_matmul,
)

comptime TEST_M = get_defined_int["M", 128]()
comptime TEST_N = get_defined_int["N", 512]()
comptime TEST_K = get_defined_int["K", 512]()
comptime DUMP_GPU_ASM = get_defined_string["DUMP_GPU_ASM", ""]()


def test_4wave[
    M: Int,
    N: Int,
    K: Int,
    enable_swizzle: Bool,
](ctx: DeviceContext) raises:
    """Test 4-wave FP8 kernel against vendor BLAS."""
    comptime in_dtype = DType.float8_e4m3fn

    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        # Small range [-0.5, 0.5] keeps values representable in FP8 while
        # exercising non-trivial accumulation.
        for i in range(M * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[in_dtype]()
        for i in range(K * N):
            host_b[i] = random_float64(-0.5, 0.5).cast[in_dtype]()

    var a_tt = TileTensor(device_a, row_major[M, K]())
    var b_tt = TileTensor(device_b, row_major[N, K]())
    var c_tt = TileTensor(device_c, row_major[M, N]())

    ctx.enqueue_memset(device_c_ref, 0)
    var c_ref_tt = TileTensor(device_c_ref, row_major[M, N]())
    vendor_blas.matmul(
        ctx,
        c_ref_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )

    ctx.enqueue_memset(device_c, 0)

    amd_4wave_matmul[enable_swizzle=enable_swizzle, dump_asm_path=DUMP_GPU_ASM](
        a_tt,
        b_tt,
        c_tt,
        ctx,
    )

    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        var printed = 0
        var max_rel_err = Float32(0.0)
        var rel_tol = Float32(0.05)  # FP8 16x16x128 precision
        var abs_tol = Float32(1e-5)
        for i in range(M * N):
            var actual = host_c[i]
            var expected = host_c_ref[i]
            var diff = abs(actual - expected)
            var denom = max(abs(expected), abs_tol)
            var rel_err = diff / denom
            max_rel_err = max(max_rel_err, rel_err)
            if rel_err > rel_tol:
                if printed < 10:
                    var row, col = divmod(i, N)
                    print(
                        "Mismatch at (",
                        row,
                        ",",
                        col,
                        "): actual=",
                        actual,
                        " expected=",
                        expected,
                        " rel_err=",
                        rel_err,
                    )
                    printed += 1
                errors += 1

        print("  Max relative error:", max_rel_err)
        if errors != 0:
            print("  Error count:", errors, "out of", M * N)
        assert_equal(errors, 0)


def main() raises:
    with DeviceContext() as ctx:
        print("Running AMD 4-wave Kernel Tests (FP8)")
        print("  Shape: M=", TEST_M, " N=", TEST_N, " K=", TEST_K, sep="")

        print("  Testing without swizzle...")
        test_4wave[TEST_M, TEST_N, TEST_K, enable_swizzle=False](ctx)
        print("  PASSED: No swizzle")

        print("  Testing with swizzle...")
        test_4wave[TEST_M, TEST_N, TEST_K, enable_swizzle=True](ctx)
        print("  PASSED: With swizzle")

        print("==== AMD 4-wave Tests passed ====")
