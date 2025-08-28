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

from sys import (
    CompilationTarget,
    align_of,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    size_of,
)

from testing import assert_equal, assert_true


fn test_size_of() raises:
    assert_equal(size_of[__mlir_type.i16](), 2)

    assert_equal(size_of[__mlir_type.ui16](), 2)

    assert_equal(size_of[DType.int16](), 2)

    assert_equal(size_of[DType.uint16](), 2)

    assert_equal(size_of[SIMD[DType.int16, 2]](), 4)


fn test_align_of() raises:
    assert_true(align_of[__mlir_type.i16]() > 0)

    assert_true(align_of[__mlir_type.ui16]() > 0)

    assert_true(align_of[DType.int16]() > 0)

    assert_true(align_of[DType.uint16]() > 0)

    assert_true(align_of[SIMD[DType.int16, 2]]() > 0)


fn test_cores() raises:
    assert_true(num_logical_cores() > 0)
    assert_true(num_physical_cores() > 0)
    assert_true(num_performance_cores() > 0)


fn test_target_has_feature():
    # Ensures target feature check functions exist and return a boolable value.
    var _has_feature: Bool = CompilationTarget.has_avx()
    _has_feature = CompilationTarget.has_avx2()
    _has_feature = CompilationTarget.has_avx512f()
    _has_feature = CompilationTarget.has_fma()
    _has_feature = CompilationTarget.has_intel_amx()
    _has_feature = CompilationTarget.has_neon()
    _has_feature = CompilationTarget.has_neon_int8_dotprod()
    _has_feature = CompilationTarget.has_neon_int8_matmul()
    _has_feature = CompilationTarget.has_sse4()
    _has_feature = CompilationTarget.has_vnni()


def main():
    test_size_of()
    test_align_of()
    test_cores()
    test_target_has_feature()
