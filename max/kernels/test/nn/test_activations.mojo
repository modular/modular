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

from math import iota
from random import randn, seed
from sys.info import CompilationTarget

from nn.activations import elu, leaky_relu, relu, relu_n1
from test_utils import compare, libm_call
from testing import assert_almost_equal


# CHECK-LABEL: test_elu
fn test_elu():
    print("== test_elu")

    var simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(elu(simd_val))

    # CHECK: [-0.86466{{[0-9]+}}, -0.63212{{[0-9]+}}, 0.0, 1.0]
    print(elu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(elu(0.5 * simd_val))


# CHECK-LABEL: test_relu
fn test_relu():
    print("== test_relu")

    var simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(relu(simd_val))

    # CHECK: [0.0, 0.0, 0.0, 1.0]
    print(relu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(relu(0.5 * simd_val))


# CHECK-LABEL: test_relu_n1
fn test_relu_n1():
    print("== test_relu_n1")

    var simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 1.0, 1.0]
    print(relu_n1(simd_val))

    # CHECK: [-1.0, -1.0, 0.0, 1.0]
    print(relu_n1(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.0]
    print(relu_n1(0.5 * simd_val))


# CHECK-LABEL: test_leaky_relu
fn test_leaky_relu():
    print("== test_leaky_relu")

    var simd_val = iota[DType.float32, 4]()

    # Test with negative slope of 0.01
    var slope_001 = Float32(0.01)

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(leaky_relu(simd_val, slope_001))

    # For negative values: [-2, -1, 0, 1] with slope 0.01
    # Expected: [-0.02, -0.01, 0.0, 1.0]
    # CHECK: [-0.02, -0.01, 0.0, 1.0]
    print(leaky_relu(simd_val - 2, slope_001))

    # Test with different slope (0.1)
    var slope_01 = Float32(0.1)

    # For negative values: [-2, -1, 0, 1] with slope 0.1
    # Expected: [-0.2, -0.1, 0.0, 1.0]
    # CHECK: [-0.2, -0.1, 0.0, 1.0]
    print(leaky_relu(simd_val - 2, slope_01))


@always_inline
fn erf_libm[
    dtype: DType, simd_width: Int
](arg: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    return libm_call["erff", "err"](arg)


def main():
    test_elu()
    test_relu()
    test_relu_n1()
    test_leaky_relu()
