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
"""Provides examples for random numbers."""

from std.random import seed
from testing import *


fn test_seed():
    from std.random import seed

    seed()


fn test_constant_seed():
    from std.random import seed

    seed(123456)


fn test_random_float64() raises:
    from std.random import random_float64, seed

    seed()
    var rnd: Float64 = random_float64(10.0, 20.0)
    assert_true(10.0 <= rnd < 20.0)


fn test_random_si64() raises:
    from std.random import random_si64, seed

    seed()
    var rnd: Int64 = random_si64(-100, 100)
    assert_true(Int64(-100) <= rnd <= Int64(100))


fn test_random_ui64() raises:
    from std.random import random_ui64, seed

    seed()
    var rnd: UInt64 = random_ui64(0, 100)
    assert_true(UInt64(0) <= rnd <= UInt64(100))


fn test_rand_count_dtype() raises:
    from std.random import randint, seed
    from memory import alloc

    seed()
    var size: Int = 10
    var ptr = alloc[Int32](size)
    randint[DType.int32](ptr, size, -50, 50)
    for i in range(size):
        assert_true(Int32(-50) <= ptr[i] <= Int32(50))
    ptr.free()


fn test_rand_count_dtype_scale() raises:
    from std.random import rand, seed
    from memory import alloc

    seed()
    var size: Int = 10
    var ptr = alloc[Float32](size)
    rand[DType.float32](ptr, size, min=0.0, max=1.0, int_scale=16)
    for i in range(size):
        assert_true(Float32(0.0) <= ptr[i] < Float32(1.0))

        _ = ptr[i]
    ptr.free()


fn test_randn_float64() raises:
    from std.random import randn_float64, seed

    seed()
    for i in range(10000):
        _ = randn_float64(0.0, 1.0)


fn test_dtype_randn() raises:
    from std.random import randn, seed
    from memory import alloc

    seed()
    var size: Int = 1000
    var ptr = alloc[Float64](size)
    std.random.randn[DType.float64](ptr, size, mean=0.0, standard_deviation=1.0)
    ptr.free()


fn test_shuffle() raises:
    from std.random import shuffle, random_ui64
    from std.collections.list import List

    var list: List[Int] = [0, 1, 2, 3, 4, 5]
    shuffle(list)
    # The list elements are now in random order
    assert_true(len(list) == 6)


fn main() raises:
    test_seed()
    test_constant_seed()
    test_random_float64()
    test_random_si64()
    test_random_ui64()
    test_rand_count_dtype()
    test_rand_count_dtype_scale()
    test_randn_float64()
    test_dtype_randn()
    test_shuffle()
