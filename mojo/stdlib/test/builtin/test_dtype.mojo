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

from sys import size_of

from testing import assert_equal, assert_false, assert_true, TestSuite

comptime uint_dtypes = [
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.uint128,
    DType.uint256,
]

comptime int_dtypes = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.int128,
    DType.int256,
]

comptime non_index_integral_dtypes = uint_dtypes + int_dtypes
comptime integral_dtypes = [DType.int, DType.uint] + non_index_integral_dtypes

comptime float_dtypes = [
    DType.float8_e3m4,
    DType.float8_e4m3fn,
    DType.float8_e4m3fnuz,
    DType.float8_e5m2,
    DType.float8_e5m2fnuz,
    DType.bfloat16,
    DType.float16,
    DType.float32,
    DType.float64,
]

comptime all_dtypes = (
    [DType.bool] + integral_dtypes + float_dtypes + [DType.invalid]
)


fn test_equality() raises:
    assert_true(DType.float32 == DType.float32)
    assert_true(DType.float32 != DType.int32)
    assert_true(DType.float32 == DType.float32)
    assert_true(DType.float32 != DType.int32)


fn test_stringable() raises:
    assert_equal(String(DType.bool), "bool")
    assert_equal(String(DType.int), "int")
    assert_equal(String(DType.uint), "uint")
    assert_equal(String(DType.int64), "int64")
    assert_equal(String(DType.float32), "float32")


fn _test_repr(value: DType, expected: String) raises:
    assert_equal(value.__repr__(), expected)
    var string = String()
    value.write_repr_to(string)
    assert_equal(string, expected)


fn test_representable() raises:
    _test_repr(DType.bool, "DType.bool")
    _test_repr(DType.int, "DType.int")
    _test_repr(DType.uint, "DType.uint")
    _test_repr(DType.int64, "DType.int64")
    _test_repr(DType.float32, "DType.float32")


fn test_is_xxx() raises:
    fn _is_category[
        test: fn(DType) -> Bool,
        true_dtypes: List[DType],
    ]() raises:
        comptime for dt in all_dtypes:
            comptime res = dt in true_dtypes
            assert_equal(test(dt), res)

    # _is_category[DType.is_integral, integral_dtypes]()
    # _is_category[DType.is_floating_point, float_dtypes]()
    _is_category[DType.is_unsigned, [DType.uint] + uint_dtypes]()
    # _is_category[DType.is_signed, [DType.int] + int_dtypes + float_dtypes]()


fn test_key_element() raises:
    var s = {DType.bool, DType.int64}
    assert_true(DType.int64 in s)
    assert_false(DType.float32 in s)


def test_from_str():
    comptime dt = DType._from_str("bool")
    assert_equal(dt, DType.bool)

    assert_equal(DType._from_str("bool"), DType.bool)
    assert_equal(DType._from_str("DType.bool"), DType.bool)

    assert_equal(DType._from_str("int64"), DType.int64)
    assert_equal(DType._from_str("DType.int64"), DType.int64)

    assert_equal(DType._from_str("bfloat16"), DType.bfloat16)
    assert_equal(DType._from_str("DType.bfloat16"), DType.bfloat16)

    assert_equal(DType._from_str("blahblah"), DType.invalid)
    assert_equal(DType._from_str("DType.blahblah"), DType.invalid)

    comptime for dt in all_dtypes:
        assert_equal(DType._from_str(String(dt)), dt)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
