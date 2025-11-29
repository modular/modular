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

from complex import ComplexScalar
from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from math import ceil
from sys.info import has_accelerator, size_of
from utils.numerics import nan

from testing import assert_almost_equal

from nn._fft.fft import (
    _cpu_fft_kernel_radix_n,
    _intra_block_fft_kernel_radix_n,
    _launch_inter_multiprocessor_fft,
)
from nn._fft.utils import _get_ordered_bases_processed_list

comptime _TestValues[complex_dtype: DType] = List[
    Tuple[List[Int], List[ComplexScalar[complex_dtype]]]
]


fn _get_test_values_2[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 2.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    res = [
        (List(0, 0), List(Complex(0), Complex(0))),
        (List(1, 0), List(Complex(1), Complex(1))),
        (List(1, -1), List(Complex(0), Complex(2))),
        (List(18, 7), List(Complex(25, 0), Complex(11, 0))),
        (List(4, 8), List(Complex(12, 0), Complex(-4, 0))),
        (List(5, 4), List(Complex(9, 0), Complex(1, 0))),
    ]


fn _get_test_values_3[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 3.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    res = [
        (List(0, 0, 0), List(Complex(0), Complex(0), Complex(0))),
        (
            List(1, 0, 1),
            List(Complex(2), Complex(0.5, 0.866), Complex(0.5, -0.866)),
        ),
        (
            List(1, -1, 1),
            List(Complex(1), Complex(1, 1.732), Complex(1, -1.732)),
        ),
        (
            List(18, 7, 29),
            List(Complex(54, 0), Complex(0, 19.053), Complex(0, -19.053)),
        ),
        (
            List(4, 8, 15),
            List(Complex(27, 0), Complex(-7.5, 6.062), Complex(-7.5, -6.062)),
        ),
        (
            List(5, 4, 3),
            List(Complex(12, 0), Complex(1.5, -0.866), Complex(1.5, 0.866)),
        ),
    ]


fn _get_test_values_4[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 4.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(0, 0, 0, 0),
            List(Complex(0), Complex(0), Complex(0), Complex(0)),
        ),
        (
            List(1, 0, 1, 0),
            List(Complex(2), Complex(0), Complex(2), Complex(0)),
        ),
        (
            List(1, -1, 1, -1),
            List(Complex(0), Complex(0), Complex(4), Complex(0)),
        ),
        (
            List(18, 7, 29, 27),
            List(
                Complex(81, 0), Complex(-11, 20), Complex(13, 0),
                Complex(-11, -20),
            ),
        ),
        (
            List(4, 8, 15, 16),
            List(
                Complex(43, 0), Complex(-11, 8), Complex(-5, 0),
                Complex(-11, -8),
            ),
        ),
        (
            List(5, 4, 3, 2),
            List(Complex(14, 0), Complex(2, -2), Complex(2, 0), Complex(2, 2)),
        ),
    ]
    # fmt: on


fn _get_test_values_5[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 5.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(0, 0, 0, 0, 0),
            List(
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(1, 0, 1, 0, 1),
            List(
                Complex(3), Complex(0.5, 0.363), Complex(0.5, 1.539),
                Complex(0.5, -1.539), Complex(0.5, -0.363),
            ),
        ),
        (
            List(18, 7, 29, 27, 42),
            List(
                Complex(123, 0), Complex(-12.163, 32.111),
                Complex(-4.337, 22.475), Complex(-4.337, -22.475),
                Complex(-12.163, -32.111),
            ),
        ),
        (
            List(4, 8, 15, 16, 23),
            List(
                Complex(66, 0), Complex(-11.5, 14.854), Complex(-11.5, 7.866),
                Complex(-11.5, -7.866), Complex(-11.5, -14.854),
            ),
        ),
        (
            List(1, -1, 1, -1, 5),
            List(
                Complex(5, 0), Complex(2.236, 4.531), Complex(-2.236, 5.429),
                Complex(-2.236, -5.429), Complex(2.236, -4.531),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_6[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 6.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(0, 0, 0, 0, 0, 0),
            List(
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0),
            ),
        ),
        (
            List(1, 0, 1, 0, 1, 0),
            List(
                Complex(3), Complex(0), Complex(0),
                Complex(3), Complex(0), Complex(0),
            ),
        ),
        (
            List(18, 7, 29, 27, 42, 34),
            List(
                Complex(157, 0), Complex(-24, 34.641), Complex(-11, 12.124),
                Complex(21, 0), Complex(-11, -12.124), Complex(-24, -34.641),
            ),
        ),
        (
            List(4, 8, 15, 16, 23, 42),
            List(
                Complex(108, 0), Complex(-6, 36.373), Complex(-24, 22.517),
                Complex(-24, 0), Complex(-24, -22.517), Complex(-6, -36.373),
            ),
        ),
        (
            List(1, -1, 1, -1, 5, 4),
            List(
                Complex(9, 0), Complex(0.5, 7.794), Complex(-4.5, 0.866),
                Complex(5, 0), Complex(-4.5, -0.866), Complex(0.5, -7.794),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_7[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 7.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(0, 0, 0, 0, 0, 0, 0),
            List(
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0),
            ),
        ),
        (
            List(1, 0, 1, 0, 1, 0, 1),
            List(
                Complex(4), Complex(0.5, 0.241), Complex(0.5, 0.627),
                Complex(0.5, 2.191), Complex(0.5, -2.191), Complex(0.5, -0.627),
                Complex(0.5, -0.241),
            ),
        ),
        (
            List(18, 7, 29, 27, 42, 34, 11),
            List(
                Complex(168, 0), Complex(-46.963, 14.51),
                Complex(0.254, -9.997), Complex(25.708, 12.45),
                Complex(25.708, -12.45), Complex(0.254, 9.997),
                Complex(-46.963, -14.51),
            ),
        ),
        (
            List(4, 8, 15, 16, 23, 42, 0),
            List(
                Complex(108, 0), Complex(-38.834, 23.106),
                Complex(-24.819, -24.987), Complex(23.653, -17.756),
                Complex(23.653, 17.756), Complex(-24.819, 24.987),
                Complex(-38.834, -23.106),
            ),
        ),
        (
            List(1, -1, 1, -1, 5, 4, 3),
            List(
                Complex(12, 0), Complex(-2.47, 8.655), Complex(-1.456, -2.093),
                Complex(1.425, 5.24), Complex(1.425, -5.24),
                Complex(-1.456, 2.093), Complex(-2.47, -8.655),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_8[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 8.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(0, 0, 0, 0, 0, 0, 0, 0),
            List(
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(4), Complex(0), Complex(0), Complex(0),
                Complex(4), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(18, 7, 29, 27, 42, 34, 11, 10),
            List(
                Complex(178, 0), Complex(-55.113, -10.929),
                Complex(20, -4), Complex(7.113, 25.071), Complex(22, 0),
                Complex(7.113, -25.071), Complex(20, 4),
                Complex(-55.113, 10.929),
            ),
        ),
        (
            List(4, 8, 15, 16, 23, 42, 0, 0),
            List(
                Complex(108, 0), Complex(-54.355, -2.272), Complex(12, -34),
                Complex(16.355, 27.728), Complex(-24, 0),
                Complex(16.355, -27.728), Complex(12, 34),
                Complex(-54.355, 2.272),
            ),
        ),
        (
            List(1, -1, 1, -1, 5, 4, 3, 2),
            List(
                Complex(14, 0), Complex(-5.414, 7.657), Complex(2, -2),
                Complex(-2.586, 3.657), Complex(6, 0), Complex(-2.586, -3.657),
                Complex(2, 2), Complex(-5.414, -7.657),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_10[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 10.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            List(
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(5), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(5), Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(18, 7, 29, 27, 42, 34, 11, 10, 21, 17),
            List(
                Complex(216, 0), Complex(-35.444, -36.12),
                Complex(5.41, 44.283), Complex(-17.556, -5.278),
                Complex(16.59, 15.54), Complex(26, 0),
                Complex(16.59, -15.54), Complex(-17.556, 5.278),
                Complex(5.41, -44.283), Complex(-35.444, 36.12),
            ),
        ),
        (
            List(4, 8, 15, 16, 23, 42, 0, 0, 0, 0),
            List(
                Complex(108, 0), Complex(-50.444, -47.704),
                Complex(30.5, 14.854), Complex(-32.556, -11.261),
                Complex(30.5, 7.866), Complex(-24, 0),
                Complex(30.5, -7.866), Complex(-32.556, 11.261),
                Complex(30.5, -14.854), Complex(-50.444, 47.704),
            ),
        ),
        (
            List(1, -1, 1, -1, 5, 4, 3, 2, 1, 1),
            List(
                Complex(16, 0), Complex(-9.163, 2.853), Complex(5.045, 2.041),
                Complex(-1.337, -1.763), Complex(-0.545, 5.204),
                Complex(6, 0), Complex(-0.545, -5.204),
                Complex(-1.337, 1.763), Complex(5.045, -2.041),
                Complex(-9.163, -2.853),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_16[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 16.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(8), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0),
                Complex(8), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(18, 7, 29, 27, 42, 34, 11, 10, 12, 15, 13, 22, 26, 31, 19, 21),
            List(
                Complex(337, 0), Complex(26.508, -21.777),
                Complex(-81.134, 5.678), Complex(-11.67, 30.958),
                Complex(26, -7), Complex(-10.271, 10.272),
                Complex(5.134, 29.678), Complex(19.434, 21.537), Complex(3, 0),
                Complex(19.434, -21.537), Complex(5.134, -29.678),
                Complex(-10.271, -10.272), Complex(26, 7),
                Complex(-11.67, -30.958), Complex(-81.134, -5.678),
                Complex(26.508, 21.777),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_20[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 20.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0),
            List(
                Complex(10), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(10), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15),
            List(
                Complex(811, 0), Complex(-62.611, 52.731),
                Complex(-187.325, -170.06), Complex(173.006, 63.944),
                Complex(-87.728, -10.045), Complex(-152, 9),
                Complex(-10.675, 45.958), Complex(10.575, -147.347),
                Complex(39.728, 5.621), Complex(-48.969, 79.866),
                Complex(101, 0), Complex(-48.969, -79.866),
                Complex(39.728, -5.621), Complex(10.575, 147.347),
                Complex(-10.675, -45.958), Complex(-152, -9),
                Complex(-87.728, 10.045), Complex(173.006, -63.944),
                Complex(-187.325, 170.06), Complex(-62.611, -52.731),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_32[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 32.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(16), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0),
                Complex(16), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0),
            ),
        ),
        (
            List(18, 7, 29, 27, 42, 34, 11, 10, 12, 15, 13, 22, 26, 31, 19, 21,
            25, 50, 64, 71, 45, 88, 91, 38, 27, 31, 90, 41, 99, 112, 76, 51),
            List(
                Complex(1336, 0), Complex(57.903, 436.536),
                Complex(45.745, 32.162), Complex(-22.34, 204.468),
                Complex(-273.543, 86.56), Complex(-128.005, 24.107),
                Complex(-22.323, -18.511), Complex(-17.104, 111.152),
                Complex(-99, -87), Complex(63.744, -111.88),
                Complex(54.364, 53.247), Complex(-7.371, 51.756),
                Complex(13.543, 84.56), Complex(-68.274, 77.156),
                Complex(-61.787, -48.081), Complex(65.447, -61.456),
                Complex(38, 0), Complex(65.447, 61.456),
                Complex(-61.787, 48.081), Complex(-68.274, -77.156),
                Complex(13.543, -84.56), Complex(-7.371, -51.756),
                Complex(54.364, -53.247), Complex(63.744, 111.88),
                Complex(-99, 87), Complex(-17.104, -111.152),
                Complex(-22.323, 18.511), Complex(-128.005, -24.107),
                Complex(-273.543, -86.56), Complex(-22.34, -204.468),
                Complex(45.745, -32.162), Complex(57.903, -436.536),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_21[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 21.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1),
            List(
                Complex(11, 0), Complex(0.5, 0.075), Complex(0.5, 0.154),
                Complex(0.5, 0.241), Complex(0.5, 0.341), Complex(0.5, 0.464),
                Complex(0.5, 0.627), Complex(0.5, 0.866), Complex(0.5, 1.274),
                Complex(0.5, 2.191), Complex(0.5, 6.672), Complex(0.5, -6.672),
                Complex(0.5, -2.191), Complex(0.5, -1.274),
                Complex(0.5, -0.866), Complex(0.5, -0.627),
                Complex(0.5, -0.464), Complex(0.5, -0.341),
                Complex(0.5, -0.241), Complex(0.5, -0.154),
                Complex(0.5, -0.075),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63),
            List(
                Complex(874, 0), Complex(-50.819, 45.598),
                Complex(-92.208, -190.546), Complex(142.842, 183.809),
                Complex(-88.629, -41.252), Complex(-98.965, -32.472),
                Complex(4.959, 95.899), Complex(109, 13.856),
                Complex(-99.639, 99.55), Complex(-126.301, 5.872),
                Complex(-0.74, 88.692), Complex(-0.74, -88.692),
                Complex(-126.301, -5.872), Complex(-99.639, -99.55),
                Complex(109, -13.856), Complex(4.959, -95.899),
                Complex(-98.965, 32.472), Complex(-88.629, 41.252),
                Complex(142.842, -183.809), Complex(-92.208, 190.546),
                Complex(-50.819, -45.598),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_30[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 30.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(15, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(15, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100),
            List(
                Complex(1427, 0), Complex(39.788, 109.878),
                Complex(52.927, 137.741), Complex(-201.977, -126.286),
                Complex(-50.673, 274.824), Complex(57.5, -172.339),
                Complex(-10.702, -49.419), Complex(11.555, -5.267),
                Complex(-62.609, 190.585), Complex(5.977, 108.279),
                Complex(135.5, 68.416), Complex(-145.144, 58.734),
                Complex(-35.298, 141.634), Complex(-271.199, 71.903),
                Complex(-97.645, 10.584), Complex(107, 0),
                Complex(-97.645, -10.584), Complex(-271.199, -71.903),
                Complex(-35.298, -141.634), Complex(-145.144, -58.734),
                Complex(135.5, -68.416), Complex(5.977, -108.279),
                Complex(-62.609, -190.585), Complex(11.555, 5.267),
                Complex(-10.702, 49.419), Complex(57.5, 172.339),
                Complex(-50.673, -274.824), Complex(-201.977, 126.286),
                Complex(52.927, -137.741), Complex(39.788, -109.878),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_35[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 35.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1),
            List(
                Complex(18, 0), Complex(0.5, 0.045), Complex(0.5, 0.091),
                Complex(0.5, 0.138), Complex(0.5, 0.188), Complex(0.5, 0.241),
                Complex(0.5, 0.299), Complex(0.5, 0.363), Complex(0.5, 0.437),
                Complex(0.5, 0.523), Complex(0.5, 0.627), Complex(0.5, 0.758),
                Complex(0.5, 0.929), Complex(0.5, 1.17), Complex(0.5, 1.539),
                Complex(0.5, 2.191), Complex(0.5, 3.691), Complex(0.5, 11.133),
                Complex(0.5, -11.133), Complex(0.5, -3.691),
                Complex(0.5, -2.191), Complex(0.5, -1.539), Complex(0.5, -1.17),
                Complex(0.5, -0.929), Complex(0.5, -0.758),
                Complex(0.5, -0.627), Complex(0.5, -0.523),
                Complex(0.5, -0.437), Complex(0.5, -0.363),
                Complex(0.5, -0.299), Complex(0.5, -0.241),
                Complex(0.5, -0.188), Complex(0.5, -0.138),
                Complex(0.5, -0.091), Complex(0.5, -0.045),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100, 92, 6,
            21, 72, 45),
            List(
                Complex(1663, 0), Complex(-46.097, 158.832),
                Complex(-123.731, 71.088), Complex(26.612, -225.737),
                Complex(-234.899, 20.137), Complex(181.511, 201.552),
                Complex(29.716, -142.556), Complex(21.82, 17.649),
                Complex(-45.288, 135.255), Complex(-308.162, 153.464),
                Complex(-142.842, 24.65), Complex(-61.186, -50.519),
                Complex(139.693, -240.477), Complex(-13.862, -3.408),
                Complex(44.18, 116.054), Complex(-198.169, -55.571),
                Complex(60.713, 18.17), Complex(65.99, 135.709),
                Complex(65.99, -135.709), Complex(60.713, -18.17),
                Complex(-198.169, 55.571), Complex(44.18, -116.054),
                Complex(-13.862, 3.408), Complex(139.693, 240.477),
                Complex(-61.186, 50.519), Complex(-142.842, -24.65),
                Complex(-308.162, -153.464), Complex(-45.288, -135.255),
                Complex(21.82, -17.649), Complex(29.716, 142.556),
                Complex(181.511, -201.552), Complex(-234.899, -20.137),
                Complex(26.612, 225.737), Complex(-123.731, -71.088),
                Complex(-46.097, -158.832),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_48[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 48.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(24, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(24, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100, 92, 6,
            21, 72, 45, 30, 62, 97, 23, 60, 8, 9, 86, 53, 75, 70, 25, 50),
            List(
                Complex(2311, 0), Complex(-155.238, 103.773),
                Complex(55.286, -12.729), Complex(-25.88, 168.922),
                Complex(100.861, -141.055), Complex(-264.564, 35.071),
                Complex(-370.685, 214.484), Complex(106.279, 4.799),
                Complex(119.5, -265.87), Complex(-18.664, -52.952),
                Complex(31.727, -63.158), Complex(-53.04, 146.406),
                Complex(-270, -155), Complex(138.721, 143.776),
                Complex(-113.846, 176.961), Complex(-11.898, -128.875),
                Complex(356.5, 45.899), Complex(-59.39, -16.344),
                Complex(-55.315, 140.484), Complex(-176.354, 210.282),
                Complex(-94.861, -199.945), Complex(-171.559, 60.998),
                Complex(80.833, 9.152), Complex(-40.415, 172.79),
                Complex(97, 0), Complex(-40.415, -172.79),
                Complex(80.833, -9.152), Complex(-171.559, -60.998),
                Complex(-94.861, 199.945), Complex(-176.354, -210.282),
                Complex(-55.315, -140.484), Complex(-59.39, 16.344),
                Complex(356.5, -45.899), Complex(-11.898, 128.875),
                Complex(-113.846, -176.961), Complex(138.721, -143.776),
                Complex(-270, 155), Complex(-53.04, -146.406),
                Complex(31.727, 63.158), Complex(-18.664, 52.952),
                Complex(119.5, 265.87), Complex(106.279, -4.799),
                Complex(-370.685, -214.484), Complex(-264.564, -35.071),
                Complex(100.861, 141.055), Complex(-25.88, -168.922),
                Complex(55.286, 12.729), Complex(-155.238, -103.773),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_60[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 60.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(30, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(30, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100, 92, 6,
            21, 72, 45, 30, 62, 97, 23, 60, 8, 9, 86, 53, 75, 70, 25, 50,
            20, 96, 27, 83, 88, 76, 82, 42, 89, 69, 94, 38),
            List(
                Complex(3115, 0), Complex(-6.356, 162.764),
                Complex(90.985, 302.903), Complex(-132.771, 100.985),
                Complex(-19.789, 219.018), Complex(24.717, -142.165),
                Complex(-312.819, -126.037), Complex(-164.982, -47.041),
                Complex(45.925, 501.211), Complex(1.913, 132.133),
                Complex(46.5, -229.497), Complex(-110.561, 8.075),
                Complex(-140.614, 23.701), Complex(-222.601, -156.299),
                Complex(7.846, 23.149), Complex(-276, -233),
                Complex(70.112, 238.54), Complex(-94.586, -26.794),
                Complex(86.319, 200.128), Complex(-213.72, -71.276),
                Complex(335.5, -32.043), Complex(-88.759, -134.253),
                Complex(-124.809, -41.428), Complex(14.802, -16.127),
                Complex(-76.886, 282.771), Complex(-198.717, -318.835),
                Complex(-125.522, -172.624), Complex(175.617, 59.896),
                Complex(88.252, -23.032), Complex(107.002, 239.96),
                Complex(93, 0), Complex(107.002, -239.96),
                Complex(88.252, 23.032), Complex(175.617, -59.896),
                Complex(-125.522, 172.624), Complex(-198.717, 318.835),
                Complex(-76.886, -282.771), Complex(14.802, 16.127),
                Complex(-124.809, 41.428), Complex(-88.759, 134.253),
                Complex(335.5, 32.043), Complex(-213.72, 71.276),
                Complex(86.319, -200.128), Complex(-94.586, 26.794),
                Complex(70.112, -238.54), Complex(-276, 233),
                Complex(7.846, -23.149), Complex(-222.601, 156.299),
                Complex(-140.614, -23.701), Complex(-110.561, -8.075),
                Complex(46.5, 229.497), Complex(1.913, -132.133),
                Complex(45.925, -501.211), Complex(-164.982, 47.041),
                Complex(-312.819, 126.037), Complex(24.717, 142.165),
                Complex(-19.789, -219.018), Complex(-132.771, -100.985),
                Complex(90.985, -302.903), Complex(-6.356, -162.764),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_64[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 64.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(32, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0),
                Complex(32, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100, 92, 6,
            21, 72, 45, 30, 62, 97, 23, 60, 8, 9, 86, 53, 75, 70, 25, 50,
            20, 96, 27, 83, 88, 76, 82, 42, 89, 69, 94, 38, 33, 35, 17, 14),
            List(
                Complex(3214, 0), Complex(-152.807, 180.606),
                Complex(-160.637, 287.126), Complex(-267.159, -80.776),
                Complex(-187.611, 14.835), Complex(97.545, -225.69),
                Complex(-124.89, -349.256), Complex(-158.067, 4.829),
                Complex(-390.463, 108.338), Complex(290.161, 7.15),
                Complex(282.136, 58.526), Complex(-32.406, -206.452),
                Complex(1.596, -14.44), Complex(-32.741, 64.737),
                Complex(-252.24, -79.6), Complex(23.526, -4.319),
                Complex(-260, -254), Complex(57.1, 251.915),
                Complex(-34.079, -64.172), Complex(14.833, 194.424),
                Complex(16.519, -203.192), Complex(255.136, 259.221),
                Complex(26.759, -223.271), Complex(-57.581, -58.344),
                Complex(-59.537, 78.338), Complex(-115.066, -60.228),
                Complex(334.685, 144.109), Complex(-263.625, 65.441),
                Complex(-334.503, 14.082), Complex(68.249, -47.171),
                Complex(32.265, -82.285), Complex(144.901, 211.738),
                Complex(94, 0), Complex(144.901, -211.738),
                Complex(32.265, 82.285), Complex(68.249, 47.171),
                Complex(-334.503, -14.082), Complex(-263.625, -65.441),
                Complex(334.685, -144.109), Complex(-115.066, 60.228),
                Complex(-59.537, -78.338), Complex(-57.581, 58.344),
                Complex(26.759, 223.271), Complex(255.136, -259.221),
                Complex(16.519, 203.192), Complex(14.833, -194.424),
                Complex(-34.079, 64.172), Complex(57.1, -251.915),
                Complex(-260, 254), Complex(23.526, 4.319),
                Complex(-252.24, 79.6), Complex(-32.741, -64.737),
                Complex(1.596, 14.44), Complex(-32.406, 206.452),
                Complex(282.136, -58.526), Complex(290.161, -7.15),
                Complex(-390.463, -108.338), Complex(-158.067, -4.829),
                Complex(-124.89, 349.256), Complex(97.545, 225.69),
                Complex(-187.611, -14.835), Complex(-267.159, 80.776),
                Complex(-160.637, -287.126), Complex(-152.807, -180.606),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_100[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 100.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0),
            List(
                Complex(50, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(50, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100, 92, 6,
            21, 72, 45, 30, 62, 97, 23, 60, 8, 9, 86, 53, 75, 70, 25, 50,
            20, 96, 27, 83, 88, 76, 82, 42, 89, 69, 94, 38, 33, 35, 17, 14,
            26, 67, 99, 32, 95, 44, 64, 36, 66, 57, 37, 93, 19, 81, 54, 7,
            59, 3, 58, 34, 46, 77, 80, 47, 18, 85, 68, 84, 65, 5, 39, 4,
            52, 56, 55, 24),
            List(
                Complex(5050.0, 0.0), Complex(-342.554, 42.072),
                Complex(-89.905, 24.461), Complex(-190.032, 206.142),
                Complex(138.232, -37.501), Complex(-401.25, 38.945),
                Complex(-65.928, -286.365), Complex(279.534, 160.471),
                Complex(115.966, -265.079), Complex(235.004, -187.957),
                Complex(-363.021, 51.122), Complex(-182.725, -63.37),
                Complex(-56.701, 6.718), Complex(-235.125, 674.109),
                Complex(241.023, 173.705), Complex(-76.584, 299.023),
                Complex(25.709, -58.387), Complex(-160.699, -324.266),
                Complex(-209.425, -135.972), Complex(-71.937, -186.774),
                Complex(-64.357, -124.305), Complex(158.868, -140.848),
                Complex(-225.606, -4.173), Complex(48.929, 133.801),
                Complex(-65.997, 14.638), Complex(-368.0, -368.0),
                Complex(-190.375, 229.322), Complex(110.749, 84.386),
                Complex(42.505, -134.671), Complex(-51.182, 167.57),
                Complex(11.521, 227.328), Complex(123.968, -237.565),
                Complex(-190.855, 160.691), Complex(221.825, 146.133),
                Complex(140.681, -421.578), Complex(128.04, -188.533),
                Complex(40.326, 83.908), Complex(-169.621, 120.613),
                Complex(-59.673, 74.887), Complex(-135.272, -125.133),
                Complex(41.857, 309.588), Complex(270.227, 68.296),
                Complex(-695.846, -89.434), Complex(-34.713, -452.209),
                Complex(-73.603, 91.407), Complex(197.794, 134.546),
                Complex(-47.446, -1.843), Complex(91.17, -129.995),
                Complex(-53.083, 284.883), Complex(203.587, -70.967),
                Complex(258.0, 0.0), Complex(203.587, 70.967),
                Complex(-53.083, -284.883), Complex(91.17, 129.995),
                Complex(-47.446, 1.843), Complex(197.794, -134.546),
                Complex(-73.603, -91.407), Complex(-34.713, 452.209),
                Complex(-695.846, 89.434), Complex(270.227, -68.296),
                Complex(41.857, -309.588), Complex(-135.272, 125.133),
                Complex(-59.673, -74.887), Complex(-169.621, -120.613),
                Complex(40.326, -83.908), Complex(128.04, 188.533),
                Complex(140.681, 421.578), Complex(221.825, -146.133),
                Complex(-190.855, -160.691), Complex(123.968, 237.565),
                Complex(11.521, -227.328), Complex(-51.182, -167.57),
                Complex(42.505, 134.671), Complex(110.749, -84.386),
                Complex(-190.375, -229.322), Complex(-368.0, 368.0),
                Complex(-65.997, -14.638), Complex(48.929, -133.801),
                Complex(-225.606, 4.173), Complex(158.868, 140.848),
                Complex(-64.357, 124.305), Complex(-71.937, 186.774),
                Complex(-209.425, 135.972), Complex(-160.699, 324.266),
                Complex(25.709, 58.387), Complex(-76.584, -299.023),
                Complex(241.023, -173.705), Complex(-235.125, -674.109),
                Complex(-56.701, -6.718), Complex(-182.725, 63.37),
                Complex(-363.021, -51.122), Complex(235.004, 187.957),
                Complex(115.966, 265.079), Complex(279.534, -160.471),
                Complex(-65.928, 286.365), Complex(-401.25, -38.945),
                Complex(138.232, 37.501), Complex(-190.032, -206.142),
                Complex(-89.905, -24.461), Complex(-342.554, -42.072),
            ),
        ),
    ]
    # fmt: on


fn _get_test_values_128[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 128.
    """
    comptime Complex = ComplexScalar[complex_dtype]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(64, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(64, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100, 92, 6,
            21, 72, 45, 30, 62, 97, 23, 60, 8, 9, 86, 53, 75, 70, 25, 50,
            20, 96, 27, 83, 88, 76, 82, 42, 89, 69, 94, 38, 33, 35, 17, 14,
            26, 67, 99, 32, 95, 44, 64, 36, 66, 57, 37, 93, 19, 81, 54, 7,
            59, 3, 58, 34, 46, 77, 80, 47, 18, 85, 68, 84, 65, 5, 39, 4,
            52, 56, 55, 24, 80, 100, 91, 31, 79, 56, 41, 1, 87, 68, 81, 83,
            55, 98, 69, 82, 25, 43, 66, 86, 8, 84, 2, 34, 65, 73, 57, 29),
            List(
                Complex(6724.0, 0.0), Complex(-156.29818818, 100.55380564),
                Complex(-154.46186294, 358.35832448), Complex(-316.70533589, 68.87259552),
                Complex(-235.36501962, 168.72629575), Complex(153.36419219, -92.41119721),
                Complex(-167.97031692, -297.19454317), Complex(129.45766604, 252.75436499),
                Complex(-276.25393942, -143.20742064), Complex(473.81747787, 144.00393585),
                Complex(320.84179105, 25.17023746), Complex(-158.88631562, 220.18592989),
                Complex(-301.98202028, -497.41277706), Complex(-303.80261787, -48.39312759),
                Complex(-31.01725790, -135.61181326), Complex(-25.07873719, 256.84842535),
                Complex(-541.98989873, 157.59797974), Complex(59.92361411, 633.35206540),
                Complex(178.19036588, 147.50626608), Complex(-198.01406228, 134.93497624),
                Complex(271.22992526, -58.42798577), Complex(169.03650037, 21.35203365),
                Complex(-367.91968034, -147.58223984), Complex(-314.27893266, -225.99217847),
                Complex(-2.47523190, -166.68442802), Complex(-152.65863234, -53.19315472),
                Complex(-204.04087748, -159.01146349), Complex(220.66531481, -369.42260824),
                Complex(-71.35655134, -37.13141518), Complex(-68.56609846, 293.66828693),
                Complex(-61.10304087, 5.44109104), Complex(-97.66658182, -62.90052013),
                Complex(-376.0, -544.0), Complex(-138.48726553, 159.68362255),
                Complex(76.08025032, 328.85094309), Complex(-14.20841142, 57.48766158),
                Complex(-84.53504416, -290.91404046), Complex(13.63508856, 10.08330984),
                Complex(150.11572968, 91.71350905), Complex(431.86537249, 229.85517425),
                Complex(-26.50543210, -88.35138354), Complex(-206.02323494, 264.85461217),
                Complex(177.55994584, 238.77452385), Complex(407.19246988, -35.05565566),
                Complex(-204.32192289, -106.42113879), Complex(-37.18822859, -219.42794172),
                Complex(-118.71555940, 173.21844419), Complex(-398.99093613, -76.39699575),
                Complex(-146.01010126, -78.40202025), Complex(-32.24435434, -86.59838054),
                Complex(-157.26272934, -267.70569965), Complex(-13.92019860, 183.79322094),
                Complex(247.72582152, -19.97028849), Complex(437.03364716, -451.81031090),
                Complex(-464.63944250, 388.84686941), Complex(35.83260476, -441.59172457),
                Complex(-114.76539655, 163.12562383), Complex(-8.35436859, 199.11883066),
                Complex(317.99397617, -39.65937037), Complex(142.14939725, 180.29153298),
                Complex(194.60481151, 232.37931206), Complex(-379.57004602, 371.85170987),
                Complex(-37.65129124, 313.45244404), Complex(-69.03079898, -118.97609904),
                Complex(196.0, 0.0), Complex(-69.03079898, 118.97609904),
                Complex(-37.65129124, -313.45244404), Complex(-379.57004602, -371.85170987),
                Complex(194.60481151, -232.37931206), Complex(142.14939725, -180.29153298),
                Complex(317.99397617, 39.65937037), Complex(-8.35436859, -199.11883066),
                Complex(-114.76539655, -163.12562383), Complex(35.83260476, 441.59172457),
                Complex(-464.63944250, -388.84686941), Complex(437.03364716, 451.81031090),
                Complex(247.72582152, 19.97028849), Complex(-13.92019860, -183.79322094),
                Complex(-157.26272934, 267.70569965), Complex(-32.24435434, 86.59838054),
                Complex(-146.01010126, 78.40202025), Complex(-398.99093613, 76.39699575),
                Complex(-118.71555940, -173.21844419), Complex(-37.18822859, 219.42794172),
                Complex(-204.32192289, 106.42113879), Complex(407.19246988, 35.05565566),
                Complex(177.55994584, -238.77452385), Complex(-206.02323494, -264.85461217),
                Complex(-26.50543210, 88.35138354), Complex(431.86537249, -229.85517425),
                Complex(150.11572968, -91.71350905), Complex(13.63508856, -10.08330984),
                Complex(-84.53504416, 290.91404046), Complex(-14.20841142, -57.48766158),
                Complex(76.08025032, -328.85094309), Complex(-138.48726553, -159.68362255),
                Complex(-376.0, 544.0), Complex(-97.66658182, 62.90052013),
                Complex(-61.10304087, -5.44109104), Complex(-68.56609846, -293.66828693),
                Complex(-71.35655134, 37.13141518), Complex(220.66531481, 369.42260824),
                Complex(-204.04087748, 159.01146349), Complex(-152.65863234, 53.19315472),
                Complex(-2.47523190, 166.68442802), Complex(-314.27893266, 225.99217847),
                Complex(-367.91968034, 147.58223984), Complex(169.03650037, -21.35203365),
                Complex(271.22992526, 58.42798577), Complex(-198.01406228, -134.93497624),
                Complex(178.19036588, -147.50626608), Complex(59.92361411, -633.35206540),
                Complex(-541.98989873, -157.59797974), Complex(-25.07873719, -256.84842535),
                Complex(-31.01725790, 135.61181326), Complex(-303.80261787, 48.39312759),
                Complex(-301.98202028, 497.41277706), Complex(-158.88631562, -220.18592989),
                Complex(320.84179105, -25.17023746), Complex(473.81747787, -144.00393585),
                Complex(-276.25393942, 143.20742064), Complex(129.45766604, -252.75436499),
                Complex(-167.97031692, 297.19454317), Complex(153.36419219, 92.41119721),
                Complex(-235.36501962, -168.72629575), Complex(-316.70533589, -68.87259552),
                Complex(-154.46186294, -358.35832448), Complex(-156.29818818, -100.55380564)
            ),
        ),
    ]
    # fmt: on


fn test_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    test_num: UInt,
    bases: List[UInt],
    inverse: Bool = False,
    target: StaticString = "cpu",
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    constrained[len(in_layout) == 3, "in_layout must have rank 3"]()
    comptime batches = UInt(in_layout.shape[0].value())
    comptime sequence_length = UInt(in_layout.shape[1].value())
    comptime do_rfft = in_layout.shape[2].value() == 1
    comptime do_complex = in_layout.shape[2].value() == 2

    constrained[
        do_rfft or do_complex,
        "The layout should match one of: {(batches, sequence_length, 1), ",
        "(batches, sequence_length, 2)}",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    comptime bases_processed = _get_ordered_bases_processed_list[
        sequence_length, bases, target
    ]()
    comptime ordered_bases = bases_processed[0]
    comptime processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
        comptime last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = Int((sequence_length // last_base) * (last_base - 1))
        var offsets = List[UInt](capacity=c * len(bases))
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (sequence_length // base) * (base - 1)
        return val, offsets^

    comptime total_offsets = _calc_total_offsets()
    comptime total_twfs = total_offsets[0]
    comptime twf_offsets = total_offsets[1]

    @parameter
    if is_cpu[target]():
        _cpu_fft_kernel_radix_n[
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
        ](output, x)
        return
    constrained[
        has_accelerator(), "The non-cpu implementation is for GPU only"
    ]()

    comptime gpu_info = ctx.default_device_info
    comptime max_threads_per_block = UInt(gpu_info.max_thread_block_size)
    comptime threads_per_sm = gpu_info.threads_per_multiprocessor
    constrained[
        threads_per_sm > 0,
        "Unknown number of threads per sm for the given device. ",
        "It is needed in order to run the gpu implementation.",
    ]()
    comptime max_threads_available = UInt(threads_per_sm * gpu_info.sm_count)
    comptime num_threads = sequence_length // ordered_bases[
        len(ordered_bases) - 1
    ]
    comptime num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.uint]()
    )
    comptime shared_mem_size = UInt(gpu_info.shared_memory_per_multiprocessor)
    comptime output_size = UInt(size_of[out_dtype]()) * sequence_length * 2
    comptime twf_size = UInt(size_of[out_dtype]()) * total_twfs * 2

    @parameter
    if test_num == 0 or test_num == 1:
        comptime batch_size = max_threads_available // num_threads
        comptime func = _intra_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            x.origin,
            output.origin,
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            warp_exec = (
                UInt(gpu_info.warp_size) >= batches * num_threads
            ) if test_num
            == 1 else False,
        ]

        @parameter
        for _ in range(batches // batch_size):
            ctx.enqueue_function_checked[func, func](
                output, x, grid_dim=(1, batch_size), block_dim=Int(num_threads)
            )
        comptime remainder = batches % batch_size

        @parameter
        if remainder > 0:
            ctx.enqueue_function_checked[func, func](
                output, x, grid_dim=(1, remainder), block_dim=Int(num_threads)
            )
    elif test_num == 2 or test_num == 3:
        comptime block_dim = UInt(
            ceil(num_threads / num_blocks).cast[DType.uint]()
        )
        _launch_inter_multiprocessor_fft[
            length=sequence_length,
            processed_list=processed_list,
            ordered_bases=ordered_bases,
            do_rfft=do_rfft,
            inverse=inverse,
            block_dim=block_dim,
            num_blocks=num_blocks,
            batches=batches,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            max_cluster_size = UInt(0 if Int(test_num) == 2 else 8),
        ](output, x, ctx)
    else:
        # TODO: Implement for sequences > max_threads_available in the same GPU
        constrained[
            False,
            "fft for sequences longer than max_threads_available",
            "is not implemented yet. max_threads_available: ",
            String(max_threads_available),
        ]()


def test_fft_radix_n[
    dtype: DType,
    bases: List[UInt],
    test_values: _TestValues[dtype],
    inverse: Bool,
    target: StaticString,
    test_num: UInt,
    debug: Bool,
]():
    comptime BATCHES = len(test_values)
    comptime SIZE = len(test_values[0][0])
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = Layout.row_major(
        BATCHES, SIZE, 2
    ) if inverse else Layout.row_major(BATCHES, SIZE, 1)
    comptime out_layout = Layout.row_major(BATCHES, SIZE, 2)

    @parameter
    if debug:
        print("----------------------------")
        print("Buffers for Bases:")
        var b = materialize[bases]()
        print(b.__str__().replace("UInt(", "").replace(")", ""))
        print("----------------------------")

    @parameter
    fn _eval[
        res_layout: Layout, res_origin: MutOrigin
    ](
        result: LayoutTensor[out_dtype, res_layout, res_origin],
        scalar_in: List[Int],
        complex_out: List[ComplexScalar[out_dtype]],
    ) raises:
        @parameter
        if debug:
            print("out: ", end="")
            for i in range(SIZE):
                if i == 0:
                    print("[", result[i, 0], ", ", result[i, 1], sep="", end="")
                else:
                    print(
                        ", ", result[i, 0], ", ", result[i, 1], sep="", end=""
                    )
            print("]")
            print("expected: ", end="")

        comptime ATOL = 1e-3 if dtype is DType.float64 else (
            1e-2 if dtype is DType.float32 else 1e-1
        )
        comptime RTOL = 1e-5

        # gather all real parts and then the imaginary parts
        @parameter
        if inverse:

            @parameter
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print("[", scalar_in[i], ".0, 0.0", sep="", end="")
                    else:
                        print(", ", scalar_in[i], ".0, 0.0", sep="", end="")
                print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    Scalar[out_dtype](scalar_in[i]),
                    atol=ATOL,
                    rtol=RTOL,
                )
                assert_almost_equal(result[i, 1], 0, atol=ATOL, rtol=RTOL)
        else:

            @parameter
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print("[", complex_out[i].re, ", ", sep="", end="")
                    else:
                        print(", ", complex_out[i].re, ", ", sep="", end="")
                    print(complex_out[i].im, end="")
                print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    complex_out[i].re.cast[out_dtype](),
                    atol=ATOL,
                    rtol=RTOL,
                )
                assert_almost_equal(
                    result[i, 1],
                    complex_out[i].im.cast[out_dtype](),
                    atol=ATOL,
                    rtol=RTOL,
                )

    with DeviceContext() as ctx:

        @parameter
        if target == "cpu":
            var out_data = List[Scalar[in_dtype]](
                length=out_layout.size(), fill=nan[in_dtype]()
            )
            var x_data = List[Scalar[out_dtype]](
                length=in_layout.size(), fill=nan[out_dtype]()
            )
            var batch_output = LayoutTensor[mut=True, out_dtype, out_layout](
                Span(out_data)
            )
            var batch_x = LayoutTensor[mut=False, in_dtype, in_layout](
                Span(x_data)
            )

            for idx, test in enumerate(materialize[test_values]()):
                comptime x_layout = Layout.row_major(
                    in_layout.shape[1].value(), in_layout.shape[2].value()
                )
                var x = LayoutTensor[mut=True, in_dtype, x_layout](
                    batch_x.ptr + batch_x.stride[0]() * idx
                )
                for i in range(SIZE):

                    @parameter
                    if inverse:
                        x[i, 0] = test[1][i].re.cast[in_dtype]()
                        x[i, 1] = test[1][i].im.cast[in_dtype]()
                    else:
                        x[i, 0] = Scalar[in_dtype](test[0][i])

            test_fft[bases=bases, inverse=inverse, target=target, test_num=0](
                batch_output, batch_x, ctx
            )

            for idx, test in enumerate(materialize[test_values]()):
                comptime output_layout = Layout.row_major(
                    out_layout.shape[1].value(), 2
                )
                var output = LayoutTensor[
                    mut=True, out_dtype, output_layout, batch_output.origin
                ](batch_output.ptr + batch_output.stride[0]() * idx)
                _eval(output, test[0], test[1])
        else:
            var x_data = ctx.enqueue_create_buffer[in_dtype](in_layout.size())
            x_data.enqueue_fill(nan[in_dtype]())
            var out_data = ctx.enqueue_create_buffer[out_dtype](
                out_layout.size()
            )
            out_data.enqueue_fill(nan[out_dtype]())
            var batch_output = LayoutTensor[mut=True, out_dtype, out_layout](
                out_data.unsafe_ptr()
            )
            var batch_x = LayoutTensor[mut=False, in_dtype, in_layout](
                x_data.unsafe_ptr()
            )
            with x_data.map_to_host() as x_host:
                for idx, test in enumerate(materialize[test_values]()):
                    comptime x_layout = Layout.row_major(
                        in_layout.shape[1].value(), in_layout.shape[2].value()
                    )
                    var x = LayoutTensor[mut=False, in_dtype, x_layout](
                        x_host.unsafe_ptr() + batch_x.stride[0]() * idx
                    )

                    for i in range(SIZE):

                        @parameter
                        if inverse:
                            x[i, 0] = test[1][i].re.cast[in_dtype]()
                            x[i, 1] = test[1][i].im.cast[in_dtype]()
                        else:
                            x[i, 0] = Scalar[in_dtype](test[0][i])

            ctx.synchronize()
            test_fft[
                bases=bases,
                inverse=inverse,
                target="gpu",
                test_num=test_num,
            ](batch_output, batch_x, ctx)
            ctx.synchronize()
            with out_data.map_to_host() as out_host:
                for idx, test in enumerate(materialize[test_values]()):
                    comptime output_layout = Layout.row_major(
                        out_layout.shape[1].value(), 2
                    )
                    var output = LayoutTensor[
                        mut=True, out_dtype, output_layout, batch_output.origin
                    ](out_host.unsafe_ptr() + batch_output.stride[0]() * idx)
                    _eval(output, test[0], test[1])

        @parameter
        if debug:
            print("----------------------------")
            print("Tests passed")
            print("----------------------------")


def _test_fft[
    dtype: DType,
    func: fn[bases: List[UInt], test_values: _TestValues[dtype]] () raises,
]():
    comptime L = List[UInt]

    comptime values_2 = _get_test_values_2[dtype]()
    func[[2], values_2]()

    comptime values_3 = _get_test_values_3[dtype]()
    func[[3], values_3]()

    comptime values_4 = _get_test_values_4[dtype]()
    func[[4], values_4]()
    func[[2], values_4]()

    comptime values_5 = _get_test_values_5[dtype]()
    func[[5], values_5]()

    comptime values_6 = _get_test_values_6[dtype]()
    func[[6], values_6]()
    func[[3, 2], values_6]()
    func[[2, 3], values_6]()

    comptime values_7 = _get_test_values_7[dtype]()
    func[[7], values_7]()

    comptime values_8 = _get_test_values_8[dtype]()
    func[[8], values_8]()
    func[[2], values_8]()
    func[[4, 2], values_8]()
    func[[2, 4], values_8]()

    comptime values_10 = _get_test_values_10[dtype]()
    func[[10], values_10]()
    func[[5, 2], values_10]()

    comptime values_16 = _get_test_values_16[dtype]()
    func[[16], values_16]()
    func[[2], values_16]()
    func[[4], values_16]()
    func[[2, 4], values_16]()
    func[[8, 2], values_16]()
    func[[2, 8], values_16]()

    comptime values_20 = _get_test_values_20[dtype]()
    func[[20], values_20]()
    func[[10, 2], values_20]()
    func[[5, 4], values_20]()
    func[[5, 2], values_20]()

    comptime values_21 = _get_test_values_21[dtype]()
    func[[7, 3], values_21]()

    comptime values_32 = _get_test_values_32[dtype]()
    func[[2], values_32]()
    func[[16, 2], values_32]()
    func[[8, 4], values_32]()
    func[[4, 2], values_32]()
    func[[8, 2], values_32]()

    comptime values_35 = _get_test_values_35[dtype]()
    func[[7, 5], values_35]()

    comptime values_48 = _get_test_values_48[dtype]()
    func[[8, 6], values_48]()
    func[[3, 2], values_48]()

    comptime values_60 = _get_test_values_60[dtype]()
    func[[10, 6], values_60]()
    func[[6, 5, 2], values_60]()
    func[[5, 4, 3], values_60]()
    func[[3, 4, 5], values_60]()
    func[[5, 3, 2], values_60]()

    comptime values_64 = _get_test_values_64[dtype]()
    func[[2], values_64]()
    func[[8], values_64]()
    func[[4], values_64]()
    func[[16, 4], values_64]()

    comptime values_100 = _get_test_values_100[dtype]()
    func[[20, 5], values_100]()
    func[[10], values_100]()
    func[[5, 4], values_100]()

    comptime values_128 = _get_test_values_128[dtype]()
    # func[[64, 2], values_128]()  # long compile times, but important to test
    # func[[32, 4], values_128]()  # long compile times, but important to test
    func[[16, 8], values_128]()
    func[[16, 4, 2], values_128]()
    func[[8, 8, 2], values_128]()
    func[[8, 4, 4], values_128]()
    func[[8, 4, 2, 2], values_128]()
    func[[8, 2, 2, 2, 2], values_128]()
    func[[4, 4, 4, 2], values_128]()
    func[[4, 4, 2, 2, 2], values_128]()
    func[[4, 2, 2, 2, 2, 2], values_128]()
    func[[2], values_128]()


comptime _test[
    dtype: DType,
    inverse: Bool,
    target: StaticString,
    test_num: UInt,
    debug: Bool,
] = _test_fft[
    dtype,
    test_fft_radix_n[
        dtype, inverse=inverse, target=target, test_num=test_num, debug=debug
    ],
]


def test_fft():
    comptime dtype = DType.float32
    _test[dtype, False, "cpu", 0, debug=False]()
    _test[dtype, False, "gpu", 0, debug=False]()
    _test[dtype, False, "gpu", 1, debug=False]()
    _test[dtype, False, "gpu", 2, debug=False]()
    _test[dtype, False, "gpu", 3, debug=False]()


def test_ifft():
    comptime dtype = DType.float32
    _test[dtype, True, "cpu", 0, debug=False]()
    _test[dtype, True, "gpu", 0, debug=False]()
    _test[dtype, True, "gpu", 1, debug=False]()
    _test[dtype, True, "gpu", 2, debug=False]()
    _test[dtype, True, "gpu", 3, debug=False]()


def main():
    test_fft()
    test_ifft()
