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
"""Provides reference values for the host and GPU `lgamma` tests."""


def lgamma_bit_refs[dtype: DType]() -> List[Tuple[UInt64, Float64]]:
    """Returns `(bits, lgamma(x))` pairs for inputs given as bit patterns.

    The inputs are signed subnormals and neighbors of the positive zeros of
    `lgamma` at 1 and 2. The references are independently rounded 600-digit
    mpmath values. The first six entries are the subnormals.

    Parameters:
        dtype: `float32` or `float64`.

    Returns:
        The pairs, with each bit pattern widened to `UInt64`.
    """
    comptime match dtype:
        case .float32:
            return [
                (UInt64(0x00000001), 103.2789306640625),
                (UInt64(0x00000200), 97.04060363769531),
                (UInt64(0x007FFFFF), 87.3365478515625),
                (UInt64(0x80000001), 103.2789306640625),
                (UInt64(0x80000200), 97.04060363769531),
                (UInt64(0x807FFFFF), 87.3365478515625),
                (UInt64(0x3F7FFFFF), 3.4404738613602603e-08),
                (UInt64(0x3F800001), -6.880945591092313e-08),
                (UInt64(0x3FFFFFFF), -5.039981587628972e-08),
                (UInt64(0x40000001), 1.0079966017428887e-07),
                (UInt64(0x3F7FF000), 0.0001409708202118054),
                (UInt64(0x3F800800), -0.00014087276940699667),
                (UInt64(0x3FFFF800), -0.00010319960711058229),
                (UInt64(0x40000400), 0.00010323805327061564),
            ]
        case .float64:
            return [
                (UInt64(0x0000000000000001), 744.4400719213812),
                (UInt64(0x0000000000004000), 734.7360113935421),
                (UInt64(0x000FFFFFFFFFFFFF), 708.3964185322641),
                (UInt64(0x8000000000000001), 744.4400719213812),
                (UInt64(0x8000000000004000), 734.7360113935421),
                (UInt64(0x800FFFFFFFFFFFFF), 708.3964185322641),
                (UInt64(0x3FEFFFFFFFFFFFFF), 6.408381213480008e-17),
                (UInt64(0x3FF0000000000001), -1.2816762426960008e-16),
                (UInt64(0x3FFFFFFFFFFFFFFF), -9.387698065543117e-17),
                (UInt64(0x4000000000000001), 1.8775396131086244e-16),
                (UInt64(0x3FEFFE0000000000), 0.00014097082187592238),
                (UInt64(0x3FF0010000000000), -0.00014087277616326634),
                (UInt64(0x3FFFFF0000000000), -0.00010319961029799209),
                (UInt64(0x4000008000000000), 0.00010323805136409636),
            ]
        case _:
            comptime assert False, "expected float32 or float64"


def lgamma_zeros_f32() -> List[Tuple[Float64, Float64]]:
    """Returns `(x, lgamma(x))` at the float32 nearest each negative zero of
    `lgamma`.

    Near these zeros the result is tiny while the reflection terms are not, so
    any absolute error is largest relative to the result. The references are
    correctly rounded MPFR values.

    Returns:
        The pairs, in float64.
    """
    return [
        (-2.747682571411133, -1.4418168348129257e-07),
        (-2.4570248126983643, -1.1287842482943233e-07),
        (-3.955294370651245, 1.7780596408556448e-06),
        (-3.143580913543701, -1.9605461432092852e-07),
        (-4.991544723510742, 9.666771802585572e-06),
        (-4.039361953735352, -3.053971795452526e-06),
        (-5.998607635498047, 0.0001113230682676658),
        (-5.008218288421631, -1.4815573194937315e-05),
        (-6.9998016357421875, 0.0006440637516789138),
        (-6.00138521194458, 5.9716250689234585e-05),
        (-7.999975204467773, 0.00029724976047873497),
        (-7.0001983642578125, -0.0001555986818857491),
        (-8.000024795532227, 0.00019109305867459625),
    ]


def lgamma_zeros_f64() -> List[Tuple[Float64, Float64]]:
    """Returns `(x, lgamma(x))` at the float64 nearest each negative zero of
    `lgamma`, as `lgamma_zeros_f32` does for float32.

    Returns:
        The pairs.
    """
    return [
        (-2.7476826467274127, 1.733509244024501e-16),
        (-2.4570247382208006, 5.619192358950097e-17),
        (-3.955294284858598, -4.14382750757705e-16),
        (-3.14358088834998, 1.6978655906121085e-15),
        (-4.991544640560048, 1.7683619350849613e-14),
        (-4.039361839740537, -5.664578074060335e-15),
        (-5.998607480080875, -2.3721063667118474e-13),
        (-5.0082181683225935, 5.4188509265538106e-15),
        (-6.999801507890638, 5.313011065735902e-14),
        (-6.001385294453155, -4.6433836942838004e-14),
        (-7.999975197095821, -2.1213071311827735e-12),
        (-7.000198333407325, -1.263203743493977e-12),
        (-8.999997244250977, -8.050618056741812e-11),
        (-8.000024800270682, 1.75595561986039e-11),
        (-9.99999972442663, 1.7719543958825935e-09),
        (-9.000002755714823, 3.444263328391509e-11),
        (-10.99999997494789, 7.92108817813105e-09),
        (-10.000000275573013, 1.2668051387565237e-09),
        (-11.999999997912324, -4.799928255018621e-08),
        (-11.000000025052106, 2.734640389948086e-08),
        (-12.99999999983941, 4.201542925270726e-06),
        (-12.000000002087676, -5.854619992113373e-08),
        (-13.99999999998853, 7.057210284625434e-05),
        (-13.00000000016059, 4.200706921268597e-06),
        (-14.999999999999236, 0.0011552549109141656),
        (-14.00000000001147, 7.057204149708364e-05),
        (-15.999999999999952, -0.0034851246475472554),
        (-15.000000000000764, 0.0011552549067268103),
    ]
