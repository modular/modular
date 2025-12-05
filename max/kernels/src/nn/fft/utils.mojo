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

from sys.info import is_64bit
from complex import ComplexScalar
from math import exp, pi, ceil, sin, cos
from bit import count_trailing_zeros
from gpu.host.info import is_cpu


fn _get_dtype[length: UInt]() -> DType:
    @parameter
    if length < UInt(UInt8.MAX):
        return DType.uint8
    elif length < UInt(UInt16.MAX):
        return DType.uint16
    elif length < UInt(UInt32.MAX):
        return DType.uint32
    elif UInt64(length) < UInt64.MAX:
        return DType.uint64
    elif UInt128(length) < UInt128.MAX:
        return DType.uint128
    else:
        return DType.uint256


fn _mixed_radix_digit_reverse[
    length: UInt, ordered_bases: List[UInt]
](idx: Scalar) -> type_of(idx):
    """Performs mixed-radix digit reversal for an index `idx` based on a
    sequence of `ordered_bases`.

    Notes:
        Given `N = R_0 * R_1 * ... * R_{M-1}`, an input index `k` is represented
        as: `k = d_0 + d_1*R_0 + d_2*R_0*R_1 + ... + d_{M-1}*R_0*...*R_{M-2}`
        where d_i is the digit for radix R_i.

        The reversed index k' is:
        `k' = d_{M-1} + d_{M-2}*R_{M-1} + ... + d_1*R_{M-1}*...*R_2 + d_0*R_{M-1
        }*...*R_1`
    """
    var reversed_idx = type_of(idx)(0)
    var current_val = idx
    var base_offset = length

    @parameter
    for i in range(len(ordered_bases)):
        comptime base = ordered_bases[i]
        base_offset //= base
        reversed_idx += (current_val % base) * base_offset
        current_val //= base
    return reversed_idx


fn _get_twiddle_factors[
    length: UInt, dtype: DType, inverse: Bool = False
](out res: InlineArray[ComplexScalar[dtype], Int(length - 1)]):
    """Get the twiddle factors for the length.

    Examples:
        for a signal with 8 datapoints:
        the result is: [W_1_8, W_2_8, W_3_8, W_4_8, W_5_8, W_6_8, W_7_8]
    """
    res = type_of(res)(uninitialized=True)
    comptime N = length
    for n in range(1, N):
        # exp((-j * 2 * pi * n) / N)
        var factor = 2 * n / Int(N)

        var num: ComplexScalar[dtype]

        if factor == 0:
            num = {1, 0}
        elif factor == 0.5:
            num = {0, -1}
        elif factor == 1:
            num = {-1, 0}
        elif factor == 1.5:
            num = {0, 1}
        else:
            var theta = Float64(-factor * pi)
            num = {cos(theta).cast[dtype](), sin(theta).cast[dtype]()}

        @parameter
        if not inverse:
            res[n - 1] = num
        else:
            res[n - 1] = {num.re, -num.im}


fn _prep_twiddle_factors[
    length: UInt, base: UInt, processed: UInt, dtype: DType, inverse: Bool
](
    out res: InlineArray[
        InlineArray[ComplexScalar[dtype], Int(base - 1)], Int(length // base)
    ]
):
    comptime twiddle_factors = _get_twiddle_factors[length, dtype, inverse]()
    res = {uninitialized = True}
    comptime Sc = Scalar[_get_dtype[length * base]()]
    comptime offset = Sc(processed)

    comptime next_offset = offset * Sc(base)
    comptime ratio = Sc(length) // next_offset
    for local_i in range(length // base):
        res[local_i] = {uninitialized = True}
        for j in range(1, base):
            var n = Sc(local_i) % offset + (Sc(local_i) // offset) * (
                offset * Sc(base)
            )
            var twiddle_idx = ((Sc(j) * n) % next_offset) * ratio
            res[local_i][j - 1] = twiddle_factors[
                twiddle_idx - 1
            ] if twiddle_idx != 0 else {1, 0}


@parameter
fn _get_flat_twfs[
    dtype: DType,
    length: UInt,
    total_twfs: UInt,
    ordered_bases: List[UInt],
    processed_list: List[UInt],
    inverse: Bool,
](out res: InlineArray[Scalar[dtype], Int(total_twfs * 2)]):
    res = {uninitialized = True}
    var idx = 0

    @parameter
    for b in range(len(ordered_bases)):
        comptime base = ordered_bases[b]
        comptime processed = processed_list[b]
        comptime amnt_threads = length // base
        var base_twfs = _prep_twiddle_factors[
            length, base, processed, dtype, inverse
        ]()

        for i in range(base_twfs.size):
            for j in range(base_twfs[0].size):
                var t = base_twfs[i][j]
                res[idx] = t.re
                idx += 1
                res[idx] = t.im
                idx += 1


fn _log_mod(x: UInt, base: UInt) -> Tuple[UInt, UInt]:
    """Get the maximum exponent of base that fully divides x and the
    remainder.
    """
    var div = x // base

    @parameter
    fn _run() -> Tuple[UInt, UInt]:
        ref res = _log_mod(div, base)
        res[0] += 1
        return res

    # TODO: benchmark whether this performs better than doing branches
    return (UInt(0), x) if x % base != 0 else (
        (UInt(1), UInt(0)) if div == 1 else _run()
    )


fn _get_ordered_bases_processed_list[
    length: UInt, bases: List[UInt], target: StaticString
]() -> Tuple[List[UInt], List[UInt]]:
    @parameter
    fn _reduce_mul[b: List[UInt]](out res: UInt):
        res = UInt(1)
        for base in materialize[b]():
            res *= base

    @parameter
    fn _is_all_two(existing_bases: List[UInt]) -> Bool:
        for base in existing_bases:
            if base != 2:
                return False
        return True

    @parameter
    fn _build_ordered_bases(out new_bases: List[UInt]):
        @parameter
        if _reduce_mul[bases]() == length:
            new_bases = materialize[bases]()
            sort(new_bases)  # FIXME: this should just be ascending=False
            new_bases.reverse()
        else:
            var existing_bases = materialize[bases]()
            sort(existing_bases)

            @parameter
            if is_cpu[target]():
                existing_bases.reverse()  # FIXME: this should just be ascending=False
            new_bases = List[UInt](capacity=len(existing_bases))

            var processed = UInt(1)
            for base in existing_bases:
                var amnt_divisible: UInt

                if (
                    _is_all_two(existing_bases)
                    and length.is_power_of_two()
                    and base == 2
                ):
                    # FIXME(#5003): this should just be Scalar[DType.index]
                    @parameter
                    if is_64bit():
                        amnt_divisible = UInt(
                            count_trailing_zeros(UInt64(length))
                        )
                    else:
                        amnt_divisible = UInt(
                            count_trailing_zeros(UInt32(length))
                        )
                else:
                    amnt_divisible = _log_mod(length // processed, base)[0]
                for _ in range(amnt_divisible):
                    new_bases.append(base)
                    processed *= base

            @parameter
            if not is_cpu[target]():
                new_bases.reverse()

    comptime ordered_bases = _build_ordered_bases()

    @parameter
    fn _build_processed_list() -> List[UInt]:
        var ordered_bases_var = materialize[ordered_bases]()
        var processed_list = List[UInt](capacity=len(ordered_bases_var))
        var processed = UInt(1)
        for base in ordered_bases_var:
            processed_list.append(processed)
            processed *= base
        return processed_list^

    comptime processed_list = _build_processed_list()
    constrained[
        processed_list[len(processed_list) - 1]
        * ordered_bases[len(ordered_bases) - 1]
        == length,
        "powers of the bases must multiply together  to equal the sequence ",
        "length. The builtin algorithm was only able to produce: ",
        ordered_bases.__str__(),
    ]()
    constrained[1 not in ordered_bases, "Cannot do an fft with base 1."]()
    return materialize[ordered_bases](), materialize[processed_list]()
