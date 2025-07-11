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

from algorithm.functional import elementwise
from gpu.random import Random
from runtime.asyncrt import DeviceContextPtr
from tensor_internal._indexing import _dot_prod, _row_major_strides

from utils import IndexList


fn random_uniform[
    dtype: DType,
    rank: Int, //,
    output_fn: fn[width: Int, _rank: Int] (
        idx: IndexList[_rank], val: SIMD[dtype, width]
    ) capturing [_],
    target: StaticString,
](
    shape: IndexList[rank],
    lower_bound: Scalar[dtype],
    upper_bound: Scalar[dtype],
    seed_value: UInt64,
    ctx: DeviceContextPtr,
) raises:
    """Call `output_fn` with values generated from a uniform distribution on
    [lower_bound, upper_bound] for floating-point types or
    [lower_bound, upper_bound) for integer types.

    Parameters:
        dtype: The data type to generate.
        rank: The rank of the underlying buffer.
        output_fn: The function which stores the generated values.
        target: The target to run on.

    Args:
        shape: The shape of the output being stored into by output_fn.
        lower_bound: The lower bound on the uniform range.
        upper_bound: The upper bound on the uniform range.
        seed_value: Seed value used to initialize the random number generator.
        ctx: The device context.
    """

    if lower_bound > upper_bound:
        raise Error("lower_bound must be less than upper_bound")

    var strides = _row_major_strides(shape)
    var delta = Scalar[DType.float32](upper_bound - lower_bound)

    @parameter
    @always_inline
    @__copy_capture(strides, delta)
    fn generate[
        width: Int, _rank: Int
    ](idx: IndexList[_rank],):
        constrained[width <= 4]()

        var offset = _dot_prod(rebind[__type_of(strides)](idx), strides)

        var generator = Random(seed=seed_value, offset=UInt64(offset))

        var values: SIMD[DType.float32, 4] = generator.step_uniform()
        values = values * delta + Scalar[DType.float32](lower_bound)

        output_fn[width=width](idx, values.cast[dtype]().slice[width]())

    elementwise[generate, simd_width=4, target=target](shape, ctx)
