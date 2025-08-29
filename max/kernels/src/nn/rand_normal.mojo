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
from gpu.random import NormalRandom
from runtime.asyncrt import DeviceContextPtr
from tensor_internal._indexing import _dot_prod, _row_major_strides

from utils import IndexList


fn random_normal[
    dtype: DType,
    rank: Int, //,
    output_fn: fn[width: Int, _rank: Int] (
        idx: IndexList[_rank], val: SIMD[dtype, width]
    ) capturing [_],
    target: StaticString,
](
    shape: IndexList[rank],
    mean: Scalar[dtype],
    stddev: Scalar[dtype],
    seed_value: UInt64,
    ctx: DeviceContextPtr,
) raises:
    """Call `output_fn` with values generated from a normal distribution with
    the specified mean and standard deviation.

    Parameters:
        dtype: The data type to generate.
        rank: The rank of the underlying buffer.
        output_fn: The function which stores the generated values.
        target: The target to run on.

    Args:
        shape: The shape of the output being stored into by output_fn.
        mean: The mean of the normal distribution.
        stddev: The standard deviation of the normal distribution.
        seed_value: Seed value used to initialize the random number generator.
        ctx: The device context.
    """

    if stddev <= 0:
        raise Error("stddev must be positive")

    var strides = _row_major_strides(shape)

    @parameter
    @always_inline
    @__copy_capture(strides)
    fn generate[
        width: Int, _rank: Int, alignment: Int = 1
    ](idx: IndexList[_rank],):
        constrained[width <= 8]()  # NormalRandom generates 8 values at a time

        var offset = _dot_prod(rebind[__type_of(strides)](idx), strides)

        var generator = NormalRandom(seed=seed_value, offset=UInt64(offset))

        var values = generator.step_normal(
            mean=Scalar[DType.float32](mean),
            stddev=Scalar[DType.float32](stddev),
        )

        output_fn[width=width](idx, values.cast[dtype]().slice[width]())

    elementwise[generate, simd_width=8, target=target](shape, ctx)
