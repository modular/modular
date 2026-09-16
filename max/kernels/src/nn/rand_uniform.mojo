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
"""Generates tensors filled with values drawn from a uniform distribution for CPU and GPU."""

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import TileTensor
from std.random import Random
from extensibility import _dot_prod

from std.utils import IndexList
from std.utils.coord import Coord, coord_to_index_list


def random_uniform[
    dtype: DType,
    rank: Int,
    //,
    target: StaticString,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, _rank: Int](
        idx: IndexList[_rank], val: SIMD[dtype, width]
    ),
](
    shape: IndexList[rank],
    lower_bound: Scalar[dtype],
    upper_bound: Scalar[dtype],
    seed_ptr: UnsafePointer[UInt64, ImmutAnyOrigin],
    ctx: DeviceContext,
    output_fn: OutputFn,
) raises:
    """Call `output_fn` with values generated from a uniform distribution on
    [lower_bound, upper_bound] for floating-point types or
    [lower_bound, upper_bound) for integer types.

    Parameters:
        dtype: The data type to generate.
        rank: The rank of the underlying buffer.
        target: The target to run on.
        OutputFn: The type of the function which stores the generated values.

    Args:
        shape: The shape of the output being stored into by output_fn.
        lower_bound: The lower bound on the uniform range.
        upper_bound: The upper bound on the uniform range.
        seed_ptr: Pointer to a single uint64 in device memory containing
            the Philox seed.
        ctx: The device context.
        output_fn: The function which stores the generated values.
    """

    if lower_bound > upper_bound:
        raise Error("lower_bound must be less than upper_bound")

    var strides = shape.get_row_major_strides()
    var delta = Float32(upper_bound - lower_bound)

    @inline(.always)
    def generate[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert width <= 4

        var offset = _dot_prod(
            rebind[type_of(strides)](coord_to_index_list(idx)), strides
        )

        var seed_value = seed_ptr[0]

        var generator = Random(seed=seed_value, offset=UInt64(offset))

        var values: SIMD[.float32, 4] = generator.step_uniform()
        values = values * delta + Float32(lower_bound)

        output_fn[width=width](
            coord_to_index_list(idx), values.cast[dtype]().slice[width]()
        )

    elementwise[simd_width=4, target=target](generate, Coord(shape), ctx)


def keyed_uniform[
    target: StaticString,
](
    output: TileTensor[mut=True, .float32, ...],
    seed: TileTensor[mut=False, .uint64, ...],
    ctx: DeviceContext,
) raises:
    """Draws one uniform value in [0, 1) per row, keyed off that row's seed.

    `random_uniform` reads `seed_ptr[0]` and walks the flat element index as
    the Philox counter, so one stream covers the whole tensor and a caller
    cannot key a row independently of its neighbours. Here every row is its
    own Philox key at counter 0: rows with equal seeds draw equal values, and
    a row's draw never depends on how many rows share the launch or on where
    it sits among them. Speculative decoding keys its accept coin per
    (request, draft position) this way.

    Parameters:
        target: The target to run on.

    Args:
        output: The drawn values, one per row.
        seed: The Philox seed for each row, one per output row.
        ctx: The device context.

    Raises:
        Error: If the seed does not carry exactly one entry per output row.
    """
    comptime assert output.flat_rank == 1, "output must be of rank 1"
    comptime assert seed.flat_rank == 1, "seed must be of rank 1"

    var rows = Int(output.dim(0))
    if Int(seed.dim(0)) != rows:
        raise Error("keyed_uniform needs exactly one seed per output row")

    var out_ptr = output.ptr
    var seed_ptr = seed.ptr

    @inline(.always)
    def draw[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert width == 1, "each row keys its own Philox stream"

        var row = coord_to_index_list(idx)[0]
        var generator = Random(seed=seed_ptr[row])
        out_ptr[row] = generator.step_uniform()[0]

    elementwise[simd_width=1, target=target](
        draw, Coord(IndexList[1](rows)), ctx
    )
