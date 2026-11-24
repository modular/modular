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
"""Generates tensors filled with values drawn from a normal (Gaussian) distribution for CPU and GPU.

On CPU the values are PyTorch's `Tensor.normal_` stream.

For seed `s`, `torch_cpu_normal_fill` gives the values of
`torch.empty(n).normal_(mean, std, generator=torch.Generator().manual_seed(s))`
bit for bit. It follows `normal_fill` in `ATen/native/cpu/DistributionTemplates.h`
as of torch 2.15, on x86 with AVX2 (an AVX-512 host runs the same kernel):

- MT19937 seeded with the low 32 bits of the seed. A float uniform is the low
  24 bits of one word over 2^24. A double uniform is the low 53 bits of two
  words, first word in the high half, over 2^53.
- `n >= 16`: draw `n` uniforms, run Box-Muller over every 16-block (element
  `j` pairs with `j + 8`), then redraw 16 uniforms for the last 16 elements
  when `n` is not a multiple of 16. The float kernel is the AVX2 `NormalFill16`
  with `avx_mathfun.h`'s `log256_ps` and `sincos256_ps`, which this file ports
  lane by lane, FMAs included. The double kernel uses libm and `std::fma`.
- `n < 16`: `at::normal_distribution<double>`, which draws two double uniforms
  per pair of values, returns `r * cos` and caches `r * sin` for the next value.
- float16 and bfloat16 run the float kernel one 16-block at a time and round
  to the narrow type, so the redraw for a ragged end starts right after the last
  full block instead of after element `n`.

Torch's scalar non-AVX2 kernel (arm64, or `ATEN_CPU_CAPABILITY=default`) rounds
`log` and `cos` differently and is not reproduced.
"""

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.ffi import external_call
from std.math import ceildiv, cos, fma, sin, sqrt
from std.math.polynomial import polynomial_evaluate
from std.memory import Allocation, Layout, alloc, dealloc
from std.random import NormalRandom
from std.sys import simd_width_of

from layout import row_major
from std.utils.coord import Coord
from std.utils.numerics import FPUtils


def random_normal[
    dtype: DType,
    //,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength](idx: Coord, val: SIMD[dtype, width]),
](
    shape: Coord,
    mean: Float32,
    stddev: Float32,
    seed_ptr: ImmPointer[UInt64, ImmutAnyOrigin],
    ctx: DeviceContext,
    output_fn: OutputFn,
) raises:
    """Call `output_fn` with values from a normal distribution, matching
    PyTorch's `torch.randn` for the device the op runs on.

    On CPU the values are the stream of `torch.empty(shape).normal_(mean,
    stddev)` for a CPU generator seeded with the low 32 bits of the seed; see
    :func:`torch_cpu_normal_fill`.

    On GPU, for element `i`, mirrors PyTorch's per-thread Philox state:

        thread_id     = i mod GRID_BLOCK
        within_thread = i div GRID_BLOCK   (0..3)

    where `GRID_BLOCK = 256 * min(num_SMs * blocks_per_sm, ceil(numel/256))`.

    A single Philox step at counter `(0, 0, thread_id, 0)` produces 4 normals
    via :func:`std.random.NormalRandom.step_normal_4`; the within_thread
    index selects which lane to write to `output[i]`.

    Bit-exact for `numel <= 4 * GRID_BLOCK_max` (≈ 1.2M elements on B200).

    Parameters:
        dtype: The data type to generate.
        OutputFn: The type of the function which stores the generated values.

    Args:
        shape: The shape of the output being stored into by output_fn.
        mean: The mean of the normal distribution.
        stddev: The standard deviation of the normal distribution.
        seed_ptr: Pointer to a single uint64 in device memory containing
            the Philox seed.
        ctx: The device context.
        output_fn: The function which stores the generated values.
    """

    if stddev <= 0:
        raise Error("stddev must be positive")

    var numel = Int(shape.product())
    if numel == 0:
        return

    comptime if ctx.target.is_gpu():
        _random_normal_gpu(shape, numel, mean, stddev, seed_ptr, ctx, output_fn)
    else:
        _random_normal_cpu(
            shape, numel, mean, stddev, seed_ptr[0], ctx, output_fn
        )


def _random_normal_cpu[
    dtype: DType,
    //,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength](idx: Coord, val: SIMD[dtype, width]),
](
    shape: Coord,
    numel: Int,
    mean: Float32,
    stddev: Float32,
    seed: UInt64,
    ctx: DeviceContext,
    output_fn: OutputFn,
) raises:
    """Draws torch's CPU stream into a scratch buffer, then stores it through
    `output_fn`, which may be fused with a consumer."""
    comptime if dtype == .float64:
        var values = List[Float64](unsafe_uninit_length=numel)
        torch_cpu_normal_fill(values.unsafe_ptr(), numel, seed, mean, stddev)
        _store_cpu_values(shape, values.unsafe_ptr(), ctx, output_fn)
    else:
        var values = List[Float32](unsafe_uninit_length=numel)
        torch_cpu_normal_fill[full_draw=dtype == .float32](
            values.unsafe_ptr(), numel, seed, mean, stddev
        )
        _store_cpu_values(shape, values.unsafe_ptr(), ctx, output_fn)


def _store_cpu_values[
    dtype: DType,
    work_dtype: DType,
    //,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength](idx: Coord, val: SIMD[dtype, width]),
](
    shape: Coord,
    values: ImmPointer[Scalar[work_dtype], _],
    ctx: DeviceContext,
    output_fn: OutputFn,
) raises:
    var layout = row_major(shape)

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def store[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var i = Int(layout(idx))
        output_fn[width=width](
            idx, values.unsafe_load[width=width](i).cast[dtype]()
        )

    elementwise[simd_width=simd_width_of[dtype]()](store, shape, ctx)


def _random_normal_gpu[
    dtype: DType,
    //,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength](idx: Coord, val: SIMD[dtype, width]),
](
    shape: Coord,
    numel: Int,
    mean: Float32,
    stddev: Float32,
    seed_ptr: ImmPointer[UInt64, ImmutAnyOrigin],
    ctx: DeviceContext,
    output_fn: OutputFn,
) raises:
    # Row-major layout over `shape`: `layout(idx)` is the element's flat index,
    # which is what the Philox counter below walks.
    var layout = row_major(shape)

    comptime BLOCK_SIZE: Int = 256

    # GRID_BLOCK mirrors PyTorch CUDA's calc_execution_policy and depends on
    # the device's SM count (comptime via default_device_info).
    comptime info = DeviceContext[].default_device_info
    comptime MAX_GRID = (
        info.sm_count * (info.threads_per_multiprocessor // BLOCK_SIZE)
    )
    var nblocks = ceildiv(numel, BLOCK_SIZE)
    var grid_x = min(nblocks, MAX_GRID)
    var grid_block = grid_x * BLOCK_SIZE

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def generate[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert (
            width == 1
        ), "PyTorch-compat normal kernel uses scalar lanes"
        var i = Int(layout(idx))
        var thread_id = UInt64(i % grid_block)
        var within_thread = i // grid_block

        var rng = NormalRandom(seed=seed_ptr[0], subsequence=thread_id)
        var four = rng.step_normal_4(mean=mean, stddev=stddev)
        var value = four[within_thread].cast[dtype]()
        output_fn[width=1](idx, SIMD[dtype, 1](value))

    elementwise[simd_width=1](generate, shape, ctx)


comptime _MT_N = 624
comptime _MT_M = 397
# The block of values torch transforms at once and the offset between the two
# values of a Box-Muller pair. They define the stream. The SIMD width of the
# code below never changes them.
comptime _BLOCK = 16
comptime _PAIR_OFFSET = _BLOCK // 2

# A float uniform keeps one more bit than the mantissa stores, a double
# uniform likewise.
comptime _FLOAT_UNIFORM_BITS = FPUtils[.float32].mantissa_width() + 1
comptime _DOUBLE_UNIFORM_BITS = FPUtils[.float64].mantissa_width() + 1
comptime _FLOAT_UNIFORM_MASK = UInt32((1 << _FLOAT_UNIFORM_BITS) - 1)
comptime _FLOAT_UNIFORM_SCALE = Float32(1) / Float32(1 << _FLOAT_UNIFORM_BITS)
comptime _DOUBLE_UNIFORM_MASK = (UInt64(1) << UInt64(_DOUBLE_UNIFORM_BITS)) - 1
comptime _DOUBLE_UNIFORM_SCALE = Float64(1) / Float64(
    UInt64(1) << UInt64(_DOUBLE_UNIFORM_BITS)
)


# ===-----------------------------------------------------------------------===#
# MT19937
# ===-----------------------------------------------------------------------===#


def _mt_seed(state: MutPointer[UInt32, _], seed: UInt64):
    var word = UInt32(seed & 0xFFFFFFFF)
    state.unsafe_store(0, word)
    for j in range(1, _MT_N):
        word = UInt32(1812433253) * (word ^ (word >> 30)) + UInt32(j)
        state.unsafe_store(j, word)


@inline(.always)
def _mt_twist_lanes[width: Int](state: MutPointer[UInt32, _], i: Int, src: Int):
    """Rewrites `state[i : i + width]` from `state[src : src + width]`."""
    var y = (state.unsafe_load[width=width](i) & 0x80000000) | (
        state.unsafe_load[width=width](i + 1) & 0x7FFFFFFF
    )
    var magic = (y & 1) * 0x9908B0DF
    state.unsafe_store(
        i, state.unsafe_load[width=width](src) ^ (y >> 1) ^ magic
    )


def _mt_twist_span[
    width: Int
](state: MutPointer[UInt32, _], lo: Int, hi: Int, offset: Int):
    var i = lo
    while i + width <= hi:
        _mt_twist_lanes[width](state, i, i + offset)
        i += width
    while i < hi:
        _mt_twist_lanes[1](state, i, i + offset)
        i += 1


def _mt_twist[width: Int](state: MutPointer[UInt32, _]):
    # Each span reads only words that earlier spans already rewrote or that no
    # span has touched yet, so its lanes are independent.
    _mt_twist_span[width](state, 0, _MT_N - _MT_M, _MT_M)
    _mt_twist_span[width](
        state, _MT_N - _MT_M, 2 * (_MT_N - _MT_M), _MT_M - _MT_N
    )
    _mt_twist_span[width](state, 2 * (_MT_N - _MT_M), _MT_N - 1, _MT_M - _MT_N)
    # The last word pairs with the rewritten first one.
    var y = (state.unsafe_load(_MT_N - 1) & 0x80000000) | (
        state.unsafe_load(0) & 0x7FFFFFFF
    )
    state.unsafe_store(
        _MT_N - 1,
        state.unsafe_load(_MT_M - 1) ^ (y >> 1) ^ ((y & 1) * 0x9908B0DF),
    )


@inline(.always)
def _mt_temper[
    width: SIMDLength, //
](word: SIMD[.uint32, width]) -> SIMD[.uint32, width]:
    var y = word
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    return y ^ (y >> 18)


struct _MT19937(Movable):
    var _words: Allocation[UInt32]
    var _next: Int

    def __init__(out self, seed: UInt64):
        self._words = alloc(Layout[UInt32](length=_MT_N))
        self._next = _MT_N
        _mt_seed(self._words.unsafe_ptr(), seed)

    def __deinit__(deinit self):
        dealloc(self._words^)

    def next_word(mut self) -> UInt32:
        var state = self._words.unsafe_ptr()
        if self._next == _MT_N:
            _mt_twist[simd_width_of[DType.uint32]()](state)
            self._next = 0
        var word = state.unsafe_load(self._next)
        self._next += 1
        return _mt_temper(word)

    def next_double_uniform(mut self) -> Float64:
        var high = UInt64(self.next_word())
        var low = UInt64(self.next_word())
        var bits = ((high << 32) | low) & _DOUBLE_UNIFORM_MASK
        return Float64(bits) * _DOUBLE_UNIFORM_SCALE

    def fill_float_uniforms[
        width: Int
    ](mut self, dst: MutPointer[Float32, _], count: Int):
        """Writes the next `count` float uniforms to `dst`."""
        var state = self._words.unsafe_ptr()
        var done = 0
        while done < count:
            if self._next == _MT_N:
                _mt_twist[width](state)
                self._next = 0
            var take = min(count - done, _MT_N - self._next)
            var j = 0
            while j + width <= take:
                var words = _mt_temper(
                    state.unsafe_load[width=width](self._next + j)
                )
                dst.unsafe_store(
                    done + j,
                    (words & _FLOAT_UNIFORM_MASK).cast[.float32]()
                    * _FLOAT_UNIFORM_SCALE,
                )
                j += width
            self._next += j
            done += j
            take -= j
            for _ in range(take):
                var word = _mt_temper(state.unsafe_load(self._next))
                dst.unsafe_store(
                    done,
                    Float32(word & _FLOAT_UNIFORM_MASK) * _FLOAT_UNIFORM_SCALE,
                )
                self._next += 1
                done += 1


# ===-----------------------------------------------------------------------===#
# avx_mathfun.h
# ===-----------------------------------------------------------------------===#


def _log256[
    width: SIMDLength, //
](x: SIMD[.float32, width]) -> SIMD[.float32, width]:
    """`log256_ps` on `(0, 1]`, with the FMAs libtorch's compiler contracts."""
    comptime FloatVec = SIMD[.float32, width]
    comptime WordVec = SIMD[.uint32, width]
    comptime IntVec = SIMD[.int32, width]
    comptime fp = FPUtils[.float32]
    var bits = x.to_bits[.uint32]()
    var exponent = (
        (bits >> WordVec(UInt32(fp.mantissa_width()))).cast[.int32]()
        - IntVec(fp.exponent_bias())
    ).cast[.float32]() + 1.0
    # The mantissa scaled into `[0.5, 1)`.
    var mantissa = FloatVec(
        from_bits=(bits & WordVec(UInt32(fp.sign_mask() | fp.mantissa_mask())))
        | WordVec(UInt32((fp.exponent_bias() - 1) << fp.mantissa_width()))
    )
    # Cephes' SQRTHF.
    var below = mantissa.lt(0.707106781186547524)
    exponent = below.select(exponent - 1.0, exponent)
    var t = (mantissa - 1.0) + below.select(mantissa, FloatVec(0))
    var t2 = t * t
    var poly = polynomial_evaluate[
        [
            Float32(3.3333331174e-1),
            -2.4999993993e-1,
            2.0000714765e-1,
            -1.6668057665e-1,
            1.4249322787e-1,
            -1.2420140846e-1,
            1.1676998740e-1,
            -1.1514610310e-1,
            7.0376836292e-2,
        ]
    ](t)
    poly *= t
    poly = fma(poly, t2, exponent * -2.12194440e-4)
    poly -= t2 * 0.5
    return fma(exponent, FloatVec(0.693359375), t + poly)


def _sincos256[
    width: SIMDLength, //
](x: SIMD[.float32, width]) -> Tuple[
    SIMD[.float32, width], SIMD[.float32, width]
]:
    """`sincos256_ps` on `[0, 2pi)`: `(sin, cos)`."""
    comptime FloatVec = SIMD[.float32, width]
    var octant = ((x * 1.27323954473516).cast[.int32]() + 1) & SIMD[
        .int32, width
    ](-2)
    var octant_f = octant.cast[.float32]()
    var r = fma(octant_f, FloatVec(-0.78515625), x)
    r = fma(octant_f, FloatVec(-2.4187564849853515625e-4), r)
    r = fma(octant_f, FloatVec(-3.77489497744594108e-8), r)
    var r2 = r * r
    var cos_poly = polynomial_evaluate[
        [
            Float32(4.166664568298827e-2),
            -1.388731625493765e-3,
            2.443315711809948e-5,
        ]
    ](r2)
    cos_poly *= r2
    cos_poly = fma(cos_poly, r2, -(r2 * 0.5)) + 1.0
    var sin_poly = polynomial_evaluate[
        [Float32(-1.6666654611e-1), 8.3321608736e-3, -1.9515295891e-4]
    ](r2)
    sin_poly *= r2
    sin_poly = fma(sin_poly, r, r)
    # The masked blend adds +0.0, which turns a -0.0 into +0.0.
    var use_sin_poly = (octant & 2).eq(0)
    var sine = use_sin_poly.select(sin_poly, cos_poly) + 0.0
    var cosine = use_sin_poly.select(cos_poly, sin_poly) + 0.0
    sine = (octant & 4).eq(0).select(sine, -sine)
    cosine = ((octant - 2) & 4).eq(0).select(-cosine, cosine)
    return (sine, cosine)


# ===-----------------------------------------------------------------------===#
# normal_fill
# ===-----------------------------------------------------------------------===#


def _normal_fill_16_float[
    width: Int
](block: MutPointer[Float32, _], mean: Float32, stddev: Float32):
    """AVX2 `NormalFill16`: lane `j` Box-Muller-pairs with lane `j + 8`.

    Each pair is independent, so `width` lanes at a time give the same bits as
    eight.
    """
    comptime assert _PAIR_OFFSET % width == 0
    comptime FloatVec = SIMD[.float32, width]
    comptime for j in range(0, _PAIR_OFFSET, width):
        var u1 = 1.0 - block.unsafe_load[width=width](j)
        var u2 = block.unsafe_load[width=width](j + _PAIR_OFFSET)
        var radius = sqrt(-2.0 * _log256(u1))
        # `2.0f * pi<double>` rounded to float.
        var sin_cos = _sincos256(6.283185307179586 * u2)
        block.unsafe_store(
            j, fma(radius * sin_cos[1], FloatVec(stddev), FloatVec(mean))
        )
        block.unsafe_store(
            j + _PAIR_OFFSET,
            fma(radius * sin_cos[0], FloatVec(stddev), FloatVec(mean)),
        )


def _normal_fill_16_double(
    block: MutPointer[Float64, _], mean: Float64, stddev: Float64
):
    for j in range(_PAIR_OFFSET):
        var radius = sqrt(
            -2.0 * external_call["log", Float64](1.0 - block.unsafe_load(j))
        )
        var theta = 6.283185307179586 * block.unsafe_load(j + _PAIR_OFFSET)
        block.unsafe_store(j, fma(radius * cos(theta), stddev, mean))
        block.unsafe_store(
            j + _PAIR_OFFSET, fma(radius * sin(theta), stddev, mean)
        )


def _fill_blocks_float[
    full_draw: Bool, vector_width: Int
](
    dst: MutPointer[Float32, _],
    numel: Int,
    mut rng: _MT19937,
    mean: Float32,
    stddev: Float32,
):
    # A block has `_PAIR_OFFSET` pairs, so the Box-Muller lanes stop there even
    # on a wider vector unit.
    comptime pair_width = min(vector_width, _PAIR_OFFSET)
    var num_blocks = numel // _BLOCK
    # The wide path draws a uniform for every element up front. The narrow one
    # only for the full blocks.
    rng.fill_float_uniforms[vector_width](
        dst, numel if full_draw else num_blocks * _BLOCK
    )
    for block in range(num_blocks):
        _normal_fill_16_float[pair_width](
            dst.unsafe_offset(_BLOCK * block), mean, stddev
        )
    if numel % _BLOCK != 0:
        var tail = dst.unsafe_offset(numel - _BLOCK)
        rng.fill_float_uniforms[vector_width](tail, _BLOCK)
        _normal_fill_16_float[pair_width](tail, mean, stddev)


def _fill_blocks_double(
    dst: MutPointer[Float64, _],
    numel: Int,
    mut rng: _MT19937,
    mean: Float64,
    stddev: Float64,
):
    for i in range(numel):
        dst.unsafe_store(i, rng.next_double_uniform())
    for block in range(numel // _BLOCK):
        _normal_fill_16_double(dst.unsafe_offset(_BLOCK * block), mean, stddev)
    if numel % _BLOCK != 0:
        var tail = dst.unsafe_offset(numel - _BLOCK)
        for i in range(_BLOCK):
            tail.unsafe_store(i, rng.next_double_uniform())
        _normal_fill_16_double(tail, mean, stddev)


def _fill_small[
    dtype: DType
](
    dst: MutPointer[Scalar[dtype], _],
    numel: Int,
    mut rng: _MT19937,
    mean: Float64,
    stddev: Float64,
):
    """`at::normal_distribution<double>` for fewer than 16 elements."""
    var cached = Float64(0)
    var has_cached = False
    for i in range(numel):
        var sample: Float64
        if has_cached:
            sample = cached
            has_cached = False
        else:
            var u1 = rng.next_double_uniform()
            var u2 = rng.next_double_uniform()
            var radius = sqrt(-2.0 * external_call["log1p", Float64](-u2))
            var theta = 6.283185307179586 * u1
            cached = radius * sin(theta)
            has_cached = True
            sample = radius * cos(theta)
        dst.unsafe_store(i, fma(sample, stddev, mean).cast[dtype]())


def torch_cpu_normal_fill[
    full_draw: Bool = True,
    vector_width: Int = simd_width_of[DType.float32](),
](
    dst: MutPointer[Float32, _],
    numel: Int,
    seed: UInt64,
    mean: Float32,
    stddev: Float32,
):
    """Fills `dst` with `torch.empty(numel).normal_(mean, stddev)` for a
    `float32` tensor and a generator seeded with `seed`.

    Parameters:
        full_draw: Whether torch draws one uniform per element before it
            transforms them. True for `float32`. False for `float16` and
            `bfloat16`, whose values are this output rounded to the narrow type.
        vector_width: The SIMD width of the Mersenne twister and the
            Box-Muller lanes, a power of two. It changes speed, not the values.

    Args:
        dst: Destination of `numel` values.
        numel: Number of values to generate.
        seed: The generator seed. Torch keeps its low 32 bits.
        mean: The mean of the distribution.
        stddev: The standard deviation of the distribution.
    """
    var rng = _MT19937(seed)
    if numel < _BLOCK:
        _fill_small(dst, numel, rng, Float64(mean), Float64(stddev))
    else:
        _fill_blocks_float[full_draw, vector_width](
            dst, numel, rng, mean, stddev
        )


def torch_cpu_normal_fill(
    dst: MutPointer[Float64, _],
    numel: Int,
    seed: UInt64,
    mean: Float32,
    stddev: Float32,
):
    """Fills `dst` with `torch.empty(numel, dtype=torch.float64).normal_(mean,
    stddev)` for a generator seeded with `seed`.

    Args:
        dst: Destination of `numel` values.
        numel: Number of values to generate.
        seed: The generator seed. Torch keeps its low 32 bits.
        mean: The mean of the distribution.
        stddev: The standard deviation of the distribution.
    """
    var rng = _MT19937(seed)
    if numel < _BLOCK:
        _fill_small(dst, numel, rng, Float64(mean), Float64(stddev))
    else:
        _fill_blocks_double(dst, numel, rng, Float64(mean), Float64(stddev))
