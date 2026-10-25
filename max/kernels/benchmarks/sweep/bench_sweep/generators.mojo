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
"""Deterministic device-side input generators and their CPU references.

Every generator writes element `i` as a pure function of its arguments and
`i`, and that function is shared between the device kernel and the `*_host`
reference, so a correctness check can rebuild any input on the host without
copying it back. Kernels are grid-stride loops that assume nothing about warp
width, so they run at any launch size on any GPU. A target the fill kernels
do not run on (`device_fill_supported`) fills from the CPU reference and
copies instead, through the same formulas, so a plugin-backed accelerator
needs no fill kernels of its own; a build for a toolchain that compiles
offload functions ahead of time drops the kernels entirely with
`-D sweep_fill_kernels=0`.
"""

from std.math import ceildiv, fma, min
from std.memory.alloc import unsafe_alloc
from std.random import rand
from std.sys import get_defined_int

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext

from .backend import device_fill_supported

# Whether this build contains the fill kernels at all. Unset, it does, and
# a launch dispatches on `device_fill_supported` at runtime: the targets the
# library knows JIT-compile device code, so a build cannot tell from its own
# target whether the kernels will run. A target whose toolchain compiles
# offload functions ahead of time, for a device the `max.gpu` intrinsics do
# not exist on, builds with `-D sweep_fill_kernels=0`: the kernels are then
# never referenced, so the toolchain never compiles them, and every fill
# runs on the host and copies.
comptime _FILL_KERNELS = get_defined_int["sweep_fill_kernels", 1]() != 0


# ===----------------------------------------------------------------------=== #
# Launch shape
# ===----------------------------------------------------------------------=== #

comptime _BLOCK = 256
# Enough resident blocks to saturate any current GPU when the SM count is
# unavailable; grid-stride loops cover the rest.
comptime _FALLBACK_GRID = 4096
comptime _BLOCKS_PER_SM = 8


def _launch_dims(ctx: DeviceContext, n: Int) -> Tuple[Int, Int]:
    var block = _BLOCK
    var max_grid = _FALLBACK_GRID
    try:
        block = min(
            block, ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK)
        )
        max_grid = (
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
            * _BLOCKS_PER_SM
        )
    except:
        pass
    return (block, max(1, min(ceildiv(n, block), max_grid)))


@inline(.always)
def _thread_start() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@inline(.always)
def _thread_stride() -> Int:
    return Int(grid_dim.x * block_dim.x)


# ===----------------------------------------------------------------------=== #
# Host fill and copy
# ===----------------------------------------------------------------------=== #


def _copy_to_device[
    dtype: DType
](
    ctx: DeviceContext,
    buf: DeviceBuffer[dtype, ...],
    values: List[Scalar[dtype]],
) raises:
    """Copies `values` into the leading `len(values)` elements of `buf`.

    The host half of the fill fallback for a target the fill kernels do not
    run on: the values are the same generator's CPU reference, and the copy
    is the host-to-device transfer every such target's benchmarks already
    use. The synchronize lets the caller's list die with this call.
    """
    if len(values) > len(buf):
        raise Error(
            "cannot fill ",
            len(buf),
            " elements from ",
            len(values),
            " host values",
        )
    var view = buf if len(values) == len(buf) else DeviceBuffer[dtype](
        ctx, buf.unsafe_ptr(), len(values), owning=False
    )
    ctx.enqueue_copy(view, values.unsafe_ptr().as_imm())
    ctx.synchronize()


def _device_to_host[
    dtype: DType
](ctx: DeviceContext, buf: DeviceBuffer[dtype, ...]) raises -> List[
    Scalar[dtype]
]:
    """Returns a copy of `buf`'s contents, through the same device-to-host
    transfer as `_copy_to_device`."""
    var host = unsafe_alloc[Scalar[dtype]](len(buf))
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var out = List[Scalar[dtype]](capacity=len(buf))
    for i in range(len(buf)):
        out.append(host[unsafe_offset=i])
    host.unsafe_free()
    return out^


# ===----------------------------------------------------------------------=== #
# Uniform
# ===----------------------------------------------------------------------=== #


@inline(.always)
def mix32(x: UInt32) -> UInt32:
    """Hashes 32 bits with full avalanche (lowbias32, Wellons)."""
    var h = x
    h ^= h >> 16
    h *= 0x7FEB352D
    h ^= h >> 15
    h *= 0x846CA68B
    h ^= h >> 16
    return h


@inline(.always)
def uniform_bits(seed: UInt32, i: Int) -> UInt32:
    """Returns 32 hashed bits for element `i` of the stream `seed`."""
    var key = mix32(seed ^ 0x9E3779B9)
    return mix32(UInt32(i) ^ mix32(UInt32(i >> 32) ^ key))


@inline(.always)
def uniform_at[
    dtype: DType
](seed: UInt32, i: Int, lo: Float32, hi: Float32) -> Scalar[dtype]:
    """Returns element `i` of the uniform stream `seed` on `[lo, hi)`.

    The explicit `fma` pins the rounding: a plain `lo + (hi - lo) * u` may or
    may not be contracted depending on the target, and the host reference must
    produce the same bits as every device. Integer results go through `int64`
    because a float out of the target's range converts differently on each
    target (GPUs saturate), while integer truncation is defined.
    """
    var u = Float32(uniform_bits(seed, i) >> 8) * (1.0 / 16777216.0)
    var x = fma(hi - lo, u, lo)
    comptime if dtype.is_integral():
        return x.cast[DType.int64]().cast[dtype]()
    else:
        return x.cast[dtype]()


def _fill_uniform_kernel[
    dtype: DType
](
    ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int64,
    seed: UInt32,
    lo: Float32,
    hi: Float32,
):
    var i = _thread_start()
    while i < Int(n):
        ptr[unsafe_offset=i] = uniform_at[dtype](seed, i, lo, hi)
        i += _thread_stride()


def fill_uniform[
    dtype: DType
](
    ctx: DeviceContext,
    buf: DeviceBuffer[dtype, ...],
    seed: Int,
    lo: Float32 = 0.0,
    hi: Float32 = 1.0,
    on_host: Bool = False,
) raises:
    """Fills `buf` with `uniform_at(seed, i, lo, hi)`.

    The stream is a pure function of (seed, index), so a fixture filled once
    at the largest size holds the same prefix a smaller one would. `on_host`
    forces the host fill and copy; it is also the path taken for a target the
    fill kernels do not run on (`device_fill_supported`) and for a build
    without them (`-D sweep_fill_kernels=0`).
    """
    comptime if _FILL_KERNELS:
        if not on_host and device_fill_supported(ctx):
            var n = len(buf)
            var dims = _launch_dims(ctx, n)
            comptime kernel = _fill_uniform_kernel[dtype]
            ctx.enqueue_function[kernel](
                buf.unsafe_ptr().as_unsafe_any_origin(),
                Int64(n),
                UInt32(seed),
                lo,
                hi,
                grid_dim=dims[1],
                block_dim=dims[0],
            )
            return
    _copy_to_device(ctx, buf, uniform_host[dtype](len(buf), seed, lo, hi))


def uniform_host[
    dtype: DType
](n: Int, seed: Int, lo: Float32 = 0.0, hi: Float32 = 1.0) -> List[
    Scalar[dtype]
]:
    """Returns the first `n` elements `fill_uniform` writes."""
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(uniform_at[dtype](UInt32(seed), i, lo, hi))
    return out^


def fill_uniform_host_rand[
    dtype: DType
](buf: DeviceBuffer[dtype, ...], lo: Float64 = 0.0, hi: Float64 = 1.0) raises:
    """The status-quo fill: map to host, `std.random.rand`, copy back.

    Kept as the baseline the device fills are measured against, and as the
    fallback for a target without a device fill. Not deterministic across
    sizes and has no reference.
    """
    with buf.map_to_host() as host:
        rand(host.as_span(), min=lo, max=hi)


# ===----------------------------------------------------------------------=== #
# Affine
# ===----------------------------------------------------------------------=== #


@inline(.always)
def affine_at[dtype: DType](i: Int, scale: Int, offset: Int) -> Scalar[dtype]:
    return Scalar[dtype](i * scale + offset)


def _fill_affine_kernel[
    dtype: DType
](
    ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int64,
    scale: Int64,
    offset: Int64,
):
    var i = _thread_start()
    while i < Int(n):
        ptr[unsafe_offset=i] = affine_at[dtype](i, Int(scale), Int(offset))
        i += _thread_stride()


def fill_affine[
    dtype: DType
](
    ctx: DeviceContext,
    buf: DeviceBuffer[dtype, ...],
    scale: Int,
    offset: Int,
    n: Int = -1,
    on_host: Bool = False,
) raises:
    """Writes `buf[i] = i * scale + offset` for the first `n` elements.

    `n < 0` fills the whole buffer. `on_host` is as in `fill_uniform`.
    """
    var count = len(buf) if n < 0 else n
    comptime if _FILL_KERNELS:
        if not on_host and device_fill_supported(ctx):
            var dims = _launch_dims(ctx, count)
            comptime kernel = _fill_affine_kernel[dtype]
            ctx.enqueue_function[kernel](
                buf.unsafe_ptr().as_unsafe_any_origin(),
                Int64(count),
                Int64(scale),
                Int64(offset),
                grid_dim=dims[1],
                block_dim=dims[0],
            )
            return
    _copy_to_device(ctx, buf, affine_host[dtype](count, scale, offset))


def affine_host[
    dtype: DType
](n: Int, scale: Int, offset: Int) -> List[Scalar[dtype]]:
    """Returns the first `n` elements `fill_affine` writes."""
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(affine_at[dtype](i, scale, offset))
    return out^


# ===----------------------------------------------------------------------=== #
# Page tables
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct PageLayout(Equatable, TrivialRegisterPassable, Writable):
    """How logical pages map onto physical blocks.

    Gather locality changes kernel time, so it is an axis to sweep rather than
    whatever the allocator happened to produce.
    """

    var _value: Int

    comptime IDENTITY = Self(0)
    """Sequence pages are consecutive blocks, in order."""
    comptime SHUFFLED = Self(1)
    """Every page lands on an unrelated block."""
    comptime FRAGMENTED = Self(2)
    """Runs of consecutive blocks, with the runs shuffled."""

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @staticmethod
    def parse(name: String) raises -> Self:
        if name == "identity":
            return Self.IDENTITY
        if name == "shuffled":
            return Self.SHUFFLED
        if name == "fragmented":
            return Self.FRAGMENTED
        raise Error("unknown page layout: ", name)

    def write_to(self, mut writer: Some[Writer]):
        if self == Self.IDENTITY:
            writer.write("identity")
        elif self == Self.SHUFFLED:
            writer.write("shuffled")
        else:
            writer.write("fragmented")


@inline(.always)
def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


@inline(.always)
def coprime_near(n: Int, target: Int) -> Int:
    """Returns the smallest `m >= max(1, target)` with `gcd(m, n) == 1`."""
    var m = max(1, target)
    while _gcd(m, n) != 1:
        m += 1
    return m


@fieldwise_init
struct PagePermutation(TrivialRegisterPassable):
    """A bijection on `[0, n)` that moves runs of `run` consecutive entries.

    Full runs are permuted by `u -> (u * mult + offset) % units`; a partial
    last run stays in place. `run == n` is the identity and `run == 1`
    shuffles single entries.
    """

    var n: Int
    var run: Int
    var mult: Int
    var offset: Int

    @staticmethod
    def make(layout: PageLayout, n: Int, seed: Int, run: Int = 4) -> Self:
        var r: Int
        if layout == PageLayout.IDENTITY:
            r = max(1, n)
        elif layout == PageLayout.SHUFFLED:
            r = 1
        else:
            r = max(1, min(run, n))
        var units = max(1, n // r)
        # A multiplier near units / phi scatters neighbors across the range.
        var mult = coprime_near(units, (units * 618) // 1000)
        return Self(n, r, mult, Int(uniform_bits(UInt32(seed), 0)) % units)

    @inline(.always)
    def __call__(self, j: Int) -> Int:
        var units = max(1, self.n // self.run)
        if j >= units * self.run:
            return j
        var u = j // self.run
        return ((u * self.mult + self.offset) % units) * self.run + (
            j % self.run
        )


def _fill_page_table_kernel(
    ptr: Pointer[UInt32, MutAnyOrigin],
    n: Int64,
    run: Int64,
    mult: Int64,
    offset: Int64,
    first_block: Int64,
):
    var perm = PagePermutation(Int(n), Int(run), Int(mult), Int(offset))
    var i = _thread_start()
    while i < Int(n):
        ptr[unsafe_offset=i] = UInt32(perm(i) + Int(first_block))
        i += _thread_stride()


def fill_page_table(
    ctx: DeviceContext,
    lut: DeviceBuffer[DType.uint32, ...],
    layout: PageLayout,
    seed: Int = 0,
    first_block: Int = 1,
    run: Int = 4,
    on_host: Bool = False,
) raises:
    """Fills a flat `[batch, pages_per_seq]` page table.

    Entry `j` is `first_block + perm(j)` for a `PagePermutation` over the
    whole table, so every sequence gets distinct blocks and block
    `first_block - 1` and below stay unreferenced (production's null block).
    `on_host` is as in `fill_uniform`.
    """
    var n = len(lut)
    comptime if _FILL_KERNELS:
        if not on_host and device_fill_supported(ctx):
            var perm = PagePermutation.make(layout, n, seed, run)
            var dims = _launch_dims(ctx, n)
            ctx.enqueue_function[_fill_page_table_kernel](
                lut.unsafe_ptr().as_unsafe_any_origin(),
                Int64(n),
                Int64(perm.run),
                Int64(perm.mult),
                Int64(perm.offset),
                Int64(first_block),
                grid_dim=dims[1],
                block_dim=dims[0],
            )
            return
    _copy_to_device(
        ctx, lut, page_table_host(n, layout, seed, first_block, run)
    )


def page_table_host(
    n: Int,
    layout: PageLayout,
    seed: Int = 0,
    first_block: Int = 1,
    run: Int = 4,
) -> List[UInt32]:
    """Returns the page table `fill_page_table` writes into `n` entries."""
    var perm = PagePermutation.make(layout, n, seed, run)
    var out = List[UInt32](capacity=n)
    for j in range(n):
        out.append(UInt32(perm(j) + first_block))
    return out^


# ===----------------------------------------------------------------------=== #
# Sparse index lists
# ===----------------------------------------------------------------------=== #


@inline(.always)
def sparse_stride(window: Int, count: Int) -> Int:
    """Returns a stride coprime with `window` that spreads `count` picks.

    Coprime means `(i * stride + c) % window` never repeats within `window`
    picks; starting near `window // count` spreads them over the whole window
    the way a real top-k list touches pages across the context.
    """
    return coprime_near(window, window // count)


@inline(.always)
def distinct_stride(m: Int, window: Int, count: Int) -> Int:
    """Returns `m`, or the next stride coprime with `window` if `m` repeats.

    `count` picks `(i * m + c) % window` are distinct whenever
    `m * (count - 1) < window`, since every pairwise difference is then a
    nonzero value below `window`. Keeping `m` there lets two benchmarks share
    an exact gather pattern; past that point only a coprime stride avoids
    repeats.
    """
    if m * (count - 1) < window:
        return m
    return coprime_near(window, m)


@inline(.always)
def sparse_slot(
    lut: Pointer[UInt32, ImmutAnyOrigin],
    lut_cols: Int,
    page_size: Int,
    pad_invalid: Bool,
    batch: Int,
    s: Int,
    i: Int,
    cache_len: Int,
    seq_len: Int,
    top_k: Int,
    index_stride: Int = 0,
) -> Int32:
    """Returns the physical KV row of pick `i` for query position `s`.

    `pad_invalid` gives each position a causal window of `cache_len + s + 1`
    keys and pads picks past it with -1, as the prefill index producer does.
    Without it every position draws `top_k` picks from all keys, wrapping, as
    the decode benchmark has always done.

    `index_stride == 0` spreads the picks over the window; a positive stride
    is kept, through `distinct_stride`, so a sweep can reproduce another
    benchmark's gather pattern.
    """
    var window = cache_len + s + 1 if pad_invalid else cache_len + seq_len
    var valid = min(window, top_k)
    if pad_invalid and i >= valid:
        return -1
    var stride = distinct_stride(
        index_stride, window, valid
    ) if index_stride > 0 else sparse_stride(window, valid)
    var t = (i * stride + s) % window
    var block = Int(lut[unsafe_offset=batch * lut_cols + t // page_size])
    return Int32(block * page_size + t % page_size)


def _sparse_indices_kernel(
    dst: Pointer[Int32, MutAnyOrigin],
    lut: Pointer[UInt32, ImmutAnyOrigin],
    lut_cols: Int64,
    page_size: Int64,
    pad_invalid: Int64,
    batch_size: Int64,
    seq_len_: Int64,
    cache_len: Int64,
    top_k_: Int64,
    index_stride: Int64,
):
    var top_k = Int(top_k_)
    var seq_len = Int(seq_len_)
    var n = Int(batch_size) * seq_len * top_k
    var idx = _thread_start()
    while idx < n:
        var row = idx // top_k
        dst[unsafe_offset=idx] = sparse_slot(
            lut,
            Int(lut_cols),
            Int(page_size),
            pad_invalid != 0,
            row // seq_len,
            row % seq_len,
            idx % top_k,
            Int(cache_len),
            seq_len,
            top_k,
            Int(index_stride),
        )
        idx += _thread_stride()


def fill_sparse_indices(
    ctx: DeviceContext,
    dst: DeviceBuffer[DType.int32, ...],
    lut: DeviceBuffer[DType.uint32, ...],
    lut_cols: Int,
    page_size: Int,
    pad_invalid: Bool,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    top_k: Int,
    on_host: Bool = False,
    index_stride: Int = 0,
) raises:
    """Writes `batch_size * seq_len` rows of `top_k` physical KV rows.

    `on_host` runs the same formula on the CPU and copies, which is also the
    path taken for a target the fill kernels do not run on and for a build
    without them. See `sparse_slot` for `index_stride`.
    """
    comptime if _FILL_KERNELS:
        if not on_host and device_fill_supported(ctx):
            var n = batch_size * seq_len * top_k
            var dims = _launch_dims(ctx, n)
            ctx.enqueue_function[_sparse_indices_kernel](
                dst.unsafe_ptr().as_unsafe_any_origin(),
                lut.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                Int64(lut_cols),
                Int64(page_size),
                Int64(Int(pad_invalid)),
                Int64(batch_size),
                Int64(seq_len),
                Int64(cache_len),
                Int64(top_k),
                Int64(index_stride),
                grid_dim=dims[1],
                block_dim=dims[0],
            )
            return
    _copy_to_device(
        ctx,
        dst,
        sparse_indices_host(
            _device_to_host(ctx, lut),
            lut_cols,
            page_size,
            pad_invalid,
            batch_size,
            seq_len,
            cache_len,
            top_k,
            index_stride,
        ),
    )


def sparse_indices_host(
    lut: List[UInt32],
    lut_cols: Int,
    page_size: Int,
    pad_invalid: Bool,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    top_k: Int,
    index_stride: Int = 0,
) -> List[Int32]:
    """Returns the rows `fill_sparse_indices` writes for page table `lut`."""
    var lut_ptr = lut.unsafe_ptr().as_imm().as_unsafe_any_origin()
    var out = List[Int32](capacity=batch_size * seq_len * top_k)
    for row in range(batch_size * seq_len):
        for i in range(top_k):
            out.append(
                sparse_slot(
                    lut_ptr,
                    lut_cols,
                    page_size,
                    pad_invalid,
                    row // seq_len,
                    row % seq_len,
                    i,
                    cache_len,
                    seq_len,
                    top_k,
                    index_stride,
                )
            )
    return out^


# ===----------------------------------------------------------------------=== #
# Ragged lengths
# ===----------------------------------------------------------------------=== #


@inline(.always)
def linear_grade_at(i: Int, n: Int, lo: Int, hi: Int) -> UInt32:
    """Returns entry `i` of `n` lengths graded linearly from `lo` to `hi`.

    The first entry is `lo` and the last `hi`. A single entry takes the
    midpoint, so a batch of one is the same with or without the grade.
    """
    if n <= 1:
        return UInt32((lo + hi) // 2)
    return UInt32(lo + (hi - lo) * i // (n - 1))


def _fill_linear_grade_kernel(
    ptr: Pointer[UInt32, MutAnyOrigin], n: Int64, lo: Int64, hi: Int64
):
    var i = _thread_start()
    while i < Int(n):
        ptr[unsafe_offset=i] = linear_grade_at(i, Int(n), Int(lo), Int(hi))
        i += _thread_stride()


def fill_linear_grade(
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.uint32, ...],
    lo: Int,
    hi: Int,
    n: Int = -1,
    on_host: Bool = False,
) raises:
    """Writes the first `n` entries of a linear grade from `lo` to `hi`.

    `n < 0` fills the whole buffer. `on_host` is as in `fill_uniform`.
    """
    var count = len(buf) if n < 0 else n
    comptime if _FILL_KERNELS:
        if not on_host and device_fill_supported(ctx):
            var dims = _launch_dims(ctx, count)
            ctx.enqueue_function[_fill_linear_grade_kernel](
                buf.unsafe_ptr().as_unsafe_any_origin(),
                Int64(count),
                Int64(lo),
                Int64(hi),
                grid_dim=dims[1],
                block_dim=dims[0],
            )
            return
    _copy_to_device(ctx, buf, linear_grade_host(count, lo, hi))


def linear_grade_host(n: Int, lo: Int, hi: Int) -> List[UInt32]:
    """Returns the `n` entries `fill_linear_grade` writes."""
    var out = List[UInt32](capacity=n)
    for i in range(n):
        out.append(linear_grade_at(i, n, lo, hi))
    return out^


# ===----------------------------------------------------------------------=== #
# Page table views
# ===----------------------------------------------------------------------=== #


@inline(.always)
def lut_view_at(
    src: Pointer[UInt32, ImmutAnyOrigin],
    src_cols: Int,
    cols: Int,
    used_cols: Int,
    j: Int,
) -> UInt32:
    """Returns entry `j` of a `[rows, cols]` view of a `src_cols`-wide table.

    Row `r` keeps the leading `used_cols` entries of source row `r` and points
    the rest at block 0, the null block.
    """
    var r = j // cols
    var c = j % cols
    if c >= used_cols:
        return 0
    return src[unsafe_offset=r * src_cols + c]


def _fill_lut_view_kernel(
    dst: Pointer[UInt32, MutAnyOrigin],
    src: Pointer[UInt32, ImmutAnyOrigin],
    n: Int64,
    src_cols: Int64,
    cols: Int64,
    used_cols: Int64,
):
    var j = _thread_start()
    while j < Int(n):
        dst[unsafe_offset=j] = lut_view_at(
            src, Int(src_cols), Int(cols), Int(used_cols), j
        )
        j += _thread_stride()


def fill_lut_view(
    ctx: DeviceContext,
    dst: DeviceBuffer[DType.uint32, ...],
    src: DeviceBuffer[DType.uint32, ...],
    src_cols: Int,
    rows: Int,
    cols: Int,
    used_cols: Int,
    on_host: Bool = False,
) raises:
    """Writes a `[rows, cols]` page table that views the leading rows of `src`.

    A captured decode graph sizes its page table from frozen metadata, wider
    than the sequences' real pages; this builds that table over a fixture's
    blocks without refilling them. `cols` may exceed `src_cols`, but
    `used_cols` may not. `on_host` is as in `fill_uniform`.
    """
    if used_cols > src_cols or used_cols > cols:
        raise Error(
            "lut view uses ",
            used_cols,
            " columns of a ",
            src_cols,
            "-wide source into ",
            cols,
        )
    comptime if _FILL_KERNELS:
        if not on_host and device_fill_supported(ctx):
            var n = rows * cols
            var dims = _launch_dims(ctx, n)
            ctx.enqueue_function[_fill_lut_view_kernel](
                dst.unsafe_ptr().as_unsafe_any_origin(),
                src.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                Int64(n),
                Int64(src_cols),
                Int64(cols),
                Int64(used_cols),
                grid_dim=dims[1],
                block_dim=dims[0],
            )
            return
    _copy_to_device(
        ctx,
        dst,
        lut_view_host(
            _device_to_host(ctx, src), src_cols, rows, cols, used_cols
        ),
    )


def lut_view_host(
    src: List[UInt32], src_cols: Int, rows: Int, cols: Int, used_cols: Int
) -> List[UInt32]:
    """Returns the table `fill_lut_view` writes from source table `src`."""
    var src_ptr = src.unsafe_ptr().as_imm().as_unsafe_any_origin()
    var out = List[UInt32](capacity=rows * cols)
    for j in range(rows * cols):
        out.append(lut_view_at(src_ptr, src_cols, cols, used_cols, j))
    return out^
