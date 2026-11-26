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
"""MI355X (gfx950) bf16/fp16 conv2d that reads the caller's FCRS filter in place.

For a small image (30x30, 96 -> 96) the 4-wave conv declines (`M % 64 != 0`)
and the FCRS -> KRSC repack launch a flat im2col kernel needs is a third of
the time. This kernel has no repack: it lays the K axis out in `(c, r, s)`
order, which is the order of an FCRS filter row (`[F, C, R, S]`, `K = C*R*S`
contiguous elements per output channel), so the MFMA B fragments are 16-byte
loads of the filter as the caller holds it. A sum does not care which
reduction element sits at which k, only that both operands agree.

One workgroup walks strips of 16 output pixels. `NL` load waves gather a
strip's whole reduction into an LDS slot (two slots, a ring), and `NOW`
compute waves split the output channels, read the A fragments from the slot
and the B fragments from the filter, and write the output from the
accumulators. The strips of different workgroups never overlap, so no
workgroup waits on another.

The epilogue applies `compute_fn` to the output-dtype value, then the
residual (`+ beta * source`), then `epilogue_fn` or the store, once per output
element.
"""

from std.math import ceildiv
from std.sys import align_of
from std.utils import StaticTuple

from layout import Coord, TensorEngine, TensorLayout, TileTensor
from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import readfirstlane
from max.gpu.memory import AddressSpace, external_memory
from max.gpu.sync import barrier, s_waitcnt

from structured_kernels.pipeline import ProducerConsumerPipeline
from structured_kernels.pipeline_backend import (
    AmdCounterBackend,
    AmdCounterOptions,
    AmdCounterPtr,
)

comptime LdsBytes = Pointer[
    UInt8, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]
"""An LDS slot, addressed in bytes."""

comptime _GlobalPtr[dt: DType] = Pointer[
    Scalar[dt], ImmUntrackedOrigin, address_space=AddressSpace.GLOBAL
]
"""Global, not generic: a generic access counts against both `vmcnt` and
`lgkmcnt`, so every LDS wait would also drain the device loads."""
comptime _MutGlobalPtr[dt: DType] = Pointer[
    Scalar[dt], MutUntrackedOrigin, address_space=AddressSpace.GLOBAL
]
comptime _Frag[dt: DType] = SIMD[dt, 8]
"""The 16 bytes one lane of a 16x16x32 MFMA A or B fragment holds."""
comptime _Acc = SIMD[.float32, 4]
"""The four fp32 cells one lane of a 16x16x32 MFMA accumulator holds."""

comptime _RING_DEPTH = 2
comptime _RING_OPTIONS = AmdCounterOptions(
    ordered=True, spin_sleep=1, spin_cap=1 << 26
)
"""The load waves fill a slot with LDS stores and the compute waves free it
with LDS reads: no workgroup barrier, each wave drains its own `lgkmcnt`
before it signals."""


@fieldwise_init
struct Conv2dFcrsConfig(ImplicitlyCopyable, Movable):
    """The geometry and the launch numbers of the FCRS-in-place conv2d."""

    var C: Int
    """Input channels; a multiple of 8."""
    var F: Int
    """Output channels; the compute waves split them in tiles of 16."""
    var R: Int
    """Filter rows."""
    var S: Int
    """Filter columns."""
    var H: Int
    """Input rows."""
    var W: Int
    """Input columns."""
    var pad: Int
    """Symmetric padding."""
    var stride: Int
    """Stride in both directions."""
    var NL: Int
    """Load waves."""
    var NOW: Int
    """Compute waves."""
    var element_bytes: Int
    """Bytes of one element."""

    def K(self) -> Int:
        """The whole reduction: one handoff per strip."""
        return self.C * self.R * self.S

    def H_out(self) -> Int:
        return (self.H + 2 * self.pad - self.R) // self.stride + 1

    def W_out(self) -> Int:
        return (self.W + 2 * self.pad - self.S) // self.stride + 1

    def pixels(self) -> Int:
        """Output pixels per image."""
        return self.H_out() * self.W_out()

    def pitch(self) -> Int:
        """Elements per LDS row: a strip row plus 16 bytes, so the 16 rows one
        MFMA reads start in different banks."""
        return self.K() + 8

    def slot_bytes(self) -> Int:
        return 16 * self.pitch() * self.element_bytes

    def channel_tiles(self) -> Int:
        """Tiles of 16 output channels each compute wave owns."""
        return self.F // 16 // self.NOW

    def chunk_groups(self) -> Int:
        """16-byte chunks a pixel's tap holds."""
        return self.C // 8

    def units(self) -> Int:
        """Chunks a slot is made of: (output pixel, tap, 8-channel group)."""
        return 16 * self.R * self.S * self.chunk_groups()

    def units_per_lane(self) -> Int:
        """Chunks each load lane gathers per slot."""
        return ceildiv(self.units(), self.NL * WARP_SIZE)

    def threads(self) -> Int:
        return (self.NL + self.NOW) * WARP_SIZE

    def words_offset(self) -> Int:
        return _RING_DEPTH * self.slot_bytes()

    def lds_bytes(self) -> Int:
        return self.words_offset() + 8 * 2 * _RING_DEPTH

    def is_valid(self, dt: DType) -> Bool:
        return (
            (dt == .bfloat16 or dt == .float16)
            and self.element_bytes == 2
            and self.C % 8 == 0
            and self.K() % 32 == 0
            and self.NL >= 1
            and self.NOW >= 1
            and self.F % (16 * self.NOW) == 0
        )


def _gather_strip[
    dt: DType, //, config: Conv2dFcrsConfig
](slot: LdsBytes, x: _GlobalPtr[dt], row0: Int, rows: Int, wave: Int):
    """Gathers the im2col strip of the 16 output pixels from `row0` into
    `slot`, this load wave's share of its chunks.

    Every chunk is 8 consecutive channels of one input pixel (16 bytes of
    device memory) and lands `R*S` elements apart in the slot row, one 2-byte
    LDS store per element. A pixel past the last one repeats it (its output
    is masked in the epilogue); a sample outside the image is zero.
    """
    comptime PER = config.units_per_lane()
    comptime STEP = config.NL * WARP_SIZE
    comptime CPR = config.chunk_groups()
    comptime RS = config.R * config.S
    var me = wave * WARP_SIZE + lane_id()
    var chunks = StaticTuple[_Frag[dt], PER](fill=SIMD[dt, 8](0))
    # Every chunk this lane owes the slot, then one wait, then the stores: a
    # wait per chunk would put the device load's latency on the critical path
    # once per chunk instead of once per slot.
    comptime for i in range(PER):
        var unit = me + i * STEP
        var rt, group = divmod(unit, CPR)
        var row, tap = divmod(rt, RS)
        var pixel = min(row0 + row, rows - 1)
        var image, rest = divmod(pixel, config.pixels())
        var oy, ox = divmod(rest, config.W_out())
        var tap_y, tap_x = divmod(tap, config.S)
        var iy = oy * config.stride + tap_y - config.pad
        var ix = ox * config.stride + tap_x - config.pad
        if (
            unit < config.units()
            and iy >= 0
            and iy < config.H
            and ix >= 0
            and ix < config.W
        ):
            chunks[i] = x.unsafe_offset(
                ((image * config.H + iy) * config.W + ix) * config.C + 8 * group
            ).unsafe_bitcast[_Frag[dt]]()[]
    s_waitcnt[vmcnt=0]()
    comptime for i in range(PER):
        var unit = me + i * STEP
        if unit < config.units():
            var rt, group = divmod(unit, CPR)
            var row, tap = divmod(rt, RS)
            var base = row * config.pitch() + 8 * group * RS + tap
            comptime for e in range(8):
                slot.unsafe_offset(
                    (base + e * RS) * config.element_bytes
                ).unsafe_bitcast[Scalar[dt]]()[] = chunks[i][e]


def _accumulate_strip[
    dt: DType, //, config: Conv2dFcrsConfig
](
    slot: LdsBytes,
    w: _GlobalPtr[dt],
    channel0: Int,
    mut acc: StaticTuple[_Acc, config.channel_tiles()],
):
    """Adds the strip's whole reduction to this compute wave's accumulators.

    Lane `l` holds A row `l % 16` and K elements `8 * (l / 16) ..` of each
    32-K step, and the filter row of output channel `channel0 + 16 * t +
    l % 16` at the same K elements. The accumulator cell `4 * (l / 16) + j`
    is output row `4 * (l / 16) + j` of channel `l % 16`.
    """
    comptime KS = config.K() // 32
    comptime NT = config.channel_tiles()
    var quad, r = divmod(lane_id(), 16)
    var a = StaticTuple[_Frag[dt], KS](fill=SIMD[dt, 8](0))
    var b = StaticTuple[_Frag[dt], KS * NT](fill=SIMD[dt, 8](0))
    # Every operand of the slot, then one wait, then the slot's MMAs: a wait
    # per MMA puts the device load's latency on the critical path once per
    # tile.
    comptime for ks in range(KS):
        var base = (channel0 + r) * config.K() + 32 * ks + 8 * quad
        comptime for t in range(NT):
            b[ks * NT + t] = w.unsafe_offset(
                base + 16 * t * config.K()
            ).unsafe_bitcast[_Frag[dt]]()[]
        a[ks] = slot.unsafe_offset(
            (r * config.pitch() + 32 * ks + 8 * quad) * config.element_bytes
        ).unsafe_bitcast[_Frag[dt]]()[]
    s_waitcnt[vmcnt=0, lgkmcnt=0]()
    comptime for ks in range(KS):
        comptime for t in range(NT):
            mma(acc[t], a[ks], b[ks * NT + t], acc[t])


def _store_strip[
    dt: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    config: Conv2dFcrsConfig,
    has_residual: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    acc: StaticTuple[_Acc, config.channel_tiles()],
    y: _MutGlobalPtr[dt],
    source: _GlobalPtr[dt],
    beta: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    row0: Int,
    rows: Int,
    channel0: Int,
):
    """Writes the strip's output, one element at a time:
    `epilogue_fn(compute_fn(acc) + beta * source)`.

    A row past `rows` is the clamped duplicate of the last pixel the gather
    read, so it runs none of the epilogue: `compute_fn` and `epilogue_fn`
    never see a coordinate outside the output."""
    var quad, r = divmod(lane_id(), 16)
    comptime for t in range(config.channel_tiles()):
        var channel = channel0 + 16 * t + r
        comptime for j in range(4):
            var row = row0 + 4 * quad + j
            if row < rows:
                var v = acc[t][j].cast[dt]()
                comptime if has_compute_fn:
                    v = compute_fn[alignment=align_of[Scalar[dt]]()](
                        (row, channel), v
                    )
                comptime if has_residual:
                    var skip = source[
                        unsafe_offset=row * config.F + channel
                    ].cast[.float32]()
                    v = (v.cast[.float32]() + beta * skip).cast[dt]()
                comptime if has_epilogue_fn:
                    epilogue_fn[dt, 1, alignment=align_of[Scalar[dt]]()](
                        Coord(row, channel), v
                    )
                else:
                    y[unsafe_offset=row * config.F + channel] = v


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.threads())
    )
)
def conv2d_fcrs_fprop_kernel[
    config: Conv2dFcrsConfig,
    x_dt: DType,
    w_dt: DType,
    y_dt: DType,
    x_layout: TensorLayout,
    w_layout: TensorLayout,
    y_layout: TensorLayout,
    x_engine: TensorEngine,
    w_engine: TensorEngine,
    y_engine: TensorEngine,
    source_layout: TensorLayout,
    source_engine: TensorEngine,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    has_residual: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    x: TileTensor[x_dt, x_layout, ImmUntrackedOrigin, Engine=x_engine],
    w: TileTensor[w_dt, w_layout, ImmUntrackedOrigin, Engine=w_engine],
    y: TileTensor[y_dt, y_layout, MutUntrackedOrigin, Engine=y_engine],
    source: TileTensor[
        y_dt, source_layout, ImmUntrackedOrigin, Engine=source_engine
    ],
    beta: Float32,
    rows: Int32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
):
    """One workgroup walks a contiguous run of 16-pixel strips.

    Waves `0 ..< NL` gather, waves `NL ..< NL + NOW` compute and store. The
    grid is one-dimensional over strips and each workgroup computes every
    output channel, so its channel base is constant.
    """
    comptime dt = x_dt
    comptime assert (
        w_dt == dt and y_dt == dt
    ), "input, filter and output share one dtype"
    comptime assert config.is_valid(dt), "unsupported conv2d geometry"
    comptime assert (
        x.is_row_major
        and w.is_row_major
        and y.is_row_major
        and source.is_row_major
    )
    var xp = readfirstlane(
        _GlobalPtr[dt](unsafe_from_address=Int(x.unsafe_ptr()))
    )
    var wp = readfirstlane(
        _GlobalPtr[dt](unsafe_from_address=Int(w.unsafe_ptr()))
    )
    var yp = readfirstlane(
        _MutGlobalPtr[dt](unsafe_from_address=Int(y.unsafe_ptr()))
    )
    var sp = readfirstlane(
        _GlobalPtr[dt](unsafe_from_address=Int(source.unsafe_ptr()))
    )

    var lds = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=128
    ]()
    var words = lds.unsafe_offset(config.words_offset()).unsafe_bitcast[Int64]()
    for k in range(thread_idx.x, 2 * _RING_DEPTH, config.threads()):
        words.unsafe_offset(k)[] = 0
    barrier()

    # A workgroup walks only the strips it owns: the static partition leaves
    # the last workgroups none, and they must not run the whole loop.
    var total_rows = Int(rows)
    var tiles = ceildiv(total_rows, 16)
    var per_block = ceildiv(tiles, grid_dim.x)
    var first = block_idx.x * per_block
    var run = max(0, min(tiles - first, per_block))
    var wave = thread_idx.x // WARP_SIZE

    var ring = ProducerConsumerPipeline[
        _RING_DEPTH,
        AmdCounterBackend[Int32(config.NL), Int32(config.NOW), _RING_OPTIONS],
    ](words)
    if wave < config.NL:
        for t in range(run):
            ring.wait_consumer()
            var slot = ring.producer_stage()
            _gather_strip[config](
                lds.unsafe_offset(Int(slot) * config.slot_bytes()),
                xp,
                (first + t) * 16,
                total_rows,
                wave,
            )
            ring.backend.arrive_full(slot)
            ring.producer_step()
    else:
        var channel0 = (wave - config.NL) * (config.channel_tiles() * 16)
        for t in range(run):
            var acc = StaticTuple[_Acc, config.channel_tiles()](fill=_Acc(0))
            ring.wait_producer()
            var slot = ring.consumer_stage()
            _accumulate_strip[config](
                lds.unsafe_offset(Int(slot) * config.slot_bytes()),
                wp,
                channel0,
                acc,
            )
            ring.backend.arrive_empty(slot)
            ring.consumer_step()
            _store_strip[config, has_residual, has_epilogue_fn, has_compute_fn](
                acc,
                yp,
                sp,
                beta,
                epilogue_fn,
                compute_fn,
                (first + t) * 16,
                total_rows,
                channel0,
            )


def launch_conv2d_fcrs_fprop[
    dt: DType,
    w_dt: DType,
    y_dt: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    config: Conv2dFcrsConfig,
    has_residual: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
](
    x: TileTensor[dt, ...],
    w: TileTensor[w_dt, ...],
    y: TileTensor[mut=True, y_dt, ...],
    ctx: DeviceContext,
    *,
    source: TileTensor[y_dt, ...],
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Enqueues the conv2d over `x` (NHWC), the FCRS filter `w` and `y` (NHWC).

    Parameters:
        dt: Element type of the input.
        w_dt: Element type of the filter; must equal `dt`.
        y_dt: Element type of the output; must equal `dt`.
        EpilogueFnType: Type of `epilogue_fn`.
        ComputeFnType: Type of `compute_fn`.
        config: The geometry; its `H`, `W`, `C` and `F` must match the tensors.
        has_residual: Fuse `+ beta * source` before the store.
        has_epilogue_fn: Store each element through `epilogue_fn`.
        has_compute_fn: Transform each element with `compute_fn` first.

    Args:
        x: Input, `[N, H, W, C]`.
        w: Filter in FCRS layout, `[F, C, R, S]`.
        y: Output, `[N, H_out, W_out, F]`.
        ctx: Device context to enqueue on.
        source: The residual, laid out like `y`; read when `has_residual`.
            Otherwise it is never dereferenced.
        beta: Residual scale; `y = epilogue(compute(conv) + beta * source)`.
        epilogue_fn: Stores each output element when `has_epilogue_fn`.
        compute_fn: Transforms each output element when `has_compute_fn`.
    """
    comptime kernel = conv2d_fcrs_fprop_kernel[
        config,
        dt,
        w_dt,
        y_dt,
        x.LayoutType,
        w.LayoutType,
        y.LayoutType,
        x.Engine,
        w.Engine,
        y.Engine,
        source.LayoutType,
        source.Engine,
        EpilogueFnType,
        ComputeFnType,
        has_residual,
        has_epilogue_fn,
        has_compute_fn,
    ]
    var rows = Int(x.dim[0]()) * config.pixels()
    ctx.enqueue_function[kernel](
        x.as_imm(),
        w.as_imm(),
        y,
        source.as_imm(),
        beta,
        Int32(rows),
        host_args=(epilogue_fn, compute_fn),
        grid_dim=ceildiv(rows, 16),
        block_dim=config.threads(),
        shared_mem_bytes=config.lds_bytes(),
    )
