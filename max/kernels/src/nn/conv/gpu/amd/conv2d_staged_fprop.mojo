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
"""MI355X (gfx950) bf16/fp16 conv2d with both operands staged in LDS by DMA.

For a wide image whose output rows hold whole groups of strips (the WAN VAE's
240x416, 96 -> 96) the im2col gather is the cost: every chunk a lane loads is
16 bytes from a different pixel, and the compute waves read the filter from L2
per MFMA batch. This kernel reads each operand once, contiguously:

- A load wave per input row DMAs (`buffer_load ... lds`) the input row of one
  filter row, the group's columns plus the halo, into an LDS slot. All `S`
  taps read it at a column offset, so the image is read `R` times instead of
  `R * S`. Columns outside the image and the pad chunk read past the buffer
  resource and come back zero.
- A load wave per filter plane DMAs the `F` filter rows of one tap into an LDS
  slot of its own ring.
- The compute waves split the group's strips (not the channels) so each
  finishes whole pixel rows. With the filter fragment as the MFMA's A operand
  and the activation as B, a lane's four accumulator cells are four
  consecutive channels of one pixel, which the epilogue stages through LDS and
  stores as contiguous 16-byte runs.

The filter is read as KRSC (`[F, R, S, C]`), so the launcher permutes the
caller's FCRS or RSCF filter once per call. The epilogue applies `compute_fn`
to the output-dtype value, then the residual (`+ beta * source`), then
`epilogue_fn` or the store, once per output element.
"""

from std.math import align_up, ceildiv
from std.sys import align_of
from std.utils import StaticTuple

from layout import Coord, TensorEngine, TensorLayout, TileTensor, row_major
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
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import AMDBufferResource, readfirstlane
from max.gpu.memory import AddressSpace, external_memory
from max.gpu.sync import barrier, s_waitcnt
from structured_kernels.amd_tile_io import _load_to_lds

from structured_kernels.pipeline import ProducerConsumerPipeline
from structured_kernels.pipeline_backend import (
    AmdCounterBackend,
    AmdCounterOptions,
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

comptime _RING_OPTIONS = AmdCounterOptions(
    ordered=True, filled_by_dma=True, spin_sleep=1, spin_cap=1 << 26
)
"""Both rings are filled by `buffer_load ... lds`, so a load wave drains
`vmcnt` before it signals; the compute waves free a slot with LDS reads."""

comptime _InputRing[config: Conv2dStagedConfig] = ProducerConsumerPipeline[
    config.depth,
    AmdCounterBackend[Int32(config.NL), Int32(config.NOW), _RING_OPTIONS],
]
"""The ring of input-row slots: `NL` load waves fill, `NOW` compute waves read."""
comptime _PlaneRing[config: Conv2dStagedConfig] = ProducerConsumerPipeline[
    config.depth_b,
    AmdCounterBackend[Int32(config.NLB), Int32(config.NOW), _RING_OPTIONS],
]
"""The ring of filter-plane slots."""

comptime _STAGED_STRIPS = 2
"""Strips a compute wave stages in LDS at once in the epilogue."""


@fieldwise_init
struct Conv2dStagedConfig(ImplicitlyCopyable, Movable):
    """The geometry and the launch numbers of the staged conv2d.

    The stride is 1, the filter is dilation 1, and a group of `G` strips of 16
    pixels lies in one output row (`W_out % (16 * G) == 0`).
    """

    var C: Int
    """Input channels, a multiple of 32: a filter plane is one handoff."""
    var F: Int
    """Output channels, a multiple of 16; every compute wave owns all of them."""
    var R: Int
    """Filter rows."""
    var S: Int
    """Filter columns, at most 3."""
    var H: Int
    """Input rows."""
    var W: Int
    """Input columns."""
    var pad: Int
    """Symmetric padding."""
    var G: Int
    """Strips of 16 pixels in a group."""
    var NL: Int
    """Waves loading input rows."""
    var NLB: Int
    """Waves loading filter planes."""
    var NOW: Int
    """Compute waves."""
    var depth: Int
    """Slots of the input-row ring."""
    var depth_b: Int
    """Slots of the filter-plane ring."""
    var max_grid: Int
    """Workgroups to launch at most; each walks whole groups."""
    var element_bytes: Int
    """Bytes of one element."""

    def H_out(self) -> Int:
        return self.H + 2 * self.pad - self.R + 1

    def W_out(self) -> Int:
        return self.W + 2 * self.pad - self.S + 1

    def pixels(self) -> Int:
        """Output pixels per image."""
        return self.H_out() * self.W_out()

    def pitch(self) -> Int:
        """Elements per LDS row: a pixel's channels plus a pad chunk, so the 16
        rows one MFMA reads start in different banks."""
        return self.C + 8

    def chunks_per_pixel(self) -> Int:
        """16-byte chunks a pixel occupies in a slot: its data and the pad."""
        return self.C // 8 + 1

    def cols(self) -> Int:
        """Input pixels one slot holds: the group's and the halo's."""
        return self.G * 16 + self.S - 1

    def a_chunks(self) -> Int:
        return self.cols() * self.chunks_per_pixel()

    def a_steps(self) -> Int:
        """DMA instructions an input-row slot takes."""
        return ceildiv(self.a_chunks(), WARP_SIZE)

    def b_chunks(self) -> Int:
        return self.F * self.chunks_per_pixel()

    def b_steps(self) -> Int:
        """DMA instructions a filter-plane slot takes."""
        return ceildiv(self.b_chunks(), WARP_SIZE)

    def a_slot_bytes(self) -> Int:
        return self.cols() * self.pitch() * self.element_bytes

    def b_slot_bytes(self) -> Int:
        return self.F * self.pitch() * self.element_bytes

    def strips_per_wave(self) -> Int:
        """Strips each compute wave runs; a wave that owns fewer repeats the
        group's last one and ignores the result."""
        return ceildiv(self.G, self.NOW)

    def first_strip(self, wave: Int) -> Int:
        """Group-relative index of the first strip compute wave `wave` runs."""
        return wave * (self.G // self.NOW) + min(wave, self.G % self.NOW)

    def strips_owned(self, wave: Int) -> Int:
        """Strips of a group that compute wave `wave` owns."""
        return self.G // self.NOW + (1 if wave < self.G % self.NOW else 0)

    def out_slot_bytes(self) -> Int:
        """One compute wave's staging buffer: `_STAGED_STRIPS` strips of
        pixel rows of `F` channels and a pad chunk."""
        return _STAGED_STRIPS * 16 * (self.F * self.element_bytes + 16)

    def b_offset(self) -> Int:
        return align_up(self.depth * self.a_slot_bytes(), 128)

    def out_offset(self) -> Int:
        return align_up(
            self.b_offset() + self.depth_b * self.b_slot_bytes(), 128
        )

    def words_offset(self) -> Int:
        return self.out_offset() + self.NOW * self.out_slot_bytes()

    def lds_bytes(self) -> Int:
        return self.words_offset() + 8 * 2 * (self.depth + self.depth_b)

    def threads(self) -> Int:
        return (self.NL + self.NLB + self.NOW) * WARP_SIZE

    def is_valid(self, dt: DType) -> Bool:
        return (
            (dt == .bfloat16 or dt == .float16)
            and self.element_bytes == 2
            and self.C % 32 == 0
            and self.F % 16 == 0
            and self.S <= 3
            and self.W_out() % (16 * self.G) == 0
            and self.NL >= 1
            and self.NLB >= 1
            and self.NOW >= 1
        )


@always_inline
def _fill_input_row[
    config: Conv2dStagedConfig, dt: DType
](
    slot: LdsBytes,
    x_address: Int,
    mut state: StaticTuple[UInt32, config.a_steps()],
    row0: Int,
    filter_row: Int,
    rows: Int,
    wave: Int,
):
    """Issues this wave's DMA steps of the input row `filter_row` of the group
    starting at output pixel `row0`.

    Lane `j` of step `d` writes slot chunk `64 * d + j`, which is chunk
    `j % CPP` of pixel `j / CPP`, and reads the matching 16 bytes of input:
    consecutive lanes read consecutive bytes. The group's columns do not depend
    on the filter row, so the first row packs each step's source offset and
    in-image bit into `state` and the others add the row's offset. With `NL`
    waves, wave `w` owns the steps `d` with `d % NL == w`.
    """
    comptime assert config.W_out() % (config.G * 16) == 0
    comptime CPP = config.chunks_per_pixel()
    comptime CPR = CPP - 1
    var me = lane_id()
    var n_elems = (rows // config.pixels()) * (config.H * config.W * config.C)
    var bc = AMDBufferResource(
        Pointer[Scalar[dt], ImmUntrackedOrigin](unsafe_from_address=x_address),
        n_elems,
    )
    var oob = Int32(n_elems)
    # the group's image and output row, and whether filter row `filter_row` of
    # it is inside the image
    var p0 = min(row0, rows - 1)
    var image, q = divmod(p0, config.pixels())
    var oy, ox0 = divmod(q, config.W_out())
    var iy = oy + filter_row - config.pad
    var row_ok = iy >= 0 and iy < config.H
    var delta = Int32((filter_row - config.pad) * config.W * config.C)
    if filter_row == 0:
        comptime for d in range(config.a_steps()):
            if d % config.NL == wave:
                var jj = UInt32(d * WARP_SIZE) + UInt32(me)
                var px = jj // UInt32(CPP)
                var cb = jj - px * UInt32(CPP)
                var ix = Int32(ox0 - config.pad) + Int32(px)
                var word = UInt32(0)
                if (
                    cb < UInt32(CPR)
                    and UInt32(ix) < UInt32(config.W)
                    and px < UInt32(config.cols())
                ):
                    word = (
                        UInt32(
                            ((image * config.H + oy) * config.W + Int(ix))
                            * config.C
                        )
                        + 8 * cb
                    ) | (UInt32(1) << 31)
                state[d] = word
    comptime for d in range(config.a_steps()):
        if d % config.NL == wave:
            var word = state[d]
            var off = oob
            if row_ok and (word >> 31) != 0:
                off = Int32(word & UInt32(0x7FFFFFFF)) + delta
            # the last instruction's tail lanes would write past the slot
            comptime ragged = (
                config.a_chunks() % WARP_SIZE != 0 and d == config.a_steps() - 1
            )
            if not ragged or d * WARP_SIZE + me < config.a_chunks():
                _load_to_lds[width=8](
                    bc,
                    off,
                    slot.unsafe_offset(d * WARP_SIZE * 16).unsafe_bitcast[
                        Scalar[dt]
                    ](),
                )


@always_inline
def _fill_filter_plane[
    config: Conv2dStagedConfig, dt: DType
](slot: LdsBytes, w_address: Int, plane_k0: Int, wave: Int):
    """Issues this wave's DMA steps of the `F` filter rows of the plane that
    starts at element `plane_k0` of the KRSC filter, at the same pitch as the
    input slot. A filter row is 192 contiguous bytes at `C = 96`, so a wave
    instruction moves about five rows."""
    comptime CPP = config.chunks_per_pixel()
    comptime K = config.C * config.R * config.S
    var me = lane_id()
    var bc = AMDBufferResource(
        Pointer[Scalar[dt], ImmUntrackedOrigin](unsafe_from_address=w_address),
        config.F * K,
    )
    var oob = Int32(config.F * K)
    comptime for d in range(config.b_steps()):
        if d % config.NLB == wave:
            var jj = UInt32(d * WARP_SIZE) + UInt32(me)
            var f = jj // UInt32(CPP)
            var cb = jj - f * UInt32(CPP)
            var off = oob
            if cb < UInt32(CPP - 1):
                off = Int32(f * UInt32(K) + UInt32(plane_k0) + 8 * cb)
            comptime ragged = (
                config.b_chunks() % WARP_SIZE != 0 and d == config.b_steps() - 1
            )
            if not ragged or jj < UInt32(config.b_chunks()):
                _load_to_lds[width=8](
                    bc,
                    off,
                    slot.unsafe_offset(d * WARP_SIZE * 16).unsafe_bitcast[
                        Scalar[dt]
                    ](),
                )


@always_inline
def _accumulate_tap[
    config: Conv2dStagedConfig, dt: DType
](
    input_row: LdsBytes,
    plane: LdsBytes,
    tap: Int,
    wave: Int,
    mut acc: StaticTuple[_Acc, config.strips_per_wave() * (config.F // 16)],
):
    """The MFMAs of one filter tap with both operands in LDS.

    The filter fragment is the MFMA's A operand and the activation fragment
    its B operand, so lane `l` holds output pixel `l % 16` and the four
    consecutive channels `4 * (l / 16) ..` of each channel tile. The
    activation fragment of strip `g` is the input row's pixels `16 * g + tap +
    (0 ..< 16)`. Every wave runs `strips_per_wave` strips, and a wave that
    owns fewer reads the group's last strip again.
    """
    comptime NT = config.F // 16
    comptime GS = config.strips_per_wave()
    var quad, r = divmod(lane_id(), 16)
    var first = config.first_strip(wave)
    comptime for ks in range(config.C // 32):
        var filter_frags = StaticTuple[_Frag[dt], NT](fill=SIMD[dt, 8](0))
        var input_frags = StaticTuple[_Frag[dt], GS](fill=SIMD[dt, 8](0))
        comptime for t in range(NT):
            filter_frags[t] = plane.unsafe_offset(
                ((16 * t + r) * config.pitch() + 32 * ks + 8 * quad)
                * config.element_bytes
            ).unsafe_bitcast[_Frag[dt]]()[]
        comptime for i in range(GS):
            input_frags[i] = input_row.unsafe_offset(
                (
                    (16 * min(first + i, config.G - 1) + r + tap)
                    * config.pitch()
                    + 32 * ks
                    + 8 * quad
                )
                * config.element_bytes
            ).unsafe_bitcast[_Frag[dt]]()[]
        s_waitcnt[lgkmcnt=0]()
        comptime for t in range(NT):
            comptime for i in range(GS):
                mma(
                    acc[i * NT + t],
                    filter_frags[t],
                    input_frags[i],
                    acc[i * NT + t],
                )


def _residual_chunks(config: Conv2dStagedConfig) -> Int:
    """Residual chunks of 8 elements a lane holds across a wave's batches."""
    return ceildiv(config.strips_per_wave(), _STAGED_STRIPS) * ceildiv(
        _STAGED_STRIPS * 16 * (config.F * config.element_bytes // 16), WARP_SIZE
    )


@always_inline
def _load_residual[
    dt: DType, //, config: Conv2dStagedConfig
](source: _GlobalPtr[dt], row0: Int, wave: Int) -> StaticTuple[
    SIMD[dt, 8], _residual_chunks(config)
]:
    """Loads the residual chunks this lane stores in `_store_group`, so the
    loads fly while the MFMAs run. The chunks are 16 contiguous bytes a lane
    and 1 KB contiguous a wave instruction."""
    comptime CHUNKS = config.F * config.element_bytes // 16
    comptime READS = ceildiv(_STAGED_STRIPS * 16 * CHUNKS, WARP_SIZE)
    var skips = StaticTuple[SIMD[dt, 8], _residual_chunks(config)](
        fill=SIMD[dt, 8](0)
    )
    var l = lane_id()
    var first = config.first_strip(wave)
    var mine = config.strips_owned(wave)
    comptime for b in range(ceildiv(config.strips_per_wave(), _STAGED_STRIPS)):
        var live = min(_STAGED_STRIPS, mine - _STAGED_STRIPS * b)
        var out_row0 = row0 + 16 * (first + _STAGED_STRIPS * b)
        if live > 0:
            comptime for j in range(READS):
                var kk = j * WARP_SIZE + l
                if kk < live * 16 * CHUNKS:
                    skips[b * READS + j] = source.unsafe_offset(
                        out_row0 * config.F + 8 * kk
                    ).unsafe_bitcast[SIMD[dt, 8]]()[]
    return skips


@always_inline
def _store_group[
    dt: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    config: Conv2dStagedConfig,
    has_residual: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    acc: StaticTuple[_Acc, config.strips_per_wave() * (config.F // 16)],
    skips: StaticTuple[SIMD[dt, 8], _residual_chunks(config)],
    stage: LdsBytes,
    y: _MutGlobalPtr[dt],
    beta: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    row0: Int,
    wave: Int,
):
    """Writes this wave's strips of the group: `epilogue_fn(compute_fn(acc) +
    beta * source)`.

    A wave stages `_STAGED_STRIPS` strips of results in its own LDS buffer,
    then reads them back 16 bytes a lane in pixel-row order, applies the
    epilogue to those 8 contiguous channels and stores them. A pixel row is
    `F` contiguous channels and consecutive pixels are consecutive rows, so
    the residual loads and the stores are 1 KB runs of contiguous memory
    instead of 32-byte pieces of sixteen different rows.
    """
    comptime NT = config.F // 16
    comptime GS = config.strips_per_wave()
    comptime CHUNKS = config.F * config.element_bytes // 16
    comptime POUT = CHUNKS * 16 + 16
    comptime READS = ceildiv(_STAGED_STRIPS * 16 * CHUNKS, WARP_SIZE)
    var l = lane_id()
    var quad, r = divmod(l, 16)
    var first = config.first_strip(wave)
    var mine = config.strips_owned(wave)
    comptime for b in range(ceildiv(GS, _STAGED_STRIPS)):
        var live = min(_STAGED_STRIPS, mine - _STAGED_STRIPS * b)
        var out_row0 = row0 + 16 * (first + _STAGED_STRIPS * b)
        # strips `_STAGED_STRIPS * b ..` of this wave, those it owns
        comptime for k in range(_STAGED_STRIPS):
            comptime i = _STAGED_STRIPS * b + k
            comptime if i < GS:
                if i < mine:
                    comptime for t in range(NT):
                        stage.unsafe_offset(
                            (16 * k + r) * POUT
                            + config.element_bytes * (16 * t + 4 * quad)
                        ).unsafe_bitcast[SIMD[dt, 4]]()[] = acc[
                            i * NT + t
                        ].cast[
                            dt
                        ]()
        s_waitcnt[lgkmcnt=0]()
        # the staged strips, as contiguous output
        if live > 0:
            comptime for j in range(READS):
                var kk = j * WARP_SIZE + l
                if kk < live * 16 * CHUNKS:
                    var pixel, chunk = divmod(kk, CHUNKS)
                    var v = stage.unsafe_offset(
                        pixel * POUT + 16 * chunk
                    ).unsafe_bitcast[SIMD[dt, 8]]()[]
                    comptime if has_compute_fn:
                        v = compute_fn[alignment=align_of[SIMD[dt, 8]]()](
                            (out_row0 + pixel, 8 * chunk), v
                        )
                    comptime if has_residual:
                        v = (
                            v.cast[.float32]()
                            + beta * skips[b * READS + j].cast[.float32]()
                        ).cast[dt]()
                    comptime if has_epilogue_fn:
                        epilogue_fn[dt, 8, alignment=align_of[SIMD[dt, 8]]()](
                            Coord(out_row0 + pixel, 8 * chunk), v
                        )
                    else:
                        y.unsafe_offset(
                            (out_row0 + pixel) * config.F + 8 * chunk
                        ).unsafe_bitcast[SIMD[dt, 8]]()[] = v
        s_waitcnt[lgkmcnt=0]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.threads())
    )
)
def conv2d_staged_fprop_kernel[
    config: Conv2dStagedConfig,
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
    """One workgroup walks whole groups of strips.

    Waves `0 ..< NL` load input rows, the next `NLB` load filter planes and the
    last `NOW` compute and store. A workgroup owns whole groups at any grid,
    so no workgroup waits on another.
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
    var x_address = Int(readfirstlane(Int64(Int(x.unsafe_ptr()))))
    var w_address = Int(readfirstlane(Int64(Int(w.unsafe_ptr()))))
    var yp = readfirstlane(
        _MutGlobalPtr[dt](unsafe_from_address=Int(y.unsafe_ptr()))
    )
    var sp = readfirstlane(
        _GlobalPtr[dt](unsafe_from_address=Int(source.unsafe_ptr()))
    )

    var lds = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=128
    ]()
    var words_a = lds.unsafe_offset(config.words_offset()).unsafe_bitcast[
        Int64
    ]()
    var words_b = words_a.unsafe_offset(2 * config.depth)
    for k in range(
        thread_idx.x, 2 * (config.depth + config.depth_b), config.threads()
    ):
        words_a.unsafe_offset(k)[] = 0
    barrier()

    # The group of `G` strips starting at a group boundary shares one staged
    # input row, so a workgroup owns whole groups.
    var total_rows = Int(rows)
    var tiles = ceildiv(total_rows, 16)
    var per_block = config.G * ceildiv(ceildiv(tiles, config.G), grid_dim.x)
    var first = block_idx.x * per_block
    var run = ceildiv(max(0, min(tiles - first, per_block)), config.G)
    var wave = thread_idx.x // WARP_SIZE

    if wave < config.NL:
        var ring = _InputRing[config](words_a)
        var state = StaticTuple[UInt32, config.a_steps()](fill=0)
        for t in range(run):
            var row0 = (first + t * config.G) * 16
            for filter_row in range(config.R):
                ring.wait_consumer()
                var slot = ring.producer_stage()
                _fill_input_row[config, dt](
                    lds.unsafe_offset(Int(slot) * config.a_slot_bytes()),
                    x_address,
                    state,
                    row0,
                    filter_row,
                    total_rows,
                    wave,
                )
                ring.backend.arrive_full(slot)
                ring.producer_step()
    elif wave < config.NL + config.NLB:
        var ring = _PlaneRing[config](words_b)
        for _ in range(run):
            for filter_row in range(config.R):
                for tap in range(config.S):
                    ring.wait_consumer()
                    var slot = ring.producer_stage()
                    _fill_filter_plane[config, dt](
                        lds.unsafe_offset(
                            config.b_offset()
                            + Int(slot) * config.b_slot_bytes()
                        ),
                        w_address,
                        filter_row * config.S * config.C + tap * config.C,
                        wave - config.NL,
                    )
                    ring.backend.arrive_full(slot)
                    ring.producer_step()
    else:
        var compute_wave = wave - config.NL - config.NLB
        var ring_a = _InputRing[config](words_a)
        var ring_b = _PlaneRing[config](words_b)
        for t in range(run):
            var acc = StaticTuple[
                _Acc, config.strips_per_wave() * (config.F // 16)
            ](fill=_Acc(0))
            var skips = StaticTuple[SIMD[dt, 8], _residual_chunks(config)](
                fill=SIMD[dt, 8](0)
            )
            comptime if has_residual:
                skips = _load_residual[config](
                    sp, (first + t * config.G) * 16, compute_wave
                )
            for _ in range(config.R):
                ring_a.wait_producer()
                var slot_a = ring_a.consumer_stage()
                for tap in range(config.S):
                    ring_b.wait_producer()
                    var slot_b = ring_b.consumer_stage()
                    _accumulate_tap[config, dt](
                        lds.unsafe_offset(Int(slot_a) * config.a_slot_bytes()),
                        lds.unsafe_offset(
                            config.b_offset()
                            + Int(slot_b) * config.b_slot_bytes()
                        ),
                        tap,
                        compute_wave,
                        acc,
                    )
                    ring_b.backend.arrive_empty(slot_b)
                    ring_b.consumer_step()
                ring_a.backend.arrive_empty(slot_a)
                ring_a.consumer_step()
            _store_group[config, has_residual, has_epilogue_fn, has_compute_fn](
                acc,
                skips,
                lds.unsafe_offset(
                    config.out_offset() + compute_wave * config.out_slot_bytes()
                ),
                yp,
                beta,
                epilogue_fn,
                compute_fn,
                (first + t * config.G) * 16,
                compute_wave,
            )


@__name(t"amd_conv2d_krsc_repack_{dtype}")
def _repack_filter_to_krsc[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
    DstLayoutType: TensorLayout,
    dst_origin: MutOrigin,
    DstEngine: TensorEngine,
    filter_is_fcrs: Bool,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst: TileTensor[dtype, DstLayoutType, dst_origin, Engine=DstEngine],
):
    """Permutes the caller's filter, FCRS `[F, C, R, S]` or RSCF `[R, S, C, F]`,
    to KRSC `[F, R, S, C]`, so the reduction is channel-innermost."""
    comptime F = dst.static_shape[0]
    comptime R = dst.static_shape[1]
    comptime S = dst.static_shape[2]
    comptime C = dst.static_shape[3]
    var tid = global_idx.x
    if tid >= F * R * S * C:
        return
    var t1, c = divmod(tid, C)
    var t2, s = divmod(t1, S)
    var f, r = divmod(t2, R)
    comptime if filter_is_fcrs:
        dst[f, r, s, c] = src.load((f, c, r, s))[0]
    else:
        dst[f, r, s, c] = src.load((r, s, c, f))[0]


def launch_conv2d_staged_fprop[
    dt: DType,
    w_dt: DType,
    y_dt: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    config: Conv2dStagedConfig,
    has_residual: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
](
    x: TileTensor[dt, ...],
    krsc: TileTensor[w_dt, ...],
    y: TileTensor[mut=True, y_dt, ...],
    ctx: DeviceContext,
    *,
    source: TileTensor[y_dt, ...],
    beta: Float32 = 0.0,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Enqueues the conv2d over `x` (NHWC), the KRSC filter `krsc` and `y`
    (NHWC).

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
        krsc: Filter, `[F, R, S, C]`.
        y: Output, `[N, H_out, W_out, F]`.
        ctx: Device context to enqueue on.
        source: The residual, laid out like `y`; read when `has_residual`.
            Otherwise it is never dereferenced.
        beta: Residual scale; `y = epilogue(compute(conv) + beta * source)`.
        epilogue_fn: Stores each output element when `has_epilogue_fn`.
        compute_fn: Transforms each output element when `has_compute_fn`.
    """
    comptime kernel = conv2d_staged_fprop_kernel[
        config,
        dt,
        w_dt,
        y_dt,
        x.LayoutType,
        krsc.LayoutType,
        y.LayoutType,
        x.Engine,
        krsc.Engine,
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
    var groups = ceildiv(ceildiv(rows, 16), config.G)
    ctx.enqueue_function[kernel](
        x.as_imm(),
        krsc.as_imm(),
        y,
        source.as_imm(),
        beta,
        Int32(rows),
        host_args=(epilogue_fn, compute_fn),
        grid_dim=min(groups, config.max_grid),
        block_dim=config.threads(),
        shared_mem_bytes=config.lds_bytes(),
    )


def conv2d_staged_fprop[
    dt: DType,
    w_dt: DType,
    y_dt: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    config: Conv2dStagedConfig,
    filter_is_fcrs: Bool,
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
    """Permutes the filter `w` to KRSC and runs the conv2d over `x` (NHWC),
    writing `y` (NHWC).

    Parameters:
        dt: Element type of the input.
        w_dt: Element type of the filter; must equal `dt`.
        y_dt: Element type of the output; must equal `dt`.
        EpilogueFnType: Type of `epilogue_fn`.
        ComputeFnType: Type of `compute_fn`.
        config: The geometry; its `H`, `W`, `C` and `F` must match the tensors.
        filter_is_fcrs: The filter is FCRS `[F, C, R, S]`, else RSCF.
        has_residual: Fuse `+ beta * source` before the store.
        has_epilogue_fn: Store each element through `epilogue_fn`.
        has_compute_fn: Transform each element with `compute_fn` first.

    Args:
        x: Input, `[N, H, W, C]`.
        w: Filter.
        y: Output, `[N, H_out, W_out, F]`.
        ctx: Device context to enqueue on.
        source: The residual, laid out like `y`; read when `has_residual`.
            Otherwise it is never dereferenced.
        beta: Residual scale; `y = epilogue(compute(conv) + beta * source)`.
        epilogue_fn: Stores each output element when `has_epilogue_fn`.
        compute_fn: Transforms each output element when `has_compute_fn`.
    """
    comptime n_filter = config.F * config.R * config.S * config.C
    var krsc_buffer = ctx.enqueue_create_buffer[w_dt](n_filter)
    var krsc = TileTensor(
        krsc_buffer, row_major[config.F, config.R, config.S, config.C]()
    )
    comptime repack = _repack_filter_to_krsc[
        w_dt,
        w.LayoutType,
        ImmOrigin(w.origin),
        w.Engine,
        krsc.LayoutType,
        krsc.origin,
        krsc.Engine,
        filter_is_fcrs,
    ]
    ctx.enqueue_function[repack](
        w.as_imm(),
        krsc,
        grid_dim=ceildiv(n_filter, 256),
        block_dim=256,
    )
    launch_conv2d_staged_fprop[
        config,
        has_residual=has_residual,
        has_epilogue_fn=has_epilogue_fn,
        has_compute_fn=has_compute_fn,
    ](
        x,
        krsc,
        y,
        ctx,
        source=source,
        beta=beta,
        epilogue_fn=epilogue_fn,
        compute_fn=compute_fn,
    )
