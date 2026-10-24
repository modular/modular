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
"""Commits a speculative verify's accepted rows to every Mamba-2 layer at once.

A Nemotron-H verify runs each Mamba layer on its live state without writing
it back. Once acceptance is known, each request's state has to advance by its
accepted prefix of the verify's rows. That is two updates per layer:

  - the conv window becomes the last `width - 1` accepted conv inputs, the
    stored window filling in when fewer were accepted;
  - the SSM state folds in each accepted row, `state = state * exp(A * dt) +
    dt * B * x`, with `dt = softplus(dt + dt_bias)`.

Neither needs the other: the scan's `x`, `B` and `C` are the verify's conv
outputs, which were computed causally, so their accepted rows already saw the
accepted conv history. And neither produces an output. One launch does both
for every layer and request, reading each layer's rows where the verify left
them, so a batch-1 replay fills the GPU rather than running one small launch
per layer. The SSM update is the token-serial scan's arithmetic, expression for
expression, so the state it lands is bit-identical to replaying with
`mamba2_ssd_chunk_scan_varlen_fwd_inplace`.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, fma
from std.sys import size_of
from std.utils.static_tuple import StaticTuple
from layout import TensorLayout, TensorEngine, TileTensor
from state_space.mamba2_ssd_scan import LOG2E, _HeadScalars

comptime PAD_SLOT_ID: Int32 = -1
"""A conv-state row id that marks a padded request, which is skipped."""

comptime _BLOCK_THREADS = 128


@inline(.always)
def _layer_pointer[
    dtype: DType, N: Int
](
    pointers: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], N],
    layer: Int,
) -> ImmPointer[Scalar[dtype], ImmutAnyOrigin]:
    """Returns `pointers[layer]` without indexing the tuple at run time.

    A run-time index into a kernel-argument tuple copies the whole tuple to
    local memory in every thread, which kept L1 saturated: the fold ran at a
    third of the speed of its DRAM traffic.
    """
    var pointer = pointers[0]
    comptime for i in range(1, N):
        if layer == i:
            pointer = pointers[i]
    return pointer


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_BLOCK_THREADS))
)
def mamba2_replay_fold_gpu[
    dtype: DType,
    conv_dtype: DType,
    state_dtype: DType,
    DSTATE: Int,
    DSTATE_SPLIT: Int,
    CH_PER_THREAD: Int,
    CONV_WIDTH: Int,
    NUM_LAYERS: Int,
    conv_pool_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    conv_rows_LT: TensorLayout,
    ssm_rows_LT: TensorLayout,
    bounds_LT: TensorLayout,
    Engine: TensorEngine,
    dt_softplus: Bool = True,
](
    projected: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS
    ],
    convolved: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS
    ],
    A: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS],
    dt_bias: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS],
    projected_stride: Int32,
    convolved_stride: Int32,
    ngroups: Int32,
    conv_pool: TileTensor[
        conv_dtype, conv_pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    ssm_pool: TileTensor[
        state_dtype, ssm_pool_LT, MutUntrackedOrigin, Engine=Engine
    ],
    conv_rows: TileTensor[
        .uint32, conv_rows_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    ssm_rows: TileTensor[
        .uint32, ssm_rows_LT, ImmUntrackedOrigin, Engine=Engine
    ],
    replay_bounds: TileTensor[
        .int32, bounds_LT, ImmUntrackedOrigin, Engine=Engine
    ],
):
    """GPU kernel: folds each request's accepted rows into every layer's state.

    The grid is one linear `x` axis. It decomposes with the channel chunk
    fastest, then the column, then the layer and request, where
    `lb = x // (head_chunks * columns)` is request `lb % batch` of layer
    `lb // batch`. Grid `y` and `z` stop at 65535 blocks, which
    `batch * NUM_LAYERS` can pass. This order keeps the launch order of a
    three-axis `(head_chunks, columns, batch * NUM_LAYERS)` grid, whose
    locality the fold is tuned for. Columns below `nheads` fold that head of
    the SSM state, laid out as the token-serial scan's: `DSTATE_SPLIT`
    threads share a channel, each owning `DSTATE // DSTATE_SPLIT` contiguous
    lanes and `CH_PER_THREAD` channels. The columns above them each move 128
    channels of the conv window.

    Grid: `(batch * NUM_LAYERS * columns * head_chunks, 1, 1)` with
    `head_chunks = ceildiv(head_dim, CH)`, `CH = (128 // DSTATE_SPLIT) *
    CH_PER_THREAD` and `columns = nheads + ceildiv(ceildiv(conv_dim, 128),
    head_chunks)`, block: `(DSTATE_SPLIT, 128 // DSTATE_SPLIT, 1)`.

    Each request continues its live state, so no initial-state flag is read.

    Parameters:
        dtype: Element type of the activations, `A` and `dt_bias`.
        conv_dtype: Storage dtype of `conv_pool`.
        state_dtype: Storage dtype of `ssm_pool`; the fold accumulates in fp32.
        DSTATE: State dimension per head.
        DSTATE_SPLIT: Threads cooperating on one channel's state.
        CH_PER_THREAD: Channels each thread carries.
        CONV_WIDTH: Convolution width; the window holds `CONV_WIDTH - 1` rows.
        NUM_LAYERS: Number of Mamba layers.
        conv_pool_LT: Tensor layout of `conv_pool`.
        ssm_pool_LT: Tensor layout of `ssm_pool`.
        conv_rows_LT: Tensor layout of `conv_rows`.
        ssm_rows_LT: Tensor layout of `ssm_rows`.
        bounds_LT: Tensor layout of `replay_bounds`.
        Engine: Engine shared by all tile operands.
        dt_softplus: Whether to apply softplus to `dt + dt_bias`.

    Args:
        projected: Each layer's `in_proj` output rows, `(rows, intermediate +
            conv_dim + nheads)` row-major: the gate, the conv input and `dt`.
        convolved: Each layer's conv output rows, `(rows, conv_dim)` row-major:
            the scan's `x`, `B` and `C`. Each row's `B` is read as
            `DSTATE // DSTATE_SPLIT`-wide vectors, which the launcher checks
            are aligned.
        A: Each layer's per-head decay, `(nheads,)`, already `-exp(A_log)`.
        dt_bias: Each layer's per-head `dt` bias, `(nheads,)`.
        projected_stride: Row stride of every `projected` tensor.
        convolved_stride: Row stride of every `convolved` tensor.
        ngroups: Number of `B` groups.
        conv_pool: Conv windows, `(max_slots, conv_dim, CONV_WIDTH - 1)`.
        ssm_pool: SSM states, `(max_slots, nheads, head_dim, DSTATE)`.
        conv_rows: Each layer's conv-window row per request,
            `(NUM_LAYERS, batch)`.
        ssm_rows: Each layer's SSM-state row per request, `(NUM_LAYERS,
            batch)`.
        replay_bounds: Each request's accepted `(start, end)` rows, `(2 *
            batch,)`.
    """
    comptime L = DSTATE // DSTATE_SPLIT
    comptime P = CH_PER_THREAD
    comptime NY = _BLOCK_THREADS // DSTATE_SPLIT
    comptime CH = NY * P
    comptime W = CONV_WIDTH - 1
    comptime assert DSTATE_SPLIT * NY == _BLOCK_THREADS, "128-thread blocks"
    comptime assert L % 8 == 0 or L < 8, "B lanes load as one vector"

    var nheads = Int(ssm_pool.dim[1]())
    var head_dim = Int(ssm_pool.dim[2]())
    var conv_dim = Int(conv_pool.dim[1]())
    var intermediate = nheads * head_dim
    var batch = Int(replay_bounds.dim[0]()) // 2
    var head_chunks = ceildiv(head_dim, CH)
    var columns = nheads + ceildiv(
        ceildiv(conv_dim, _BLOCK_THREADS), head_chunks
    )
    var rest, chunk = divmod(block_idx.x, head_chunks)
    var lb, column = divmod(rest, columns)
    var layer, b = divmod(lb, batch)
    var start = Int(replay_bounds.load[width=1]((2 * b,)))
    var end = Int(replay_bounds.load[width=1]((2 * b + 1,)))
    if end <= start:
        return

    var tx = thread_idx.x
    var ty = thread_idx.y
    var proj = _layer_pointer(projected, layer)
    var conv = _layer_pointer(convolved, layer)
    var proj_stride = Int(projected_stride)
    var conv_stride = Int(convolved_stride)

    if column >= nheads:
        var d = (
            ((column - nheads) * head_chunks + chunk) * _BLOCK_THREADS
            + ty * DSTATE_SPLIT
            + tx
        )
        var slot = Int(Int32(conv_rows.load[width=1]((layer, b))))
        if d >= conv_dim or slot == Int(PAD_SLOT_ID):
            return
        var count = end - start
        # Read the whole window before writing any of it: a short prefix
        # keeps part of the stored window, shifted down.
        var window = Array[Scalar[conv_dtype], W](fill=0)
        comptime for s in range(W):
            var row = count - W + s
            if row >= 0:
                window[s] = (
                    proj[
                        unsafe_offset=(start + row) * proj_stride
                        + intermediate
                        + d
                    ]
                    .cast[.float32]()
                    .cast[conv_dtype]()
                )
            else:
                window[s] = (
                    conv_pool.load[width=1]((slot, d, W + row))
                    .cast[dtype]()
                    .cast[.float32]()
                    .cast[conv_dtype]()
                )
        comptime for s in range(W):
            conv_pool.store((slot, d, s), window[s])
        return

    var h = column
    var p0 = chunk * CH + ty
    var n_base = tx * L
    var heads_per_group = UInt32(nheads) // UInt32(ngroups)
    var group_id = Int(UInt32(h) // heads_per_group)
    var head = _HeadScalars(
        _layer_pointer(A, layer)[unsafe_offset=h].cast[.float32]() * LOG2E,
        _layer_pointer(dt_bias, layer)[unsafe_offset=h].cast[.float32](),
        0,
    )
    var slot = Int(ssm_rows.load[width=1]((layer, b)))
    var dt_col = intermediate + conv_dim + h
    var b_col = intermediate + group_id * DSTATE + n_base

    var state = Array[SIMD[.float32, L], P](fill=0)
    comptime for i in range(P):
        if p0 + i * NY < head_dim:
            state[i] = ssm_pool.load[width=L](
                (slot, h, p0 + i * NY, n_base)
            ).cast[.float32]()

    for t in range(start, end):
        var delta = head.delta[dt_softplus](
            proj[unsafe_offset=t * proj_stride + dt_col].cast[.float32]()
        )
        var dA = head.decay(delta)
        var B_vals = conv.unsafe_load[width=L, alignment=L * size_of[dtype]()](
            t * conv_stride + b_col
        ).cast[.float32]()
        comptime for i in range(P):
            var x_val = Float32(0)
            if p0 + i * NY < head_dim:
                x_val = conv[
                    unsafe_offset=t * conv_stride + h * head_dim + p0 + i * NY
                ].cast[.float32]()
            state[i] = fma(
                B_vals, SIMD[.float32, L](delta * x_val), state[i] * dA
            )

    comptime for i in range(P):
        if p0 + i * NY < head_dim:
            ssm_pool.store[width=L](
                (slot, h, p0 + i * NY, n_base), state[i].cast[state_dtype]()
            )


def mamba2_replay_fold_launch[
    dtype: DType,
    conv_dtype: DType,
    state_dtype: DType,
    NUM_LAYERS: Int,
    conv_pool_LT: TensorLayout,
    ssm_pool_LT: TensorLayout,
    conv_rows_LT: TensorLayout,
    ssm_rows_LT: TensorLayout,
    bounds_LT: TensorLayout,
    Engine: TensorEngine,
    //,
    dt_softplus: Bool = True,
](
    projected: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS
    ],
    convolved: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS
    ],
    A: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS],
    dt_bias: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], NUM_LAYERS],
    projected_stride: Int,
    convolved_stride: Int,
    ngroups: Int,
    conv_pool: TileTensor[conv_dtype, conv_pool_LT, _, Engine=Engine],
    ssm_pool: TileTensor[state_dtype, ssm_pool_LT, _, Engine=Engine],
    conv_rows: TileTensor[.uint32, conv_rows_LT, _, Engine=Engine],
    ssm_rows: TileTensor[.uint32, ssm_rows_LT, _, Engine=Engine],
    replay_bounds: TileTensor[.int32, bounds_LT, _, Engine=Engine],
    ctx: DeviceContext,
) raises:
    """Launches `mamba2_replay_fold_gpu` for every layer and request.

    Parameters:
        dtype: Element type of the activations, `A` and `dt_bias`.
        conv_dtype: Storage dtype of `conv_pool`.
        state_dtype: Storage dtype of `ssm_pool`.
        NUM_LAYERS: Number of Mamba layers.
        conv_pool_LT: Tensor layout of `conv_pool`.
        ssm_pool_LT: Tensor layout of `ssm_pool`.
        conv_rows_LT: Tensor layout of `conv_rows`.
        ssm_rows_LT: Tensor layout of `ssm_rows`.
        bounds_LT: Tensor layout of `replay_bounds`.
        Engine: Engine shared by all tile operands.
        dt_softplus: Whether to apply softplus to `dt + dt_bias`.

    Args:
        projected: See `mamba2_replay_fold_gpu`.
        convolved: See `mamba2_replay_fold_gpu`.
        A: See `mamba2_replay_fold_gpu`.
        dt_bias: See `mamba2_replay_fold_gpu`.
        projected_stride: See `mamba2_replay_fold_gpu`.
        convolved_stride: See `mamba2_replay_fold_gpu`.
        ngroups: See `mamba2_replay_fold_gpu`.
        conv_pool: See `mamba2_replay_fold_gpu`.
        ssm_pool: See `mamba2_replay_fold_gpu`.
        conv_rows: See `mamba2_replay_fold_gpu`.
        ssm_rows: See `mamba2_replay_fold_gpu`.
        replay_bounds: See `mamba2_replay_fold_gpu`.
        ctx: The device context to launch on.
    """
    var batch = Int(replay_bounds.dim[0]()) // 2
    if batch == 0:
        return
    var nheads = Int(ssm_pool.dim[1]())
    var head_dim = Int(ssm_pool.dim[2]())
    var dstate = Int(ssm_pool.dim[3]())
    var conv_dim = Int(conv_pool.dim[1]())
    var width = Int(conv_pool.dim[2]()) + 1

    @inline(.always)
    def launch[DSTATE: Int, CONV_WIDTH: Int]() raises {imm}:
        # The token-serial scan's tile, which the fold reproduces exactly.
        comptime WIDE = DSTATE == 128 and ctx.target.is_nvidia_gpu()
        comptime DSTATE_SPLIT = 16 if WIDE else 8
        comptime CH_PER_THREAD = 4 if WIDE else 1
        comptime ROWS = _BLOCK_THREADS // DSTATE_SPLIT
        comptime CH = ROWS * CH_PER_THREAD
        comptime kernel = mamba2_replay_fold_gpu[
            dtype,
            conv_dtype,
            state_dtype,
            DSTATE,
            DSTATE_SPLIT,
            CH_PER_THREAD,
            CONV_WIDTH,
            NUM_LAYERS,
            conv_pool_LT,
            ssm_pool_LT,
            conv_rows_LT,
            ssm_rows_LT,
            bounds_LT,
            Engine,
            dt_softplus,
        ]
        # `B` is read a thread's lanes at a time; prove those vectors aligned.
        comptime B_ALIGN = DSTATE // DSTATE_SPLIT * size_of[dtype]()
        var b_offset = nheads * head_dim * size_of[dtype]()
        if (convolved_stride * size_of[dtype]()) % B_ALIGN != 0 or (
            b_offset % B_ALIGN != 0
        ):
            raise Error("the B rows of `convolved` are not vector aligned")
        comptime for i in range(NUM_LAYERS):
            if Int(convolved[i]) % B_ALIGN != 0:
                raise Error("a `convolved` tensor is not vector aligned")
        var head_chunks = ceildiv(head_dim, CH)
        var conv_blocks = ceildiv(
            ceildiv(conv_dim, _BLOCK_THREADS), head_chunks
        )
        ctx.enqueue_function[kernel](
            projected,
            convolved,
            A,
            dt_bias,
            Int32(projected_stride),
            Int32(convolved_stride),
            Int32(ngroups),
            conv_pool,
            ssm_pool,
            conv_rows,
            ssm_rows,
            replay_bounds,
            grid_dim=(
                batch * NUM_LAYERS * (nheads + conv_blocks) * head_chunks,
                1,
                1,
            ),
            block_dim=(DSTATE_SPLIT, ROWS, 1),
        )

    @inline(.always)
    def with_width[DSTATE: Int]() raises {imm}:
        match width:
            case 2:
                launch[DSTATE, 2]()
            case 3:
                launch[DSTATE, 3]()
            case 4:
                launch[DSTATE, 4]()
            case _:
                raise Error(
                    "Unsupported conv width: "
                    + String(width)
                    + ". Expected 2, 3 or 4."
                )

    match dstate:
        case 16:
            with_width[16]()
        case 64:
            with_width[64]()
        case 128:
            with_width[128]()
        case 256:
            with_width[256]()
        case _:
            raise Error(
                "Unsupported dstate: "
                + String(dstate)
                + ". Expected 16, 64, 128, or 256."
            )
