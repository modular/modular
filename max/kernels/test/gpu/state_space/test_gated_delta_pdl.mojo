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
"""PDL race test for the gated-delta kernels.

Each case runs one chain twice from the same inputs and demands bit-identical
results. The reference arm copies the inputs to the device ahead of time and
launches the kernels without PDL. The PDL arm poisons the device inputs and
publishes them through slow producers, each of which waits for the grid
before it, releases its dependent grid at once, sleeps, and only then writes.
A kernel that reads an operand ahead of its wait, or that releases a
dependent before its own wait has passed, reads poison and the arms differ.

Every operand a kernel reads is written by the producer launched immediately
before it. A wait only orders a kernel after the grid before it, so an operand
written any earlier would already be complete and a read hoisted above the
wait would go unnoticed. A slow stage also sits between the two kernels of
each chain (an activation after the conv1d, the accepted counts after the
verify), so each second kernel is released while its own input is still being
written.

Chains: conv1d -> recurrence (decode and ragged), and verify-ring -> state
fold. Grids run several waves deep so a dependent released at entry cannot
starve its predecessor's unscheduled blocks.
"""

from std.math import ceildiv
from std.random import rand, seed
from std.sys import size_of
from std.sys.intrinsics import llvm_intrinsic
from std.testing import TestSuite, assert_equal

from layout import TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from state_space.gated_delta import (
    gated_delta_recurrence_fwd_gpu,
    gated_delta_recurrence_verify_ring_gpu,
    gated_delta_ring_record_elements,
    gated_delta_state_fold_gpu,
)
from state_space.gated_delta_conv1d import gated_delta_conv1d_fwd_gpu

comptime WORK_DTYPE = DType.float32
comptime STATE_DTYPE = DType.bfloat16
comptime KEY_HEAD_DIM = 128
comptime VALUE_HEAD_DIM = 128
comptime KERNEL_SIZE = 4
comptime CONV1D_BLOCK_DIM = 128
comptime TOKENS_PER_BLOCK = 8
comptime RING_LEN = 4
comptime POISON = Float32(1.0e3)
comptime PRODUCER_SLEEP_NS = 1_000_000


# The most copies one producer carries.
comptime MAX_COPIES = 6


struct Copy(Copyable):
    """A byte range a producer publishes, as device addresses."""

    var dst: Int
    var src: Int
    var nbytes: Int

    def __init__[
        dtype: DType
    ](out self, dst: DeviceBuffer[dtype], src: DeviceBuffer[dtype]):
        self.dst = Int(dst.unsafe_ptr())
        self.src = Int(src.unsafe_ptr())
        self.nbytes = len(dst) * size_of[dtype]()


def delayed_publish[
    max_copies: Int
](table: MutPointer[UInt64, MutAnyOrigin], count: Int32):
    """Waits for the grid before it, releases its dependent, sleeps, then copies.

    `table` holds a (destination, source, byte count) triple per copy.
    """
    wait_on_dependent_grids()
    launch_dependent_grids()
    llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(PRODUCER_SLEEP_NS))
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for j in range(min(Int(count), max_copies)):
        var dst = MutPointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(table[3 * j])
        )
        var src = MutPointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(table[3 * j + 1])
        )
        for i in range(tid, Int(table[3 * j + 2]), stride):
            dst[i] = src[i]


def publish(ctx: DeviceContext, pdl_level: PDLLevel, *copies: Copy) raises:
    """Queues one slow producer that writes every copy."""
    debug_assert(len(copies) <= MAX_COPIES, "too many copies for a producer")
    var host = List[UInt64]()
    var nbytes = 0
    for j in range(len(copies)):
        host.append(UInt64(copies[j].dst))
        host.append(UInt64(copies[j].src))
        host.append(UInt64(copies[j].nbytes))
        nbytes = max(nbytes, copies[j].nbytes)
    var table = ctx.enqueue_create_buffer[DType.uint64](len(host))
    ctx.enqueue_copy(table, host)
    ctx.enqueue_function[delayed_publish[MAX_COPIES]](
        table,
        Int32(len(copies)),
        grid_dim=min(ceildiv(nbytes, 256), 16),
        block_dim=256,
        attributes=pdl_launch_attributes(pdl_level),
    )


struct Staged[dtype: DType](Movable):
    """One input: a staging copy on the device and the buffer a kernel reads."""

    var staging: DeviceBuffer[Self.dtype]
    var live: DeviceBuffer[Self.dtype]
    var length: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        host: List[Scalar[Self.dtype]],
        pdl: Bool,
    ) raises:
        self.length = len(host)
        self.staging = ctx.enqueue_create_buffer[Self.dtype](self.length)
        self.live = ctx.enqueue_create_buffer[Self.dtype](self.length)
        ctx.enqueue_copy(self.staging, host)
        if pdl:
            comptime if Self.dtype.is_floating_point():
                ctx.enqueue_memset[Self.dtype](
                    self.live, Scalar[Self.dtype](POISON)
                )
            else:
                ctx.enqueue_memset[Self.dtype](self.live, Scalar[Self.dtype](0))
        else:
            ctx.enqueue_copy(self.live, host)

    def copy(self) -> Copy:
        """The copy that publishes the staging buffer to `live`."""
        return Copy(self.live, self.staging)


struct Outputs(Movable):
    var out: List[Float32]
    var conv_out: List[Float32]
    var conv_state: List[Scalar[STATE_DTYPE]]
    var rec_state: List[Scalar[STATE_DTYPE]]
    var ring: List[Float32]

    def __init__(out self):
        self.out = List[Float32]()
        self.conv_out = List[Float32]()
        self.conv_state = List[Scalar[STATE_DTYPE]]()
        self.rec_state = List[Scalar[STATE_DTYPE]]()
        self.ring = List[Float32]()


def _random[dtype: DType](n: Int) -> List[Scalar[dtype]]:
    var data = List[Scalar[dtype]](unsafe_uninit_length=n)
    rand[dtype](data.unsafe_ptr(), n)
    return data^


def _permutation(n: Int, stride: Int) -> List[UInt32]:
    """A distinct slot per request; `stride` must be coprime to `n`."""
    var slots = List[UInt32](unsafe_uninit_length=n)
    for i in range(n):
        slots[i] = UInt32((i * stride + 1) % n)
    return slots^


def _offsets(lengths: List[Int]) -> List[UInt32]:
    var offsets = List[UInt32](unsafe_uninit_length=len(lengths) + 1)
    offsets[0] = 0
    for b in range(len(lengths)):
        offsets[b + 1] = offsets[b] + UInt32(lengths[b])
    return offsets^


def run_conv_recurrence_arm[
    pdl_level: PDLLevel
](ctx: DeviceContext, lengths: List[Int], nk: Int, nv: Int) raises -> Outputs:
    """Runs conv1d then the recurrence from a seeded input at one PDL level."""
    var batch_size = len(lengths)
    var offsets_h = _offsets(lengths)
    var total = Int(offsets_h[batch_size])
    var key_dim = nk * KEY_HEAD_DIM
    var value_dim = nv * VALUE_HEAD_DIM
    var conv_dim = 2 * key_dim + value_dim
    var pdl = pdl_level != PDLLevel.OFF

    var decay = Staged[WORK_DTYPE](ctx, _random[WORK_DTYPE](total * nv), pdl)
    var beta = Staged[WORK_DTYPE](ctx, _random[WORK_DTYPE](total * nv), pdl)
    var rec_slots = Staged[DType.uint32](ctx, _permutation(batch_size, 7), pdl)
    var rec_offsets = Staged[DType.uint32](ctx, offsets_h, pdl)
    var conv_offsets = Staged[DType.uint32](ctx, offsets_h, pdl)
    var conv_weight = Staged[WORK_DTYPE](
        ctx, _random[WORK_DTYPE](conv_dim * KERNEL_SIZE), pdl
    )
    var conv_slots = Staged[DType.uint32](ctx, _permutation(batch_size, 5), pdl)
    var qkv = Staged[WORK_DTYPE](
        ctx, _random[WORK_DTYPE](total * conv_dim), pdl
    )

    var conv_state_h = _random[STATE_DTYPE](
        batch_size * conv_dim * (KERNEL_SIZE - 1)
    )
    var rec_state_h = _random[STATE_DTYPE](
        batch_size * nv * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )
    var conv_state_dev = ctx.enqueue_create_buffer[STATE_DTYPE](
        len(conv_state_h)
    )
    var rec_state_dev = ctx.enqueue_create_buffer[STATE_DTYPE](len(rec_state_h))
    ctx.enqueue_copy(conv_state_dev, conv_state_h)
    ctx.enqueue_copy(rec_state_dev, rec_state_h)
    var conv_out_dev = ctx.enqueue_create_buffer[WORK_DTYPE](total * conv_dim)
    var out_dev = ctx.enqueue_create_buffer[WORK_DTYPE](total * value_dim)
    var act_dev = ctx.enqueue_create_buffer[WORK_DTYPE](total * conv_dim)
    ctx.enqueue_memset[WORK_DTYPE](conv_out_dev, 0)
    ctx.enqueue_memset[WORK_DTYPE](out_dev, 0)
    ctx.enqueue_memset[WORK_DTYPE](act_dev, Scalar[WORK_DTYPE](POISON))
    ctx.synchronize()

    comptime if pdl_level != PDLLevel.OFF:
        publish(
            ctx,
            pdl_level,
            qkv.copy(),
            conv_weight.copy(),
            conv_slots.copy(),
            conv_offsets.copy(),
        )

    var qkv_tt = TileTensor(qkv.live, row_major(total, conv_dim))
    var weight_tt = TileTensor(
        conv_weight.live, row_major(conv_dim, KERNEL_SIZE)
    )
    var conv_state_tt = TileTensor(
        conv_state_dev, row_major(batch_size, conv_dim, KERNEL_SIZE - 1)
    )
    var conv_slots_tt = TileTensor(conv_slots.live, row_major(batch_size))
    var conv_offsets_tt = TileTensor(
        conv_offsets.live, row_major(batch_size + 1)
    )
    var conv_out_tt = TileTensor(conv_out_dev, row_major(total, conv_dim))
    # Never written without CAPTURE_INPUT, but it must not alias the output.
    var window_dev = ctx.enqueue_create_buffer[WORK_DTYPE](total * conv_dim)
    var window_tt = TileTensor(window_dev, row_major(total, conv_dim))

    var tokens_per_block = min(
        TOKENS_PER_BLOCK, max(1, ceildiv(total, batch_size))
    )
    comptime conv_kernel = gated_delta_conv1d_fwd_gpu[
        WORK_DTYPE,
        STATE_DTYPE,
        KERNEL_SIZE,
        CONV1D_BLOCK_DIM,
        True,
        type_of(qkv_tt).LayoutType,
        type_of(weight_tt).LayoutType,
        type_of(conv_state_tt).LayoutType,
        type_of(conv_slots_tt).LayoutType,
        type_of(conv_offsets_tt).LayoutType,
        type_of(conv_out_tt).LayoutType,
        type_of(qkv_tt).Engine,
        pdl_level=pdl_level,
    ]
    ctx.enqueue_function[conv_kernel](
        Int32(batch_size),
        Int32(total),
        Int32(conv_dim),
        Int32(tokens_per_block),
        qkv_tt,
        weight_tt,
        conv_state_tt,
        conv_slots_tt,
        conv_offsets_tt,
        conv_out_tt,
        window_tt,
        grid_dim=(ceildiv(total, tokens_per_block), ceildiv(conv_dim, 128)),
        block_dim=(CONV1D_BLOCK_DIM,),
        attributes=pdl_launch_attributes(pdl_level),
    )

    # A slow stage between the two kernels, like the activation that sits
    # between them in the model, which also writes the rest of the
    # recurrence's operands.
    var act_copy = Copy(act_dev, conv_out_dev)
    comptime if pdl_level != PDLLevel.OFF:
        publish(
            ctx,
            pdl_level,
            act_copy,
            decay.copy(),
            beta.copy(),
            rec_slots.copy(),
            rec_offsets.copy(),
        )
    else:
        publish(ctx, pdl_level, act_copy)

    var act_tt = TileTensor(act_dev, row_major(total, conv_dim))
    var decay_tt = TileTensor(decay.live, row_major(total, nv))
    var beta_tt = TileTensor(beta.live, row_major(total, nv))
    var rec_slots_tt = TileTensor(rec_slots.live, row_major(batch_size))
    var rec_offsets_tt = TileTensor(rec_offsets.live, row_major(batch_size + 1))
    var rec_state_tt = TileTensor(
        rec_state_dev, row_major(batch_size, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM)
    )
    var out_tt = TileTensor(out_dev, row_major(total, value_dim))
    comptime rec_kernel = gated_delta_recurrence_fwd_gpu[
        WORK_DTYPE,
        STATE_DTYPE,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        type_of(out_tt).LayoutType,
        type_of(act_tt).LayoutType,
        type_of(decay_tt).LayoutType,
        type_of(beta_tt).LayoutType,
        type_of(rec_state_tt).LayoutType,
        type_of(rec_slots_tt).LayoutType,
        type_of(rec_offsets_tt).LayoutType,
        type_of(out_tt).Engine,
        pdl_level=pdl_level,
    ]
    ctx.enqueue_function[rec_kernel](
        Int32(batch_size),
        Int32(nv),
        Int32(nk),
        Int32(key_dim),
        out_tt,
        rec_state_tt,
        rec_slots_tt,
        act_tt,
        decay_tt,
        beta_tt,
        rec_offsets_tt,
        grid_dim=(batch_size * nv,),
        block_dim=(VALUE_HEAD_DIM,),
        attributes=pdl_launch_attributes(pdl_level),
    )

    var result = Outputs()
    result.out = List[Float32](unsafe_uninit_length=total * value_dim)
    result.conv_out = List[Float32](unsafe_uninit_length=total * conv_dim)
    result.conv_state = List[Scalar[STATE_DTYPE]](
        unsafe_uninit_length=len(conv_state_h)
    )
    result.rec_state = List[Scalar[STATE_DTYPE]](
        unsafe_uninit_length=len(rec_state_h)
    )
    ctx.enqueue_copy(result.out, out_dev)
    ctx.enqueue_copy(result.conv_out, conv_out_dev)
    ctx.enqueue_copy(result.conv_state, conv_state_dev)
    ctx.enqueue_copy(result.rec_state, rec_state_dev)
    ctx.synchronize()
    return result^


def run_verify_fold_arm[
    pdl_level: PDLLevel
](ctx: DeviceContext, lengths: List[Int], nk: Int, nv: Int) raises -> Outputs:
    """Runs verify-ring then the state fold from a seeded input at one PDL level.
    """
    var batch_size = len(lengths)
    var offsets_h = _offsets(lengths)
    var total = Int(offsets_h[batch_size])
    var key_dim = nk * KEY_HEAD_DIM
    var value_dim = nv * VALUE_HEAD_DIM
    var conv_dim = 2 * key_dim + value_dim
    var pdl = pdl_level != PDLLevel.OFF

    var accepted_h = List[UInt32](unsafe_uninit_length=batch_size)
    for b in range(batch_size):
        accepted_h[b] = UInt32(b % (lengths[b] + 1))

    var decay = Staged[WORK_DTYPE](ctx, _random[WORK_DTYPE](total * nv), pdl)
    var beta = Staged[WORK_DTYPE](ctx, _random[WORK_DTYPE](total * nv), pdl)
    var slots = Staged[DType.uint32](ctx, _permutation(batch_size, 7), pdl)
    var ring_slots = Staged[DType.uint32](ctx, _permutation(batch_size, 5), pdl)
    var offsets = Staged[DType.uint32](ctx, offsets_h, pdl)
    var accepted = Staged[DType.uint32](ctx, accepted_h, pdl)
    var fold_slots = Staged[DType.uint32](ctx, _permutation(batch_size, 7), pdl)
    var fold_ring_slots = Staged[DType.uint32](
        ctx, _permutation(batch_size, 5), pdl
    )
    var qkv = Staged[WORK_DTYPE](
        ctx, _random[WORK_DTYPE](total * conv_dim), pdl
    )

    var rec_state_h = _random[STATE_DTYPE](
        batch_size * nv * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )
    var rec_state_dev = ctx.enqueue_create_buffer[STATE_DTYPE](len(rec_state_h))
    ctx.enqueue_copy(rec_state_dev, rec_state_h)
    var record_stride = gated_delta_ring_record_elements[
        KEY_HEAD_DIM, VALUE_HEAD_DIM
    ](nv // nk)
    var ring_elems = batch_size * nk * RING_LEN * record_stride
    var ring_dev = ctx.enqueue_create_buffer[WORK_DTYPE](ring_elems)
    ctx.enqueue_memset[WORK_DTYPE](ring_dev, 0)
    var out_dev = ctx.enqueue_create_buffer[WORK_DTYPE](total * value_dim)
    ctx.enqueue_memset[WORK_DTYPE](out_dev, 0)
    ctx.synchronize()

    comptime if pdl_level != PDLLevel.OFF:
        publish(
            ctx,
            pdl_level,
            qkv.copy(),
            decay.copy(),
            beta.copy(),
            slots.copy(),
            ring_slots.copy(),
            offsets.copy(),
        )

    var qkv_tt = TileTensor(qkv.live, row_major(total, conv_dim))
    var decay_tt = TileTensor(decay.live, row_major(total, nv))
    var beta_tt = TileTensor(beta.live, row_major(total, nv))
    var slots_tt = TileTensor(slots.live, row_major(batch_size))
    var ring_slots_tt = TileTensor(ring_slots.live, row_major(batch_size))
    var offsets_tt = TileTensor(offsets.live, row_major(batch_size + 1))
    var accepted_tt = TileTensor(accepted.live, row_major(batch_size))
    var rec_state_tt = TileTensor(
        rec_state_dev, row_major(batch_size, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM)
    )
    var ring_tt = TileTensor(
        ring_dev, row_major(batch_size, nk, RING_LEN, record_stride)
    )
    var out_tt = TileTensor(out_dev, row_major(total, value_dim))
    comptime verify_kernel = gated_delta_recurrence_verify_ring_gpu[
        WORK_DTYPE,
        STATE_DTYPE,
        WORK_DTYPE,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        RING_LEN,
        type_of(out_tt).LayoutType,
        type_of(qkv_tt).LayoutType,
        type_of(decay_tt).LayoutType,
        type_of(beta_tt).LayoutType,
        type_of(rec_state_tt).LayoutType,
        type_of(slots_tt).LayoutType,
        type_of(offsets_tt).LayoutType,
        type_of(ring_tt).LayoutType,
        type_of(ring_slots_tt).LayoutType,
        type_of(out_tt).Engine,
        True,
        pdl_level=pdl_level,
    ]
    ctx.enqueue_function[verify_kernel](
        Int32(batch_size),
        Int32(nv),
        Int32(nk),
        Int32(key_dim),
        out_tt,
        rec_state_tt,
        slots_tt,
        qkv_tt,
        decay_tt,
        beta_tt,
        offsets_tt,
        ring_tt,
        ring_slots_tt,
        grid_dim=(batch_size * nv,),
        block_dim=(VALUE_HEAD_DIM,),
        attributes=pdl_launch_attributes(pdl_level),
    )

    # The accepted counts arrive after the verify, as they do in the model,
    # so the fold's own wait is the only thing keeping it off the poison.
    comptime if pdl_level != PDLLevel.OFF:
        publish(
            ctx,
            pdl_level,
            accepted.copy(),
            fold_slots.copy(),
            fold_ring_slots.copy(),
        )

    # The fold takes a `[num_layers, batch_size]` table; one layer here.
    var slots_2d = TileTensor(fold_slots.live, row_major(1, batch_size))
    var ring_slots_2d = TileTensor(
        fold_ring_slots.live, row_major(1, batch_size)
    )
    comptime fold_kernel = gated_delta_state_fold_gpu[
        STATE_DTYPE,
        WORK_DTYPE,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        RING_LEN,
        type_of(rec_state_tt).LayoutType,
        type_of(slots_2d).LayoutType,
        type_of(ring_tt).LayoutType,
        type_of(ring_slots_2d).LayoutType,
        type_of(accepted_tt).LayoutType,
        type_of(rec_state_tt).Engine,
        pdl_level=pdl_level,
    ]
    ctx.enqueue_function[fold_kernel](
        Int32(batch_size),
        Int32(1),
        Int32(nv),
        Int32(nk),
        rec_state_tt,
        slots_2d,
        ring_tt,
        ring_slots_2d,
        accepted_tt,
        grid_dim=(batch_size * nv,),
        block_dim=(VALUE_HEAD_DIM,),
        attributes=pdl_launch_attributes(pdl_level),
    )

    var result = Outputs()
    result.out = List[Float32](unsafe_uninit_length=total * value_dim)
    result.ring = List[Float32](unsafe_uninit_length=ring_elems)
    result.rec_state = List[Scalar[STATE_DTYPE]](
        unsafe_uninit_length=len(rec_state_h)
    )
    ctx.enqueue_copy(result.out, out_dev)
    ctx.enqueue_copy(result.ring, ring_dev)
    ctx.enqueue_copy(result.rec_state, rec_state_dev)
    ctx.synchronize()
    return result^


def _count_mismatches[
    dtype: DType
](a: List[Scalar[dtype]], b: List[Scalar[dtype]]) -> Int:
    var bad = 0
    if len(a) != len(b):
        return len(a) + len(b)
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            bad += 1
    return bad


def _check(label: String, want: Outputs, got: Outputs) raises:
    assert_equal(
        _count_mismatches(want.out, got.out), 0, msg=label + " readout"
    )
    assert_equal(
        _count_mismatches(want.conv_out, got.conv_out), 0, label + " conv out"
    )
    assert_equal(
        _count_mismatches(want.conv_state, got.conv_state),
        0,
        label + " conv state",
    )
    assert_equal(
        _count_mismatches(want.rec_state, got.rec_state),
        0,
        label + " recurrent state",
    )
    assert_equal(_count_mismatches(want.ring, got.ring), 0, msg=label + " ring")


def _check_conv_recurrence_level[
    level: PDLLevel
](
    ctx: DeviceContext, lengths: List[Int], nk: Int, nv: Int, want: Outputs
) raises:
    # The first launch of a kernel loads its module, which delays the host's
    # enqueue of the next launch past the producers' sleep and hides the race.
    # Only the later runs overlap the way the model's launches do.
    for _ in range(2):
        seed(1234)
        var got = run_conv_recurrence_arm[level](ctx, lengths, nk, nv)
        _check("PDL level " + String(level._level), want, got)


def _check_verify_fold_level[
    level: PDLLevel
](
    ctx: DeviceContext, lengths: List[Int], nk: Int, nv: Int, want: Outputs
) raises:
    for _ in range(2):
        seed(4321)
        var got = run_verify_fold_arm[level](ctx, lengths, nk, nv)
        _check("PDL level " + String(level._level), want, got)


def run_conv_recurrence_case(
    ctx: DeviceContext, lengths: List[Int], nk: Int, nv: Int
) raises:
    # `rand` is unseeded, so each arm must see identical inputs: reseed.
    seed(1234)
    var want = run_conv_recurrence_arm[PDLLevel.OFF](ctx, lengths, nk, nv)
    _check_conv_recurrence_level[PDLLevel.ON](ctx, lengths, nk, nv, want)
    _check_conv_recurrence_level[PDLLevel.OVERLAP_AT_BEGINNING](
        ctx, lengths, nk, nv, want
    )


def run_verify_fold_case(
    ctx: DeviceContext, lengths: List[Int], nk: Int, nv: Int
) raises:
    seed(4321)
    var want = run_verify_fold_arm[PDLLevel.OFF](ctx, lengths, nk, nv)
    _check_verify_fold_level[PDLLevel.ON](ctx, lengths, nk, nv, want)
    _check_verify_fold_level[PDLLevel.OVERLAP_AT_BEGINNING](
        ctx, lengths, nk, nv, want
    )


def _uniform(n: Int, length: Int) -> List[Int]:
    var lengths = List[Int]()
    for _ in range(n):
        lengths.append(length)
    return lengths^


def test_decode_chain_many_waves() raises:
    """64 decode requests x 24 heads is 1536 blocks, several waves deep."""
    var ctx = DeviceContext()
    run_conv_recurrence_case(ctx, _uniform(64, 1), 8, 24)


def test_ragged_chain_production_heads() raises:
    """A prefill-and-decode mix at the 48-head, 16-key-head production shape."""
    var ctx = DeviceContext()
    run_conv_recurrence_case(ctx, [1, 5, 1, 3], 16, 48)


def test_single_request_chain() raises:
    var ctx = DeviceContext()
    run_conv_recurrence_case(ctx, [1], 16, 48)


def test_verify_fold_chain_many_waves() raises:
    """48 requests with 1 to 4 token windows; the fold keeps a prefix of each.
    """
    var ctx = DeviceContext()
    var lengths = List[Int]()
    for b in range(48):
        lengths.append(1 + b % RING_LEN)
    run_verify_fold_case(ctx, lengths, 8, 24)


def test_verify_fold_chain_production_heads() raises:
    var ctx = DeviceContext()
    run_verify_fold_case(ctx, [4, 2, 1, 3], 16, 48)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
