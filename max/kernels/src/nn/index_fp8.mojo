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
from sys import size_of, simd_width_of
from math import ceildiv
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout.layout_tensor import (
    LayoutTensor,
    ThreadScope,
    copy_dram_to_sram,
    copy_dram_to_local,
    copy_local_to_shared,
)
from layout.tma_async import TMATensorTile, create_tensor_tile, SharedMemBarrier
from gpu import block_idx, thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.sync import barrier
from gpu.memory import external_memory
from nn.mha_operand import RaggedMHAOperand, MHAOperand
from nn.mha_fa3_utils import q_tma
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


struct IndexSmemStorage[
    dtype: DType,
    num_heads: Int,
    depth: Int,
    BN: Int,
]:
    var q_smem: InlineArray[Scalar[Self.dtype], Self.num_heads * Self.depth]
    var k_smem: InlineArray[Scalar[Self.dtype], Self.BN * Self.depth]
    var scratch: InlineArray[Scalar[DType.float32], Self.BN * 8]


fn fp8_index_kernel[
    dtype: DType,
    output_layout: Layout,
    q_layout: Layout,
    qs_layout: Layout,
    k_operand_type: MHAOperand,
    ks_operand_type: MHAOperand,
    block_tile_shape: InlineArray[Int, 2],
    valid_length_layout: Layout,
    num_heads: Int,
    depth: Int,
    # When False: num_keys = cache_length + seq_len (cache_length excludes new tokens).
    # When True: num_keys = cache_length (cache_length already includes new tokens).
    _is_cache_length_accurate: Bool = False,
](
    output: LayoutTensor[DType.float32, output_layout, MutAnyOrigin],
    # [total_seq_len, num_heads, depth]
    q: LayoutTensor[dtype, q_layout, MutAnyOrigin],
    # [total_seq_len, num_heads]
    q_s: LayoutTensor[DType.float32, qs_layout, MutAnyOrigin],
    # MHAOperand for K values
    k_operand: k_operand_type,
    # MHAOperand for K scales
    ks_operand: ks_operand_type,
    valid_length: LayoutTensor[
        DType.uint32, valid_length_layout, ImmutAnyOrigin
    ],
):
    comptime assert valid_length_layout.rank() == 1, "valid_length must be 1D"
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]

    comptime thread_dim_x = 16
    comptime thread_dim_y = 8

    comptime simd_width = simd_width_of[dtype]()

    var batch_idx = block_idx.x
    var seq_offset = block_idx.y
    var key_offset = block_idx.z * UInt(BM)
    var tid = thread_idx.x * 8 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var num_keys = k_operand.cache_length(Int(batch_idx))

    @parameter
    if not _is_cache_length_accurate:
        num_keys += Int(seq_len)

    if seq_offset >= UInt(seq_len) or key_offset >= UInt(num_keys):
        return

    ref smem_ptr = external_memory[
        Scalar[DType.uint8], address_space = AddressSpace.SHARED, alignment=128
    ]().bitcast[IndexSmemStorage[dtype, num_heads, depth, BN]]()[]

    ref q_smem = smem_ptr.q_smem
    ref k_smem = smem_ptr.k_smem
    ref scratch_smem = smem_ptr.scratch

    var q_smem_tile = LayoutTensor[
        dtype,
        Layout.row_major(num_heads, depth),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ](q_smem.unsafe_ptr())

    var k_smem_tile = LayoutTensor[
        dtype,
        Layout.row_major(BN, depth),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ](k_smem.unsafe_ptr())

    var k_smem_ptr = k_smem.unsafe_ptr()

    var q_ptr = q.ptr_at_offset(Index(start_of_seq + UInt32(seq_offset), 0, 0))
    var q_s_ptr = q_s.ptr_at_offset(Index(start_of_seq + UInt32(seq_offset), 0))
    var o_ptr = output.ptr_at_offset(
        Index(start_of_seq + UInt32(seq_offset), UInt32(key_offset))
    )

    comptime QTileType = LayoutTensor[
        dtype, Layout.row_major(num_heads, depth), MutAnyOrigin
    ]

    comptime QSTileType = LayoutTensor[
        DType.float32, Layout.row_major(1, num_heads), MutAnyOrigin
    ]

    comptime LogitsType = LayoutTensor[
        DType.float32,
        Layout.row_major(BN // thread_dim_x, num_heads // thread_dim_y),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    comptime QSRegTileType = LayoutTensor[
        DType.float32,
        Layout.row_major(1, num_heads // thread_dim_y),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    comptime LogitsSumType = LayoutTensor[
        DType.float32,
        Layout.row_major(BN // thread_dim_x, 1),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    comptime ScratchType = LayoutTensor[
        DType.float32,
        Layout.row_major(BN, thread_dim_y),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ]

    var q_tile = QTileType(q_ptr)
    var q_s_tile = QSTileType(q_s_ptr)
    var logits = LogitsType.stack_allocation()
    var q_s_reg_tile = QSRegTileType.stack_allocation()
    var logits_sum = LogitsSumType.stack_allocation()
    var scratch = ScratchType(scratch_smem.unsafe_ptr())

    var q_s_frag = q_s_tile.tile[1, num_heads // thread_dim_y](
        Int(thread_idx.x // thread_dim_x), Int(thread_idx.y)
    )

    comptime for q_frag_idx in range(num_heads // thread_dim_y):
        q_s_reg_tile[0, q_frag_idx] = q_s_frag[0, q_frag_idx][0]

    copy_dram_to_sram[
        thread_layout = Layout.row_major(16, 8),
        thread_scope = ThreadScope.BLOCK,
        block_dim_count=2,
    ](
        q_smem_tile.vectorize[1, simd_width](),
        q_tile.vectorize[1, simd_width](),
    )

    for i in range(BM // BN):
        var current_key_offset = Int(key_offset) + i * Int(BN)
        if current_key_offset >= num_keys:
            break

        # Load K tile via MHAOperand
        for k_row in range(BN):
            if current_key_offset + k_row >= num_keys:
                # Zero-fill OOB rows
                for d_idx in range(depth):
                    if Int(tid) == k_row % 128:
                        k_smem_ptr[k_row * depth + d_idx] = Scalar[dtype](0)
            else:
                var k_ptr = k_operand.block_paged_ptr[1](
                    UInt32(batch_idx),
                    UInt32(current_key_offset + k_row),
                    UInt32(0),  # head_idx = 0 for MLA (single head for K)
                    UInt32(0),
                )
                for d_idx in range(depth):
                    if Int(tid) == k_row % 128:
                        k_smem_ptr[k_row * depth + d_idx] = k_ptr[d_idx].cast[
                            dtype
                        ]()

        barrier()

        # Load K scales for current tile
        var k_s_reg: Float32 = 0.0
        if current_key_offset + Int(tid) < num_keys:
            var ks_ptr = ks_operand.block_paged_ptr[1](
                UInt32(batch_idx),
                UInt32(current_key_offset + Int(tid)),
                UInt32(0),
                UInt32(0),
            )
            k_s_reg = ks_ptr[0].cast[DType.float32]()

        q_smem_frag = q_smem_tile.tile[num_heads // thread_dim_y, depth](
            Int(thread_idx.y), 0
        )
        k_smem_frag = k_smem_tile.tile[BN // thread_dim_x, depth](
            Int(thread_idx.x), 0
        )

        _ = logits.fill(0)
        _ = logits_sum.fill(0)

        for k in range(depth):
            comptime for mma_m in range(BN // thread_dim_x):
                comptime for mma_n in range(num_heads // thread_dim_y):
                    logits[mma_m, mma_n] += (
                        k_smem_frag[mma_m, k][0] * q_smem_frag[mma_n, k][0]
                    ).cast[DType.float32]()

        comptime for l_i in range(BN // thread_dim_x):
            comptime for l_j in range(num_heads // thread_dim_y):
                logits[l_i, l_j] = (
                    max(logits[l_i, l_j], 0) * q_s_reg_tile[0, l_j][0]
                )

                logits_sum[l_i, 0] += logits[l_i, l_j]

            scratch[
                thread_idx.x * UInt(BN // thread_dim_x) + UInt(l_i),
                thread_idx.y,
            ] = logits_sum[l_i, 0]

        barrier()

        if current_key_offset + Int(tid) < num_keys:
            # Sum logits across heads
            var row_sum: Float32 = 0.0

            for col_idx in range(thread_dim_y):
                row_sum += scratch[tid, col_idx][0]

            o_ptr[Scalar[DType.uint](i * BN) + tid] = k_s_reg * row_sum


@always_inline
fn fp8_index[
    dtype: DType,
    output_layout: Layout,
    q_layout: Layout,
    qs_layout: Layout,
    k_layout: Layout,
    ks_layout: Layout,
    //,
    num_heads: Int,
    depth: Int,
](
    output: LayoutTensor[
        mut=True,
        DType.float32,
        output_layout,
        address_space = AddressSpace.GENERIC,
        ...,
    ],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, ...],
    q_s: LayoutTensor[
        DType.float32, qs_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[dtype, k_layout, address_space = AddressSpace.GENERIC, ...],
    k_s: LayoutTensor[
        DType.float32, ks_layout, address_space = AddressSpace.GENERIC, ...
    ],
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    # Create RaggedMHAOperand wrappers for K and K_s
    # These allow accessing ragged data via cache_row_offsets indirection
    var k_operand = RaggedMHAOperand(
        LayoutTensor[k.dtype, k.layout, ImmutAnyOrigin](
            k.ptr,
            RuntimeLayout[k.layout].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[
            cache_row_offsets.dtype,
            cache_row_offsets.layout,
            ImmutAnyOrigin,
        ](
            cache_row_offsets.ptr,
            RuntimeLayout[cache_row_offsets.layout].row_major(
                cache_row_offsets.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    # K_s is 1D [total_keys], reshape to 3D [total_keys, 1, 1] for MHAOperand interface
    comptime ks_3d_layout = Layout.row_major(UNKNOWN_VALUE, 1, 1)
    var ks_operand = RaggedMHAOperand(
        LayoutTensor[k_s.dtype, ks_3d_layout, ImmutAnyOrigin](
            k_s.ptr,
            RuntimeLayout[ks_3d_layout].row_major(
                Index(k_s.runtime_layout.shape.value[0], 1, 1)
            ),
        ),
        LayoutTensor[
            cache_row_offsets.dtype,
            cache_row_offsets.layout,
            ImmutAnyOrigin,
        ](
            cache_row_offsets.ptr,
            RuntimeLayout[cache_row_offsets.layout].row_major(
                cache_row_offsets.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    comptime block_tile_shape: InlineArray[Int, 2] = [512, 128]
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime smem_use = size_of[IndexSmemStorage[dtype, num_heads, depth, BN]]()
    comptime smem_available = ctx.default_device_info.shared_memory_per_multiprocessor - 1024

    # RaggedMHAOperand.cache_length() returns full key length directly,
    # so _is_cache_length_accurate=True skips adding seq_len in the kernel.
    comptime kernel = fp8_index_kernel[
        dtype,
        output_layout,
        q.layout,
        qs_layout,
        type_of(k_operand),
        type_of(ks_operand),
        block_tile_shape,
        type_of(valid_length).layout,
        num_heads,
        depth,
        _is_cache_length_accurate=True,
    ]

    ctx.enqueue_function[kernel, kernel](
        output,
        q,
        q_s,
        k_operand,
        ks_operand,
        valid_length,
        grid_dim=(
            batch_size,
            max_seq_len,
            ceildiv(max_num_keys, BM),
        ),
        block_dim=(16, 8, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_available)
        ),
    )


fn _index_matmul_max[
    dtype: DType,
    output_layout: Layout,
    q_layout: Layout,
    qs_layout: Layout,
    k_layout: Layout,
    k_type: MHAOperand,
](
    output: LayoutTensor[DType.float32, output_layout, MutAnyOrigin],
    q: LayoutTensor[dtype, q_layout, MutAnyOrigin],
    q_s: LayoutTensor[DType.float32, qs_layout, MutAnyOrigin],
    k: LayoutTensor[dtype, k_layout, MutAnyOrigin],
    valid_length: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    k_lut: k_type,
    max_seq_len: Int,
):
    comptime num_heads = q_layout.shape[1].value()
    comptime depth = q_layout.shape[2].value()

    var batch_idx = block_idx.z // UInt(num_heads)
    var head_idx = block_idx.z % UInt(num_heads)
    var seq_idx = block_idx.x * 16 + thread_idx.x
    var key_idx = block_idx.y * 16 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var num_keys = k_lut.cache_length(Int(batch_idx))
    var k_row_start = k_lut.row_idx(UInt32(Int(batch_idx)), 0)

    if key_idx >= UInt(num_keys) or seq_idx >= UInt(seq_len):
        return

    var q_ptr = q.ptr_at_offset(Index(start_of_seq, 0, 0))
    var k_ptr = k.ptr_at_offset(Index(k_row_start, 0, 0))
    var o_ptr = output.ptr_at_offset(Index(start_of_seq, 0, 0))

    var q_runtime_layout = RuntimeLayout[q_layout].row_major(
        Index(seq_len, num_heads, depth)
    )
    var q_batch = LayoutTensor[dtype, q_layout, MutAnyOrigin](
        q_ptr, q_runtime_layout
    )

    var k_runtime_layout = RuntimeLayout[k_layout].row_major(
        Index(num_keys, 1, depth)
    )
    var k_batch = LayoutTensor[dtype, k_layout, MutAnyOrigin](
        k_ptr, k_runtime_layout
    )

    var o_runtime_layout = RuntimeLayout[output_layout].row_major(
        Index(seq_len, num_keys, num_heads)
    )
    var o_batch = LayoutTensor[DType.float32, output_layout, MutAnyOrigin](
        o_ptr, o_runtime_layout
    )

    var accum = Float32(0.0)
    for d in range(Int(depth)):
        accum += (
            k_batch[key_idx, 0, d][0] * q_batch[seq_idx, head_idx, d][0]
        ).cast[DType.float32]()

    accum = max(accum, 0) * q_s[start_of_seq + UInt32(seq_idx), head_idx][0]
    o_batch[seq_idx, key_idx, head_idx] = accum


fn _reduce_logits[
    logits_layout: Layout,
    output_layout: Layout,
    ks_layout: Layout,
    k_type: MHAOperand,
](
    logits: LayoutTensor[DType.float32, logits_layout, MutAnyOrigin],
    output: LayoutTensor[DType.float32, output_layout, MutAnyOrigin],
    k_s: LayoutTensor[DType.float32, ks_layout, MutAnyOrigin],
    valid_length: LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    k_lut: k_type,
):
    comptime num_heads = logits_layout.shape[2].value()
    var batch_idx = block_idx.z
    var seq_idx = block_idx.x * 16 + thread_idx.x
    var key_idx = block_idx.y * 16 + thread_idx.y

    var start_of_seq = valid_length[batch_idx][0]
    var end_of_seq = valid_length[batch_idx + 1][0]
    var seq_len = end_of_seq - start_of_seq

    var num_keys = k_lut.cache_length(Int(batch_idx))
    var k_row_offset = k_lut.row_idx(UInt32(Int(batch_idx)), 0)

    if seq_idx >= UInt(seq_len) or key_idx >= UInt(num_keys):
        return

    var o_ptr = output.ptr_at_offset(Index(start_of_seq, 0))
    var logits_ptr = logits.ptr_at_offset(Index(start_of_seq, 0, 0))
    var k_s_ptr = k_s.ptr_at_offset(Index(k_row_offset))

    var o_runtime_layout = RuntimeLayout[output_layout].row_major(
        Index(seq_len, num_keys)
    )
    var o_batch = LayoutTensor[DType.float32, output_layout, MutAnyOrigin](
        o_ptr, o_runtime_layout
    )
    var k_s_runtime_layout = RuntimeLayout[ks_layout].row_major(Index(num_keys))

    var logits_runtime_layout = RuntimeLayout[logits_layout].row_major(
        Index(seq_len, num_keys, num_heads)
    )
    var logits_batch = LayoutTensor[DType.float32, logits_layout, MutAnyOrigin](
        logits_ptr, logits_runtime_layout
    )
    var k_s_batch = LayoutTensor[DType.float32, ks_layout, MutAnyOrigin](
        k_s_ptr, k_s_runtime_layout
    )

    var sum = Float32(0.0)
    for head in range(num_heads):
        sum += logits_batch[seq_idx, key_idx, head][0]

    o_batch[seq_idx, key_idx] = sum * k_s_batch[key_idx][0]


@always_inline
fn fp8_index_naive[
    dtype: DType,
    output_layout: Layout,
    q_layout: Layout,
    qs_layout: Layout,
    k_layout: Layout,
    ks_layout: Layout,
    //,
    num_heads: Int,
    depth: Int,
](
    output: LayoutTensor[
        mut=True,
        DType.float32,
        output_layout,
        address_space = AddressSpace.GENERIC,
        ...,
    ],
    q: LayoutTensor[dtype, q_layout, address_space = AddressSpace.GENERIC, ...],
    q_s: LayoutTensor[
        DType.float32, qs_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: LayoutTensor[dtype, k_layout, address_space = AddressSpace.GENERIC, ...],
    k_s: LayoutTensor[
        DType.float32, ks_layout, address_space = AddressSpace.GENERIC, ...
    ],
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    cache_row_offsets: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    var k_operand = RaggedMHAOperand(
        LayoutTensor[k.dtype, k.layout, MutAnyOrigin](
            k.ptr,
            RuntimeLayout[k.layout].row_major(
                k.runtime_layout.shape.value.canonicalize()
            ),
        ),
        LayoutTensor[
            cache_row_offsets.dtype,
            cache_row_offsets.layout,
            MutAnyOrigin,
        ](
            cache_row_offsets.ptr,
            RuntimeLayout[cache_row_offsets.layout].row_major(
                cache_row_offsets.runtime_layout.shape.value.canonicalize()
            ),
        ),
    )

    var logits_size = batch_size * max_seq_len * max_num_keys * num_heads

    var logits_dev = ctx.enqueue_create_buffer[DType.float32](logits_size)

    comptime logits_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads
    )

    var logits_runtime_layout = RuntimeLayout[logits_layout].row_major(
        Index(batch_size * max_seq_len, max_num_keys, num_heads)
    )

    var logits_tensor = LayoutTensor[
        DType.float32, logits_layout, MutAnyOrigin
    ](logits_dev.unsafe_ptr(), logits_runtime_layout)

    comptime mm = _index_matmul_max[
        dtype,
        logits_layout,
        q_layout,
        qs_layout,
        k.layout,
        type_of(k_operand),
    ]

    ctx.enqueue_function[mm, mm](
        logits_tensor,
        q,
        q_s,
        k,
        valid_length,
        k_operand,
        max_seq_len,
        grid_dim=(
            ceildiv(max_seq_len, 16),
            ceildiv(max_num_keys, 16),
            batch_size * num_heads,
        ),
        block_dim=(16, 16, 1),
    )

    comptime reduce_logits = _reduce_logits[
        logits_layout,
        output_layout,
        ks_layout,
        type_of(k_operand),
    ]

    ctx.enqueue_function[reduce_logits, reduce_logits](
        logits_tensor,
        output,
        k_s,
        valid_length,
        k_operand,
        grid_dim=(
            ceildiv(max_seq_len, 16),
            ceildiv(max_num_keys, 16),
            batch_size,
        ),
        block_dim=(16, 16, 1),
    )

    _ = logits_dev
