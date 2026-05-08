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

from std.math import align_up, ceildiv
from std.math.uutils import uceildiv, udivmod, ufloordiv, umod
from std.os import abort
from std.atomic import Atomic, Ordering
from std.sys import is_amd_gpu, is_nvidia_gpu
from std.sys.info import CompilationTarget, align_of, simd_width_of, size_of
from std.ffi import c_size_t

from linalg.fp4_utils import (
    MXFP4_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    cast_fp32_to_fp4e2m1,
    cast_float_to_fp4e2m1_amd,
    compute_mxfp4_even_scale,
    set_scale_factor,
)

import std.gpu.primitives.warp as warp
from std.collections import OptionalReg
from std.gpu import (
    PDL,
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    thread_idx,
    block_idx,
    lane_id,
    warp_id,
)
from std.gpu.host import get_gpu_target, DeviceBuffer, DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
    cp_async_bulk_global_shared_cta,
    cp_async_bulk_shared_cluster_global,
)
from std.gpu.primitives import elect_one_sync
from std.gpu.sync import (
    syncwarp,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout.tile_tensor import _get_index_type
from layout.tile_layout import Layout
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    _default_desc_shape,
)
from std.math import exp, recip
from std.memory import stack_allocation
from std.memory.unsafe import bitcast
from shmem import SHMEM_SIGNAL_SET, SHMEMScope, shmem_put_nbi, shmem_signal_op

from std.utils.index import Index, IndexList, StaticTuple
from std.utils.numerics import get_accum_type

from std.builtin.device_passable import DevicePassable

comptime elementwise_epilogue_type = def[
    dtype: DType, width: SIMDSize, *, alignment: Int = 1
](IndexList[2], SIMD[dtype, width]) capturing -> None

comptime router_weights_wrapper_type = def[width: Int](
    token_idx: Int, topk_id: Int
) capturing -> SIMD[DType.float32, width]


comptime input_scales_wrapper_type = def[dtype: DType](
    Int,
) capturing -> Scalar[dtype]

comptime EP_DATA_READY_FLAG = 1 << 10

# Maximum number of GPUs per node for P2P signaling.
# Used to track per-rank expert completion.
comptime MAX_GPUS_PER_NODE = 8


@always_inline
def _BLOCK_SCOPE() -> StaticString:
    comptime if is_nvidia_gpu():
        return "block"
    elif is_amd_gpu():
        return "workgroup"
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@always_inline
def _DEVICE_SCOPE() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


comptime DEVICE_SCOPE = _DEVICE_SCOPE()
comptime BLOCK_SCOPE = _BLOCK_SCOPE()

comptime _counter_atomic = Atomic[DType.int32, scope=DEVICE_SCOPE]
comptime _signal_atomic = Atomic[DType.uint64]


@always_inline
def block_memcpy[
    dst_addr_space: AddressSpace,
    src_addr_space: AddressSpace,
    //,
    num_bytes: Int,
    block_size: Int,
](
    dst_p: UnsafePointer[mut=True, UInt8, _, address_space=dst_addr_space],
    src_p: UnsafePointer[mut=False, UInt8, _, address_space=src_addr_space],
    thread_idx: Int,
) -> None:
    """
    Copies a memory area from source to destination. This function will use the
    vectorized store and load instructions to copy the memory area. User should
    make sure pointers are aligned to the simd width.
    """
    comptime simd_width = simd_width_of[DType.uint8]()
    for i in range(thread_idx, num_bytes // simd_width, block_size):
        dst_p.store[alignment=simd_width](
            i * simd_width,
            src_p.load[
                width=simd_width,
                alignment=simd_width,
                invariant=True,
            ](i * simd_width),
        )


@always_inline
def block_prefix_sum[
    dtype: DType, //, num_elements: Int
](_val: Scalar[dtype]) -> Scalar[dtype]:
    """
    Performs a prefix sum (scan) operation across all threads in a block.
    """
    comptime n_elements_aligned = align_up(num_elements, WARP_SIZE)
    comptime n_warps = n_elements_aligned // WARP_SIZE
    comptime assert (
        n_warps <= WARP_SIZE
    ), "Number of warps must be less than or equal to warp size"

    var warp_prefix_sum = stack_allocation[
        n_warps,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()

    var val = Scalar[dtype](0)
    if thread_idx.x < num_elements:
        val = _val

    if thread_idx.x < n_elements_aligned:
        val = warp.prefix_sum(val)
        if lane_id() == WARP_SIZE - 1:
            warp_prefix_sum[warp_id()] = val
    barrier()

    if warp_id() == 0:
        var warp_sum = Scalar[dtype](0)
        if lane_id() < n_warps:
            warp_sum = warp_prefix_sum[lane_id()]
        warp_sum = warp.prefix_sum[exclusive=True](warp_sum)
        if lane_id() < n_warps:
            warp_prefix_sum[lane_id()] = warp_sum
    barrier()

    if thread_idx.x < num_elements:
        val += warp_prefix_sum[warp_id()]
    barrier()

    return val


@always_inline
@parameter
def ep_signal_completion[
    p2p_world_size: Int,
    //,
    use_shmem: Bool,
    n_experts_per_device: Int = 0,
    skip_a2a: Bool = False,
](
    my_rank: Int32,
    dst_rank: Int32,
    recv_count_ptrs: InlineArray[
        UnsafePointer[UInt64, MutExternalOrigin], p2p_world_size
    ],
    signal_offset: Int32,
    signal: UInt64,
    rank_completion_counter: UnsafePointer[Int32, MutExternalOrigin],
) -> None:
    """
    Signals the completion of the communication by writing to the receive count
    buffer. Will use direct memory access if the target device is on the same
    node, and use the SHMEM API if the target device is on a different node.

    For same-node signaling, uses normal stores and only issues a release store
    when the last expert for a destination rank is completed. This reduces the
    number of release stores from n_experts to p2p_world_size.
    """

    var my_p2p_world, my_p2p_rank = udivmod(Int(my_rank), p2p_world_size)
    var dst_p2p_world, dst_p2p_rank = udivmod(Int(dst_rank), p2p_world_size)

    comptime scope = DEVICE_SCOPE if skip_a2a else ""

    # If the target device is on the same node, we can directly write to its
    # receive count buffer.
    if my_p2p_world == dst_p2p_world:
        var dst_p2p_ptr = recv_count_ptrs[dst_p2p_rank] + signal_offset
        var old_count = _counter_atomic.fetch_add[ordering=Ordering.RELAXED](
            rank_completion_counter + Int(dst_p2p_rank), 1
        )

        # If this is the last expert for this destination rank,
        # use a release store to flush all pending stores.
        if old_count < Int32(n_experts_per_device - 1):
            comptime if is_nvidia_gpu():
                dst_p2p_ptr[] = signal
            else:
                # TODO(KERN-2792): Investigate why AMD GPUs require this to be
                # store-release instead of a normal store as done above. Without
                # store-release, the kernel can hang spinning on stale data.
                Atomic[scope=scope].store[ordering=Ordering.RELEASE](
                    dst_p2p_ptr, signal
                )
        else:
            # Technically, this release store only guarantees the arrival of
            # all experts' messages to the target device. It doesn't guarantee
            # the arrival of the previous experts' signals to the target
            # device. However, this does not matter as we will check the
            # arrival signal individually in the dispatch_wait/combine_wait kernel.
            Atomic[scope=scope].store[ordering=Ordering.RELEASE](
                dst_p2p_ptr, signal
            )
            # Reset counter for next kernel invocation.
            rank_completion_counter[dst_p2p_rank] = 0
    else:
        comptime if use_shmem:
            # This signal operation is sent using the same RC as the one used
            # for token transfer. Since RC guarantees the message is delivered
            # in order, the remote device can confirm all the tokens for the
            # expert has been received once the signal operation is received.
            shmem_signal_op(
                recv_count_ptrs[my_p2p_rank] + signal_offset,
                signal,
                SHMEM_SIGNAL_SET,
                dst_rank,
            )


@always_inline
def get_device_alignment() -> Int:
    comptime gpu_target = get_gpu_target()
    comptime gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
    comptime gpu_alignment = align_of[
        SIMD[DType.uint8, gpu_simd_width], target=gpu_target
    ]()

    return gpu_alignment


trait TokenFormat(DevicePassable, ImplicitlyDestructible):
    comptime hid_dim: Int
    comptime top_k: Int
    comptime alignment: Int

    # We process received tokens in tiles. This tuple specifies number of tokens
    # in a tile, and number of blocks needed to process the tile.
    comptime dispatch_wait_tile_shape: Tuple[Int, Int]

    # The size of the dynamic shared memory resources needed for the dispatch
    # kernel.
    comptime dispatch_smem_size: Int

    @always_inline
    @staticmethod
    def token_size() -> Int:
        "Returns the size of the (quantized) token in bytes."
        ...

    @always_inline
    @staticmethod
    def src_info_size() -> Int:
        "Returns the size of the source info in bytes. Currently, source info is a single int32 that stores a token's index in the original rank."
        return align_up(size_of[Int32](), Self.alignment)

    @always_inline
    @staticmethod
    def topk_info_size() -> Int:
        "Returns the size of the top-k info in bytes. Currently, top-k info is an array of uint16 that stores a token's top-k expert IDs."
        return align_up(size_of[UInt16]() * Self.top_k, Self.alignment)

    @always_inline
    @staticmethod
    def msg_size() -> Int:
        "Returns the size of the message in bytes."
        return Self.token_size() + Self.src_info_size() + Self.topk_info_size()

    @always_inline
    @staticmethod
    def src_info_offset() -> Int:
        "Returns the offset of the source info in the message."
        return Self.token_size()

    @always_inline
    @staticmethod
    def topk_info_offset() -> Int:
        "Returns the offset of the top-k info in the message."
        return Self.token_size() + Self.src_info_size()

    @always_inline
    def pad_expert_offsets[
        n_groups: Int
    ](self, row_offsets: UnsafePointer[mut=True, UInt32, ...]) -> None:
        """
        Pad the offsets to satisfy the grouped matmul alignment requirement.
        """
        pass

    @always_inline
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, _, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        "Copy the token to the send buffer. This function needs to be called by all threads in the block."
        ...

    @always_inline
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, _, address_space=buf_addr_space],
        token_index: Int,
    ) -> None:
        "Copy the message to the output tensor. This function needs to be called by all threads in a warp."
        ...

    @always_inline
    def init_smem_resources(self) -> None:
        "Initialize the shared memory resources for the token format."
        pass

    @always_inline
    def copy_msg_tile_to_output_tensor[
        extract_topk_info_func: def(
            UnsafePointer[UInt8, MutExternalOrigin], Int
        ) -> None,
        recv_buf_ptr_func: def(Int) -> UnsafePointer[UInt8, MutExternalOrigin],
        //,
        n_warps: Int,
        shared_expert_offset: Int = 0,
    ](
        self,
        expert_id: Int,
        expert_start_pos: Int,
        tile_id: Int,
        tile_end: Int,
        extract_topk_info_functor: extract_topk_info_func,
        recv_buf_ptr_functor: recv_buf_ptr_func,
    ) -> None:
        "Copy a tile of tokens from the receive buffer to the output tensor."

        # Let each warp process a single token in the tile at a time.
        comptime tile_size = Self.dispatch_wait_tile_shape[0]
        comptime n_k_tiles = Self.dispatch_wait_tile_shape[1]
        var tile_start = ufloordiv(tile_id, n_k_tiles) * tile_size
        var w = warp_id()
        for tok_id_in_tile in range(w, tile_end - tile_start, n_warps):
            var msg_ptr = recv_buf_ptr_functor(tok_id_in_tile)
            var output_pos = expert_start_pos + tile_start + tok_id_in_tile
            self.copy_msg_to_output_tensor(msg_ptr, output_pos)

            if umod(tile_id, n_k_tiles) == 0:
                extract_topk_info_functor(msg_ptr, output_pos)


struct BF16TokenFormat[
    output_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](TokenFormat, TrivialRegisterPassable):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()

    comptime dispatch_wait_tile_shape = (128, 1)
    comptime dispatch_smem_size = 0

    comptime TensorType = TileTensor[
        DType.bfloat16, Self.output_layout, MutExternalOrigin
    ]
    var output_tokens: Self.TensorType

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return String(
            "BF16TokenFormat[hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @always_inline
    def __init__(
        out self,
        output_tokens: TileTensor[DType.bfloat16, Self.output_layout, ...],
    ):
        self.output_tokens = {
            UnsafePointer[BFloat16, MutExternalOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }

    @always_inline
    @staticmethod
    def token_size() -> Int:
        return align_up(
            Self.hid_dim * size_of[DType.bfloat16](), Self.alignment
        )

    @always_inline
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, _, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        block_memcpy[Self.hid_dim * size_of[BFloat16](), block_size](
            buf_p,
            src_p.bitcast[UInt8](),
            thread_idx.x,
        )

    @always_inline
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, _, address_space=buf_addr_space],
        token_index: Int,
    ) -> None:
        comptime assert (
            Self.TensorType.flat_rank >= 2
        ), "output_tokens expects rank >= 2"
        comptime bf16_width = simd_width_of[DType.bfloat16]()
        comptime byte_width = bf16_width * size_of[BFloat16]()
        for i in range(lane_id(), Self.hid_dim // bf16_width, WARP_SIZE):
            self.output_tokens.store(
                (Idx(token_index), Idx(i * bf16_width)),
                bitcast[DType.bfloat16, bf16_width](
                    buf_p.load[
                        width=byte_width,
                        invariant=True,
                        alignment=Self.alignment,
                    ](
                        i * byte_width,
                    )
                ),
            )


struct BlockwiseFP8TokenFormat[
    fp8_dtype: DType,
    scales_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](TokenFormat, TrivialRegisterPassable):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime expert_m_padding = 16 // size_of[Self.scales_dtype]()

    comptime dispatch_wait_tile_shape = (128, 1)
    comptime dispatch_smem_size = 0

    comptime TensorType = TileTensor[
        Self.fp8_dtype, Self.output_layout, MutExternalOrigin
    ]
    comptime ScalesTensorType = TileTensor[
        Self.scales_dtype, Self.scales_layout, MutExternalOrigin
    ]
    var output_tokens: Self.TensorType
    var output_scales: Self.ScalesTensorType

    comptime group_size = 128

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return String(
            "BlockwiseFP8TokenFormat[fp8_dtype = ",
            String(Self.fp8_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @always_inline
    def __init__(
        out self,
        output_tokens: TileTensor[Self.fp8_dtype, Self.output_layout, ...],
        output_scales: TileTensor[Self.scales_dtype, Self.scales_layout, ...],
    ):
        self.output_tokens = {
            UnsafePointer[Scalar[Self.fp8_dtype], MutExternalOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }
        self.output_scales = {
            UnsafePointer[Scalar[Self.scales_dtype], MutExternalOrigin](
                unsafe_from_address=Int(output_scales.ptr)
            ),
            output_scales.layout,
        }

    @always_inline
    @staticmethod
    def fp8_quant_size() -> Int:
        return align_up(
            Self.hid_dim * size_of[Self.fp8_dtype](), Self.alignment
        )

    @always_inline
    @staticmethod
    def scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by 128"
        return align_up(
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype](),
            Self.alignment,
        )

    @always_inline
    @staticmethod
    def token_size() -> Int:
        return Self.fp8_quant_size() + Self.scales_size()

    @always_inline
    @staticmethod
    def scales_offset() -> Int:
        return Self.fp8_quant_size()

    @always_inline
    def pad_expert_offsets[
        n_groups: Int
    ](self, row_offsets: UnsafePointer[mut=True, UInt32, ...]) -> None:
        """
        The mojo blockwise FP8 grouped matmul requires each group's m to be
        aligned to the expert_m_padding. This function updates the row_offsets
        tensor to satisfy this requirement.

        For example, if the expert_m_padding is 4, and the row_offsets tensor
        is [0, 10, 20, 30, 40], the function will update the row_offsets tensor
        to [0, 12, 24, 36, 48].
        """
        var tid = thread_idx.x
        var per_expert_m = UInt32(0)
        if tid < n_groups:
            per_expert_m = row_offsets[tid + 1] - row_offsets[tid]
            per_expert_m = UInt32(
                align_up(Int(per_expert_m), Self.expert_m_padding)
            )

        var aligned_exp_end = block_prefix_sum[n_groups](per_expert_m)

        if tid < n_groups:
            row_offsets[tid + 1] = aligned_exp_end

    @always_inline
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, _, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        comptime src_width = simd_width_of[src_type]()
        comptime byte_width = src_width * size_of[Self.fp8_dtype]()

        comptime fp8_max = Scalar[Self.fp8_dtype].MAX_FINITE
        comptime fp8_max_t = Scalar[Self.fp8_dtype].MAX_FINITE.cast[
            Self.scales_dtype
        ]()

        comptime n_threads_per_group = Self.group_size // src_width
        comptime assert (
            WARP_SIZE % n_threads_per_group == 0
        ), "Each warp must process a multiple of quantization groups"

        for i in range(thread_idx.x, Self.hid_dim // src_width, block_size):
            var loaded_vec = src_p.load[
                width=src_width, alignment=Self.alignment, invariant=True
            ](i * src_width).cast[Self.scales_dtype]()
            var thread_max = abs(loaded_vec).reduce_max()
            var group_max = warp.lane_group_max[n_threads_per_group](thread_max)

            # 1e-4 is taken from DeepEP.
            var scale_factor = max(group_max, 1e-4) / fp8_max_t
            var output_vec = loaded_vec / scale_factor
            output_vec = output_vec.clamp(-fp8_max_t, fp8_max_t)

            buf_p.store[alignment=byte_width](
                i * byte_width,
                bitcast[DType.uint8, byte_width](
                    output_vec.cast[Self.fp8_dtype]()
                ),
            )

            # The first thread in each group stores the scale factor.
            comptime scale_bytes = size_of[Self.scales_dtype]()
            if umod(lane_id(), n_threads_per_group) == 0:
                scale_idx = ufloordiv(i * src_width, Self.group_size)
                buf_p.store[alignment=scale_bytes](
                    Self.scales_offset() + scale_idx * scale_bytes,
                    bitcast[DType.uint8, scale_bytes](scale_factor),
                )

    @always_inline
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, _, address_space=buf_addr_space],
        token_index: Int,
    ) -> None:
        comptime assert (
            Self.TensorType.flat_rank >= 2
        ), "output_tokens expects rank >= 2"
        # First we copy the FP8 quants.
        comptime fp8_width = simd_width_of[Self.fp8_dtype]()
        for i in range(lane_id(), Self.hid_dim // fp8_width, WARP_SIZE):
            self.output_tokens.store(
                (Idx(token_index), Idx(i * fp8_width)),
                bitcast[Self.fp8_dtype, fp8_width](
                    buf_p.load[
                        width=fp8_width,
                        invariant=True,
                        alignment=Self.alignment,
                    ](
                        i * fp8_width,
                    )
                ),
            )

        comptime assert (
            Self.ScalesTensorType.flat_rank >= 2
        ), "output_scales expects rank >= 2"
        # Unlike the output tensor, the scales tensor is stored in a transposed way.
        comptime scale_bytes = size_of[Self.scales_dtype]()
        for i in range(lane_id(), Self.hid_dim // Self.group_size, WARP_SIZE):
            self.output_scales.store(
                (Idx(i), Idx(token_index)),
                bitcast[Self.scales_dtype, 1](
                    buf_p.load[
                        width=scale_bytes,
                        invariant=True,
                        alignment=scale_bytes,
                    ](
                        Self.scales_offset() + i * scale_bytes,
                    )
                ),
            )


@align(64)
struct NVFP4TokenFormat[
    fp4_dtype: DType,
    scales_dtype: DType,
    output_layout: TensorLayout,
    scales_offset_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](ImplicitlyCopyable, TokenFormat):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime group_size = NVFP4_SF_VECTOR_SIZE

    comptime dispatch_wait_tile_shape = (128, 2)

    comptime TensorType = TileTensor[
        Self.fp4_dtype, Self.output_layout, MutExternalOrigin
    ]
    comptime ScalesOffsetTensorType = TileTensor[
        DType.uint32, Self.scales_offset_layout, MutExternalOrigin
    ]

    comptime _n_k_tiles = Self.dispatch_wait_tile_shape[1]
    comptime _n_warps = 32  # Always use 32 warps per block on Nvidia GPUs.
    comptime tma_tile_shape = Index(
        1,
        Self._hid_dim // NVFP4_SF_VECTOR_SIZE // SF_ATOM_K // Self._n_k_tiles,
        1,
        SF_ATOM_K * SF_ATOM_M[1],
    )
    comptime _scales_smem_per_warp = align_up(
        Int(Coord(Self.tma_tile_shape).product()), 128
    ) * size_of[Self.scales_dtype]()
    comptime _fp4_smem_per_warp = align_up(
        Self._hid_dim // 2 // Self._n_k_tiles, 16
    )
    comptime _mbar_smem_offset = Self._n_warps * (
        Self._scales_smem_per_warp + Self._fp4_smem_per_warp
    )
    comptime _mbar_smem_size = align_up(
        Self._n_warps * size_of[SharedMemBarrier](), 8
    )
    comptime dispatch_smem_size = Self._mbar_smem_offset + Self._mbar_smem_size

    comptime ScalesTMATensorTileType = TMATensorTile[
        Self.scales_dtype,
        4,
        Self.tma_tile_shape,
        _default_desc_shape[
            4,
            Self.scales_dtype,
            Self.tma_tile_shape,
            TensorMapSwizzle.SWIZZLE_NONE,
        ](),
    ]
    var scales_tma_op: Self.ScalesTMATensorTileType
    var output_tokens: Self.TensorType
    var output_scales_offset: Self.ScalesOffsetTensorType

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return String(
            "NVFP4TokenFormat[fp4_dtype = ",
            String(Self.fp4_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @always_inline
    def __init__(
        out self,
        output_tokens: TileTensor[Self.fp4_dtype, Self.output_layout, ...],
        output_scales: TileTensor[Self.scales_dtype, ...],
        output_scales_offset: TileTensor[
            DType.uint32, Self.scales_offset_layout, ...
        ],
        ctx: DeviceContext,
    ):
        self.output_tokens = {
            UnsafePointer[Scalar[Self.fp4_dtype], MutExternalOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }
        self.output_scales_offset = {
            UnsafePointer[Scalar[DType.uint32], MutExternalOrigin](
                unsafe_from_address=Int(output_scales_offset.ptr)
            ),
            output_scales_offset.layout,
        }

        # Merge the last two dimensions of the output_scales tensor into a single
        # dimension. This is required by the TMA instructions that the leading
        # dimensions must be multiples of 16-byte strides
        var scales_tensor_view = TileTensor(
            output_scales.ptr,
            row_major(
                (
                    Idx(Int(output_scales.dim(0))),
                    Idx[Self._hid_dim // NVFP4_SF_VECTOR_SIZE // SF_ATOM_K](),
                    Idx[SF_ATOM_M[0]](),
                    Idx[SF_ATOM_K * SF_ATOM_M[1]](),
                ),
            ),
        )

        try:
            self.scales_tma_op = create_tensor_tile[Self.tma_tile_shape](
                ctx, scales_tensor_view
            )
        except e:
            abort(String(e))

    @always_inline
    @staticmethod
    def fp4_quant_size() -> Int:
        return align_up(Self.hid_dim // 2, Self.alignment)

    @always_inline
    @staticmethod
    def scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by group_size"
        return align_up(
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype](),
            Self.alignment,
        )

    @always_inline
    @staticmethod
    def token_size() -> Int:
        return Self.fp4_quant_size() + Self.scales_size()

    @always_inline
    @staticmethod
    def scales_offset() -> Int:
        return Self.fp4_quant_size()

    @always_inline
    def pad_expert_offsets[
        n_groups: Int
    ](self, row_offsets: UnsafePointer[mut=True, UInt32, ...]) -> None:
        """
        The mojo NVFP4 grouped matmul doesn't require padding for each group's
        FP4 quants. However, it requires each group's scales to be aligned to
        the SF_MN_GROUP_SIZE=128. This function updates the output_scales_offset
        tensor to satisfy this requirement.

        For example, if the row_offsets tensor is [0, 100, 300, 400], this
        function will update the output_scales_offset tensor to [0, 1, 1]. The
        formula is:
            For group i, its first scales block index is row_offsets[i] //
            SF_MN_GROUP_SIZE + output_scales_offset[i].
        Group 0, 1 and 2 have 100, 200, 100 tokens respectively, so the number
        of scales blocks are 1, 2, 1 respectively. The scales blocks for group 1
        start at 100 // 128 + output_scales_offset[1] = 1, and the scales blocks
        for group 2 start at 300 // 128 + output_scales_offset[2] = 3.
        """
        comptime assert (
            Self.ScalesOffsetTensorType.flat_rank >= 1
        ), "output_scales_offset expects rank >= 1"
        var tid = thread_idx.x
        comptime n_rounds = ceildiv(n_groups, WARP_SIZE)

        if warp_id() == 0:
            var prev_round_blocks_num: UInt32 = 0

            comptime for round_i in range(n_rounds):
                var per_expert_m: UInt32 = 0
                var group_idx = round_i * WARP_SIZE + tid
                if group_idx < n_groups:
                    per_expert_m = (
                        row_offsets[group_idx + 1] - row_offsets[group_idx]
                    )

                var scales_blocks = ceildiv(
                    per_expert_m, UInt32(SF_MN_GROUP_SIZE)
                )
                var group_scales_start = (
                    prev_round_blocks_num
                    + warp.prefix_sum[exclusive=True](scales_blocks)
                )

                if group_idx < n_groups:
                    self.output_scales_offset.store(
                        (Idx(group_idx),),
                        group_scales_start
                        - row_offsets[group_idx] // UInt32(SF_MN_GROUP_SIZE),
                    )

                var group_scales_end = group_scales_start + scales_blocks
                prev_round_blocks_num = warp.shuffle_idx(
                    group_scales_end, UInt32(WARP_SIZE - 1)
                )

    @always_inline
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, _, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        comptime src_width = 8
        comptime byte_width = src_width // 2
        comptime NUM_THREADS_PER_SF = NVFP4_SF_VECTOR_SIZE // src_width

        for i in range(thread_idx.x, Self.hid_dim // src_width, block_size):
            var loaded_vec = src_p.load[
                width=src_width, alignment=Self.alignment, invariant=True
            ](i * src_width)

            # each thread finds maximum value in its local 8 elements
            var thread_max = abs(loaded_vec).reduce_max().cast[DType.float32]()
            # find the maximum value among all 16 elements
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )

            # get the scale factor for these 16 elements by dividing it by the maximum value of fp4-e2m1
            var scale_factor = input_scale * (group_max * recip(Float32(6.0)))

            # NOTE: NVFP4 uses FP8-UE4M3 format for the scale factor but we know that scale_factor is always positive, so we can use E4M3 instead of UE4M3.
            var fp8_scale_factor = scale_factor.cast[Self.scales_dtype]()

            var output_scale = Float32(0.0)
            if group_max != 0:
                output_scale = recip(
                    fp8_scale_factor.cast[DType.float32]() * recip(input_scale)
                )

            # write back the scale factor
            comptime scale_bytes = size_of[Self.scales_dtype]()
            if i % NUM_THREADS_PER_SF == 0:
                buf_p.store[alignment=scale_bytes](
                    Self.scales_offset()
                    + ufloordiv(i, NUM_THREADS_PER_SF) * scale_bytes,
                    bitcast[DType.uint8, scale_bytes](fp8_scale_factor),
                )

            var input_f32 = loaded_vec.cast[DType.float32]() * output_scale
            var output_vector = bitcast[Self.fp4_dtype, byte_width](
                cast_fp32_to_fp4e2m1(input_f32)
            )
            buf_p.store[alignment=byte_width](
                i * byte_width,
                bitcast[DType.uint8, byte_width](output_vector),
            )

    @always_inline
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, _, address_space=buf_addr_space],
        token_index: Int,
    ) -> None:
        "NVFP4 format directly uses tile based copy."
        pass

    @always_inline
    def init_smem_resources(self) -> None:
        if thread_idx.x == 0:
            self.scales_tma_op.prefetch_descriptor()

        var smem_base = external_memory[
            UInt8, address_space=AddressSpace.SHARED, alignment=128
        ]()
        var mbar_base = (smem_base + Self._mbar_smem_offset).bitcast[
            SharedMemBarrier
        ]()
        if elect_one_sync():
            mbar_base[warp_id()].init()

    @always_inline
    def copy_msg_tile_to_output_tensor[
        extract_topk_info_func: def(
            UnsafePointer[UInt8, MutExternalOrigin], Int
        ) -> None,
        recv_buf_ptr_func: def(Int) -> UnsafePointer[UInt8, MutExternalOrigin],
        //,
        n_warps: Int,
        shared_expert_offset: Int = 0,
    ](
        self,
        expert_id: Int,
        expert_start_pos: Int,
        tile_id: Int,
        tile_end: Int,
        extract_topk_info_functor: extract_topk_info_func,
        recv_buf_ptr_functor: recv_buf_ptr_func,
    ) -> None:
        comptime tile_size = Self.dispatch_wait_tile_shape[0]
        var k_tile_idx = umod(tile_id, Self._n_k_tiles)
        var tile_start = ufloordiv(tile_id, Self._n_k_tiles) * tile_size
        var tile_token_count = tile_end - tile_start
        var w = Int(warp_id())
        var is_warp_leader = elect_one_sync()

        # --- Scales: sub-warp shuffle into SMEM, then 2D TMA store ---
        comptime aligned_tile_size = align_up(
            Int(Coord(Self.tma_tile_shape).product()), 128
        )
        var smem_ptr = external_memory[
            Scalar[Self.scales_dtype],
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]()
        var scales_tile = TileTensor(
            smem_ptr + aligned_tile_size * w,
            row_major(Coord(Self.tma_tile_shape)),
        )

        # Each warp is divided into SF_ATOM_M[1] sub-warps. Each sub-warp
        # handles one token: warp W, sub-warp S processes tile-local token
        # W + SF_ATOM_M[0] * S.
        comptime sub_warp_size = WARP_SIZE // SF_ATOM_M[1]
        var sub_warp_id, lane_in_sub_warp = udivmod(lane_id(), sub_warp_size)
        var scales_tok = w + SF_ATOM_M[0] * sub_warp_id
        var oob = scales_tok >= tile_token_count

        comptime n_scales_per_token = (
            Self.hid_dim // NVFP4_SF_VECTOR_SIZE // Self._n_k_tiles
        )
        comptime n_scales_simd_per_token = n_scales_per_token // SF_ATOM_K

        var scales_gmem_ptr = Optional[
            UnsafePointer[Scalar[Self.scales_dtype], MutExternalOrigin]
        ]()
        if not oob:
            scales_gmem_ptr = (
                recv_buf_ptr_functor(scales_tok) + Self.scales_offset()
            ).bitcast[Scalar[Self.scales_dtype]]()
            scales_gmem_ptr.unsafe_value() += n_scales_per_token * k_tile_idx

        comptime for i in range(0, n_scales_simd_per_token, sub_warp_size):
            var _i = i + lane_in_sub_warp
            var scales_simd = SIMD[Self.scales_dtype, SF_ATOM_K](0.0)
            if not oob:
                scales_simd = scales_gmem_ptr.unsafe_value().load[
                    width=SF_ATOM_K, invariant=True, alignment=SF_ATOM_K
                ](_i * SF_ATOM_K)

            scales_tile.store(
                (Idx(0), Idx(_i), Idx(0), Idx(sub_warp_id * 4)),
                scales_simd,
            )
        syncwarp()

        if is_warp_leader:
            comptime assert (
                Self.ScalesOffsetTensorType.flat_rank == 1
            ), "output_scales_offset expects rank == 1"

            # The first expert always starts at scales block 0, so
            # `output_scales_offset[0]` is 0 by construction. Skip the load
            # for that case so that `pack_shared_expert_inputs` does not
            # race with the auxiliary SM that populates
            # `output_scales_offset` in `pad_expert_offsets`.
            var output_scales_offset: UInt32 = 0
            if expert_id + shared_expert_offset != 0:
                output_scales_offset = rebind[UInt32](
                    self.output_scales_offset[expert_id + shared_expert_offset]
                )
            var scales_block_id = (
                UInt32(
                    ufloordiv(expert_start_pos + tile_start, SF_MN_GROUP_SIZE)
                )
                + output_scales_offset
            )
            fence_async_view_proxy()
            self.scales_tma_op.async_store(
                scales_tile,
                StaticTuple[UInt32, 4](
                    0,
                    UInt32(w),
                    UInt32(k_tile_idx * n_scales_simd_per_token),
                    scales_block_id,
                ),
            )

            self.scales_tma_op.commit_group()

        # --- FP4 values: 1D TMA g2s then s2g per warp ---
        comptime fp4_bytes_per_ktile = Self.hid_dim // 2 // Self._n_k_tiles
        var k_byte_offset = k_tile_idx * fp4_bytes_per_ktile

        var smem_base = smem_ptr.bitcast[UInt8]()
        var warp_fp4_smem = (
            smem_base
            + 32 * Self._scales_smem_per_warp
            + w * Self._fp4_smem_per_warp
        )
        var mbar_base = (smem_base + Self._mbar_smem_offset).bitcast[
            SharedMemBarrier
        ]()
        var output_tokens_base = self.output_tokens.ptr.bitcast[UInt8]()
        var phase = UInt32(0)

        # Process the tokens in reverse order. This reduce latency when there
        # are only a few tokens in the tile.
        for tok_local in range(n_warps - 1 - w, tile_token_count, n_warps):
            var token_ptr = recv_buf_ptr_functor(tok_local)
            var output_pos = expert_start_pos + tile_start + tok_local

            if is_warp_leader:
                # g2s: load FP4 bytes from recv_buf to SMEM via 1D TMA.
                var mbar = mbar_base + w
                mbar[].expect_bytes(Int32(fp4_bytes_per_ktile))
                cp_async_bulk_shared_cluster_global(
                    warp_fp4_smem,
                    token_ptr + k_byte_offset,
                    Int32(fp4_bytes_per_ktile),
                    mbar[].unsafe_ptr(),
                )
                mbar[].wait(phase=phase)
                phase ^= 1

                # s2g: write FP4 bytes from SMEM to output_tokens via 1D TMA.
                fence_async_view_proxy()
                cp_async_bulk_global_shared_cta(
                    output_tokens_base
                    + output_pos * (Self.hid_dim // 2)
                    + k_byte_offset,
                    warp_fp4_smem,
                    Int32(fp4_bytes_per_ktile),
                )
                cp_async_bulk_commit_group()
                cp_async_bulk_wait_group[0]()

            if k_tile_idx == 0:
                extract_topk_info_functor(token_ptr, output_pos)

        # Filp the mbarrier phase to even if it is odd.
        if is_warp_leader:
            if phase == 1:
                var mbar = mbar_base + w
                mbar[].expect_bytes(0)
                mbar[].wait(phase=phase)

            # Flush any pending scales TMA store.
            cp_async_bulk_wait_group[0]()


struct MXFP4TokenFormat[
    fp4_dtype: DType,
    scales_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](TokenFormat, TrivialRegisterPassable):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime group_size = MXFP4_SF_VECTOR_SIZE

    comptime dispatch_wait_tile_shape = (128, 1)
    comptime dispatch_smem_size = 0

    comptime TensorType = TileTensor[
        Self.fp4_dtype, Self.output_layout, MutExternalOrigin
    ]
    comptime ScalesTensorType = TileTensor[
        Self.scales_dtype, Self.scales_layout, MutExternalOrigin
    ]
    var output_tokens: Self.TensorType
    var output_scales: Self.ScalesTensorType

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return String(
            "MXFP4TokenFormat[fp4_dtype = ",
            String(Self.fp4_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @always_inline
    def __init__(
        out self,
        output_tokens: TileTensor[Self.fp4_dtype, Self.output_layout, ...],
        output_scales: TileTensor[Self.scales_dtype, Self.scales_layout, ...],
    ):
        self.output_tokens = {
            UnsafePointer[Scalar[Self.fp4_dtype], MutExternalOrigin](
                unsafe_from_address=Int(output_tokens.ptr)
            ),
            output_tokens.layout,
        }
        self.output_scales = {
            UnsafePointer[Scalar[Self.scales_dtype], MutExternalOrigin](
                unsafe_from_address=Int(output_scales.ptr)
            ),
            output_scales.layout,
        }

    @always_inline
    @staticmethod
    def fp4_quant_size() -> Int:
        return align_up(Self.hid_dim // 2, Self.alignment)

    @always_inline
    @staticmethod
    def scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by group_size"
        return align_up(
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype](),
            Self.alignment,
        )

    @always_inline
    @staticmethod
    def token_size() -> Int:
        return Self.fp4_quant_size() + Self.scales_size()

    @always_inline
    @staticmethod
    def scales_offset() -> Int:
        return Self.fp4_quant_size()

    @always_inline
    @staticmethod
    def copy_token_to_send_buf[
        src_type: DType,
        block_size: Int,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, _, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type], ...],
        input_scale: Float32,
    ) -> None:
        comptime src_width = 8
        comptime byte_width = src_width // 2
        comptime NUM_THREADS_PER_SF = MXFP4_SF_VECTOR_SIZE // src_width

        for i in range(thread_idx.x, Self.hid_dim // src_width, block_size):
            var loaded_vec = src_p.load[
                width=src_width, alignment=Self.alignment, invariant=True
            ](i * src_width).cast[DType.float32]()

            # each thread finds maximum value in its local 8 elements
            var thread_max = abs(loaded_vec).reduce_max().cast[DType.float32]()
            # find the maximum value among all 32 elements
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )

            # Use MXFP4 even-mode rounding for the E8M0 scale.
            var fp8_scale_factor = compute_mxfp4_even_scale(group_max).cast[
                Self.scales_dtype
            ]()
            var scale_f32 = fp8_scale_factor.cast[DType.float32]()

            # write back the scale factor
            comptime scale_bytes = size_of[Self.scales_dtype]()
            if i % NUM_THREADS_PER_SF == 0:
                buf_p.store[alignment=scale_bytes](
                    Self.scales_offset()
                    + i // NUM_THREADS_PER_SF * scale_bytes,
                    bitcast[DType.uint8, scale_bytes](fp8_scale_factor),
                )

            var output_vector = bitcast[Self.fp4_dtype, byte_width](
                cast_float_to_fp4e2m1_amd(
                    loaded_vec,
                    scale_f32,
                )
            )
            buf_p.store[alignment=byte_width](
                i * byte_width,
                bitcast[DType.uint8, byte_width](output_vector),
            )

    @always_inline
    def copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, _, address_space=buf_addr_space],
        token_index: Int,
    ) -> None:
        comptime assert (
            Self.TensorType.flat_rank >= 2
        ), "output_tokens expects rank >= 2"
        # First we copy the FP4 quants.
        comptime fp4_width = simd_width_of[Self.fp4_dtype]()
        comptime quant_bytes = Self.hid_dim // 2
        comptime assert (
            quant_bytes % fp4_width == 0
        ), "quant_bytes must be divisible by fp4_width"

        for i in range(lane_id(), quant_bytes // fp4_width, WARP_SIZE):
            self.output_tokens.store(
                (Idx(token_index), Idx(i * fp4_width)),
                bitcast[Self.fp4_dtype, fp4_width](
                    buf_p.load[
                        width=fp4_width,
                        invariant=True,
                        alignment=Self.alignment,
                    ](
                        i * fp4_width,
                    )
                ),
            )

        comptime assert (
            Self.ScalesTensorType.flat_rank >= 2
        ), "output_scales expects rank >= 2"
        comptime scale_bytes = size_of[Self.scales_dtype]()
        for i in range(lane_id(), Self.hid_dim // Self.group_size, WARP_SIZE):
            self.output_scales.store(
                (Idx(token_index), Idx(i)),
                bitcast[Self.scales_dtype, 1](
                    buf_p.load[
                        width=scale_bytes,
                        invariant=True,
                        alignment=scale_bytes,
                    ](
                        Self.scales_offset() + i * scale_bytes,
                    )
                ),
            )


# ===-----------------------------------------------------------------------===#
# EP Atomic Counters
# ===-----------------------------------------------------------------------===#


struct EPLocalSyncCounters[n_experts: Int](
    DevicePassable, TrivialRegisterPassable
):
    """Manages atomic counters for EP kernel synchronization within a device.

    This struct provides dedicated atomic counter space for each of the four
    EP kernels: dispatch_async, dispatch_wait, combine_async, and combine_wait.
    Each kernel has its own memory region to avoid conflicts, except
    dispatch_wait and combine_async which must share memory since combine_async
    reads data that dispatch_wait writes.

    The struct is used to synchronize between thread blocks within the same
    device.

    Memory Layout (all sizes in Int32 elements):
    - dispatch_async: 2 * n_experts + MAX_GPUS_PER_NODE
    - dispatch_wait/combine_async: 4 * n_experts + 4
    - combine_wait: 2 * n_experts
    """

    var ptr: UnsafePointer[Int32, MutExternalOrigin]
    """Base pointer to the allocated atomic counter memory."""

    comptime device_type: AnyType = Self

    @always_inline
    def __init__(out self, ptr: UnsafePointer[mut=True, Int32, ...]):
        self.ptr = ptr.unsafe_origin_cast[
            MutExternalOrigin
        ]().address_space_cast[AddressSpace.GENERIC]()

    @always_inline
    def __init__(out self, buffer: DeviceBuffer[DType.int32]):
        self.ptr = buffer.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin]()

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return String(t"EPLocalSyncCounters[n_experts={Self.n_experts}]")

    @always_inline
    @staticmethod
    def dispatch_async_size() -> Int:
        """Returns the size in Int32 elements needed by dispatch_async kernel.
        """
        return 2 * Self.n_experts + MAX_GPUS_PER_NODE

    @always_inline
    @staticmethod
    def dispatch_wait_size() -> Int:
        """Returns the size in Int32 elements needed by dispatch_wait kernel.

        Layout (see EPDispatchKernel for exact offset constants):
          Region A [0, 2*n_experts): per expert-rank combine_async compat data
          Region B [2*n_experts, 3*n_experts): within-expert rank prefix sums
          Region C [3*n_experts, 4*n_experts): per-expert work counters
            (only first n_local_experts entries used; rest unused)
          Region D [4*n_experts]: cleanup ref counter
          Region E [4*n_experts + 1]: global ready flag
          Region F [4*n_experts + 2]: send_buf_ready counter
          Region G [4*n_experts + 3]: shared_expert_started counter

        Region A will be used by combine_async kernel to track the number of
        tokens of each expert-rank pair. Region D, E, F and G needs to be reset
        to 0 once the dispatch_wait kernel is done.
        """
        return 4 * Self.n_experts + 4

    @always_inline
    @staticmethod
    def combine_async_size() -> Int:
        """Returns the size in Int32 elements needed by combine_async kernel.

        Must match dispatch_wait_size() since combine_async reuses the same
        memory region.
        """
        return 4 * Self.n_experts + 4

    @always_inline
    @staticmethod
    def combine_wait_size() -> Int:
        """Returns the size in Int32 elements needed by combine_wait kernel."""
        return 2 * Self.n_experts

    @always_inline
    @staticmethod
    def total_size() -> Int:
        """Returns the total size in Int32 elements needed for all counters."""

        comptime assert (
            Self.combine_async_size() == Self.dispatch_wait_size()
        ), "combine_async_size must be equal to dispatch_wait_size"

        # The combine_async_size is omitted because the combine_async kernel reuses the
        # counters of dispatch_wait kernel.
        return (
            Self.dispatch_async_size()
            + Self.dispatch_wait_size()
            + Self.combine_wait_size()
        )

    @always_inline
    def get_dispatch_async_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
        """Returns pointer to dispatch_async kernel atomic counters.

        Layout:
            [0, n_experts): reserved counters per expert
            [n_experts, 2*n_experts): finished counters per expert
        """
        return self.ptr

    @always_inline
    def get_dispatch_wait_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
        """Returns pointer to dispatch_wait kernel atomic counters."""
        return self.ptr + Self.dispatch_async_size()

    @always_inline
    def get_combine_async_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
        """Returns pointer to combine_async kernel atomic counters.

        Note: Returns the same pointer as get_dispatch_wait_ptr() because
        combine_async_kernel reads the offset/count data that dispatch_wait_kernel writes.
        """
        return self.ptr + Self.dispatch_async_size()

    @always_inline
    def get_combine_wait_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
        """Returns pointer to combine_wait kernel atomic counters."""
        return self.ptr + Self.dispatch_async_size() + Self.dispatch_wait_size()


# ===-----------------------------------------------------------------------===#
# EPDispatchKernel - Dispatch Phase Implementation
# ===-----------------------------------------------------------------------===#


struct EPDispatchKernel[
    num_threads: Int,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    use_shmem: Bool = True,
    fused_shared_expert: Bool = False,
    skip_a2a: Bool = False,
]:
    """Implements dispatch_async and dispatch_wait kernel logic for Expert Parallelism.

    This struct encapsulates the token dispatch operations used in MoE (Mixture
    of Experts) models with expert parallelism. It provides methods for:

    1. Async Dispatch:
       - `monitor_and_signal_completion`: Aux SMs count tokens per expert and
         signal completion when all tokens for an expert have been sent.
       - `copy_and_send_tokens`: Comm SMs copy tokens to send buffer and
         transfer them to destination ranks.

    2. Wait for Arrivals:
       - `wait_for_arrivals_and_compute_offsets`: Aux SMs wait for token
         arrivals and compute output tensor offsets. Also signals other SMs to
         copy the tokens to the output tensor once data is ready.
       - `copy_received_tokens_to_output`: Comm SMs copy received tokens to the
         output tensor.

    Parameters:
        num_threads: The number of threads per block.
        n_sms: The total number of SMs in the device.
        n_experts: The total number of experts in the model.
        n_ranks: The number of devices participating in communication.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a high-speed GPU interconnect group.
        token_fmt_type: Type conforming to TokenFormat trait.
        use_shmem: Whether to use the SHMEM API for communication.
        fused_shared_expert: Whether to pack the shared expert inputs with the
            routed experts' inputs.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
    """

    comptime n_local_experts = Self.n_experts // Self.n_ranks
    comptime n_warps = Self.num_threads // WARP_SIZE
    comptime top_k = Self.token_fmt_type.top_k
    comptime hid_dim = Self.token_fmt_type.hid_dim
    comptime msg_bytes = Self.token_fmt_type.msg_size()

    # Aux SMs for dispatch_async kernel: one SM handles n_warps experts for
    # monitoring.
    comptime n_signal_sms = ceildiv(Self.n_experts, Self.n_warps)
    # Aux SMs for dispatch_wait kernel: single SM computes offsets.
    comptime n_offset_sms = 1
    # Communication SMs for each kernel phase.
    comptime n_dispatch_async_comm_sms = Self.n_sms - Self.n_signal_sms
    comptime n_dispatch_wait_comm_sms = Self.n_sms - Self.n_offset_sms

    # Atomic counter layout offsets for dispatch_wait kernel.
    comptime rank_prefix_offset = 2 * Self.n_experts
    comptime work_counter_offset = 3 * Self.n_experts
    comptime cleanup_counter_offset = 4 * Self.n_experts
    comptime ready_flag_offset = 4 * Self.n_experts + 1
    # These two offsets are only used when fused_shared_expert is True.
    comptime send_buf_ready_offset = 4 * Self.n_experts + 2
    comptime shared_expert_started_offset = 4 * Self.n_experts + 3

    comptime _recv_layout = row_major[
        Self.n_local_experts,
        Self.n_ranks,
        Self.max_tokens_per_rank,
        Self.msg_bytes,
    ]()
    comptime _recv_count_layout = row_major[
        Self.n_local_experts, Self.n_ranks
    ]()
    comptime _send_layout = row_major[
        Self.max_tokens_per_rank, Self.msg_bytes
    ]()

    @staticmethod
    @always_inline
    def recv_buf_layout[
        out_dtype: DType = _get_index_type[type_of(Self._recv_layout)](
            AddressSpace.GENERIC
        ),
    ](coord: Coord, out offset: Scalar[out_dtype]):
        comptime if Self.skip_a2a:
            _coord = Coord((coord[0], Idx(0), coord[2], coord[3]))
            offset = Self._recv_layout[linear_idx_type=out_dtype](_coord)
        else:
            offset = Self._recv_layout[linear_idx_type=out_dtype](coord)

    @staticmethod
    @always_inline
    def recv_count_layout(coord: Coord, out offset: Scalar[DType.int32]):
        comptime if Self.skip_a2a:
            _coord = Coord((coord[0], Idx(0)))
            offset = Self._recv_count_layout[linear_idx_type=DType.int32](
                _coord
            )
        else:
            offset = Self._recv_count_layout[linear_idx_type=DType.int32](coord)

    @staticmethod
    @always_inline
    def send_buf_layout(coord: Coord, out offset: Scalar[DType.int32]):
        offset = Self._send_layout[linear_idx_type=DType.int32](coord)

    # ===-------------------------------------------------------------------===#
    # Dispatch Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    def monitor_and_signal_completion(
        topk_ids: TileTensor[mut=False, DType.int32, ...],
        recv_count_ptrs: InlineArray[
            UnsafePointer[UInt64, MutExternalOrigin], Self.p2p_world_size
        ],
        expert_reserved_counter: UnsafePointer[Int32, MutExternalOrigin],
        expert_finished_counter: UnsafePointer[Int32, MutExternalOrigin],
        rank_completion_counter: UnsafePointer[Int32, MutExternalOrigin],
        my_rank: Int32,
    ) -> None:
        """Auxiliary SM logic for dispatch_kernel.

        Counts tokens per expert and signals completion when all tokens for an
        expert have been sent. Each warp handles one expert.

        Args:
            topk_ids: The top-k expert IDs for each token.
            recv_count_ptrs: Array of pointers to receive count buffers.
            expert_reserved_counter: Counter for reserved slots per expert.
            expert_finished_counter: Counter for finished sends per expert.
            rank_completion_counter: Counter for per-rank completion tracking.
            my_rank: The rank of the current device.
        """
        var num_tokens = Int(topk_ids.dim(0))

        var expert_idx = Int32(block_idx.x * Self.n_warps + warp_id())
        var global_expert_idx = expert_idx
        var expert_count: Int32 = 0

        comptime if Self.skip_a2a:
            global_expert_idx = (
                expert_idx + Int32(Self.n_local_experts) * my_rank
            )

        if expert_idx < Int32(Self.n_experts):
            for i in range(lane_id(), num_tokens * Self.top_k, WARP_SIZE):
                if topk_ids.raw_load(i) == global_expert_idx:
                    expert_count += 1

            expert_count = warp.sum(expert_count)

            if lane_id() == 0:
                var dst_rank, dst_expert_local_idx = udivmod(
                    Int(global_expert_idx), Self.n_local_experts
                )
                var signal_offset = Self.recv_count_layout(
                    (Idx(dst_expert_local_idx), Idx(my_rank))
                )
                var counter_offset = Self.recv_count_layout(
                    (Idx(dst_expert_local_idx), Idx(dst_rank))
                )

                # Wait until all the tokens for the expert have been sent.
                while (
                    _counter_atomic.load[ordering=Ordering.ACQUIRE](
                        expert_finished_counter + counter_offset
                    )
                    != expert_count
                ):
                    pass

                ep_signal_completion[
                    Self.use_shmem,
                    n_experts_per_device=Self.n_local_experts,
                    skip_a2a=Self.skip_a2a,
                ](
                    my_rank,
                    Int32(dst_rank),
                    recv_count_ptrs,
                    signal_offset,
                    UInt64(expert_count),
                    rank_completion_counter,
                )

                expert_reserved_counter[counter_offset] = 0
                expert_finished_counter[counter_offset] = 0

    @staticmethod
    @always_inline
    def copy_and_send_tokens[
        input_type: DType,
        //,
        input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
    ](
        input_tokens: TileTensor[mut=False, input_type, ...],
        topk_ids: TileTensor[mut=False, DType.int32, ...],
        send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
        recv_buf_ptrs: InlineArray[
            UnsafePointer[UInt8, MutExternalOrigin], Self.p2p_world_size
        ],
        expert_reserved_counter: UnsafePointer[Int32, MutExternalOrigin],
        expert_finished_counter: UnsafePointer[Int32, MutExternalOrigin],
        my_rank: Int32,
    ) -> None:
        """Communication SM logic for dispatch_kernel.

        Copies tokens to send buffer and transfers them to destination ranks.
        Uses direct P2P transfers for same-node destinations and SHMEM for
        cross-node destinations.

        Args:
            input_tokens: The input tokens to be dispatched.
            topk_ids: The top-k expert IDs for each token.
            send_buf_p: Pointer to the send buffer.
            recv_buf_ptrs: Array of pointers to receive buffers.
            expert_reserved_counter: Counter for reserved slots per expert.
            expert_finished_counter: Counter for finished sends per expert.
            my_rank: The rank of the current device.
        """
        comptime assert (
            input_tokens.flat_rank == 2
        ), "input_tokens expects rank == 2"
        comptime assert topk_ids.flat_rank == 2, "topk_ids expects rank == 2"
        var tid = thread_idx.x
        var num_tokens = input_tokens.dim(0)
        var my_p2p_world, my_p2p_rank = divmod(
            my_rank, Int32(Self.p2p_world_size)
        )

        var input_scale = Float32(1.0)

        comptime if input_scales_wrapper is not None:
            comptime input_scale_fn = input_scales_wrapper.value()
            input_scale = input_scale_fn[DType.float32](0)

        for token_idx in range(
            block_idx.x - Self.n_signal_sms,
            Int(num_tokens),
            Self.n_dispatch_async_comm_sms,
        ):
            # First, all threads in the block copy the input token to the send
            # buffer.
            var curr_send_buf_ptr = send_buf_p + Self.send_buf_layout(
                (Idx(token_idx), Idx(0))
            )
            var input_tensor_ptr = input_tokens.ptr_at_offset(
                (Idx(token_idx), Idx(0))
            )
            Self.token_fmt_type.copy_token_to_send_buf[
                input_type, Self.num_threads
            ](curr_send_buf_ptr, input_tensor_ptr, input_scale)

            if tid < Self.top_k:
                # Store all the top-k expert IDs in current token's message.
                # The remote device will use the expert ID to determine a
                # token's top-k id.
                # Cast the expert ID to a 16-bit integer to save space.
                var top_k_idx = rebind[Int32](topk_ids[token_idx, tid])
                curr_send_buf_ptr.store[
                    width=size_of[UInt16](),
                    alignment=align_of[DType.uint16](),
                ](
                    Self.token_fmt_type.topk_info_offset()
                    + tid * size_of[UInt16](),
                    bitcast[DType.uint8, size_of[UInt16]()](UInt16(top_k_idx)),
                )

                # Store the source token index in current token's message.
                if tid == 0:
                    curr_send_buf_ptr.store[
                        width=size_of[Int32](),
                        alignment=align_of[DType.int32](),
                    ](
                        Self.token_fmt_type.src_info_offset(),
                        bitcast[DType.uint8, size_of[Int32]()](
                            Int32(token_idx)
                        ),
                    )

            barrier()

            # Try to copy the message to the target expert's recv_buf if the
            # target device is on the same node.
            for topk_idx in range(warp_id(), Self.top_k, Self.n_warps):
                var target_expert = rebind[Int32](topk_ids[token_idx, topk_idx])
                var dst_rank, dst_expert_local_idx = divmod(
                    target_expert, Int32(Self.n_local_experts)
                )
                var dst_p2p_world, dst_p2p_rank = divmod(
                    dst_rank, Int32(Self.p2p_world_size)
                )
                var counter_offset = Self.recv_count_layout(
                    (Idx(dst_expert_local_idx), Idx(dst_rank))
                )

                comptime if Self.skip_a2a:
                    # skip send token if the target expert is not on the current device
                    if dst_rank != my_rank:
                        continue

                if my_p2p_world == dst_p2p_world:
                    var slot_idx: Int32 = 0
                    if lane_id() == 0:
                        slot_idx = _counter_atomic.fetch_add[
                            ordering=Ordering.RELAXED
                        ](expert_reserved_counter + counter_offset, 1)
                    slot_idx = warp.broadcast(slot_idx)

                    var dst_recv_buf_ptr = recv_buf_ptrs[
                        dst_p2p_rank
                    ] + Self.recv_buf_layout(
                        (
                            Idx(dst_expert_local_idx),
                            Idx(my_rank),
                            Idx(slot_idx),
                            Idx(0),
                        )
                    )

                    block_memcpy[Self.msg_bytes, WARP_SIZE](
                        dst_recv_buf_ptr,
                        curr_send_buf_ptr,
                        lane_id(),
                    )

                    syncwarp()

                    if lane_id() == 0:
                        _ = _counter_atomic.fetch_add[
                            ordering=Ordering.RELEASE
                        ](expert_finished_counter + counter_offset, 1)

            # We set up `n_rcs` Reliable Communications (RCs) for each
            # remote device. We would like to use the same RC for each expert.
            # However, NVSHMEM does not allow us to explicitly specify the RC
            # for each transfer. Instead, we set the environment variable
            # `NVSHMEM_IBGDA_RC_MAP_BY=warp` so that the RC is selected by the
            # warp ID using round-robin. We can then control the RC for each
            # expert by using the correct warp.
            comptime if Self.use_shmem:
                comptime n_rcs = min(Self.n_local_experts, Self.n_warps)
                var rc_map_offset = Int32(
                    umod(block_idx.x * Self.n_warps + warp_id(), n_rcs)
                )

                var topk_idx = lane_id()
                if topk_idx < Self.top_k and warp_id() < n_rcs:
                    var target_expert = rebind[Int32](
                        topk_ids[token_idx, topk_idx]
                    )
                    var dst_rank, dst_expert_local_idx = divmod(
                        target_expert, Int32(Self.n_local_experts)
                    )
                    var counter_offset = Self.recv_count_layout(
                        (Idx(dst_expert_local_idx), Idx(dst_rank))
                    )
                    var dst_p2p_world = dst_rank // Int32(Self.p2p_world_size)
                    if (
                        rc_map_offset == target_expert % Int32(n_rcs)
                        and my_p2p_world != dst_p2p_world
                    ):
                        var slot_idx = _counter_atomic.fetch_add[
                            ordering=Ordering.RELAXED
                        ](expert_reserved_counter + counter_offset, 1)
                        var dst_recv_buf_ptr = recv_buf_ptrs[
                            my_p2p_rank
                        ] + Self.recv_buf_layout(
                            (
                                Idx(dst_expert_local_idx),
                                Idx(my_rank),
                                Idx(slot_idx),
                                Idx(0),
                            )
                        )
                        shmem_put_nbi[kind=SHMEMScope.default](
                            dst_recv_buf_ptr,
                            curr_send_buf_ptr,
                            c_size_t(Self.msg_bytes),
                            dst_rank,
                        )

                        _ = _counter_atomic.fetch_add[
                            ordering=Ordering.RELEASE
                        ](expert_finished_counter + counter_offset, 1)

    # ===-------------------------------------------------------------------===#
    # Dispatch Callback Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    def wait_for_arrivals_and_compute_offsets(
        format_handler: Self.token_fmt_type,
        row_offsets: TileTensor[mut=True, DType.uint32, ...],
        expert_ids: TileTensor[mut=True, DType.int32, ...],
        recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
        atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
        my_rank: Int32,
        reserved_shared_expert_tokens: UInt32 = 0,
    ) -> None:
        """Auxiliary SM logic for dispatch_wait_kernel.

        Waits for token arrivals from all ranks and computes the output tensor
        offsets for each local expert. Also signals other SMs to copy the tokens
        to the output tensor once data is ready.

        Args:
            format_handler: Instance of token_fmt_type for token decoding.
            row_offsets: Output row offsets for grouped matmul.
            expert_ids: Output expert IDs for grouped matmul.
            recv_count_p: Pointer to receive count buffer.
            atomic_counter: Atomic counter for synchronization.
            my_rank: The rank of the current device.
            reserved_shared_expert_tokens: The number of tokens reserved for the
                shared expert.
        """
        comptime assert (
            row_offsets.flat_rank == 1
        ), "row_offsets expects rank == 1"
        comptime assert (
            expert_ids.flat_rank == 1
        ), "expert_ids expects rank == 1"
        comptime shared_expert_offset = 1 if Self.fused_shared_expert else 0
        var tid = thread_idx.x

        var prefix_sum_arr = stack_allocation[
            Self.n_experts, DType.uint32, address_space=AddressSpace.SHARED
        ]()

        if tid < Self.n_local_experts + shared_expert_offset:
            expert_ids[tid] = Int32(tid)

        if tid == 0:
            row_offsets[0] = 0

            comptime if Self.fused_shared_expert:
                # Place the shared expert's inputs before all routed experts'
                # inputs.
                row_offsets[1] = reserved_shared_expert_tokens

        var token_count: UInt32 = 0
        if tid < Self.n_experts:
            var target_count_ptr = recv_count_p + tid
            var _token_count = _signal_atomic.load[ordering=Ordering.ACQUIRE](
                target_count_ptr
            )
            while _token_count == UInt64.MAX_FINITE:
                _token_count = _signal_atomic.load[ordering=Ordering.ACQUIRE](
                    target_count_ptr
                )
            token_count = UInt32(_token_count)
        barrier()

        token_count = block_prefix_sum[Self.n_experts](token_count)
        if tid < Self.n_experts:
            prefix_sum_arr[tid] = token_count + reserved_shared_expert_tokens

            if tid % Self.n_ranks == Self.n_ranks - 1:
                var local_expert_id = ufloordiv(tid, Self.n_ranks)
                row_offsets[local_expert_id + shared_expert_offset + 1] = (
                    token_count + reserved_shared_expert_tokens
                )
        barrier()

        # Some token format handlers may require padding the expert offsets to
        # satisfy the grouped matmul alignment requirement.
        comptime n_groups = Self.n_local_experts + shared_expert_offset
        format_handler.pad_expert_offsets[n_groups](row_offsets.ptr)

        # Write out data needed for other SMs to copy the tokens to the output
        # tensor.
        if tid < Self.n_experts:
            var local_expert_id = ufloordiv(tid, Self.n_ranks)

            var raw_expert_start_offset = (
                reserved_shared_expert_tokens if local_expert_id
                == 0 else prefix_sum_arr[local_expert_id * Self.n_ranks - 1]
            )

            # Region B: within-expert rank prefix sums. Each value is the
            # cumulative token count for (expert, ranks 0..r) relative to the
            # expert's start position (independent of alignment).
            var within_expert_prefix = (
                prefix_sum_arr[tid] - raw_expert_start_offset
            )
            atomic_counter.store(
                Self.rank_prefix_offset + tid, Int32(within_expert_prefix)
            )

        # Region C: initialize per-expert work-claiming counters to 0.
        if tid < Self.n_local_experts:
            atomic_counter.store(Self.work_counter_offset + tid, Int32(0))

        barrier()

        # Signal other SMs to copy the tokens to the output tensor.
        if tid == 0:
            atomic_counter.store(
                Self.cleanup_counter_offset,
                Int32(Self.n_dispatch_wait_comm_sms),
            )
            _counter_atomic.store[ordering=Ordering.RELEASE](
                atomic_counter + Self.ready_flag_offset,
                Int32(EP_DATA_READY_FLAG),
            )

        # Write out data needed by the combine kernel, and reset the receive
        # count buffer.
        if tid < Self.n_experts:
            var local_expert_id = ufloordiv(tid, Self.n_ranks)

            # The row offsets might be padded to satisfy the grouped matmul
            # alignment requirement. We check the row_offsets tensor again to
            # get the updated start offset of each expert-rank pair.
            var aligned_expert_start_offset = rebind[UInt32](
                row_offsets[local_expert_id + shared_expert_offset]
            )
            var raw_expert_start_offset = (
                reserved_shared_expert_tokens if local_expert_id
                == 0 else prefix_sum_arr[local_expert_id * Self.n_ranks - 1]
            )
            var alignment_delta = Int32(aligned_expert_start_offset) - Int32(
                raw_expert_start_offset
            )

            # Region A: combine_async-compatible per-expert-rank data.
            # Stores (flag + cumulative_end, per_pair_token_count) so that
            # combine_async can derive token_start and token_end.
            atomic_counter.store(
                tid * 2,
                Int32(
                    EP_DATA_READY_FLAG
                    + Int32(prefix_sum_arr[tid])
                    + alignment_delta
                ),
            )
            var pair_token_count: UInt32
            if tid == 0:
                pair_token_count = (
                    prefix_sum_arr[0] - reserved_shared_expert_tokens
                )
            else:
                pair_token_count = prefix_sum_arr[tid] - prefix_sum_arr[tid - 1]
            atomic_counter.store(tid * 2 + 1, Int32(pair_token_count))

            # Reset the receive count buffer.
            recv_count_p.store(tid, UInt64.MAX_FINITE)

    @staticmethod
    @always_inline
    def copy_received_tokens_to_output(
        format_handler: Self.token_fmt_type,
        row_offsets: TileTensor[mut=True, DType.uint32, ...],
        src_info: TileTensor[mut=True, DType.int32, ...],
        recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
        atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
        my_rank: Int32,
    ) -> None:
        """Communication SM logic for dispatch_wait_kernel.

        Copies received tokens from the receive buffer to the output tensor.
        Each SM is assigned to one local expert and dynamically claims tiles via
        per-expert atomic counters. Tokens within a tile may come from multiple
        source ranks; rank boundaries are resolved via the within-expert prefix
        sums written by the auxiliary SM.

        Args:
            format_handler: Instance of token_fmt_type for token decoding.
            row_offsets: Output row offsets for grouped matmul.
            src_info: Output tensor for source token info.
            recv_buf_p: Pointer to the receive buffer.
            atomic_counter: Atomic counter for synchronization.
            my_rank: The rank of the current device.
        """
        comptime assert (
            row_offsets.flat_rank == 1
        ), "row_offsets expects rank == 1"
        comptime assert src_info.flat_rank == 2, "src_info expects rank == 2"
        comptime assert Self.n_local_experts <= Self.n_dispatch_wait_comm_sms
        comptime shared_expert_offset = 1 if Self.fused_shared_expert else 0

        comptime tile_size = Self.token_fmt_type.dispatch_wait_tile_shape[0]
        comptime sms_per_tile = Self.token_fmt_type.dispatch_wait_tile_shape[1]

        var sm_id = block_idx.x
        var tid = thread_idx.x
        var local_expert_id = umod(sm_id, Self.n_local_experts)
        var global_expert_idx = (
            Int(my_rank) * Self.n_local_experts + local_expert_id
        )

        # Shared memory: rank prefix sums, per-tile token-to-rank map,
        # expert start, and chunk_start broadcast slot.
        var rank_prefix = stack_allocation[
            Self.n_ranks, DType.int32, address_space=AddressSpace.SHARED
        ]()
        var tok_rank_map = stack_allocation[
            tile_size, DType.int32, address_space=AddressSpace.SHARED
        ]()
        var smem_vals = stack_allocation[
            2, DType.int32, address_space=AddressSpace.SHARED
        ]()

        @always_inline
        def fetch_tile_id() {read} -> Int32:
            """Fetch the start of the next tile for the current expert. Should
            be called by a single thread.
            """
            return Atomic[scope=DEVICE_SCOPE].fetch_add[
                ordering=Ordering.ACQUIRE
            ](atomic_counter + Self.work_counter_offset + local_expert_id, 1)

        @always_inline
        def fill_tok_rank_map(tile_id: Int, _total: Int) {mut} -> None:
            """Fill tok_rank_map for a tile. Must be called by warp 0 only,
            after rank_prefix is loaded."""
            var _tile_start = ufloordiv(tile_id, sms_per_tile) * tile_size
            var _count = min(_tile_start + tile_size, _total) - _tile_start
            for _tok in range(tid, _count, WARP_SIZE):
                var _rank = Int32(Self.n_ranks - 1)
                for r in range(Self.n_ranks):
                    if _tile_start + _tok < Int(rank_prefix[r]):
                        _rank = Int32(r)
                        break
                tok_rank_map[_tok] = _rank

        # Wait for the auxiliary SM to signal that all offsets are ready.
        if warp_id() == 0:
            var flag = _counter_atomic.load[ordering=Ordering.ACQUIRE](
                atomic_counter + Self.ready_flag_offset
            )
            while flag != EP_DATA_READY_FLAG:
                flag = _counter_atomic.load[ordering=Ordering.ACQUIRE](
                    atomic_counter + Self.ready_flag_offset
                )

            if tid == 0:
                smem_vals[0] = Int32(
                    rebind[UInt32](
                        row_offsets[local_expert_id + shared_expert_offset]
                    )
                )
                smem_vals[1] = fetch_tile_id()

            # Load within-expert rank prefix sums for this expert.
            var base = Self.rank_prefix_offset + local_expert_id * Self.n_ranks
            comptime assert (
                Self.n_ranks <= WARP_SIZE
            ), "n_ranks must be less than or equal to warp size"
            if tid < Self.n_ranks:
                rank_prefix[tid] = (atomic_counter + base)[tid]
            syncwarp()

            # Fill tok_rank_map for the first tile.
            fill_tok_rank_map(
                Int(smem_vals[1]), Int(rank_prefix[Self.n_ranks - 1])
            )
        barrier()

        var expert_start_val = Int(smem_vals[0])
        var tile_id = Int(smem_vals[1])
        var total_tokens = Int(rank_prefix[Self.n_ranks - 1])

        # Dynamic tile claiming loop. All warps in this SM cooperate on
        # each claimed tile.
        var last_tile = False
        while True:
            var tile_start = ufloordiv(tile_id, sms_per_tile) * tile_size
            if tile_start >= total_tokens or last_tile:
                break

            var tile_end = min(tile_start + tile_size, total_tokens)
            last_tile = tile_end == total_tokens

            @always_inline
            def _recv_buf_ptr_for(
                tok_local: Int,
            ) {read} -> UnsafePointer[UInt8, MutExternalOrigin]:
                """Return the pointer to the token in the receive buffer."""
                var wep = tile_start + tok_local
                var src_rank = Int(tok_rank_map[tok_local])
                var rank_base = (
                    Int(rank_prefix[src_rank - 1]) if src_rank > 0 else 0
                )
                return recv_buf_p + Self.recv_buf_layout(
                    (
                        Idx(local_expert_id),
                        Idx(src_rank),
                        Idx(wep - rank_base),
                        Idx(0),
                    )
                )

            @always_inline
            def extract_topk_info(
                token_ptr: UnsafePointer[UInt8, MutExternalOrigin],
                output_pos: Int,
            ) {read} -> None:
                """Extract the top-k info from the token ans save it to the
                src_info tensor. Should be called by whole warp.
                """
                if lane_id() < Self.top_k:
                    var src_topk_idx = bitcast[DType.uint16, 1](
                        token_ptr.load[
                            width=size_of[UInt16](),
                            alignment=size_of[UInt16](),
                        ](
                            Self.token_fmt_type.topk_info_offset()
                            + lane_id() * size_of[UInt16](),
                        )
                    )
                    var src_idx = bitcast[DType.int32, 1](
                        token_ptr.load[
                            width=size_of[Int32](),
                            alignment=size_of[Int32](),
                        ](Self.token_fmt_type.src_info_offset())
                    )
                    if UInt16(global_expert_idx) == src_topk_idx:
                        src_info[output_pos, 0] = src_idx
                        src_info[output_pos, 1] = Int32(lane_id())

            format_handler.copy_msg_tile_to_output_tensor[
                n_warps=Self.n_warps,
                shared_expert_offset=shared_expert_offset,
            ](
                local_expert_id,
                expert_start_val,
                tile_id,
                tile_end,
                extract_topk_info,
                _recv_buf_ptr_for,
            )

            # Warp 0 claims the next tile and fills tok_rank_map.
            if not last_tile:
                barrier()
                if warp_id() == 0:
                    if tid == 0:
                        smem_vals[1] = fetch_tile_id()
                    syncwarp()
                    fill_tok_rank_map(Int(smem_vals[1]), total_tokens)
                barrier()
                tile_id = Int(smem_vals[1])

        # Cleanup: the last SM to finish resets the flag.
        if warp_id() == 0 and tid == 0:
            var count = Atomic[scope=DEVICE_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](atomic_counter + Self.cleanup_counter_offset, Int32(-1))
            if count == 1:
                atomic_counter.store(Self.ready_flag_offset, Int32(0))

    @staticmethod
    @always_inline
    def pack_shared_expert_inputs(
        format_handler: Self.token_fmt_type,
        send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
        fused_se_counter: UnsafePointer[Int32, MutExternalOrigin],
        shared_expert_token_count: Int,
    ) -> None:
        """Copies already-quantized shared expert tokens from send_buf to output.

        Waits for dispatch_async signal SMs to indicate all tokens have been
        written to the send buffer, then uses tile-based copy via
        copy_msg_tile_to_output_tensor. Only SMs needed for the copy participate.

        Args:
            format_handler: Instance of token_fmt_type for token decoding.
            send_buf_p: Pointer to the send buffer containing serialized tokens.
            fused_se_counter: Pointer to the two fused shared expert atomic
                counters (send_buf_ready at [0], started at [1]).
            shared_expert_token_count: Number of shared expert tokens to copy.
        """
        comptime tile_size = Self.token_fmt_type.dispatch_wait_tile_shape[0]
        comptime sms_per_tile = Self.token_fmt_type.dispatch_wait_tile_shape[1]

        var sm_id = Self.n_dispatch_wait_comm_sms - block_idx.x - 1
        var n_tiles = (
            ceildiv(shared_expert_token_count, tile_size) * sms_per_tile
        )
        var n_sms_for_shared = min(n_tiles, Self.n_dispatch_wait_comm_sms)

        if sm_id >= n_sms_for_shared:
            return

        # Wait for all dispatch_async signal SMs to finish writing to send_buf.
        if warp_id() == 0 and thread_idx.x == 0:
            var ready = _counter_atomic.load[ordering=Ordering.ACQUIRE](
                fused_se_counter
            )
            while ready != Int32(Self.n_signal_sms):
                ready = _counter_atomic.load[ordering=Ordering.ACQUIRE](
                    fused_se_counter
                )

            # Signal that this SM has started; the last one resets both counters.
            var started = Atomic[scope=DEVICE_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](fused_se_counter + 1, 1)
            if started == Int32(n_sms_for_shared) - 1:
                fused_se_counter.store(Int32(0))
                fused_se_counter.store(1, Int32(0))
        barrier()

        for tile_id in range(sm_id, n_tiles, n_sms_for_shared):
            var tile_start = ufloordiv(tile_id, sms_per_tile) * tile_size
            var tile_end = min(
                tile_start + tile_size, shared_expert_token_count
            )

            @always_inline
            def _send_buf_ptr_for(
                tok_local: Int,
            ) {read} -> UnsafePointer[UInt8, MutExternalOrigin]:
                return send_buf_p + Self.send_buf_layout(
                    (Idx(tile_start + tok_local), Idx(0))
                )

            @always_inline
            def extract_topk_info(
                token_ptr: UnsafePointer[UInt8, MutExternalOrigin],
                output_pos: Int,
            ) {read} -> None:
                pass

            format_handler.copy_msg_tile_to_output_tensor[
                n_warps=Self.n_warps,
            ](
                0,
                0,
                tile_id,
                tile_end,
                extract_topk_info,
                _send_buf_ptr_for,
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_dispatch_async_{input_type}_{num_threads}_{n_sms}_{n_experts}_{n_ranks}_{max_tokens_per_rank}_{p2p_world_size}_{use_shmem}",
    mangle=True,
)
def dispatch_async_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    topk_ids_layout: TensorLayout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
    use_shmem: Bool = True,
](
    input_tokens: TileTensor[
        input_type, input_tokens_layout, ImmutExternalOrigin
    ],
    topk_ids: TileTensor[DType.int32, topk_ids_layout, ImmutExternalOrigin],
    send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_buf_ptrs: InlineArray[
        UnsafePointer[UInt8, MutExternalOrigin], p2p_world_size
    ],
    recv_count_ptrs: InlineArray[
        UnsafePointer[UInt64, MutExternalOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
):
    """
    Dispatch tokens to experts on remote ranks based on the top-k expert IDs.
    This kernel utilizes the non-blocking SHMEM API if `use_shmem` is True, and
    would return immediately after initiating the communication. The
    communication is considered complete after calling the `dispatch_wait_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        topk_ids_layout: The layout of the top-k expert IDs.
        n_sms: The total number of SMs in the device.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        token_fmt_type: Type conforming to TokenFormat trait that defines the
            token encoding scheme.
        input_scales_wrapper: The wrapper for the input scales.
        use_shmem: Whether to use the SHMEM API for the communication.

    Args:
        input_tokens: The input tokens to be dispatched.
        topk_ids: The top-k expert IDs for each token.
        send_buf_p: The pointer to the send buffer. The underlying buffer is of
            shape `(max_tokens_per_rank, msg_bytes)`. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`. Need
            to be allocated using `shmem_alloc` if `use_shmem` is True.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks)`. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
    """

    comptime dispatch_impl = EPDispatchKernel[
        num_threads,
        n_sms,
        n_experts,
        n_ranks,
        max_tokens_per_rank,
        p2p_world_size,
        token_fmt_type,
        use_shmem,
    ]

    # The reserved counter is incremented once a warp is ready to send.
    # The finished counter is incremented once the token is sent.
    var atomic_counter = ep_counters.get_dispatch_async_ptr()
    var expert_reserved_counter = atomic_counter
    var expert_finished_counter = atomic_counter + n_experts
    # Per-rank completion counter for same-node signaling.
    var rank_completion_counter = atomic_counter + 2 * n_experts

    # The auxiliary SMs are used for counting the number of tokens that need to
    # be sent to each expert. It also monitors the completion of
    # the communication for each expert.
    if block_idx.x < dispatch_impl.n_signal_sms:
        dispatch_impl.monitor_and_signal_completion(
            topk_ids,
            recv_count_ptrs,
            expert_reserved_counter,
            expert_finished_counter,
            rank_completion_counter,
            my_rank,
        )

    # All the other SMs are used for sending the tokens to the experts.
    else:
        dispatch_impl.copy_and_send_tokens[input_scales_wrapper](
            input_tokens,
            topk_ids,
            send_buf_p,
            recv_buf_ptrs,
            expert_reserved_counter,
            expert_finished_counter,
            my_rank,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__llvm_arg_metadata(format_handler, `nvvm.grid_constant`)
@__name(
    t"ep_wait_{num_threads}_{n_sms}_{n_experts}_{n_ranks}_{max_tokens_per_rank}",
    mangle=True,
)
def dispatch_wait_kernel[
    num_threads: Int,
    row_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    src_info_layout: TensorLayout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    token_fmt_type: TokenFormat,
    input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
](
    format_handler: token_fmt_type,
    row_offsets: TileTensor[
        DType.uint32, row_offsets_layout, MutExternalOrigin
    ],
    expert_ids: TileTensor[DType.int32, expert_ids_layout, MutExternalOrigin],
    src_info: TileTensor[DType.int32, src_info_layout, MutExternalOrigin],
    recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
):
    """
    This kernel is called after the `dispatch_kernel` to complete the
    communication. It will keep polling the receive count buffer, and once the
    count is no longer MAX_FINITE, it can confirm that the communication is
    complete from a remote rank.

    The kernel will also aggregate the tokens from all the experts, and store
    them in the output tensor using a ragged representation.

    Parameters:
        num_threads: The number of threads in the block.
        row_offsets_layout: The layout of the row offsets.
        expert_ids_layout: The layout of the expert IDs.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        n_experts: The number of experts in the device.
        n_ranks: The number of ranks.
        max_tokens_per_rank: The maximum number of tokens per rank.
        token_fmt_type: Type conforming to TokenFormat trait that defines the
            token encoding scheme.
        input_scales_wrapper: The wrapper for the input scales.

    Args:
        format_handler: Instance of token_fmt_type that performs token decoding
            and manages output tensor writes.
        row_offsets: The row offsets to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        expert_ids: The expert IDs to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        src_info: The source token info to be updated. Once the expert
            computation is complete, tokens will be send back to the original
            rank using information in this tensor.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated
            using `shmem_alloc` if `use_shmem` is True. The underlying buffer is
            of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be
            allocated using `shmem_alloc` if `use_shmem` is True. The underlying
            buffer is of shape `(n_local_experts, n_ranks)`.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
    """

    comptime dispatch_impl = EPDispatchKernel[
        num_threads,
        n_sms,
        n_experts,
        n_ranks,
        max_tokens_per_rank,
        1,  # p2p world size
        token_fmt_type,
        use_shmem=False,
    ]

    var atomic_counter = ep_counters.get_dispatch_wait_ptr()

    # The last SM is used for checking if any of a local expert has received
    # tokens from all the remote ranks. It will also calculate the offset where
    # the tokens start in the output tensor.
    if block_idx.x >= dispatch_impl.n_dispatch_wait_comm_sms:
        dispatch_impl.wait_for_arrivals_and_compute_offsets(
            format_handler,
            row_offsets,
            expert_ids,
            recv_count_p,
            atomic_counter,
            my_rank,
        )

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        format_handler.init_smem_resources()
        dispatch_impl.copy_received_tokens_to_output(
            format_handler,
            row_offsets,
            src_info,
            recv_buf_p,
            atomic_counter,
            my_rank,
        )


# ===-----------------------------------------------------------------------===#
# EPCombineKernel - Combine Phase Implementation
# ===-----------------------------------------------------------------------===#


struct EPCombineKernel[
    num_threads: Int,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    use_shmem: Bool = True,
    fused_shared_expert: Bool = False,
    skip_a2a: Bool = False,
]:
    """Implements combine_async and combine_wait kernel logic for Expert Parallelism.

    This struct encapsulates the token combine operations used in MoE (Mixture
    of Experts) models with expert parallelism. It provides methods for:

    1. Async Combine:
       - `send_tokens_back`: Send processed tokens back to their original ranks.

    2. Wait for Arrivals:
       - `wait_for_all_arrivals`: Aux SMs wait for all tokens to arrive.
       - `reduce_and_copy_to_output`: Comm SMs reduce and copy tokens to output.

    Parameters:
        num_threads: The number of threads per block.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of devices participating in communication.
        msg_bytes: The number of bytes per token message.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a high-speed GPU interconnect group.
        use_shmem: Whether to use the SHMEM API for communication.
        fused_shared_expert: Whether to filter out the shared expert's outputs.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            receive tokens from the current device.
    """

    comptime n_local_experts = Self.n_experts // Self.n_ranks
    comptime n_warps = Self.num_threads // WARP_SIZE

    # Aux SMs for combine_wait kernel: single SM waits for arrivals.
    comptime n_wait_sms = 1
    # Reduce SMs for combine_wait kernel.
    comptime n_reduce_sms = Self.n_sms - Self.n_wait_sms

    comptime _send_layout = row_major[
        Self.n_local_experts * Self.n_ranks * Self.max_tokens_per_rank,
        Self.msg_bytes,
    ]()
    comptime _recv_layout = row_major[
        Self.max_tokens_per_rank, Self.top_k, Self.msg_bytes
    ]()
    comptime _recv_count_layout = row_major[
        Self.n_local_experts, Self.n_ranks
    ]()

    @staticmethod
    @always_inline
    def send_buf_layout[
        out_dtype: DType = _get_index_type[type_of(Self._send_layout)](
            AddressSpace.GENERIC
        ),
    ](coord: Coord) -> Scalar[out_dtype]:
        return Self._send_layout[linear_idx_type=out_dtype](coord)

    @staticmethod
    @always_inline
    def recv_buf_layout(coord: Coord) -> Scalar[DType.int32]:
        return Self._recv_layout[linear_idx_type=DType.int32](coord)

    @staticmethod
    @always_inline
    def recv_count_layout(coord: Coord) -> Scalar[DType.int32]:
        comptime if Self.skip_a2a:
            var _coord = Coord((coord[0], Idx(0)))
            return Self._recv_count_layout[linear_idx_type=DType.int32](_coord)
        else:
            return Self._recv_count_layout[linear_idx_type=DType.int32](coord)

    # ===-------------------------------------------------------------------===#
    # Combine Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    def copy_shared_expert_outputs[
        input_type: DType,
        //,
    ](
        input_tokens: TileTensor[input_type, ...],
        output_tokens: TileTensor[mut=True, input_type, ...],
    ) -> None:
        """Copies shared expert outputs to the output tensor.

        This method copies the shared expert's output tokens from the input
        tensor to the output tensor when fused_shared_expert is enabled.

        Args:
            input_tokens: The input tokens containing shared expert outputs.
            output_tokens: The output tensor to copy shared expert outputs to.
        """
        comptime assert (
            input_tokens.flat_rank >= 2
        ), "input_tokens expects rank >= 2"
        comptime assert (
            output_tokens.flat_rank >= 2
        ), "output_tokens expects rank >= 2"
        comptime hid_dim = input_tokens.static_shape[1]
        var tid = thread_idx.x
        var sm_id = block_idx.x
        var shared_expert_token_count = output_tokens.dim(0)

        for token_idx in range(
            sm_id, Int(shared_expert_token_count), Self.n_sms
        ):
            var output_tokens_p = output_tokens.ptr + token_idx * hid_dim
            var input_tokens_p = input_tokens.ptr + token_idx * hid_dim
            block_memcpy[hid_dim * size_of[input_type](), Self.num_threads](
                output_tokens_p.bitcast[UInt8](),
                input_tokens_p.bitcast[UInt8](),
                tid,
            )

    @staticmethod
    @always_inline
    def send_tokens_back[
        input_type: DType,
        //,
    ](
        input_tokens: TileTensor[input_type, ...],
        src_info: TileTensor[DType.int32, ...],
        send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
        recv_buf_ptrs: InlineArray[
            UnsafePointer[UInt8, MutExternalOrigin], Self.p2p_world_size
        ],
        recv_count_ptrs: InlineArray[
            UnsafePointer[UInt64, MutExternalOrigin], Self.p2p_world_size
        ],
        atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
        rank_completion_counter: UnsafePointer[Int32, MutExternalOrigin],
        my_rank: Int32,
    ) -> None:
        """Send processed tokens back to their original ranks.

        Each SM handles one expert-rank pair, sending all tokens for that pair
        back to the original rank. Uses direct P2P transfers for same-node
        destinations and SHMEM for cross-node destinations.

        Args:
            input_tokens: The tokens to be sent back.
            src_info: Source token info (original position and top-k ID).
            send_buf_p: Pointer to the send buffer.
            recv_buf_ptrs: Array of pointers to receive buffers.
            recv_count_ptrs: Array of pointers to receive count buffers.
            atomic_counter: Atomic counter for synchronization.
            rank_completion_counter: Counter for per-rank completion tracking.
            my_rank: The rank of the current device.
        """
        comptime assert (
            input_tokens.flat_rank == 2
        ), "input_tokens expects rank == 2"
        comptime assert src_info.flat_rank >= 2, "src_info expects rank >= 2"
        comptime hid_dim = input_tokens.static_shape[1]

        comptime assert (
            Self.msg_bytes == hid_dim * size_of[Scalar[input_type]]()
        ), "EP combine_async: input shape doesn't match message size."

        var tid = thread_idx.x
        var sm_id = block_idx.x
        var my_p2p_world, my_p2p_rank = udivmod(
            Int(my_rank), Self.p2p_world_size
        )

        # Each rank holds `n_local_experts` experts, and for each expert, it
        # needs to send back different tokens to `n_ranks` remote ranks. We use
        # one block per-expert-per-rank to send back the tokens.
        for _global_idx in range(sm_id, Self.n_experts, Self.n_sms):
            var global_idx = _global_idx
            comptime if Self.skip_a2a:
                global_idx = _global_idx + Self.n_local_experts * Int(my_rank)

            var target_rank, local_expert_id = udivmod(
                global_idx, Self.n_local_experts
            )
            var expert_rank_offset = Self.recv_count_layout(
                (Idx(local_expert_id), Idx(target_rank))
            )
            var dst_p2p_world, dst_p2p_rank = udivmod(
                target_rank, Self.p2p_world_size
            )

            # Info for where the tokens for the current expert and rank start
            # and end are stored in the atomic counter by the
            # `dispatch_wait_kernel`.
            comptime DATA_READY_FLAG = 1024
            var token_end_count = atomic_counter.load[
                width=2,
                alignment=align_of[SIMD[DType.int32, 2]](),
                invariant=True,
            ](2 * expert_rank_offset)
            var token_end = token_end_count[0] - DATA_READY_FLAG
            var token_start = token_end - token_end_count[1]

            # If the target device is on the same node, we can directly copy the
            # tokens to the receive buffer, skipping the send buffer.
            if dst_p2p_world == my_p2p_world:
                for token_idx in range(token_start, token_end):
                    var src_token_info = src_info.load[width=2](
                        (Idx(token_idx), Idx(0))
                    )
                    var src_idx = src_token_info[0]
                    var src_topk_idx = src_token_info[1]

                    var dst_recv_buf_ptr = recv_buf_ptrs[
                        dst_p2p_rank
                    ] + Self.recv_buf_layout(
                        (Idx(src_idx), Idx(src_topk_idx), Idx(0))
                    )
                    block_memcpy[
                        hid_dim * size_of[input_type](), Self.num_threads
                    ](
                        dst_recv_buf_ptr,
                        input_tokens.ptr_at_offset(
                            (Idx(token_idx), Idx(0))
                        ).bitcast[UInt8](),
                        tid,
                    )

            # If the target device is on a different node, we need to send the
            # tokens to the target device using the SHMEM API.
            else:
                comptime if Self.use_shmem:
                    # The tokens are sent back to the original rank using the
                    # same RC as the one they come from.
                    comptime n_rcs = min(Self.n_local_experts, Self.n_warps)
                    var rc_map_offset = (
                        sm_id * Self.n_warps + warp_id()
                    ) % n_rcs

                    var n_rounds = ceildiv(
                        token_end - token_start, Int32(Self.n_warps)
                    )
                    for round_i in range(n_rounds):
                        var token_idx = (
                            token_start
                            + round_i * Int32(Self.n_warps)
                            + Int32(warp_id())
                        )
                        if token_idx < token_end:
                            var curr_send_buf_ptr = (
                                send_buf_p
                                + Self.send_buf_layout((Idx(token_idx), Idx(0)))
                            )

                            # To use SHMEM API, we need to copy the tokens to
                            # the send buffer first.
                            block_memcpy[
                                hid_dim * size_of[input_type](), WARP_SIZE
                            ](
                                curr_send_buf_ptr,
                                input_tokens.ptr_at_offset(
                                    (Idx(token_idx), Idx(0))
                                ).bitcast[UInt8](),
                                lane_id(),
                            )

                        barrier()

                        if (
                            warp_id() < n_rcs
                            and local_expert_id % n_rcs == rc_map_offset
                        ):
                            var token_idx = (
                                token_start
                                + round_i * Int32(Self.n_warps)
                                + Int32(lane_id())
                            )
                            if token_idx < token_end:
                                var src_token_info = src_info.load[width=2](
                                    (Idx(token_idx), Idx(0))
                                )
                                var src_idx = src_token_info[0]
                                var src_topk_idx = src_token_info[1]

                                var curr_send_buf_ptr = (
                                    send_buf_p
                                    + Self.send_buf_layout(
                                        (Idx(token_idx), Idx(0))
                                    )
                                )
                                var dst_recv_buf_ptr = recv_buf_ptrs[
                                    my_p2p_rank
                                ] + Self.recv_buf_layout(
                                    (
                                        Idx(Int(src_idx)),
                                        Idx(Int(src_topk_idx)),
                                        Idx(0),
                                    )
                                )

                                shmem_put_nbi[kind=SHMEMScope.default](
                                    dst_recv_buf_ptr,
                                    curr_send_buf_ptr,
                                    c_size_t(Self.msg_bytes),
                                    Int32(target_rank),
                                )

            barrier()

            # Once all the tokens for the current expert and rank have been
            # sent, signal the completion of the communication.
            comptime n_rcs = min(Self.n_local_experts, Self.n_warps)
            var rc_map_offset = (sm_id * Self.n_warps + warp_id()) % n_rcs
            if warp_id() < n_rcs and local_expert_id % n_rcs == rc_map_offset:
                if lane_id() == 0:
                    var signal_offset = Self.recv_count_layout(
                        (Idx(local_expert_id), Idx(my_rank))
                    )

                    ep_signal_completion[
                        Self.use_shmem,
                        n_experts_per_device=Self.n_local_experts,
                        skip_a2a=Self.skip_a2a,
                    ](
                        my_rank,
                        Int32(target_rank),
                        recv_count_ptrs,
                        signal_offset,
                        UInt64(token_end - token_start),
                        rank_completion_counter,
                    )

                    atomic_counter.store[
                        width=2, alignment=align_of[SIMD[DType.int32, 2]]()
                    ](expert_rank_offset * 2, 0)

    # ===-------------------------------------------------------------------===#
    # Combine Callback Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    def wait_for_all_arrivals(
        recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
        atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
    ) -> None:
        """Auxiliary SM logic for combine_wait_kernel.

        Waits for all tokens to arrive from all ranks, then signals other SMs
        that they can start copying tokens to the output tensor.

        Args:
            recv_count_p: Pointer to the receive count buffer.
            atomic_counter: Atomic counter for synchronization.
        """
        comptime DATA_READY_FLAG = 1024

        if thread_idx.x < Self.n_experts:
            var target_count_ptr = recv_count_p + thread_idx.x
            while (
                _signal_atomic.load[ordering=Ordering.ACQUIRE](target_count_ptr)
                == UInt64.MAX_FINITE
            ):
                pass

            target_count_ptr[] = UInt64.MAX_FINITE
        barrier()

        # Once all the tokens have been received, set flags for other SMs to
        # copy the tokens to the output tensor.
        if thread_idx.x < Self.n_reduce_sms:
            _counter_atomic.store[ordering=Ordering.RELEASE](
                atomic_counter + Self.n_wait_sms + thread_idx.x,
                Int32(DATA_READY_FLAG),
            )

    @staticmethod
    @always_inline
    def reduce_and_copy_to_output[
        output_type: DType,
        router_weights_wrapper: Optional[router_weights_wrapper_type] = None,
        elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    ](
        output_tokens: TileTensor[mut=True, output_type, ...],
        recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
        atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
        my_rank: Int32,
        topk_ids_p: Optional[UnsafePointer[Int32, ImmutExternalOrigin]] = None,
    ) -> None:
        """Communication SM logic for combine_wait_kernel.

        Copies received tokens to the output tensor, optionally applying
        router weights and reduction across top-k experts.

        Args:
            output_tokens: The tensor to store the output tokens.
            recv_buf_p: Pointer to the receive buffer.
            atomic_counter: Atomic counter for synchronization.
            my_rank: The rank of the current device.
            topk_ids_p: Pointer to the top-k IDs for each token, only required
                if skip_a2a is True.
        """
        comptime DATA_READY_FLAG = 1024
        var num_tokens = Int(output_tokens.dim(0))
        comptime dst_simd_width = simd_width_of[output_type]()
        comptime byte_simd_width = simd_width_of[DType.uint8]()

        comptime last_dim = 1 if router_weights_wrapper else 2
        comptime hid_dim = output_tokens.static_shape[last_dim]
        comptime _align = align_of[SIMD[DType.uint8, byte_simd_width]]()

        comptime assert (
            Self.msg_bytes == hid_dim * size_of[Scalar[output_type]]()
        ), "EP combine_async: output shape doesn't match message size."
        comptime assert (
            Self.msg_bytes % byte_simd_width == 0
        ), "EP combine_async: message size must be divisible by " + String(
            byte_simd_width
        )

        var sm_id = block_idx.x

        if thread_idx.x == 0:
            while (
                _counter_atomic.load[ordering=Ordering.ACQUIRE](
                    atomic_counter + sm_id
                )
                != DATA_READY_FLAG
            ):
                pass

            # Reset the atomic counter for the next round.
            atomic_counter.store(sm_id, 0)
        barrier()

        comptime n_chunk_elems = WARP_SIZE * dst_simd_width
        comptime n_chunk_bytes = WARP_SIZE * byte_simd_width
        comptime n_chunks_per_tok = hid_dim // n_chunk_elems

        comptime assert (
            hid_dim % n_chunk_elems == 0
        ), "EP combine_async: hid_dim must be divisible by n_chunk_elems"

        # This will allow a single token to be processed by multiple blocks.
        # Reduce the latency when there is only a small number of tokens.
        var global_id = sm_id - Self.n_wait_sms + warp_id() * Self.n_reduce_sms

        for chunk_idx in range(
            global_id,
            num_tokens * n_chunks_per_tok,
            Self.n_warps * Self.n_reduce_sms,
        ):
            var token_idx, chunk_idx_in_token = udivmod(
                chunk_idx, n_chunks_per_tok
            )

            var accum = SIMD[DType.float32, dst_simd_width](0)

            comptime for topk_idx in range(Self.top_k):
                comptime if Self.skip_a2a:
                    var topk_ids_ptr = topk_ids_p.unsafe_value()
                    var expert_id = topk_ids_ptr.load(
                        token_idx * Self.top_k + topk_idx
                    )

                    if ufloordiv(Int(expert_id), Self.n_local_experts) != Int(
                        my_rank
                    ):
                        continue

                var recv_buf_ptr = recv_buf_p + Self.recv_buf_layout(
                    (
                        Idx(token_idx),
                        Idx(topk_idx),
                        Idx(chunk_idx_in_token * n_chunk_bytes),
                    )
                )
                var recv_chunk = bitcast[output_type, dst_simd_width](
                    recv_buf_ptr.load[
                        width=byte_simd_width,
                        invariant=True,
                        alignment=_align,
                    ](
                        lane_id() * byte_simd_width,
                    )
                )

                comptime if router_weights_wrapper:
                    comptime router_weights_fn = router_weights_wrapper.value()

                    var weight = router_weights_fn[1](token_idx, topk_idx)
                    accum += weight * recv_chunk.cast[DType.float32]()

                else:
                    # The output tensor is of shape
                    # `(num_tokens, top_k, hid_dim)`.
                    comptime assert output_tokens.flat_rank >= 3, (
                        "output_tokens expects rank >= 3 for (token_idx,"
                        " topk_idx, elem_offset)"
                    )
                    var elem_offset = (
                        chunk_idx_in_token * n_chunk_elems
                        + lane_id() * dst_simd_width
                    )
                    output_tokens.store(
                        (
                            Idx(token_idx),
                            Idx(topk_idx),
                            Idx(elem_offset),
                        ),
                        recv_chunk,
                    )

            comptime if router_weights_wrapper:
                comptime if elementwise_lambda_fn:
                    comptime lambda_fn = elementwise_lambda_fn.value()
                    lambda_fn[alignment=_align](
                        (
                            token_idx,
                            chunk_idx_in_token * n_chunk_elems
                            + lane_id() * dst_simd_width,
                        ),
                        accum.cast[output_type](),
                    )

                else:
                    comptime assert (
                        output_tokens.flat_rank >= 2
                    ), "output_tokens expects rank >= 2 for reduced output"
                    output_tokens.store(
                        (
                            Idx(token_idx),
                            Idx(
                                chunk_idx_in_token * n_chunk_elems
                                + lane_id() * dst_simd_width
                            ),
                        ),
                        accum.cast[output_type](),
                    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_combine_async_{input_type}_{num_threads}_{n_sms}_{top_k}_{n_experts}_{n_ranks}_{msg_bytes}_{max_tokens_per_rank}_{p2p_world_size}_{use_shmem}",
    mangle=True,
)
def combine_async_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    src_info_layout: TensorLayout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    use_shmem: Bool = True,
](
    input_tokens: TileTensor[
        input_type, input_tokens_layout, ImmutExternalOrigin
    ],
    src_info: TileTensor[DType.int32, src_info_layout, ImmutExternalOrigin],
    send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_buf_ptrs: InlineArray[
        UnsafePointer[UInt8, MutExternalOrigin], p2p_world_size
    ],
    recv_count_ptrs: InlineArray[
        UnsafePointer[UInt64, MutExternalOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
):
    """
    Send tokens to the original rank based on the src_info tensor.
    This kernel utilizes the non-blocking SHMEM API, and would return
    immediately after initiating the communication. The communication is
    considered complete after calling the `combine_wait_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        msg_bytes: This is the total number of bytes we need to send for each
            token.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        use_shmem: Whether to use the SHMEM API for the communication.

    Args:
        input_tokens: The tokens to be sent back to the original rank.
        src_info: The source token info tensor of shape
            `(n_local_experts * n_ranks * max_tokens_per_rank, 2)`. The first
            column stores a token's position in the original rank's tensor, and
            the second column stores the top-k ID for the token.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True. The underlying buffer is of
            shape `(n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes)`.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world. Each buffer is of shape
            `(max_tokens_per_rank, top_k, msg_bytes)`. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks)`.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
    """

    comptime combine_impl = EPCombineKernel[
        num_threads,
        n_sms,
        top_k,
        n_experts,
        n_ranks,
        msg_bytes,
        max_tokens_per_rank,
        p2p_world_size,
        use_shmem,
    ]

    var atomic_counter = ep_counters.get_combine_async_ptr()
    # Per-rank completion counter for same-node signaling.
    var rank_completion_counter = atomic_counter + 2 * n_experts

    combine_impl.send_tokens_back(
        input_tokens,
        src_info,
        send_buf_p,
        recv_buf_ptrs,
        recv_count_ptrs,
        atomic_counter,
        rank_completion_counter,
        my_rank,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_combine_wait_{output_type}_{num_threads}_{n_sms}_{top_k}_{n_experts}_{n_ranks}_{msg_bytes}_{max_tokens_per_rank}",
    mangle=True,
)
def combine_wait_kernel[
    output_type: DType,
    num_threads: Int,
    output_tokens_layout: TensorLayout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    router_weights_wrapper: Optional[router_weights_wrapper_type] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    output_tokens: TileTensor[
        output_type, output_tokens_layout, MutExternalOrigin
    ],
    recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
):
    """
    This kernel is called after the `combine_kernel` to complete the
    communication. It will keep polling the receive count buffer, and once the
    count is no longer MAX_FINITE, it can confirm that the communication is
    complete from a remote rank.

    Parameters:
        output_type: The type of the output tokens.
        num_threads: The number of threads in the block.
        output_tokens_layout: The layout of the output tokens.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The number of experts in the device.
        n_ranks: The number of ranks.
        msg_bytes: The number of bytes in the message for each token.
        max_tokens_per_rank: The maximum number of tokens per rank.
        router_weights_wrapper: The wrapper for the router weights. If provided,
            all routed experts' outputs for a token will be weighted and summed.
        elementwise_lambda_fn: Optional output lambda function.

    Args:
        output_tokens: The tensor to store the output tokens.
        recv_buf_p: The pointer to the receive buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(max_tokens_per_rank, top_k, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be
            allocated using `shmem_alloc` if `use_shmem` is True. The underlying
            buffer is of shape `(n_local_experts, n_ranks)`.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
    """

    comptime combine_impl = EPCombineKernel[
        num_threads,
        n_sms,
        top_k,
        n_experts,
        n_ranks,
        msg_bytes,
        max_tokens_per_rank,
        1,  # p2p world size is not used in combine_wait
        use_shmem=False,
    ]

    var atomic_counter = ep_counters.get_combine_wait_ptr()
    var sm_id = block_idx.x

    # The first SM is used for checking if we have received tokens from all the
    # remote ranks.
    if sm_id < combine_impl.n_wait_sms:
        combine_impl.wait_for_all_arrivals(recv_count_p, atomic_counter)

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        combine_impl.reduce_and_copy_to_output[
            output_type,
            router_weights_wrapper,
            elementwise_lambda_fn,
        ](
            output_tokens,
            recv_buf_p,
            atomic_counter,
            my_rank,
        )


# ===-----------------------------------------------------------------------===#
# Fused EP Kernels
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__llvm_arg_metadata(format_handler, `nvvm.grid_constant`)
@__name(
    t"ep_fused_dispatch_{input_type}_{num_threads}_{n_sms}_{n_experts}_{n_ranks}_{max_tokens_per_rank}_{p2p_world_size}_{fused_shared_expert}_{use_shmem}",
    mangle=True,
)
def dispatch_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    topk_ids_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    src_info_layout: TensorLayout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    fused_shared_expert: Bool = False,
    input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
    skip_a2a: Bool = False,
    use_shmem: Bool = True,
    allreduce_world_size: Int = 1,
](
    input_tokens: TileTensor[
        input_type, input_tokens_layout, ImmutExternalOrigin
    ],
    topk_ids: TileTensor[DType.int32, topk_ids_layout, ImmutExternalOrigin],
    format_handler: token_fmt_type,
    row_offsets: TileTensor[
        DType.uint32, row_offsets_layout, MutExternalOrigin
    ],
    expert_ids: TileTensor[DType.int32, expert_ids_layout, MutExternalOrigin],
    src_info: TileTensor[DType.int32, src_info_layout, MutExternalOrigin],
    send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_buf_ptrs: InlineArray[
        UnsafePointer[UInt8, MutExternalOrigin], p2p_world_size
    ],
    recv_count_ptrs: InlineArray[
        UnsafePointer[UInt64, MutExternalOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
):
    """
    Fused dispatch kernel that combines dispatch_async and dispatch_wait
    functionality in a single kernel launch.

    This kernel dispatches tokens to experts on remote ranks based on the top-k
    expert IDs, then waits for all tokens to arrive and aggregates them for
    grouped matmul computation.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        topk_ids_layout: The layout of the top-k expert IDs.
        row_offsets_layout: The layout of the row offsets.
        expert_ids_layout: The layout of the expert IDs.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        token_fmt_type: Type conforming to TokenFormat trait that defines the
            token encoding scheme.
        fused_shared_expert: Whether to pack the shared expert inputs with the
            routed experts' inputs. When enabled, input_tokens is used as the
            shared expert inputs.
        input_scales_wrapper: The wrapper for the input scales.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to use the SHMEM API for the communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).

    Args:
        input_tokens: The input tokens to be dispatched. Also used as shared
            expert inputs when fused_shared_expert is True.
        topk_ids: The top-k expert IDs for each token.
        format_handler: Instance of token_fmt_type that performs token decoding
            and manages output tensor writes.
        row_offsets: The row offsets to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        expert_ids: The expert IDs to be updated. Will be consumed by the
            `grouped_matmul` kernel.
        src_info: The source token info to be updated. Once the expert
            computation is complete, tokens will be sent back to the original
            rank using information in this tensor.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world.
        ep_counters: EP atomic counters for kernel synchronization.
        my_rank: The rank of the current device.
    """

    comptime dispatch_impl = EPDispatchKernel[
        num_threads,
        n_sms,
        n_experts,
        n_ranks,
        max_tokens_per_rank,
        p2p_world_size,
        token_fmt_type,
        use_shmem,
        fused_shared_expert,
        skip_a2a,
    ]

    comptime _allreduce_world_size = UInt32(allreduce_world_size)

    var num_tokens = UInt32(input_tokens.dim(0))
    var shared_expert_token_count = UInt32(0)
    comptime if fused_shared_expert:
        shared_expert_token_count = num_tokens
        comptime if skip_a2a:
            shared_expert_token_count = (
                num_tokens + _allreduce_world_size - UInt32(my_rank) - 1
            ) // _allreduce_world_size

    # ===== dispatch_async =====
    var async_atomic_counter = ep_counters.get_dispatch_async_ptr()
    var expert_reserved_counter = async_atomic_counter
    var expert_finished_counter = async_atomic_counter + n_experts
    var rank_completion_counter = async_atomic_counter + 2 * n_experts

    var wait_atomic_counter = ep_counters.get_dispatch_wait_ptr()
    var my_p2p_rank = my_rank % Int32(p2p_world_size)

    with PDL():
        if block_idx.x < dispatch_impl.n_signal_sms:
            dispatch_impl.monitor_and_signal_completion(
                topk_ids,
                recv_count_ptrs,
                expert_reserved_counter,
                expert_finished_counter,
                rank_completion_counter,
                my_rank,
            )
            comptime if fused_shared_expert:
                # Skip signaling if there are no tokens for shared experts.
                if shared_expert_token_count > 0:
                    barrier()
                    if thread_idx.x == 0:
                        _ = Atomic[scope=DEVICE_SCOPE].fetch_add[
                            ordering=Ordering.RELEASE
                        ](
                            wait_atomic_counter
                            + dispatch_impl.send_buf_ready_offset,
                            1,
                        )
        else:
            dispatch_impl.copy_and_send_tokens[input_scales_wrapper](
                input_tokens,
                topk_ids,
                send_buf_p,
                recv_buf_ptrs,
                expert_reserved_counter,
                expert_finished_counter,
                my_rank,
            )

        # ===== dispatch_wait =====
        if block_idx.x >= dispatch_impl.n_dispatch_wait_comm_sms:
            dispatch_impl.wait_for_arrivals_and_compute_offsets(
                format_handler,
                row_offsets,
                expert_ids,
                recv_count_ptrs[my_p2p_rank],
                wait_atomic_counter,
                my_rank,
                shared_expert_token_count,
            )
        else:
            format_handler.init_smem_resources()
            comptime if fused_shared_expert:
                var _send_buf_p = send_buf_p
                comptime if skip_a2a:
                    var rank_offset = UInt32(my_rank) * (
                        num_tokens // _allreduce_world_size
                    ) + min(
                        UInt32(my_rank),
                        num_tokens % _allreduce_world_size,
                    )
                    _send_buf_p += rank_offset * UInt32(
                        token_fmt_type.msg_size()
                    )

                dispatch_impl.pack_shared_expert_inputs(
                    format_handler,
                    _send_buf_p,
                    wait_atomic_counter + dispatch_impl.send_buf_ready_offset,
                    Int(shared_expert_token_count),
                )

            dispatch_impl.copy_received_tokens_to_output(
                format_handler,
                row_offsets,
                src_info,
                recv_buf_ptrs[my_p2p_rank],
                wait_atomic_counter,
                my_rank,
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_combine_{input_type}_{num_threads}_{n_sms}_{top_k}_{n_experts}_{n_ranks}_{msg_bytes}_{max_tokens_per_rank}_{p2p_world_size}_{fused_shared_expert}_{use_shmem}",
    mangle=True,
)
def combine_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: TensorLayout,
    src_info_layout: TensorLayout,
    output_tokens_layout: TensorLayout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    router_weights_wrapper: Optional[router_weights_wrapper_type] = None,
    fused_shared_expert: Bool = False,
    epilogue_fn: Optional[elementwise_epilogue_type] = None,
    skip_a2a: Bool = False,
    use_shmem: Bool = True,
    allreduce_world_size: Int = 1,
](
    input_tokens: TileTensor[
        input_type, input_tokens_layout, ImmutExternalOrigin
    ],
    src_info: TileTensor[DType.int32, src_info_layout, ImmutExternalOrigin],
    output_tokens: TileTensor[
        input_type, output_tokens_layout, MutExternalOrigin
    ],
    send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_buf_ptrs: InlineArray[
        UnsafePointer[UInt8, MutExternalOrigin], p2p_world_size
    ],
    recv_count_ptrs: InlineArray[
        UnsafePointer[UInt64, MutExternalOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    topk_ids_p: Optional[UnsafePointer[Int32, ImmutExternalOrigin]],
    my_rank: Int32,
):
    """
    Fused combine kernel that combines combine_async and combine_wait
    functionality in a single kernel launch.

    This kernel sends processed tokens back to their original ranks, then waits
    for all tokens to arrive and computes the weighted sum of routed expert
    outputs for each token.

    For fused_shared_expert mode, the shared expert outputs are added to the
    reduced routed expert outputs using an elementwise lambda. This requires
    router_weights_wrapper to be provided (output must be reduced).

    Parameters:
        input_type: The type of the input/output tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        src_info_layout: The layout of the source token info.
        output_tokens_layout: The layout of the output tokens.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        msg_bytes: The number of bytes per token message.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        router_weights_wrapper: The wrapper for the router weights. If provided,
            all routed experts' outputs for a token will be weighted and summed.
            REQUIRED when fused_shared_expert is True.
        fused_shared_expert: Whether to add the shared expert's output to the
            combined output. Requires router_weights_wrapper to be provided.
        epilogue_fn: Optional elementwise epilogue function applied after
            computing combined output. If provided, this function is called with
            coordinates and values instead of directly storing to output.
        skip_a2a: Whether to skip the A2A communication. If true, we will only
            send tokens within the current device.
        use_shmem: Whether to use the SHMEM API for the communication.
        allreduce_world_size: The world size of the allreduce operation. Only
            needed for skip_a2a. Used to calculate the workload distribution for
            the shared expert (if has one).
    Args:
        input_tokens: The tokens to be sent back to the original rank.
        src_info: The source token info tensor.
        output_tokens: The tensor to store the output tokens.
        send_buf_p: The pointer to the send buffer. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world. The local device's buffer at index
            my_p2p_rank is used for both send (combine_async) and receive
            (combine_wait) operations.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world. The local device's buffer at index
            my_p2p_rank is used for receive count tracking.
        ep_counters: EP atomic counters for kernel synchronization.
        topk_ids_p: Pointer to the top-k IDs for each token, only required if
            skip_a2a is True.
        my_rank: The rank of the current device.

    """

    comptime if fused_shared_expert:
        comptime assert router_weights_wrapper, (
            "EP combine_kernel: fused_shared_expert requires "
            "router_weights_wrapper to be provided. Cannot add shared expert "
            "output to non-reduced routed expert outputs."
        )
    comptime _allreduce_world_size = UInt32(allreduce_world_size)

    comptime combine_impl = EPCombineKernel[
        num_threads,
        n_sms,
        top_k,
        n_experts,
        n_ranks,
        msg_bytes,
        max_tokens_per_rank,
        p2p_world_size,
        use_shmem,
        fused_shared_expert,
        skip_a2a,
    ]

    # ===== combine_async =====
    var async_atomic_counter = ep_counters.get_combine_async_ptr()
    var rank_completion_counter = async_atomic_counter + 2 * n_experts

    with PDL():
        combine_impl.send_tokens_back(
            input_tokens,
            src_info,
            send_buf_p,
            recv_buf_ptrs,
            recv_count_ptrs,
            async_atomic_counter,
            rank_completion_counter,
            my_rank,
        )

        # ===== combine_wait =====
        var wait_atomic_counter = ep_counters.get_combine_wait_ptr()
        var my_p2p_rank = my_rank % Int32(p2p_world_size)

        if block_idx.x < combine_impl.n_wait_sms:
            combine_impl.wait_for_all_arrivals(
                recv_count_ptrs[my_p2p_rank], wait_atomic_counter
            )
        else:
            # Create an elementwise lambda that adds shared expert output if enabled
            comptime if fused_shared_expert:
                comptime assert (
                    input_tokens.flat_rank >= 2
                ), "input_tokens expects rank >= 2"
                comptime assert (
                    output_tokens.flat_rank >= 2
                ), "output_tokens expects rank >= 2"
                comptime hid_dim = input_tokens.static_shape[1]

                @always_inline
                @parameter
                def add_shared_expert_output[
                    dtype: DType, width: SIMDSize, *, alignment: Int = 1
                ](
                    idx: IndexList[2], combined_val: SIMD[dtype, width]
                ) capturing:
                    var shared_expert_val = SIMD[dtype, width]()

                    comptime if combine_impl.skip_a2a:
                        var num_tokens = UInt32(output_tokens.dim(0))
                        var shared_expert_token_count = (
                            num_tokens
                            + _allreduce_world_size
                            - UInt32(my_rank)
                            - 1
                        ) // _allreduce_world_size
                        var rank_start = UInt32(my_rank) * (
                            num_tokens // _allreduce_world_size
                        ) + min(
                            UInt32(my_rank),
                            num_tokens % _allreduce_world_size,
                        )

                        if (
                            Int(rank_start)
                            <= idx[0]
                            < Int(rank_start + shared_expert_token_count)
                        ):
                            shared_expert_val = input_tokens.load[width=width](
                                (Idx(idx[0] - Int(rank_start)), Idx(idx[1]))
                            ).cast[dtype]()
                    else:
                        shared_expert_val = input_tokens.load[width=width](
                            (Idx(idx[0]), Idx(idx[1]))
                        ).cast[dtype]()

                    var result = combined_val + shared_expert_val

                    comptime if epilogue_fn:
                        comptime epilogue = epilogue_fn.value()
                        epilogue[width=width, alignment=alignment](idx, result)
                    else:
                        output_tokens.store(
                            (Idx(idx[0]), Idx(idx[1])),
                            result.cast[input_type](),
                        )

                combine_impl.reduce_and_copy_to_output[
                    input_type,
                    router_weights_wrapper,
                    add_shared_expert_output,
                ](
                    output_tokens,
                    recv_buf_ptrs[my_p2p_rank],
                    wait_atomic_counter,
                    my_rank,
                    topk_ids_p,
                )

            else:
                combine_impl.reduce_and_copy_to_output[
                    input_type,
                    router_weights_wrapper,
                    epilogue_fn,
                ](
                    output_tokens,
                    recv_buf_ptrs[my_p2p_rank],
                    wait_atomic_counter,
                    my_rank,
                    topk_ids_p,
                )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_{input_dtype}_{output_dtype}", mangle=True)
def fused_silu_kernel[
    output_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    input_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
](
    output_tensor: TileTensor[output_dtype, output_layout, MutExternalOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmutExternalOrigin],
    row_offsets: TileTensor[
        DType.uint32, row_offsets_layout, ImmutExternalOrigin
    ],
):
    """
    This kernel performs the SILU operation for all the MLPs in the EP MoE
    module. We need to manually implement the kernel here is because after the
    EP dispatch phase, the actual number of received tokens is not known to the
    host. This kernel will read the row offsets to determine the actual number of
    received tokens in the input tensor.

    Arguments:
        output_tensor: The output tensor to store the result.
        input_tensor: The input tensor to perform the SILU operation.
        row_offsets: The row offsets to determine the actual number of received tokens.
    """
    comptime accum_dtype = get_accum_type[input_dtype]()
    comptime assert (
        accum_dtype.is_floating_point()
    ), "accum_dtype must be floating point"
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime simd_width = simd_width_of[input_dtype]()

    # This should also make sure the input and output tensors has static shape.
    comptime assert (
        input_dim == output_dim * 2
    ), "Input dimension must be twice the output dimension."
    comptime assert (
        output_dim % simd_width == 0
    ), "Output dimension must be divisible by the SIMD width."

    var tid = thread_idx.x
    var bid = block_idx.x
    var gid = tid + bid * num_threads

    with PDL():
        var num_tokens = row_offsets[row_offsets.static_shape[0] - 1]
        var num_elem = num_tokens * UInt32(output_dim)

        for i in range(
            gid,
            Int(num_elem // UInt32(simd_width)),
            num_threads * num_sms,
        ):
            var m = (i * simd_width) // output_dim
            var k = (i * simd_width) % output_dim

            var gate_proj = input_tensor.load[width=simd_width](
                (Idx(m), Idx(k))
            ).cast[accum_dtype]()
            var up_proj = input_tensor.load[width=simd_width](
                (Idx(m), Idx(k + output_dim))
            ).cast[accum_dtype]()

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            output_tensor.store(
                (Idx(m), Idx(k)), output_val.cast[output_dtype]()
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_fp8_{input_dtype}_{fp8_dtype}", mangle=True)
def fused_silu_fp8_kernel[
    fp8_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
    group_size: Int = 128,
](
    output_tensor: TileTensor[fp8_dtype, output_layout, MutExternalOrigin],
    scales_tensor: TileTensor[scales_dtype, scales_layout, MutExternalOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmutExternalOrigin],
    row_offsets: TileTensor[DType.uint32, offsets_layout, ImmutExternalOrigin],
):
    """
    This kernel performs the SILU operation for all the MLPs in the EP MoE
    module. We need to manually implement the kernel here is because after the
    EP dispatch phase, the actual number of received tokens is not known to the
    host. This kernel will read the row offsets to determine the actual number of
    received tokens in the input tensor.

    Once the SILU operation is performed, the output tensor will be quantized to
    the FP8 format. The scales tensor will be stored in a transposed way.

    Arguments:
        output_tensor: The output tensor to store the result.
        scales_tensor: The tensor to store the scales.
        input_tensor: The input tensor to perform the SILU operation.
        row_offsets: The row offsets to determine the actual number of received tokens.
    """
    comptime accum_dtype = get_accum_type[input_dtype]()
    comptime assert (
        accum_dtype.is_floating_point()
    ), "accum_dtype must be floating point"
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert (
        scales_tensor.flat_rank >= 2
    ), "scales_tensor must be at least 2D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime simd_width = simd_width_of[input_dtype]()

    comptime assert (
        input_dim == output_dim * 2
    ), "Input dimension must be twice the output dimension."
    comptime assert (
        output_dim % simd_width == 0
    ), "Output dimension must be divisible by the SIMD width."

    comptime n_threads_per_group = group_size // simd_width
    comptime assert (
        WARP_SIZE % n_threads_per_group == 0
    ), "Each warp must process a multiple of quantization groups"
    comptime fp8_max_t = Scalar[fp8_dtype].MAX_FINITE.cast[accum_dtype]()

    # Scatter processing of a single token across different thread blocks
    # to improve the memory access performance.
    var global_warp_id = block_idx.x + warp_id() * num_sms
    var gid = lane_id() + global_warp_id * WARP_SIZE

    with PDL():
        var num_tokens = row_offsets[row_offsets.static_shape[0] - 1]
        var num_elem = num_tokens * UInt32(output_dim)

        for i in range(
            gid,
            Int(num_elem // UInt32(simd_width)),
            num_threads * num_sms,
        ):
            var m = (i * simd_width) // output_dim
            var k = (i * simd_width) % output_dim

            var gate_proj = input_tensor.load[width=simd_width](
                (Idx(m), Idx(k))
            ).cast[accum_dtype]()
            var up_proj = input_tensor.load[width=simd_width](
                (Idx(m), Idx(k + output_dim))
            ).cast[accum_dtype]()

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            # Quantization logic.
            var thread_max = abs(output_val).reduce_max()
            var group_max = warp.lane_group_max[n_threads_per_group](thread_max)
            var scale_factor = max(group_max, 1e-4) / fp8_max_t
            output_val = (output_val / scale_factor).clamp(
                -fp8_max_t, fp8_max_t
            )

            output_tensor.store((Idx(m), Idx(k)), output_val.cast[fp8_dtype]())

            # The first thread in each group stores the scale factor.
            if umod(lane_id(), n_threads_per_group) == 0:
                scales_tensor.store(
                    (Idx(k // group_size), Idx(m)),
                    scale_factor.cast[scales_dtype](),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"ep_fused_silu_nvfp4_{input_dtype}_{fp4_dtype}", mangle=True)
def fused_silu_nvfp4_kernel[
    fp4_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    input_scales_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
](
    output_tensor: TileTensor[fp4_dtype, output_layout, MutExternalOrigin],
    scales_tensor: TileTensor[scales_dtype, scales_layout, MutExternalOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmutExternalOrigin],
    row_offsets: TileTensor[DType.uint32, offsets_layout, ImmutExternalOrigin],
    scales_offsets: TileTensor[
        DType.uint32, scales_offsets_layout, ImmutExternalOrigin
    ],
    input_scales: TileTensor[
        DType.float32, input_scales_layout, ImmutExternalOrigin
    ],
):
    """
    This kernel performs the SILU operation for all the MLPs in the EP MoE
    module. We need to manually implement the kernel here is because after the
    EP dispatch phase, the actual number of received tokens is not known to the
    host. This kernel will read the row offsets to determine the actual number of
    received tokens in the input tensor.

    Once the SILU operation is performed, the output tensor will be quantized to
    the NVFP4 format. The scales tensor will be padded and zero-filled.

    Arguments:
        output_tensor: The output tensor to store the result.
        scales_tensor: The tensor to store the scales.
        input_tensor: The input tensor to perform the SILU operation.
        row_offsets: The row offsets to determine the actual number of received tokens.
        scales_offsets: The offsets to determine the position of the scales tiles.
        input_scales: Per-expert input scale factors.
    """
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert scales_tensor.flat_rank == 5, "scales_tensor must be 5D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime assert scales_offsets.flat_rank == 1, "scales_offsets must be 1D"
    comptime assert input_scales.flat_rank == 1, "input_scales must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime hidden_size = output_dim * 2
    comptime src_width = 8
    comptime byte_width = src_width // 2
    comptime NUM_THREADS_PER_SF = NVFP4_SF_VECTOR_SIZE // src_width
    comptime scales_simds_per_tok = hidden_size // (
        NVFP4_SF_VECTOR_SIZE * SF_ATOM_K
    )
    comptime n_threads_per_token = hidden_size // src_width

    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be four times the packed output dimension."
    comptime assert (
        hidden_size % (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "Hidden size must be divisible by (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)."

    comptime n_groups = scales_offsets.static_shape[0]
    comptime n_sms_per_group = num_sms // n_groups
    comptime assert (
        n_groups <= num_sms
    ), "num_sms must be >= number of expert groups."
    comptime assert (
        input_scales.static_shape[0] == n_groups
    ), "input_scales must match number of expert groups."

    var tid = thread_idx.x
    var sm_id = block_idx.x
    var group_id, sm_id_in_group = udivmod(sm_id, n_sms_per_group)
    var tid_in_group = (
        lane_id()
        + sm_id_in_group * WARP_SIZE
        + warp_id() * WARP_SIZE * n_sms_per_group
    )
    if group_id >= n_groups:
        return

    with PDL():
        var expert_start = Int(row_offsets[group_id])
        var expert_end = Int(row_offsets[group_id + 1])
        var expert_m = expert_end - expert_start
        var scales_block_id = expert_start // SF_MN_GROUP_SIZE + Int(
            scales_offsets[group_id]
        )

        var _scales_tensor = TileTensor[
            scales_dtype, scales_layout, MutExternalOrigin
        ](
            ptr=scales_tensor.ptr_at_offset(
                (Idx(scales_block_id), Idx(0), Idx(0), Idx(0), Idx(0))
            ),
            layout=scales_tensor.layout,
        )

        var tensor_sf = rebind[Float32](input_scales[group_id])

        for group_linear in range(
            tid_in_group,
            n_threads_per_token * expert_m,
            num_threads * n_sms_per_group,
        ):
            var token_idx, hid_idx = udivmod(group_linear, n_threads_per_token)
            var m = expert_start + token_idx
            var k = hid_idx * src_width

            var gate_proj = input_tensor.load[width=src_width](
                (Idx(m), Idx(k))
            ).cast[accum_dtype]()
            var up_proj = input_tensor.load[width=src_width](
                (Idx(m), Idx(k + hidden_size))
            ).cast[accum_dtype]()

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            # Quantization logic (NVFP4).
            var thread_max = abs(output_val).reduce_max().cast[DType.float32]()
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )
            var scale_factor = tensor_sf * (group_max * recip(Float32(6.0)))

            # NOTE: NVFP4 uses FP8-UE4M3 format for the scale factor but we know
            # that scale_factor is always positive, so we can use E4M3 instead.
            var fp8_scale_factor = scale_factor.cast[scales_dtype]()
            var output_scale = Float32(0.0)
            if group_max != 0:
                output_scale = recip(
                    fp8_scale_factor.cast[DType.float32]() * recip(tensor_sf)
                )

            var input_f32 = output_val.cast[DType.float32]() * output_scale
            var output_vector = bitcast[fp4_dtype, byte_width](
                cast_fp32_to_fp4e2m1(input_f32)
            )
            output_tensor.store((Idx(m), Idx(k // 2)), output_vector)

            # The first thread in each group stores the scale factor.
            if tid % NUM_THREADS_PER_SF == 0:
                set_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE](
                    _scales_tensor,
                    token_idx,
                    k,
                    fp8_scale_factor,
                )

        # Zero pad the scales tensor to satisfy the grouped matmul requirement.
        if expert_m % SF_MN_GROUP_SIZE != 0:
            var tokens_to_zero_pad = SF_MN_GROUP_SIZE - (
                expert_m % SF_MN_GROUP_SIZE
            )
            for i in range(
                tid_in_group,
                scales_simds_per_tok * tokens_to_zero_pad,
                num_threads * n_sms_per_group,
            ):
                var token_idx, scale_simd_idx = udivmod(i, scales_simds_per_tok)
                set_scale_factor[SF_VECTOR_SIZE=1](
                    _scales_tensor,
                    token_idx + expert_m,
                    scale_simd_idx * SF_ATOM_K,
                    SIMD[scales_dtype, SF_ATOM_K](0.0),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"ep_fused_silu_nvfp4_interleaved_{input_dtype}_{fp4_dtype}", mangle=True
)
def fused_silu_nvfp4_interleaved_kernel[
    fp4_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    input_scales_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
](
    output_tensor: TileTensor[fp4_dtype, output_layout, MutExternalOrigin],
    scales_tensor: TileTensor[scales_dtype, scales_layout, MutExternalOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmutExternalOrigin],
    row_offsets: TileTensor[DType.uint32, offsets_layout, ImmutExternalOrigin],
    scales_offsets: TileTensor[
        DType.uint32, scales_offsets_layout, ImmutExternalOrigin
    ],
    input_scales: TileTensor[
        DType.float32, input_scales_layout, ImmutExternalOrigin
    ],
):
    """SwiGLU + NVFP4 quantization for interleaved gate/up layout.

    Variant of fused_silu_nvfp4_kernel that consumes inputs in the
    `[gate_0, up_0, gate_1, up_1, ...]` interleaved layout produced by
    permuting the MoE up-projection weight on the N axis with
    `σ(2i)=i, σ(2i+1)=H+i`. Used by `grouped_matmul_swiglu_nvfp4_dispatch`'s
    fallback path for tile sizes that cannot fuse SwiGLU+quant in the
    matmul epilogue (BN < 32).

    The only difference from `fused_silu_nvfp4_kernel` is the load pattern
    in the inner loop: instead of loading gate from `[k, k+8)` and up from
    `[k+H, k+H+8)`, this loads a 16-wide chunk at `[2k, 2k+16)` and
    stride-2 splits it into gate (even lanes) and up (odd lanes). All
    downstream steps (SwiGLU, two-thread-per-SF reduction, scale math,
    packed nibble store, trailing zero-pad to SF_MN_GROUP_SIZE) are
    identical.
    """
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert scales_tensor.flat_rank == 5, "scales_tensor must be 5D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime assert scales_offsets.flat_rank == 1, "scales_offsets must be 1D"
    comptime assert input_scales.flat_rank == 1, "input_scales must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime hidden_size = output_dim * 2
    comptime src_width = 8
    comptime byte_width = src_width // 2
    comptime NUM_THREADS_PER_SF = NVFP4_SF_VECTOR_SIZE // src_width
    comptime scales_simds_per_tok = hidden_size // (
        NVFP4_SF_VECTOR_SIZE * SF_ATOM_K
    )
    comptime n_threads_per_token = hidden_size // src_width

    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be twice the unpacked output dimension."
    comptime assert (
        hidden_size % (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "Hidden size must be divisible by (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)."

    comptime n_groups = scales_offsets.static_shape[0]
    comptime n_sms_per_group = num_sms // n_groups
    comptime assert (
        n_groups <= num_sms
    ), "num_sms must be >= number of expert groups."
    comptime assert (
        input_scales.static_shape[0] == n_groups
    ), "input_scales must match number of expert groups."

    var tid = thread_idx.x
    var sm_id = block_idx.x
    var group_id, sm_id_in_group = udivmod(sm_id, n_sms_per_group)
    var tid_in_group = (
        lane_id()
        + sm_id_in_group * WARP_SIZE
        + warp_id() * WARP_SIZE * n_sms_per_group
    )
    if group_id >= n_groups:
        return

    with PDL():
        var expert_start = Int(row_offsets[group_id])
        var expert_end = Int(row_offsets[group_id + 1])
        var expert_m = expert_end - expert_start
        var scales_block_id = expert_start // SF_MN_GROUP_SIZE + Int(
            scales_offsets[group_id]
        )

        var _scales_tensor = TileTensor[
            scales_dtype, scales_layout, MutExternalOrigin
        ](
            ptr=scales_tensor.ptr_at_offset(
                (Idx(scales_block_id), Idx(0), Idx(0), Idx(0), Idx(0))
            ),
            layout=scales_tensor.layout,
        )

        var tensor_sf = rebind[Float32](input_scales[group_id])

        for group_linear in range(
            tid_in_group,
            n_threads_per_token * expert_m,
            num_threads * n_sms_per_group,
        ):
            var token_idx, hid_idx = udivmod(group_linear, n_threads_per_token)
            var m = expert_start + token_idx
            var k = hid_idx * src_width

            # Interleaved load: 16 contiguous BF16 cols at 2k; even/odd split
            # into gate/up. This is the only line that differs from
            # fused_silu_nvfp4_kernel.
            var pair = input_tensor.load[width=2 * src_width](
                (Idx(m), Idx(2 * k))
            ).cast[accum_dtype]()
            var gate_proj = SIMD[accum_dtype, src_width](
                pair[0],
                pair[2],
                pair[4],
                pair[6],
                pair[8],
                pair[10],
                pair[12],
                pair[14],
            )
            var up_proj = SIMD[accum_dtype, src_width](
                pair[1],
                pair[3],
                pair[5],
                pair[7],
                pair[9],
                pair[11],
                pair[13],
                pair[15],
            )

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            var thread_max = abs(output_val).reduce_max().cast[DType.float32]()
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )
            var scale_factor = tensor_sf * (group_max * recip(Float32(6.0)))
            var fp8_scale_factor = scale_factor.cast[scales_dtype]()
            var output_scale = Float32(0.0)
            if group_max != 0:
                output_scale = recip(
                    fp8_scale_factor.cast[DType.float32]() * recip(tensor_sf)
                )

            var input_f32 = output_val.cast[DType.float32]() * output_scale
            var output_vector = bitcast[fp4_dtype, byte_width](
                cast_fp32_to_fp4e2m1(input_f32)
            )
            output_tensor.store((Idx(m), Idx(k // 2)), output_vector)

            if tid % NUM_THREADS_PER_SF == 0:
                set_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE](
                    _scales_tensor,
                    token_idx,
                    k,
                    fp8_scale_factor,
                )

        # Trailing scale-tile zero pad — same logic as fused_silu_nvfp4_kernel.
        if expert_m % SF_MN_GROUP_SIZE != 0:
            var tokens_to_zero_pad = SF_MN_GROUP_SIZE - (
                expert_m % SF_MN_GROUP_SIZE
            )
            for i in range(
                tid_in_group,
                scales_simds_per_tok * tokens_to_zero_pad,
                num_threads * n_sms_per_group,
            ):
                var token_idx, scale_simd_idx = udivmod(i, scales_simds_per_tok)
                set_scale_factor[SF_VECTOR_SIZE=1](
                    _scales_tensor,
                    token_idx + expert_m,
                    scale_simd_idx * SF_ATOM_K,
                    SIMD[scales_dtype, SF_ATOM_K](0.0),
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"fused_silu_mxfp4_{input_dtype}_{fp4_dtype}", mangle=True)
def fused_silu_mxfp4_kernel[
    fp4_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    offsets_layout: TensorLayout,
    num_threads: Int,
    num_sms: Int,
](
    output_tensor: TileTensor[fp4_dtype, output_layout, MutExternalOrigin],
    scales_tensor: TileTensor[scales_dtype, scales_layout, MutExternalOrigin],
    input_tensor: TileTensor[input_dtype, input_layout, ImmutExternalOrigin],
    row_offsets: TileTensor[DType.uint32, offsets_layout, ImmutExternalOrigin],
):
    """
    This kernel performs the SILU operation for all the MLPs in the EP MoE
    module. We need to manually implement the kernel here is because after the
    EP dispatch phase, the actual number of received tokens is not known to the
    host. This kernel will read the row offsets to determine the actual number of
    received tokens in the input tensor.

    Once the SILU operation is performed, the output tensor will be quantized to
    the NVFP4 format. The scales tensor will be padded and zero-filled.

    Arguments:
        output_tensor: The output tensor to store the result.
        scales_tensor: The tensor to store the scales.
        input_tensor: The input tensor to perform the SILU operation.
        row_offsets: The row offsets to determine the actual number of received tokens.
        scales_offsets: The offsets to determine the position of the scales tiles.
        input_scales: Per-expert input scale factors.
    """
    comptime accum_dtype = DType.float32
    comptime assert (
        output_tensor.flat_rank >= 2
    ), "output_tensor must be at least 2D"
    comptime assert scales_tensor.flat_rank == 2, "scales_tensor must be 2D"
    comptime assert (
        input_tensor.flat_rank >= 2
    ), "input_tensor must be at least 2D"
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be 1D"
    comptime input_dim = input_tensor.static_shape[1]
    comptime output_dim = output_tensor.static_shape[1]
    comptime hidden_size = output_dim * 2
    comptime src_width = 8
    comptime byte_width = src_width // 2
    comptime NUM_THREADS_PER_SF = MXFP4_SF_VECTOR_SIZE // src_width

    comptime assert (
        input_dim == hidden_size * 2
    ), "Input dimension must be four times the packed output dimension."
    comptime assert (
        hidden_size % MXFP4_SF_VECTOR_SIZE == 0
    ), "Hidden size must be divisible by MXFP4_SF_VECTOR_SIZE."

    # Scatter processing of a single token across different thread blocks
    # to improve the memory access performance.
    var global_warp_id = block_idx.x + warp_id() * num_sms
    var gid = lane_id() + global_warp_id * WARP_SIZE

    with PDL():
        var num_tokens = row_offsets[row_offsets.static_shape[0] - 1]
        var num_elem = num_tokens * UInt32(output_dim) * 2

        for i in range(
            gid,
            Int(num_elem // UInt32(src_width)),
            num_threads * num_sms,
        ):
            var m = (i * src_width) // (output_dim * 2)
            var k = (i * src_width) % (output_dim * 2)

            var gate_proj = input_tensor.load[width=src_width](
                (Idx(m), Idx(k))
            ).cast[accum_dtype]()
            var up_proj = input_tensor.load[width=src_width](
                (Idx(m), Idx(k + hidden_size))
            ).cast[accum_dtype]()

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            # Quantization logic (MXFP4).
            var thread_max = abs(output_val).reduce_max().cast[DType.float32]()
            var group_max = warp.lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )

            # Use MXFP4 even-mode rounding for the E8M0 scale.
            var fp8_scale_factor = compute_mxfp4_even_scale(group_max).cast[
                scales_dtype
            ]()
            var scale_f32 = fp8_scale_factor.cast[DType.float32]()

            # The first thread in each group stores the scale factor.
            if i % NUM_THREADS_PER_SF == 0:
                scales_tensor.store(
                    (Idx(m), Idx(k // MXFP4_SF_VECTOR_SIZE)), fp8_scale_factor
                )

            var output_vector = bitcast[fp4_dtype, byte_width](
                cast_float_to_fp4e2m1_amd(output_val, scale_f32)
            )
            output_tensor.store((Idx(m), Idx(k // 2)), output_vector)
