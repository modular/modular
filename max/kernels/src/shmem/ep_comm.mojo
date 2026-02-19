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

from math import align_up, ceildiv
from os.atomic import Atomic, Consistency
from sys import is_amd_gpu, is_nvidia_gpu
from sys.info import CompilationTarget, align_of, simd_width_of, size_of

from linalg.fp4_utils import (
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    cast_fp32_to_fp4e2m1,
    set_scale_factor,
)

import gpu.primitives.warp as warp
from collections import OptionalReg
from gpu import (
    PDL,
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from gpu.host import get_gpu_target
from gpu.intrinsics import Scope, load_acquire, store_release, threadfence
from gpu.sync import syncwarp
from layout import Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from layout.int_tuple import (
    UNKNOWN_VALUE,
    IntTuple,
    _get_index_type,
    _get_layout_type,
)
from math import exp, recip
from memory import stack_allocation
from memory.unsafe import bitcast
from shmem import SHMEM_SIGNAL_SET, SHMEMScope, shmem_put_nbi, shmem_signal_op

from utils.index import IndexList, StaticTuple
from utils.numerics import get_accum_type

from builtin.device_passable import DevicePassable

comptime RtTuple_2 = RuntimeTuple[
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE), element_type = DType.int32
]
comptime RtTuple_3 = RuntimeTuple[
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    element_type = DType.int32,
]
comptime RtTuple_4 = RuntimeTuple[
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    element_type = DType.int32,
]

comptime elementwise_epilogue_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
](IndexList[2], SIMD[dtype, width]) capturing -> None

comptime router_weights_wrapper_type = fn[width: Int](
    token_idx: Int, topk_id: Int
) capturing -> SIMD[DType.float32, width]


comptime input_scales_wrapper_type = fn[dtype: DType](Int,) capturing -> Scalar[
    dtype
]

comptime EP_DATA_READY_FLAG = 1 << 10

# Maximum number of GPUs per node for P2P signaling.
# Used to track per-rank expert completion.
comptime MAX_GPUS_PER_NODE = 8


@always_inline
fn _BLOCK_SCOPE() -> StaticString:
    comptime if is_nvidia_gpu():
        return "block"
    elif is_amd_gpu():
        return "workgroup"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString, operation = __get_current_function_name()
        ]()


@always_inline
fn _DEVICE_SCOPE() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString, operation = __get_current_function_name()
        ]()


comptime DEVICE_SCOPE = _DEVICE_SCOPE()
comptime BLOCK_SCOPE = _BLOCK_SCOPE()


@always_inline
fn block_memcpy[
    dst_addr_space: AddressSpace,
    src_addr_space: AddressSpace,
    //,
    num_bytes: Int,
    block_size: Int,
](
    dst_p: UnsafePointer[mut=True, UInt8, address_space=dst_addr_space],
    src_p: UnsafePointer[mut=False, UInt8, address_space=src_addr_space],
    thread_idx: UInt,
) -> None:
    """
    Copies a memory area from source to destination. This function will use the
    vectorized store and load instructions to copy the memory area. User should
    make sure pointers are aligned to the simd width.
    """
    comptime simd_width = simd_width_of[DType.uint8]()
    for i in range(thread_idx, num_bytes // simd_width, block_size):
        dst_p.store[width=simd_width, alignment=simd_width](
            i * simd_width,
            src_p.load[
                width=simd_width,
                alignment=simd_width,
                invariant=True,
            ](i * simd_width),
        )


@always_inline
fn block_prefix_sum[
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
        address_space = AddressSpace.SHARED,
    ]()

    var val = Scalar[dtype](0)
    if thread_idx.x < UInt(num_elements):
        val = _val

    if thread_idx.x < UInt(n_elements_aligned):
        val = warp.prefix_sum(val)
        if lane_id() == UInt(WARP_SIZE - 1):
            warp_prefix_sum[warp_id()] = val
    barrier()

    if warp_id() == 0:
        var warp_sum = Scalar[dtype](0)
        if lane_id() < UInt(n_warps):
            warp_sum = warp_prefix_sum[lane_id()]
        warp_sum = warp.prefix_sum[exclusive=True](warp_sum)
        if lane_id() < UInt(n_warps):
            warp_prefix_sum[lane_id()] = warp_sum
    barrier()

    if thread_idx.x < UInt(num_elements):
        val += warp_prefix_sum[warp_id()]
    barrier()

    return val


@always_inline
@parameter
fn ep_signal_completion[
    p2p_world_size: Int, //, use_shmem: Bool, n_experts_per_device: Int = 0
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

    For same-node signaling, uses normal stores and only issues a store_release
    when the last expert for a destination rank is completed. This reduces the
    number of store_release operations from n_experts to p2p_world_size.
    """

    var my_p2p_world, my_p2p_rank = divmod(my_rank, Int32(p2p_world_size))
    var dst_p2p_world, dst_p2p_rank = divmod(dst_rank, Int32(p2p_world_size))

    # If the target device is on the same node, we can directly write to its
    # receive count buffer.
    if my_p2p_world == dst_p2p_world:
        var dst_p2p_ptr = recv_count_ptrs[dst_p2p_rank] + signal_offset
        var old_count = Atomic[DType.int32, scope=DEVICE_SCOPE].fetch_add[
            ordering = Consistency.MONOTONIC
        ](rank_completion_counter + Int(dst_p2p_rank), 1)

        # If this is the last expert for this destination rank,
        # use store_release to flush all pending stores.
        if old_count < Int32(n_experts_per_device - 1):
            dst_p2p_ptr[] = signal
        else:
            # Technically, this store_release only guarantees the arrival of
            # all experts' messages to the target device. It doesn't guarantee
            # the arrival of the previous experts' signals to the target
            # device. However, this does not matter as we will check the
            # arrival signal individually in the dispatch_wait/combine_wait kernel.
            store_release[scope = Scope.SYSTEM](dst_p2p_ptr, signal)
            # Reset counter for next kernel invocation.
            rank_completion_counter[dst_p2p_rank] = 0
    else:

        @parameter
        if use_shmem:
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
fn get_device_alignment() -> Int:
    comptime gpu_target = get_gpu_target()
    comptime gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
    comptime gpu_alignment = align_of[
        SIMD[DType.uint8, gpu_simd_width], target=gpu_target
    ]()

    return gpu_alignment


trait TokenFormat(DevicePassable, TrivialRegisterPassable):
    comptime hid_dim: Int
    comptime top_k: Int
    comptime alignment: Int
    comptime expert_m_padding: Int

    @always_inline
    @staticmethod
    fn token_size() -> Int:
        "Returns the size of the (quantized) token in bytes."
        ...

    @always_inline
    @staticmethod
    fn src_info_size() -> Int:
        "Returns the size of the source info in bytes. Currently, source info is a single int32 that stores a token's index in the original rank."
        return align_up(size_of[Int32](), Self.alignment)

    @always_inline
    @staticmethod
    fn topk_info_size() -> Int:
        "Returns the size of the top-k info in bytes. Currently, top-k info is an array of uint16 that stores a token's top-k expert IDs."
        return align_up(size_of[UInt16]() * Self.top_k, Self.alignment)

    @always_inline
    @staticmethod
    fn msg_size() -> Int:
        "Returns the size of the message in bytes."
        return Self.token_size() + Self.src_info_size() + Self.topk_info_size()

    @always_inline
    @staticmethod
    fn src_info_offset() -> Int:
        "Returns the offset of the source info in the message."
        return Self.token_size()

    @always_inline
    @staticmethod
    fn topk_info_offset() -> Int:
        "Returns the offset of the top-k info in the message."
        return Self.token_size() + Self.src_info_size()

    @always_inline
    fn pad_expert_offsets(
        self, row_offsets: UnsafePointer[UInt32, MutAnyOrigin]
    ) -> None:
        """
        Pad the offsets to satisfy the grouped matmul alignment requirement.
        """
        pass

    @always_inline
    fn zero_pad_scales[
        block_size: Int, n_sms: Int
    ](
        self, row_offsets: UnsafePointer[UInt32, MutAnyOrigin], sm_id: Int
    ) -> None:
        """
        Zero pad the scales tensor to satisfy the grouped matmul requirement.
        """
        pass

    @always_inline
    @staticmethod
    fn copy_token_to_send_buf[
        src_type: DType,
        block_size: UInt,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type]],
        input_scale: Float32,
    ) -> None:
        "Copy the token to the send buffer. This function needs to be called by all threads in the block."
        ...

    @always_inline
    fn copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, address_space=buf_addr_space],
        token_index: Int,
        expert_id: Int,
        expert_token_index: Int,
    ) -> None:
        "Copy the message to the output tensor. This function needs to be called by all threads in a warp."
        ...


struct BF16TokenFormat[
    output_layout: Layout, //, _hid_dim: Int, _top_k: Int, _alignment: Int = 0
](TokenFormat, TrivialRegisterPassable):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime expert_m_padding = 0

    comptime TensorType = LayoutTensor[
        DType.bfloat16, Self.output_layout, MutAnyOrigin
    ]
    var output_tokens: Self.TensorType

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return String(
            "BF16TokenFormat[output_layout = ",
            String(materialize[Self.output_layout]()),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @always_inline
    fn __init__(out self, output_tokens: Self.TensorType):
        self.output_tokens = output_tokens

    @always_inline
    @staticmethod
    fn token_size() -> Int:
        return align_up(
            Self.hid_dim * size_of[DType.bfloat16](), Self.alignment
        )

    @always_inline
    @staticmethod
    fn copy_token_to_send_buf[
        src_type: DType,
        block_size: UInt,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type]],
        input_scale: Float32,
    ) -> None:
        block_memcpy[Self.hid_dim * size_of[BFloat16](), Int(block_size)](
            buf_p,
            src_p.bitcast[UInt8](),
            thread_idx.x,
        )

    @always_inline
    fn copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, address_space=buf_addr_space],
        token_index: Int,
        expert_id: Int,
        expert_token_index: Int,
    ) -> None:
        comptime bf16_width = simd_width_of[DType.bfloat16]()
        comptime byte_width = bf16_width * size_of[BFloat16]()
        for i in range(lane_id(), Self.hid_dim // bf16_width, WARP_SIZE):
            self.output_tokens.aligned_store[width=bf16_width](
                token_index,
                i * bf16_width,
                bitcast[DType.bfloat16, bf16_width](
                    buf_p.load[
                        width=byte_width,
                        invariant=True,
                        alignment = Self.alignment,
                    ](
                        i * byte_width,
                    )
                ),
            )


struct BlockwiseFP8TokenFormat[
    fp8_dtype: DType,
    scales_dtype: DType,
    output_layout: Layout,
    scales_layout: Layout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](TokenFormat, TrivialRegisterPassable):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime expert_m_padding = 16 // size_of[Self.scales_dtype]()

    comptime TensorType = LayoutTensor[
        Self.fp8_dtype, Self.output_layout, MutAnyOrigin
    ]
    comptime ScalesTensorType = LayoutTensor[
        Self.scales_dtype, Self.scales_layout, MutAnyOrigin
    ]
    var output_tokens: Self.TensorType
    var output_scales: Self.ScalesTensorType

    comptime group_size = 128

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return String(
            "BlockwiseFP8TokenFormat[fp8_dtype = ",
            String(Self.fp8_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", output_layout = ",
            String(materialize[Self.output_layout]()),
            ", scales_layout = ",
            String(materialize[Self.scales_layout]()),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @always_inline
    fn __init__(
        out self,
        output_tokens: Self.TensorType,
        output_scales: Self.ScalesTensorType,
    ):
        self.output_tokens = output_tokens
        self.output_scales = output_scales

    @always_inline
    @staticmethod
    fn fp8_quant_size() -> Int:
        return align_up(
            Self.hid_dim * size_of[Self.fp8_dtype](), Self.alignment
        )

    @always_inline
    @staticmethod
    fn scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by 128"
        return align_up(
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype](),
            Self.alignment,
        )

    @always_inline
    @staticmethod
    fn token_size() -> Int:
        return Self.fp8_quant_size() + Self.scales_size()

    @always_inline
    @staticmethod
    fn scales_offset() -> Int:
        return Self.fp8_quant_size()

    @always_inline
    @staticmethod
    fn copy_token_to_send_buf[
        src_type: DType,
        block_size: UInt,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type]],
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
                width=src_width, alignment = Self.alignment, invariant=True
            ](i * src_width).cast[Self.scales_dtype]()
            var thread_max = abs(loaded_vec).reduce_max()
            var group_max = warp.lane_group_max_and_broadcast[
                n_threads_per_group
            ](thread_max)

            # 1e-4 is taken from DeepEP.
            var scale_factor = max(group_max, 1e-4) / fp8_max_t
            var output_vec = loaded_vec / scale_factor
            output_vec = output_vec.clamp(-fp8_max_t, fp8_max_t)

            buf_p.store[width=byte_width, alignment=byte_width](
                i * byte_width,
                bitcast[DType.uint8, byte_width](
                    output_vec.cast[Self.fp8_dtype]()
                ),
            )

            # The first thread in each group stores the scale factor.
            comptime scale_bytes = size_of[Self.scales_dtype]()
            if lane_id() % UInt(n_threads_per_group) == 0:
                scale_idx = i * src_width // Self.group_size
                buf_p.store[width=scale_bytes, alignment=scale_bytes](
                    Self.scales_offset() + scale_idx * scale_bytes,
                    bitcast[DType.uint8, scale_bytes](scale_factor),
                )

    @always_inline
    fn copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, address_space=buf_addr_space],
        token_index: Int,
        expert_id: Int,
        expert_token_index: Int,
    ) -> None:
        # First we copy the FP8 quants.
        comptime fp8_width = simd_width_of[Self.fp8_dtype]()
        for i in range(lane_id(), Self.hid_dim // fp8_width, WARP_SIZE):
            self.output_tokens.aligned_store[width=fp8_width](
                token_index,
                i * fp8_width,
                bitcast[Self.fp8_dtype, fp8_width](
                    buf_p.load[
                        width=fp8_width,
                        invariant=True,
                        alignment = Self.alignment,
                    ](
                        i * fp8_width,
                    )
                ),
            )

        # Unlike the output tensor, the scales tensor is stored in a transposed way.
        comptime scale_bytes = size_of[Self.scales_dtype]()
        for i in range(lane_id(), Self.hid_dim // Self.group_size, WARP_SIZE):
            self.output_scales.store(
                i,
                token_index,
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


struct NVFP4TokenFormat[
    fp4_dtype: DType,
    scales_dtype: DType,
    output_layout: Layout,
    scales_layout: Layout,
    scales_offset_layout: Layout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int = 0,
](TokenFormat, TrivialRegisterPassable):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment or get_device_alignment()
    comptime expert_m_padding = 0
    comptime group_size = NVFP4_SF_VECTOR_SIZE

    comptime TensorType = LayoutTensor[
        Self.fp4_dtype, Self.output_layout, MutAnyOrigin
    ]
    comptime ScalesTensorType = LayoutTensor[
        Self.scales_dtype, Self.scales_layout, MutAnyOrigin
    ]
    comptime ScalesOffsetTensorType = LayoutTensor[
        DType.uint32, Self.scales_offset_layout, MutAnyOrigin
    ]
    var output_tokens: Self.TensorType
    var output_scales: Self.ScalesTensorType
    var output_scales_offset: Self.ScalesOffsetTensorType

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return String(
            "NVFP4TokenFormat[fp4_dtype = ",
            String(Self.fp4_dtype),
            ", scales_dtype = ",
            String(Self.scales_dtype),
            ", output_layout = ",
            String(materialize[Self.output_layout]()),
            ", scales_layout = ",
            String(materialize[Self.scales_layout]()),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @always_inline
    fn __init__(
        out self,
        output_tokens: Self.TensorType,
        output_scales: Self.ScalesTensorType,
        output_scales_offset: Self.ScalesOffsetTensorType,
    ):
        self.output_tokens = output_tokens
        self.output_scales = output_scales
        self.output_scales_offset = output_scales_offset

    @always_inline
    @staticmethod
    fn fp4_quant_size() -> Int:
        return align_up(Self.hid_dim // 2, Self.alignment)

    @always_inline
    @staticmethod
    fn scales_size() -> Int:
        comptime assert (
            Self.hid_dim % Self.group_size == 0
        ), "hid_dim must be divisible by group_size"
        return align_up(
            Self.hid_dim // Self.group_size * size_of[Self.scales_dtype](),
            Self.alignment,
        )

    @always_inline
    @staticmethod
    fn token_size() -> Int:
        return Self.fp4_quant_size() + Self.scales_size()

    @always_inline
    @staticmethod
    fn scales_offset() -> Int:
        return Self.fp4_quant_size()

    @always_inline
    fn pad_expert_offsets(
        self, row_offsets: UnsafePointer[UInt32, MutAnyOrigin]
    ) -> None:
        """
        Update the output_scales_offset tensor.
        """
        var tid = thread_idx.x
        comptime n_groups = Self.ScalesOffsetTensorType.layout.shape[0].value()
        comptime n_rounds = ceildiv(n_groups, WARP_SIZE)

        if warp_id() == 0:
            var prev_round_blocks_num: UInt32 = 0

            @parameter
            for round_i in range(n_rounds):
                var per_expert_m: UInt32 = 0
                var group_idx = UInt(round_i * WARP_SIZE) + tid
                if group_idx < UInt(n_groups):
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

                if group_idx < UInt(n_groups):
                    self.output_scales_offset.store(
                        IndexList[1](Int(group_idx)),
                        group_scales_start
                        - row_offsets[group_idx] // UInt32(SF_MN_GROUP_SIZE),
                    )

                prev_round_blocks_num = warp.shuffle_idx(
                    group_scales_start + scales_blocks, UInt32(WARP_SIZE - 1)
                )

    @always_inline
    fn zero_pad_scales[
        block_size: Int, n_sms: Int
    ](
        self, row_offsets: UnsafePointer[UInt32, MutAnyOrigin], sm_id: Int
    ) -> None:
        """
        Zero pad the scales tensor to satisfy the grouped matmul requirement.
        """
        comptime n_groups = Self.ScalesOffsetTensorType.layout.shape[0].value()
        comptime n_sms_per_group = n_sms // n_groups

        var tid = thread_idx.x
        var group_id, sm_id_in_group = divmod(sm_id, n_sms_per_group)
        var tid_in_group = UInt(sm_id_in_group * block_size) + tid

        if group_id >= n_groups:
            return

        var per_expert_m = row_offsets[group_id + 1] - row_offsets[group_id]
        if per_expert_m % UInt32(SF_MN_GROUP_SIZE) == 0:
            return
        var tokens_to_zero_pad = UInt32(SF_MN_GROUP_SIZE) - (
            per_expert_m % UInt32(SF_MN_GROUP_SIZE)
        )

        var scales_block_id = (
            row_offsets[group_id] // UInt32(SF_MN_GROUP_SIZE)
            + self.output_scales_offset[group_id]
        )
        var _scales_tensor = Self.ScalesTensorType(
            self.output_scales.ptr_at_offset(
                IndexList[5](Int(scales_block_id), 0, 0, 0, 0)
            ),
        )

        comptime scales_simds_per_tok = Self.hid_dim // (
            NVFP4_SF_VECTOR_SIZE * SF_ATOM_K
        )
        comptime assert (
            Self.hid_dim % (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K) == 0
        ), "hid_dim must be divisible by (NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)"
        for i in range(
            tid_in_group,
            UInt32(scales_simds_per_tok) * tokens_to_zero_pad,
            block_size * n_sms_per_group,
        ):
            token_idx, scale_simd_idx = divmod(i, scales_simds_per_tok)
            set_scale_factor[SF_VECTOR_SIZE=1](
                _scales_tensor,
                token_idx + Int(per_expert_m),
                scale_simd_idx * SF_ATOM_K,
                SIMD[Self.scales_dtype, SF_ATOM_K](0.0),
            )

    @always_inline
    @staticmethod
    fn copy_token_to_send_buf[
        src_type: DType,
        block_size: UInt,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type]],
        input_scale: Float32,
    ) -> None:
        comptime src_width = 8
        comptime byte_width = src_width // 2
        comptime NUM_THREADS_PER_SF = NVFP4_SF_VECTOR_SIZE // src_width

        for i in range(thread_idx.x, Self.hid_dim // src_width, block_size):
            var loaded_vec = src_p.load[
                width=src_width, alignment = Self.alignment, invariant=True
            ](i * src_width)

            # each thread finds maximum value in its local 8 elements
            var thread_max = abs(loaded_vec).reduce_max()
            # find the maximum value among all 16 elements (two threads for 16)
            thread_max = max(warp.shuffle_xor(thread_max, 1), thread_max)
            var group_max = thread_max.cast[DType.float32]()

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
                buf_p.store[width=scale_bytes, alignment=scale_bytes](
                    Self.scales_offset()
                    + i // NUM_THREADS_PER_SF * scale_bytes,
                    bitcast[DType.uint8, scale_bytes](fp8_scale_factor),
                )

            var input_f32 = loaded_vec.cast[DType.float32]() * output_scale
            var output_vector = bitcast[Self.fp4_dtype, byte_width](
                cast_fp32_to_fp4e2m1(input_f32)
            )
            buf_p.store[width=byte_width, alignment=byte_width](
                i * byte_width,
                bitcast[DType.uint8, byte_width](output_vector),
            )

    @always_inline
    fn copy_msg_to_output_tensor[
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        self,
        buf_p: UnsafePointer[mut=False, UInt8, address_space=buf_addr_space],
        token_index: Int,
        expert_id: Int,
        expert_token_index: Int,
    ) -> None:
        # First we copy the FP4 quants.
        comptime fp4_width = simd_width_of[Self.fp4_dtype]()
        comptime quant_bytes = Self.hid_dim // 2
        comptime assert (
            quant_bytes % fp4_width == 0
        ), "quant_bytes must be divisible by fp4_width"

        for i in range(lane_id(), quant_bytes // fp4_width, WARP_SIZE):
            self.output_tokens.aligned_store[width=fp4_width](
                token_index,
                i * fp4_width,
                bitcast[Self.fp4_dtype, fp4_width](
                    buf_p.load[
                        width=fp4_width,
                        invariant=True,
                        alignment = Self.alignment,
                    ](
                        i * fp4_width,
                    )
                ),
            )

        # The scales tensor is a 5D tensor.
        var expert_start_index = token_index - expert_token_index
        var output_scales_offset: UInt32 = 0
        if expert_id != 0:
            output_scales_offset = rebind[UInt32](
                self.output_scales_offset[expert_id]
            )
        var scales_block_id = (
            UInt32(expert_start_index // SF_MN_GROUP_SIZE)
            + output_scales_offset
        )

        var _scales_tensor = Self.ScalesTensorType(
            self.output_scales.ptr_at_offset(
                IndexList[5](Int(scales_block_id), 0, 0, 0, 0)
            )
        )

        comptime n_scales = Self.hid_dim // NVFP4_SF_VECTOR_SIZE
        comptime n_scales_simd = n_scales // SF_ATOM_K
        comptime assert (
            n_scales % SF_ATOM_K == 0
        ), "n_scales must be divisible by SF_ATOM_K"
        comptime scale_bytes = size_of[Self.scales_dtype]() * SF_ATOM_K
        for i in range(lane_id(), n_scales_simd, WARP_SIZE):
            var scale_factors = bitcast[Self.scales_dtype, SF_ATOM_K](
                buf_p.load[
                    width=scale_bytes,
                    invariant=True,
                    alignment=scale_bytes,
                ](
                    Self.scales_offset() + i * scale_bytes,
                )
            )

            # Here we set SF_VECTOR_SIZE=1 because i is already the index
            # of the scale factor (not the index of the fp4 elements).
            set_scale_factor[SF_VECTOR_SIZE=1](
                _scales_tensor,
                expert_token_index,
                i * SF_ATOM_K,
                scale_factors,
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
    - dispatch_wait/combine_async: 2 * n_experts + MAX_GPUS_PER_NODE
    - combine_wait: 2 * n_experts
    """

    var ptr: UnsafePointer[Int32, MutExternalOrigin]
    """Base pointer to the allocated atomic counter memory."""

    comptime device_type: AnyType = Self

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Int32]):
        self.ptr = UnsafePointer[Int32, MutExternalOrigin](ptr)

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return String("EPLocalSyncCounters[n_experts=", Self.n_experts, "]")

    @always_inline
    @staticmethod
    fn dispatch_async_size() -> Int:
        """Returns the size in Int32 elements needed by dispatch_async kernel.
        """
        return 2 * Self.n_experts + MAX_GPUS_PER_NODE

    @always_inline
    @staticmethod
    fn dispatch_wait_size() -> Int:
        """Returns the size in Int32 elements needed by dispatch_wait kernel."""
        return 2 * Self.n_experts + MAX_GPUS_PER_NODE

    @always_inline
    @staticmethod
    fn combine_async_size() -> Int:
        """Returns the size in Int32 elements needed by combine_async kernel."""
        return 2 * Self.n_experts + MAX_GPUS_PER_NODE

    @always_inline
    @staticmethod
    fn combine_wait_size() -> Int:
        """Returns the size in Int32 elements needed by combine_wait kernel."""
        return 2 * Self.n_experts

    @always_inline
    @staticmethod
    fn total_size() -> Int:
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
    fn get_dispatch_async_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
        """Returns pointer to dispatch_async kernel atomic counters.

        Layout:
            [0, n_experts): reserved counters per expert
            [n_experts, 2*n_experts): finished counters per expert
        """
        return self.ptr

    @always_inline
    fn get_dispatch_wait_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
        """Returns pointer to dispatch_wait kernel atomic counters."""
        return self.ptr + Self.dispatch_async_size()

    @always_inline
    fn get_combine_async_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
        """Returns pointer to combine_async kernel atomic counters.

        Note: Returns the same pointer as get_dispatch_wait_ptr() because
        combine_async_kernel reads the offset/count data that dispatch_wait_kernel writes.
        """
        return self.ptr + Self.dispatch_async_size()

    @always_inline
    fn get_combine_wait_ptr(self) -> UnsafePointer[Int32, MutExternalOrigin]:
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
    expert_m_padding: Int = 0,
    fused_shared_expert: Bool = False,
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
        expert_m_padding: The padding size for each expert.
        fused_shared_expert: Whether to pack the shared expert inputs with the
            routed experts' inputs.
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

    comptime recv_layout_static = Layout.row_major(
        Self.n_local_experts,
        Self.n_ranks,
        Self.max_tokens_per_rank,
        Self.msg_bytes,
    )

    @staticmethod
    @always_inline
    fn _get_recv_buf_layout(
        out result: RuntimeLayout[
            Self.recv_layout_static,
            element_type = _get_layout_type(
                Self.recv_layout_static, AddressSpace.GENERIC
            ),
            linear_idx_type = _get_index_type(
                Self.recv_layout_static, AddressSpace.GENERIC
            ),
        ]
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    fn _get_recv_count_layout(
        out result: RuntimeLayout[
            Layout.row_major(Self.n_local_experts, Self.n_ranks),
            element_type = DType.int32,
            linear_idx_type = DType.int32,
        ]
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    fn _get_send_buf_layout(
        out result: RuntimeLayout[
            Layout.row_major(Self.max_tokens_per_rank, Self.msg_bytes),
            element_type = DType.int32,
            linear_idx_type = DType.int32,
        ]
    ):
        return type_of(result)()

    # ===-------------------------------------------------------------------===#
    # Dispatch Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn monitor_and_signal_completion[
        topk_ids_layout: Layout, //
    ](
        topk_ids: LayoutTensor[DType.int32, topk_ids_layout, ImmutAnyOrigin],
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

        comptime assert Self.n_local_experts <= Self.n_warps, (
            "EP dispatch_async: number of experts per rank must be less than or"
            " equal to "
            + String(Self.n_warps)
        )

        var recv_count_layout = Self._get_recv_count_layout()
        var num_tokens = topk_ids.dim[0]()

        var expert_idx = Int32(block_idx.x * UInt(Self.n_warps) + warp_id())
        var expert_count: Int32 = 0

        if expert_idx < Int32(Self.n_experts):
            for i in range(lane_id(), num_tokens * Self.top_k, WARP_SIZE):
                if topk_ids.ptr[i] == expert_idx:
                    expert_count += 1

            expert_count = warp.sum(expert_count)

            if lane_id() == 0:
                # Wait until all the tokens for the expert have been sent.
                while (
                    load_acquire[scope = Scope.GPU](
                        expert_finished_counter + expert_idx
                    )
                    != expert_count
                ):
                    pass

                var dst_rank = expert_idx // Int32(Self.n_local_experts)
                var dst_expert_local_idx = expert_idx % Int32(
                    Self.n_local_experts
                )
                var signal_offset = recv_count_layout(
                    RtTuple_2(Int(dst_expert_local_idx), Int(my_rank))
                )

                ep_signal_completion[
                    Self.use_shmem, n_experts_per_device = Self.n_local_experts
                ](
                    my_rank,
                    dst_rank,
                    recv_count_ptrs,
                    signal_offset,
                    UInt64(expert_count),
                    rank_completion_counter,
                )

                expert_reserved_counter[expert_idx] = 0
                expert_finished_counter[expert_idx] = 0

    @staticmethod
    @always_inline
    fn copy_and_send_tokens[
        input_type: DType,
        input_tokens_layout: Layout,
        topk_ids_layout: Layout,
        //,
        input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
    ](
        input_tokens: LayoutTensor[
            input_type, input_tokens_layout, ImmutAnyOrigin
        ],
        topk_ids: LayoutTensor[DType.int32, topk_ids_layout, ImmutAnyOrigin],
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
        var send_buf_layout = Self._get_send_buf_layout()
        var recv_buf_layout = Self._get_recv_buf_layout()

        var tid = thread_idx.x
        var num_tokens = input_tokens.dim[0]()
        var my_p2p_world, my_p2p_rank = divmod(
            my_rank, Int32(Self.p2p_world_size)
        )

        var input_scale = Float32(1.0)

        @parameter
        if input_scales_wrapper is not None:
            comptime input_scale_fn = input_scales_wrapper.value()
            input_scale = input_scale_fn[DType.float32](0)

        for token_idx in range(
            block_idx.x - UInt(Self.n_signal_sms),
            num_tokens,
            Self.n_dispatch_async_comm_sms,
        ):
            # First, all threads in the block copy the input token to the send
            # buffer.
            var curr_send_buf_ptr = send_buf_p + send_buf_layout(
                RtTuple_2(token_idx, 0)
            )
            var input_tensor_ptr = input_tokens.ptr + input_tokens._offset(
                token_idx, 0
            )
            Self.token_fmt_type.copy_token_to_send_buf[
                input_type, UInt(Self.num_threads)
            ](curr_send_buf_ptr, input_tensor_ptr, input_scale)

            if tid < UInt(Self.top_k):
                # Store all the top-k expert IDs in current token's message.
                # The remote device will use the expert ID to determine a
                # token's top-k id.
                # Cast the expert ID to a 16-bit integer to save space.
                var top_k_idx = topk_ids.load[width=1](token_idx, Int(tid))
                curr_send_buf_ptr.store[
                    width = size_of[UInt16](),
                    alignment = align_of[DType.uint16](),
                ](
                    Self.token_fmt_type.topk_info_offset()
                    + Int(tid * UInt(size_of[UInt16]())),
                    bitcast[DType.uint8, size_of[UInt16]()](UInt16(top_k_idx)),
                )

                # Store the source token index in current token's message.
                if tid == 0:
                    curr_send_buf_ptr.store[
                        width = size_of[Int32](),
                        alignment = align_of[DType.int32](),
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
                var target_expert = topk_ids.load[width=1](token_idx, topk_idx)
                var dst_rank, dst_expert_local_idx = divmod(
                    target_expert, Int32(Self.n_local_experts)
                )
                var dst_p2p_world, dst_p2p_rank = divmod(
                    dst_rank, Int32(Self.p2p_world_size)
                )

                if my_p2p_world == dst_p2p_world:
                    var slot_idx: Int32 = 0
                    if lane_id() == 0:
                        slot_idx = Atomic[scope=DEVICE_SCOPE].fetch_add[
                            ordering = Consistency.MONOTONIC
                        ](expert_reserved_counter + target_expert, 1)
                    slot_idx = warp.broadcast(slot_idx)

                    var dst_recv_buf_ptr = recv_buf_ptrs[
                        dst_p2p_rank
                    ] + recv_buf_layout(
                        RtTuple_4(
                            Int(dst_expert_local_idx),
                            Int(my_rank),
                            Int(slot_idx),
                            0,
                        )
                    )

                    block_memcpy[Self.msg_bytes, WARP_SIZE](
                        dst_recv_buf_ptr,
                        curr_send_buf_ptr,
                        lane_id(),
                    )

                    syncwarp()

                    if lane_id() == 0:
                        _ = Atomic[scope=DEVICE_SCOPE].fetch_add[
                            ordering = Consistency.RELEASE
                        ](expert_finished_counter + target_expert, 1)

            # We set up `n_local_experts` Reliable Communications (RCs) for each
            # remote device. We would like to use the same RC for each expert.
            # However, NVSHMEM does not allow us to explicitly specify the RC
            # for each transfer. Instead, we set the environment variable
            # `NVSHMEM_IBGDA_RC_MAP_BY=warp` so that the RC is selected by the
            # warp ID using round-robin. We can then control the RC for each
            # expert by using the correct warp.
            @parameter
            if Self.use_shmem:
                var rc_map_offset = Int32(
                    (block_idx.x * UInt(Self.n_warps) + warp_id())
                    % UInt(Self.n_local_experts)
                )

                var topk_idx = lane_id()
                if topk_idx < UInt(Self.top_k) and warp_id() < UInt(
                    Self.n_local_experts
                ):
                    var target_expert = topk_ids.load[width=1](
                        token_idx, Int(topk_idx)
                    )
                    var dst_rank, dst_expert_local_idx = divmod(
                        target_expert, Int32(Self.n_local_experts)
                    )
                    var dst_p2p_world = dst_rank // Int32(Self.p2p_world_size)
                    if (
                        rc_map_offset == dst_expert_local_idx
                        and my_p2p_world != dst_p2p_world
                    ):
                        var slot_idx = Atomic[scope=DEVICE_SCOPE].fetch_add[
                            ordering = Consistency.MONOTONIC
                        ](expert_reserved_counter + target_expert, 1)
                        var dst_recv_buf_ptr = recv_buf_ptrs[
                            my_p2p_rank
                        ] + recv_buf_layout(
                            RtTuple_4(
                                Int(dst_expert_local_idx),
                                Int(my_rank),
                                Int(slot_idx),
                                0,
                            )
                        )
                        shmem_put_nbi[kind = SHMEMScope.default](
                            dst_recv_buf_ptr,
                            curr_send_buf_ptr,
                            UInt(Self.msg_bytes),
                            dst_rank,
                        )

                        _ = Atomic[scope=DEVICE_SCOPE].fetch_add[
                            ordering = Consistency.RELEASE
                        ](expert_finished_counter + target_expert, 1)

    # ===-------------------------------------------------------------------===#
    # Dispatch Callback Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn wait_for_arrivals_and_compute_offsets[
        row_offsets_layout: Layout, expert_ids_layout: Layout, //
    ](
        format_handler: Self.token_fmt_type,
        row_offsets: LayoutTensor[
            DType.uint32, row_offsets_layout, MutAnyOrigin
        ],
        expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin],
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
        comptime shared_expert_offset = 1 if Self.fused_shared_expert else 0
        var tid = thread_idx.x

        var prefix_sum_arr = stack_allocation[
            Self.n_experts, DType.uint32, address_space = AddressSpace.SHARED
        ]()
        var shared_mem = stack_allocation[
            1, DType.uint32, address_space = AddressSpace.SHARED
        ]()
        if tid == 0:
            row_offsets[0] = 0

            @parameter
            if Self.fused_shared_expert:
                var shared_expert_token_count = reserved_shared_expert_tokens

                @parameter
                if Self.expert_m_padding != 0:
                    shared_expert_token_count = UInt32(
                        align_up(
                            Int(shared_expert_token_count),
                            Self.expert_m_padding,
                        )
                    )

                # Place the shared expert's inputs before all routed experts'
                # inputs.
                shared_mem[] = shared_expert_token_count | 0x00100000
                expert_ids[0] = 0
                row_offsets[1] = shared_expert_token_count

            else:
                shared_mem[] = 0

        var token_count: UInt32 = 0
        if tid < UInt(Self.n_experts):
            var target_count_ptr = recv_count_p + tid
            var _token_count = load_acquire[scope = Scope.SYSTEM](
                target_count_ptr
            )
            while _token_count == UInt64.MAX_FINITE:
                _token_count = load_acquire[scope = Scope.SYSTEM](
                    target_count_ptr
                )
            token_count = UInt32(_token_count)
        barrier()

        token_count = block_prefix_sum[Self.n_experts](token_count)
        if tid < UInt(Self.n_experts):
            prefix_sum_arr[tid] = token_count
        barrier()

        if tid < UInt(Self.n_local_experts):
            var local_expert_id = tid
            var last_rank_idx = (
                tid * UInt(Self.n_ranks) + UInt(Self.n_ranks) - 1
            )
            var prev_expert_end = (
                0 if local_expert_id
                == 0 else prefix_sum_arr[last_rank_idx - UInt(Self.n_ranks)]
            )
            var local_expert_tok_count = (
                prefix_sum_arr[last_rank_idx] - prev_expert_end
            )

            @parameter
            if Self.expert_m_padding != 0:
                local_expert_tok_count = UInt32(
                    align_up(Int(local_expert_tok_count), Self.expert_m_padding)
                )

            # Conduct a atomic add to get how many experts have already completed
            # the communication, and the offset where the previous expert end in the
            # output tensor.
            var packed_expert_idx_offset = Atomic[scope=BLOCK_SCOPE].fetch_add[
                ordering = Consistency.MONOTONIC
            ](shared_mem, local_expert_tok_count | 0x00100000)

            var expert_idx = packed_expert_idx_offset >> 20
            var prev_expert_offset = packed_expert_idx_offset & 0x000FFFFF

            expert_ids[Int(expert_idx)] = Int32(
                Int(local_expert_id) + shared_expert_offset
            )
            row_offsets[Int(expert_idx) + 1] = (
                prev_expert_offset + local_expert_tok_count
            )
        barrier()

        # Some token format handlers may require padding the expert offsets to
        # satisfy the grouped matmul alignment requirement.
        format_handler.pad_expert_offsets(row_offsets.ptr)

        # Make sure row_offsets and expert_ids are visible to other threads.
        barrier()

        if tid < UInt(Self.n_experts):
            var expert_lut_idx = tid // UInt(Self.n_ranks) + UInt(
                shared_expert_offset
            )
            var local_expert_id = expert_ids[expert_lut_idx] - Int32(
                shared_expert_offset
            )

            var aligned_expert_start_offset = rebind[UInt32](
                row_offsets[expert_lut_idx]
            )
            var raw_expert_start_offset = (
                0 if local_expert_id
                == 0 else prefix_sum_arr[
                    local_expert_id * Int32(Self.n_ranks) - 1
                ]
            )
            var alignment_delta = Int32(aligned_expert_start_offset) - Int32(
                raw_expert_start_offset
            )

            var expert_rank_linear_idx = local_expert_id * Int32(
                UInt(Self.n_ranks)
            ) + Int32(tid % UInt(Self.n_ranks))
            atomic_counter.store(
                expert_rank_linear_idx * 2 + 1,
                Int32(aligned_expert_start_offset),
            )
            store_release[scope = Scope.GPU](
                atomic_counter + expert_rank_linear_idx * 2,
                Int32(
                    EP_DATA_READY_FLAG
                    + Int32(prefix_sum_arr[expert_rank_linear_idx])
                    + alignment_delta
                ),
            )

    @staticmethod
    @always_inline
    fn copy_received_tokens_to_output[
        src_info_layout: Layout, row_offsets_layout: Layout, //
    ](
        format_handler: Self.token_fmt_type,
        row_offsets: LayoutTensor[
            DType.uint32, row_offsets_layout, MutAnyOrigin
        ],
        src_info: LayoutTensor[DType.int32, src_info_layout, MutAnyOrigin],
        recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
        recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
        atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
        my_rank: Int32,
    ) -> None:
        """Communication SM logic for dispatch_wait_kernel.

        Copies received tokens from the receive buffer to the output tensor.

        Args:
            format_handler: Instance of token_fmt_type for token decoding.
            row_offsets: Output row offsets for grouped matmul.
            src_info: Output tensor for source token info.
            recv_buf_p: Pointer to the receive buffer.
            recv_count_p: Pointer to the receive count buffer.
            atomic_counter: Atomic counter for synchronization.
            my_rank: The rank of the current device.
        """
        comptime shared_expert_offset = 1 if Self.fused_shared_expert else 0
        var recv_buf_layout = Self._get_recv_buf_layout()
        var recv_count_layout = Self._get_recv_count_layout()

        var sm_id = block_idx.x - UInt(Self.n_offset_sms)

        comptime n_wg_per_sm = ceildiv(
            Self.n_experts, Self.n_dispatch_wait_comm_sms
        )
        comptime wg_size = Self.n_warps // n_wg_per_sm
        comptime wg_threads = wg_size * WARP_SIZE

        var wg_idx = warp_id() // UInt(wg_size)
        var global_wg_idx = sm_id * UInt(n_wg_per_sm) + wg_idx
        var warp_id_in_wg = warp_id() % UInt(wg_size)

        if wg_idx >= UInt(n_wg_per_sm) or global_wg_idx >= UInt(Self.n_experts):
            return

        var local_expert_id = global_wg_idx % UInt(Self.n_local_experts)
        var target_rank = global_wg_idx // UInt(Self.n_local_experts)
        var expert_rank_offset = recv_count_layout(
            RtTuple_2(Int(local_expert_id), Int(target_rank))
        )

        # Wait until the auxiliary SM has signaled that the data is ready, and
        # provided the offset where the tokens end in the output tensor.
        var offset_ptr = atomic_counter + expert_rank_offset * 2
        var output_offset = load_acquire[scope = Scope.GPU](offset_ptr)
        while output_offset < EP_DATA_READY_FLAG:
            output_offset = load_acquire[scope = Scope.GPU](offset_ptr)
        output_offset -= EP_DATA_READY_FLAG

        var token_count = Int32(recv_count_p.load(expert_rank_offset))
        output_offset -= token_count
        var local_expert_start_offset = offset_ptr.load(1)

        for token_idx in range(warp_id_in_wg, token_count, wg_size):
            var token_pos = Int(Int32(token_idx) + output_offset)
            var recv_buf_ptr = recv_buf_p + recv_buf_layout(
                RtTuple_4(
                    Int(local_expert_id),
                    Int(target_rank),
                    token_idx,
                    0,
                )
            )

            format_handler.copy_msg_to_output_tensor(
                recv_buf_ptr,
                token_pos,
                Int(local_expert_id) + shared_expert_offset,
                token_pos - Int(local_expert_start_offset),
            )

            if lane_id() < UInt(Self.top_k):
                # Load top-k expert IDs from the token's message.
                var src_topk_idx = bitcast[DType.uint16, 1](
                    recv_buf_ptr.load[width = size_of[UInt16]()](
                        Self.token_fmt_type.topk_info_offset()
                        + Int(lane_id() * UInt(size_of[UInt16]())),
                    )
                )
                var global_expert_idx = my_rank * Int32(
                    Self.n_local_experts
                ) + Int32(local_expert_id)
                if global_expert_idx == Int32(src_topk_idx):
                    # Store the source token index and the top-k id.
                    var src_idx = bitcast[DType.int32, 1](
                        recv_buf_ptr.load[width = size_of[Int32]()](
                            Self.token_fmt_type.src_info_offset()
                        )
                    )

                    src_info[token_pos, 0] = src_idx
                    src_info[token_pos, 1] = Int32(lane_id())

        barrier()
        if lane_id() == 0 and warp_id_in_wg == 0:
            recv_count_p.store(expert_rank_offset, UInt64.MAX_FINITE)
            offset_ptr.store(1, token_count)

        format_handler.zero_pad_scales[
            Self.num_threads, Self.n_dispatch_wait_comm_sms
        ](row_offsets.ptr, Int(sm_id))

    @staticmethod
    @always_inline
    fn pack_shared_expert_inputs[
        shared_expert_input_dtype: DType,
        shared_expert_input_layout: Layout,
        //,
        input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
    ](
        format_handler: Self.token_fmt_type,
        input_tokens: LayoutTensor[
            shared_expert_input_dtype,
            shared_expert_input_layout,
            ImmutAnyOrigin,
        ],
    ) -> None:
        """Packs shared expert inputs before waiting for routed expert arrivals.

        This method copies the shared expert's input tokens to the output buffer
        using the format handler. It is called before waiting for the arrival of
        routed experts' inputs when fused_shared_expert is enabled.

        Args:
            format_handler: Instance of token_fmt_type for token encoding.
            input_tokens: Input tokens for shared experts.
        """
        var smem_buf_p = stack_allocation[
            format_handler.token_size(),
            DType.uint8,
            alignment = simd_width_of[DType.uint8](),
            address_space = AddressSpace.SHARED,
        ]()
        var sm_id = block_idx.x - UInt(Self.n_offset_sms)
        var shared_expert_token_count = input_tokens.dim(0)

        var input_scale = Float32(1.0)

        @parameter
        if input_scales_wrapper is not None:
            comptime input_scale_fn = input_scales_wrapper.value()
            input_scale = input_scale_fn[DType.float32](0)

        for token_idx in range(
            sm_id, shared_expert_token_count, Self.n_dispatch_wait_comm_sms
        ):
            var input_tensor_p = input_tokens.ptr + token_idx * Self.hid_dim
            format_handler.copy_token_to_send_buf[
                shared_expert_input_dtype,
                UInt(Self.num_threads),
                buf_addr_space = AddressSpace.SHARED,
            ](smem_buf_p, input_tensor_p, input_scale)
            barrier()
            # Shared expert's input tokens are always packed at the beginning.
            if warp_id() == 0:
                format_handler.copy_msg_to_output_tensor(
                    smem_buf_p, token_idx, 0, token_idx
                )
            barrier()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn dispatch_async_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: Layout,
    topk_ids_layout: Layout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
    use_shmem: Bool = True,
](
    input_tokens: LayoutTensor[input_type, input_tokens_layout, ImmutAnyOrigin],
    topk_ids: LayoutTensor[DType.int32, topk_ids_layout, ImmutAnyOrigin],
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
    if block_idx.x < UInt(dispatch_impl.n_signal_sms):
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
fn dispatch_wait_kernel[
    num_threads: Int,
    row_offsets_layout: Layout,
    expert_ids_layout: Layout,
    src_info_layout: Layout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    token_fmt_type: TokenFormat,
    expert_m_padding: Int = 0,
    fused_shared_expert: Bool = False,
    shared_expert_input_dtype: DType = DType.bfloat16,
    input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
](
    format_handler: token_fmt_type,
    row_offsets: LayoutTensor[DType.uint32, row_offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin],
    src_info: LayoutTensor[DType.int32, src_info_layout, MutAnyOrigin],
    recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
    maybe_input_tokens: OptionalReg[
        LayoutTensor[
            shared_expert_input_dtype, Layout.row_major[2](), ImmutAnyOrigin
        ]
    ],
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
        expert_m_padding: If non-zero, the number of tokens for each local
            expert will be padded to the next multiple of `expert_m_padding`.
        fused_shared_expert: Whether to pack the shared expert inputs with the
            routed experts' inputs.
        shared_expert_input_dtype: The data type of the shared expert inputs.
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
        maybe_input_tokens: The optional input tokens for the shared experts.
            If fused_shared_expert is True, this will be used to load the input
            tokens for the shared experts.
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
        expert_m_padding=expert_m_padding,
        fused_shared_expert=fused_shared_expert,
    ]

    var atomic_counter = ep_counters.get_dispatch_wait_ptr()

    # The first SM is used for checking if any of a local expert has received
    # tokens from all the remote ranks. It will also calculate the offset where
    # the tokens start in the output tensor.
    if block_idx.x < UInt(dispatch_impl.n_offset_sms):
        var reserved_shared_expert_tokens = 0

        @parameter
        if fused_shared_expert:
            reserved_shared_expert_tokens = maybe_input_tokens.value().dim(0)

        dispatch_impl.wait_for_arrivals_and_compute_offsets(
            format_handler,
            row_offsets,
            expert_ids,
            recv_count_p,
            atomic_counter,
            my_rank,
            UInt32(reserved_shared_expert_tokens),
        )

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        # If we need to pack the shared expert's inputs, we do that before
        # waiting for the arrival of the routed experts' inputs.
        @parameter
        if fused_shared_expert:
            dispatch_impl.pack_shared_expert_inputs[input_scales_wrapper](
                format_handler, maybe_input_tokens.value()
            )

        dispatch_impl.copy_received_tokens_to_output(
            format_handler,
            row_offsets,
            src_info,
            recv_buf_p,
            recv_count_p,
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
    """

    comptime n_local_experts = Self.n_experts // Self.n_ranks
    comptime n_warps = Self.num_threads // WARP_SIZE

    # Aux SMs for combine_wait kernel: single SM waits for arrivals.
    comptime n_wait_sms = 1
    # Reduce SMs for combine_wait kernel.
    comptime n_reduce_sms = Self.n_sms - Self.n_wait_sms

    comptime send_layout_static = Layout.row_major(
        Self.n_local_experts * Self.n_ranks * Self.max_tokens_per_rank,
        Self.msg_bytes,
    )

    @staticmethod
    @always_inline
    fn _get_send_buf_layout(
        out result: RuntimeLayout[
            Self.send_layout_static,
            element_type = _get_layout_type(
                Self.send_layout_static, AddressSpace.GENERIC
            ),
            linear_idx_type = _get_index_type(
                Self.send_layout_static, AddressSpace.GENERIC
            ),
        ]
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    fn _get_recv_buf_layout(
        out result: RuntimeLayout[
            Layout.row_major(
                Self.max_tokens_per_rank, Self.top_k, Self.msg_bytes
            ),
            element_type = DType.int32,
            linear_idx_type = DType.int32,
        ]
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    fn _get_recv_count_layout(
        out result: RuntimeLayout[
            Layout.row_major(Self.n_local_experts, Self.n_ranks),
            element_type = DType.int32,
            linear_idx_type = DType.int32,
        ]
    ):
        return type_of(result)()

    # ===-------------------------------------------------------------------===#
    # Combine Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn copy_shared_expert_outputs[
        input_type: DType,
        input_tokens_layout: Layout,
        output_tokens_layout: Layout,
        //,
    ](
        input_tokens: LayoutTensor[
            input_type, input_tokens_layout, MutAnyOrigin
        ],
        output_tokens: LayoutTensor[
            input_type, output_tokens_layout, MutAnyOrigin
        ],
    ) -> None:
        """Copies shared expert outputs to the output tensor.

        This method copies the shared expert's output tokens from the input
        tensor to the output tensor when fused_shared_expert is enabled.

        Args:
            input_tokens: The input tokens containing shared expert outputs.
            output_tokens: The output tensor to copy shared expert outputs to.
        """
        comptime hid_dim = input_tokens.shape[1]()
        var tid = Int(thread_idx.x)
        var sm_id = Int(block_idx.x)
        var shared_expert_token_count = output_tokens.dim(0)

        for token_idx in range(sm_id, shared_expert_token_count, Self.n_sms):
            var output_tokens_p = output_tokens.ptr + token_idx * hid_dim
            block_memcpy[hid_dim * size_of[input_type](), Self.num_threads](
                output_tokens_p.bitcast[UInt8](),
                input_tokens.ptr_at_offset(IndexList[2](token_idx, 0)).bitcast[
                    UInt8
                ](),
                UInt(tid),
            )

    @staticmethod
    @always_inline
    fn send_tokens_back[
        input_type: DType,
        input_tokens_layout: Layout,
        src_info_layout: Layout,
        //,
    ](
        input_tokens: LayoutTensor[
            input_type, input_tokens_layout, MutAnyOrigin
        ],
        src_info: LayoutTensor[DType.int32, src_info_layout, MutAnyOrigin],
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
        comptime hid_dim = input_tokens.shape[1]()

        comptime assert (
            Self.msg_bytes == hid_dim * size_of[Scalar[input_type]]()
        ), "EP combine_async: input shape doesn't match message size."

        var send_buf_layout = Self._get_send_buf_layout()
        var recv_buf_layout = Self._get_recv_buf_layout()
        var recv_count_layout = Self._get_recv_count_layout()

        var tid = Int(thread_idx.x)
        var sm_id = Int(block_idx.x)
        var my_p2p_world, my_p2p_rank = divmod(
            my_rank, Int32(Self.p2p_world_size)
        )

        # Each rank holds `n_local_experts` experts, and for each expert, it
        # needs to send back different tokens to `n_ranks` remote ranks. We use
        # one block per-expert-per-rank to send back the tokens.
        for global_idx in range(sm_id, Self.n_experts, Self.n_sms):
            var local_expert_id = global_idx % Self.n_local_experts
            var target_rank = global_idx // Self.n_local_experts
            var expert_rank_offset = recv_count_layout(
                RtTuple_2(local_expert_id, target_rank)
            )
            var dst_p2p_world, dst_p2p_rank = divmod(
                target_rank, Self.p2p_world_size
            )

            # Info for where the tokens for the current expert and rank start
            # and end are stored in the atomic counter by the
            # `dispatch_wait_kernel`.
            comptime DATA_READY_FLAG = 1024
            var token_end_count = atomic_counter.load[
                width=2,
                alignment = align_of[SIMD[DType.int32, 2]](),
                invariant=True,
            ](2 * expert_rank_offset)
            var token_end = token_end_count[0] - DATA_READY_FLAG
            var token_start = token_end - token_end_count[1]

            # If the target device is on the same node, we can directly copy the
            # tokens to the receive buffer, skipping the send buffer.
            if Int32(dst_p2p_world) == my_p2p_world:
                for token_idx in range(token_start, token_end):
                    var src_token_info = src_info.aligned_load[2](
                        Int(token_idx), 0
                    )
                    var src_idx = src_token_info[0]
                    var src_topk_idx = src_token_info[1]

                    var dst_recv_buf_ptr = recv_buf_ptrs[
                        dst_p2p_rank
                    ] + recv_buf_layout(
                        RtTuple_3(Int(src_idx), Int(src_topk_idx), 0)
                    )
                    block_memcpy[
                        hid_dim * size_of[input_type](), Self.num_threads
                    ](
                        dst_recv_buf_ptr,
                        input_tokens.ptr_at_offset(
                            IndexList[2](Int(token_idx), 0)
                        ).bitcast[UInt8](),
                        UInt(tid),
                    )

            # If the target device is on a different node, we need to send the
            # tokens to the target device using the SHMEM API.
            else:

                @parameter
                if Self.use_shmem:
                    # The tokens are sent back to the original rank using the
                    # same RC as the one they come from.
                    var rc_map_offset = (
                        sm_id * Self.n_warps + Int(warp_id())
                    ) % Self.n_local_experts

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
                                + send_buf_layout(RtTuple_2(Int(token_idx), 0))
                            )

                            # To use SHMEM API, we need to copy the tokens to
                            # the send buffer first.
                            block_memcpy[
                                hid_dim * size_of[input_type](), WARP_SIZE
                            ](
                                curr_send_buf_ptr,
                                input_tokens.ptr_at_offset(
                                    IndexList[2](Int(token_idx), 0)
                                ).bitcast[UInt8](),
                                lane_id(),
                            )

                        barrier()

                        if (
                            warp_id() < UInt(Self.n_local_experts)
                            and local_expert_id == rc_map_offset
                        ):
                            var token_idx = (
                                token_start
                                + round_i * Int32(Self.n_warps)
                                + Int32(lane_id())
                            )
                            if token_idx < token_end:
                                var src_token_info = src_info.aligned_load[2](
                                    Int(token_idx), 0
                                )
                                var src_idx = src_token_info[0]
                                var src_topk_idx = src_token_info[1]

                                var curr_send_buf_ptr = (
                                    send_buf_p
                                    + send_buf_layout(
                                        RtTuple_2(Int(token_idx), 0)
                                    )
                                )
                                var dst_recv_buf_ptr = recv_buf_ptrs[
                                    my_p2p_rank
                                ] + recv_buf_layout(
                                    RtTuple_3(
                                        Int(src_idx), Int(src_topk_idx), 0
                                    )
                                )

                                shmem_put_nbi[kind = SHMEMScope.default](
                                    dst_recv_buf_ptr,
                                    curr_send_buf_ptr,
                                    UInt(Self.msg_bytes),
                                    Int32(target_rank),
                                )

            barrier()

            # Once all the tokens for the current expert and rank have been
            # sent, signal the completion of the communication.
            var rc_map_offset = (
                sm_id * Self.n_warps + Int(warp_id())
            ) % Self.n_local_experts
            if (
                warp_id() < UInt(Self.n_local_experts)
                and local_expert_id == rc_map_offset
            ):
                if lane_id() == 0:
                    var signal_offset = my_rank * Int32(
                        Self.n_local_experts
                    ) + Int32(local_expert_id)

                    ep_signal_completion[
                        Self.use_shmem,
                        n_experts_per_device = Self.n_local_experts,
                    ](
                        my_rank,
                        Int32(target_rank),
                        recv_count_ptrs,
                        signal_offset,
                        UInt64(token_end - token_start),
                        rank_completion_counter,
                    )

                    atomic_counter.store[
                        width=2, alignment = align_of[SIMD[DType.int32, 2]]()
                    ](expert_rank_offset * 2, 0)

    # ===-------------------------------------------------------------------===#
    # Combine Callback Kernel Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn wait_for_all_arrivals(
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

        if thread_idx.x < UInt(Self.n_experts):
            var target_count_ptr = recv_count_p + thread_idx.x
            while (
                load_acquire[scope = Scope.SYSTEM](target_count_ptr)
                == UInt64.MAX_FINITE
            ):
                pass

            target_count_ptr[] = UInt64.MAX_FINITE
        barrier()

        # Once all the tokens have been received, set flags for other SMs to
        # copy the tokens to the output tensor.
        if thread_idx.x < UInt(Self.n_reduce_sms):
            atomic_counter.store(
                Self.n_wait_sms + Int(thread_idx.x), DATA_READY_FLAG
            )

    @staticmethod
    @always_inline
    fn reduce_and_copy_to_output[
        output_type: DType,
        output_tokens_layout: Layout,
        router_weights_wrapper: Optional[router_weights_wrapper_type] = None,
        elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    ](
        output_tokens: LayoutTensor[
            output_type, output_tokens_layout, MutAnyOrigin
        ],
        recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
        atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
    ) -> None:
        """Communication SM logic for combine_wait_kernel.

        Copies received tokens to the output tensor, optionally applying
        router weights and reduction across top-k experts.

        Args:
            output_tokens: The tensor to store the output tokens.
            recv_buf_p: Pointer to the receive buffer.
            atomic_counter: Atomic counter for synchronization.
        """
        comptime DATA_READY_FLAG = 1024
        var num_tokens = output_tokens.dim[0]()
        comptime dst_simd_width = simd_width_of[output_type]()
        comptime byte_simd_width = simd_width_of[DType.uint8]()

        comptime last_dim = 1 if router_weights_wrapper else 2
        comptime hid_dim = output_tokens_layout.shape[last_dim].value()
        comptime _align = align_of[SIMD[DType.uint8, byte_simd_width]]()

        comptime assert (
            Self.msg_bytes == hid_dim * size_of[Scalar[output_type]]()
        ), "EP combine_async: output shape doesn't match message size."
        comptime assert (
            Self.msg_bytes % byte_simd_width == 0
        ), "EP combine_async: message size must be divisible by " + String(
            byte_simd_width
        )

        var recv_buf_layout = Self._get_recv_buf_layout()

        var sm_id = Int(block_idx.x)

        if thread_idx.x == 0:
            while (
                load_acquire[scope = Scope.GPU](atomic_counter + sm_id)
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
        var global_id = (
            sm_id - Self.n_wait_sms + Int(warp_id()) * Self.n_reduce_sms
        )

        for chunk_idx in range(
            global_id,
            num_tokens * n_chunks_per_tok,
            Self.n_warps * Self.n_reduce_sms,
        ):
            var token_idx, chunk_idx_in_token = divmod(
                chunk_idx, n_chunks_per_tok
            )

            var accum = SIMD[DType.float32, dst_simd_width](0)
            var recv_chunk = SIMD[output_type, dst_simd_width](0)

            @parameter
            for topk_idx in range(Self.top_k):
                var recv_buf_ptr = recv_buf_p + recv_buf_layout(
                    RtTuple_3(
                        token_idx,
                        topk_idx,
                        chunk_idx_in_token * n_chunk_bytes,
                    )
                )
                recv_chunk = bitcast[output_type, dst_simd_width](
                    recv_buf_ptr.load[
                        width=byte_simd_width,
                        invariant=True,
                        alignment=_align,
                    ](
                        Int(lane_id()) * byte_simd_width,
                    )
                )

                @parameter
                if router_weights_wrapper:
                    comptime router_weights_fn = router_weights_wrapper.value()

                    var weight = router_weights_fn[1](token_idx, topk_idx)
                    accum += weight * recv_chunk.cast[DType.float32]()

                else:
                    # The output tensor is of shape
                    # `(num_tokens, top_k, hid_dim)`.
                    var output_token_slice = output_tokens.slice[:, :, (1, 2)](
                        IndexList[1](token_idx)
                    )

                    output_token_slice.aligned_store[width=dst_simd_width](
                        topk_idx,
                        chunk_idx_in_token * n_chunk_elems
                        + Int(lane_id()) * dst_simd_width,
                        recv_chunk,
                    )

            @parameter
            if router_weights_wrapper:

                @parameter
                if elementwise_lambda_fn:
                    comptime lambda_fn = elementwise_lambda_fn.value()
                    lambda_fn[alignment=_align](
                        (
                            token_idx,
                            chunk_idx_in_token * n_chunk_elems
                            + Int(lane_id()) * dst_simd_width,
                        ),
                        accum.cast[output_type](),
                    )

                else:
                    output_tokens.aligned_store[width=dst_simd_width](
                        token_idx,
                        chunk_idx_in_token * n_chunk_elems
                        + Int(lane_id()) * dst_simd_width,
                        accum.cast[output_type](),
                    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn combine_async_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: Layout,
    src_info_layout: Layout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    use_shmem: Bool = True,
    fused_shared_expert: Bool = False,
](
    input_tokens: LayoutTensor[input_type, input_tokens_layout, MutAnyOrigin],
    src_info: LayoutTensor[DType.int32, src_info_layout, MutAnyOrigin],
    send_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_buf_ptrs: InlineArray[
        UnsafePointer[UInt8, MutExternalOrigin], p2p_world_size
    ],
    recv_count_ptrs: InlineArray[
        UnsafePointer[UInt64, MutExternalOrigin], p2p_world_size
    ],
    ep_counters: EPLocalSyncCounters[n_experts],
    my_rank: Int32,
    maybe_output_tokens: OptionalReg[
        LayoutTensor[input_type, Layout.row_major[2](), MutAnyOrigin]
    ],
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
        fused_shared_expert: Whether to filter out the shared expert's outputs.

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
        maybe_output_tokens: The optional output for the shared experts.
            If fused_shared_expert is True, this will be used to store the
            output tokens for the shared experts.
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
        fused_shared_expert,
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

    # Copy the shared expert's outputs to the output tensor.
    @parameter
    if fused_shared_expert:
        combine_impl.copy_shared_expert_outputs(
            input_tokens, maybe_output_tokens.value()
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn combine_wait_kernel[
    output_type: DType,
    num_threads: Int,
    output_tokens_layout: Layout,
    n_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    router_weights_wrapper: Optional[router_weights_wrapper_type] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    output_tokens: LayoutTensor[
        output_type, output_tokens_layout, MutAnyOrigin
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
    var sm_id = Int(block_idx.x)

    # The first SM is used for checking if we have received tokens from all the
    # remote ranks.
    if sm_id < combine_impl.n_wait_sms:
        combine_impl.wait_for_all_arrivals(recv_count_p, atomic_counter)

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        combine_impl.reduce_and_copy_to_output[
            output_type,
            output_tokens_layout,
            router_weights_wrapper,
            elementwise_lambda_fn,
        ](
            output_tokens,
            recv_buf_p,
            atomic_counter,
        )


# ===-----------------------------------------------------------------------===#
# Fused EP Kernels
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn dispatch_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: Layout,
    topk_ids_layout: Layout,
    row_offsets_layout: Layout,
    expert_ids_layout: Layout,
    src_info_layout: Layout,
    n_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
    expert_m_padding: Int = 0,
    fused_shared_expert: Bool = False,
    input_scales_wrapper: Optional[input_scales_wrapper_type] = None,
    use_shmem: Bool = True,
](
    input_tokens: LayoutTensor[input_type, input_tokens_layout, ImmutAnyOrigin],
    topk_ids: LayoutTensor[DType.int32, topk_ids_layout, ImmutAnyOrigin],
    format_handler: token_fmt_type,
    row_offsets: LayoutTensor[DType.uint32, row_offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin],
    src_info: LayoutTensor[DType.int32, src_info_layout, MutAnyOrigin],
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
        expert_m_padding: If non-zero, the number of tokens for each local
            expert will be padded to the next multiple of `expert_m_padding`.
        fused_shared_expert: Whether to pack the shared expert inputs with the
            routed experts' inputs. When enabled, input_tokens is used as the
            shared expert inputs.
        input_scales_wrapper: The wrapper for the input scales.
        use_shmem: Whether to use the SHMEM API for the communication.

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
        expert_m_padding,
        fused_shared_expert,
    ]

    # ===== dispatch_async =====
    var async_atomic_counter = ep_counters.get_dispatch_async_ptr()
    var expert_reserved_counter = async_atomic_counter
    var expert_finished_counter = async_atomic_counter + n_experts
    var rank_completion_counter = async_atomic_counter + 2 * n_experts

    with PDL():
        if block_idx.x < UInt(dispatch_impl.n_signal_sms):
            dispatch_impl.monitor_and_signal_completion(
                topk_ids,
                recv_count_ptrs,
                expert_reserved_counter,
                expert_finished_counter,
                rank_completion_counter,
                my_rank,
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
        var wait_atomic_counter = ep_counters.get_dispatch_wait_ptr()
        var my_p2p_rank = my_rank % Int32(p2p_world_size)

        if block_idx.x < UInt(dispatch_impl.n_offset_sms):
            var reserved_shared_expert_tokens: UInt32 = 0

            @parameter
            if fused_shared_expert:
                reserved_shared_expert_tokens = UInt32(input_tokens.dim(0))

            dispatch_impl.wait_for_arrivals_and_compute_offsets(
                format_handler,
                row_offsets,
                expert_ids,
                recv_count_ptrs[my_p2p_rank],
                wait_atomic_counter,
                my_rank,
                reserved_shared_expert_tokens,
            )
        else:

            @parameter
            if fused_shared_expert:
                dispatch_impl.pack_shared_expert_inputs[input_scales_wrapper](
                    format_handler, input_tokens
                )

            dispatch_impl.copy_received_tokens_to_output(
                format_handler,
                row_offsets,
                src_info,
                recv_buf_ptrs[my_p2p_rank],
                recv_count_ptrs[my_p2p_rank],
                wait_atomic_counter,
                my_rank,
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn combine_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: Layout,
    src_info_layout: Layout,
    output_tokens_layout: Layout,
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
    use_shmem: Bool = True,
](
    input_tokens: LayoutTensor[input_type, input_tokens_layout, MutAnyOrigin],
    src_info: LayoutTensor[DType.int32, src_info_layout, MutAnyOrigin],
    output_tokens: LayoutTensor[input_type, output_tokens_layout, MutAnyOrigin],
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
        use_shmem: Whether to use the SHMEM API for the communication.

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
        my_rank: The rank of the current device.
    """

    @parameter
    if fused_shared_expert:
        comptime assert router_weights_wrapper, (
            "EP combine_kernel: fused_shared_expert requires "
            "router_weights_wrapper to be provided. Cannot add shared expert "
            "output to non-reduced routed expert outputs."
        )

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
            @parameter
            if fused_shared_expert:
                comptime hid_dim = input_tokens_layout.shape[1].value()

                @always_inline
                @parameter
                fn add_shared_expert_output[
                    dtype: DType, width: Int, *, alignment: Int = 1
                ](
                    idx: IndexList[2], combined_val: SIMD[dtype, width]
                ) capturing:
                    """Add shared expert output to the reduced routed expert output.
                    """

                    var shared_expert_val = input_tokens.aligned_load[
                        width=width
                    ](idx).cast[dtype]()

                    # Add and store the result
                    var result = combined_val + shared_expert_val

                    @parameter
                    if epilogue_fn:
                        comptime epilogue = epilogue_fn.value()
                        epilogue[width=width, alignment=alignment](idx, result)
                    else:
                        output_tokens.aligned_store[width=width](
                            idx[0], idx[1], result.cast[input_type]()
                        )

                combine_impl.reduce_and_copy_to_output[
                    input_type,
                    output_tokens_layout,
                    router_weights_wrapper,
                    add_shared_expert_output,
                ](
                    output_tokens,
                    recv_buf_ptrs[my_p2p_rank],
                    wait_atomic_counter,
                )

            else:
                combine_impl.reduce_and_copy_to_output[
                    input_type,
                    output_tokens_layout,
                    router_weights_wrapper,
                    epilogue_fn,
                ](
                    output_tokens,
                    recv_buf_ptrs[my_p2p_rank],
                    wait_atomic_counter,
                )


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn fused_silu_kernel[
    output_dtype: DType,
    input_dtype: DType,
    output_layput: Layout,
    input_layout: Layout,
    row_offsets_layout: Layout,
    num_threads: Int,
    num_sms: Int,
](
    output_tensor: LayoutTensor[output_dtype, output_layput, MutAnyOrigin],
    input_tensor: LayoutTensor[input_dtype, input_layout, ImmutAnyOrigin],
    row_offsets: LayoutTensor[DType.uint32, row_offsets_layout, ImmutAnyOrigin],
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
    comptime input_dim = input_tensor.shape[1]()
    comptime output_dim = output_tensor.shape[1]()
    comptime simd_width = simd_width_of[input_dtype]()

    # This should also make sure the input and output tensors has static shape.
    comptime assert (
        input_dim == output_dim * 2
    ), "Input dimension must be twice the output dimension."
    comptime assert (
        output_dim % simd_width == 0
    ), "Output dimension must be divisible by the SIMD width."

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var gid = tid + bid * num_threads

    with PDL():
        var num_tokens = row_offsets[row_offsets.size() - 1]
        var num_elem = num_tokens * UInt32(output_dim)

        for i in range(
            gid,
            num_elem // UInt32(simd_width),
            num_threads * num_sms,
        ):
            var m = (i * simd_width) // output_dim
            var k = (i * simd_width) % output_dim

            var gate_proj = input_tensor.aligned_load[width=simd_width](
                m, k
            ).cast[accum_dtype]()
            var up_proj = input_tensor.aligned_load[width=simd_width](
                m, k + output_dim
            ).cast[accum_dtype]()

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            output_tensor.aligned_store[width=simd_width](
                m, k, output_val.cast[output_dtype]()
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn fused_silu_fp8_kernel[
    fp8_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layput: Layout,
    scales_layput: Layout,
    input_layout: Layout,
    offsets_layout: Layout,
    num_threads: Int,
    num_sms: Int,
    group_size: Int = 128,
](
    output_tensor: LayoutTensor[fp8_dtype, output_layput, MutAnyOrigin],
    scales_tensor: LayoutTensor[scales_dtype, scales_layput, MutAnyOrigin],
    input_tensor: LayoutTensor[input_dtype, input_layout, ImmutAnyOrigin],
    row_offsets: LayoutTensor[DType.uint32, offsets_layout, ImmutAnyOrigin],
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
    comptime input_dim = input_tensor.shape[1]()
    comptime output_dim = output_tensor.shape[1]()
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
    var global_warp_id = block_idx.x + warp_id() * UInt(num_sms)
    var gid = lane_id() + global_warp_id * UInt(WARP_SIZE)

    with PDL():
        var num_tokens = row_offsets[row_offsets.size() - 1]
        var num_elem = num_tokens * UInt32(output_dim)

        for i in range(
            gid,
            num_elem // UInt32(simd_width),
            num_threads * num_sms,
        ):
            var m = (i * simd_width) // output_dim
            var k = (i * simd_width) % output_dim

            var gate_proj = input_tensor.aligned_load[width=simd_width](
                m, k
            ).cast[accum_dtype]()
            var up_proj = input_tensor.aligned_load[width=simd_width](
                m, k + output_dim
            ).cast[accum_dtype]()

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            # Quantization logic.
            var thread_max = abs(output_val).reduce_max()
            var group_max = warp.lane_group_max_and_broadcast[
                n_threads_per_group
            ](thread_max)
            var scale_factor = max(group_max, 1e-4) / fp8_max_t
            output_val = (output_val / scale_factor).clamp(
                -fp8_max_t, fp8_max_t
            )

            output_tensor.aligned_store[width=simd_width](
                m, k, output_val.cast[fp8_dtype]()
            )

            # The first thread in each group stores the scale factor.
            if lane_id() % UInt(n_threads_per_group) == 0:
                scales_tensor.store(
                    k // group_size, m, scale_factor.cast[scales_dtype]()
                )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn fused_silu_nvfp4_kernel[
    fp4_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layput: Layout,
    scales_layput: Layout,
    input_layout: Layout,
    offsets_layout: Layout,
    scales_offsets_layout: Layout,
    input_scales_layout: Layout,
    num_threads: Int,
    num_sms: Int,
](
    output_tensor: LayoutTensor[fp4_dtype, output_layput, MutAnyOrigin],
    scales_tensor: LayoutTensor[scales_dtype, scales_layput, MutAnyOrigin],
    input_tensor: LayoutTensor[input_dtype, input_layout, ImmutAnyOrigin],
    row_offsets: LayoutTensor[DType.uint32, offsets_layout, ImmutAnyOrigin],
    scales_offsets: LayoutTensor[
        DType.uint32, scales_offsets_layout, ImmutAnyOrigin
    ],
    input_scales: LayoutTensor[
        DType.float32, input_scales_layout, ImmutAnyOrigin
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
    comptime input_dim = input_tensor.shape[1]()
    comptime output_dim = output_tensor.shape[1]()
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

    comptime n_groups = scales_offsets_layout.shape[0].value()
    comptime n_sms_per_group = num_sms // n_groups
    comptime assert (
        n_groups <= num_sms
    ), "num_sms must be >= number of expert groups."
    comptime assert (
        input_scales_layout.shape[0].value() == n_groups
    ), "input_scales must match number of expert groups."

    var tid = Int(thread_idx.x)
    var sm_id = Int(block_idx.x)
    var group_id, sm_id_in_group = divmod(sm_id, n_sms_per_group)
    var tid_in_group = (
        lane_id()
        + UInt(sm_id_in_group) * UInt(WARP_SIZE)
        + warp_id() * UInt(WARP_SIZE) * UInt(n_sms_per_group)
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

        var _scales_tensor = LayoutTensor[
            scales_dtype, scales_layput, MutAnyOrigin
        ](
            scales_tensor.ptr_at_offset(
                IndexList[5](Int(scales_block_id), 0, 0, 0, 0)
            ),
        )

        var tensor_sf = rebind[Float32](input_scales[group_id])

        for group_linear in range(
            tid_in_group,
            n_threads_per_token * expert_m,
            num_threads * n_sms_per_group,
        ):
            var token_idx, hid_idx = divmod(group_linear, n_threads_per_token)
            var m = expert_start + token_idx
            var k = hid_idx * src_width

            var gate_proj = input_tensor.aligned_load[width=src_width](
                m, k
            ).cast[accum_dtype]()
            var up_proj = input_tensor.aligned_load[width=src_width](
                m, k + hidden_size
            ).cast[accum_dtype]()

            gate_proj = gate_proj / (1.0 + exp(-gate_proj))
            var output_val = gate_proj * up_proj

            # Quantization logic (NVFP4).
            var thread_max = abs(output_val).reduce_max()
            thread_max = max(warp.shuffle_xor(thread_max, 1), thread_max)
            var group_max = thread_max.cast[DType.float32]()
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
            output_tensor.aligned_store[width=byte_width](
                m, k // 2, output_vector
            )

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
                var token_idx, scale_simd_idx = divmod(i, scales_simds_per_tok)
                set_scale_factor[SF_VECTOR_SIZE=1](
                    _scales_tensor,
                    token_idx + expert_m,
                    scale_simd_idx * SF_ATOM_K,
                    SIMD[scales_dtype, SF_ATOM_K](0.0),
                )
