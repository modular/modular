# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import OptionalReg
from math import align_up, ceildiv
from os.atomic import Atomic, Consistency
from sys.info import align_of, simd_width_of, size_of

import gpu.warp as warp
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
from gpu.intrinsics import Scope, load_acquire, store_release, threadfence
from gpu.sync import syncwarp
from layout import Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from layout.int_tuple import (
    UNKNOWN_VALUE,
    IntTuple,
    _get_index_type,
    _get_layout_type,
)
from math import exp
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
] (IndexList[2], SIMD[dtype, width]) capturing -> None

comptime EP_DATA_READY_FLAG = 1 << 10


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
@parameter
fn ep_signal_completion[
    p2p_world_size: Int, //, use_shmem: Bool
](
    my_rank: Int32,
    dst_rank: Int32,
    recv_count_ptrs: InlineArray[
        UnsafePointer[UInt64, MutExternalOrigin], p2p_world_size
    ],
    signal_offset: Int32,
    signal: UInt64,
) -> None:
    """
    Signals the completion of the communication by writing to the receive count
    buffer. Will use direct memory access if the target device is on the same
    node, and use the SHMEM API if the target device is on a different node.
    """

    var my_p2p_world, my_p2p_rank = divmod(my_rank, p2p_world_size)
    var dst_p2p_world, dst_p2p_rank = divmod(dst_rank, p2p_world_size)

    # If the target device is on the same node, we can directly write to its
    # receive count buffer.
    if my_p2p_world == dst_p2p_world:
        var dst_p2p_ptr = recv_count_ptrs[dst_p2p_rank] + signal_offset
        store_release[scope = Scope.SYSTEM](
            dst_p2p_ptr,
            signal,
        )
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


@register_passable("trivial")
trait TokenFormat(DevicePassable):
    comptime hid_dim: Int
    comptime top_k: Int
    comptime alignment: Int

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
    @staticmethod
    fn copy_token_to_send_buf[
        src_type: DType,
        block_size: UInt,
        buf_addr_space: AddressSpace = AddressSpace.GENERIC,
    ](
        buf_p: UnsafePointer[mut=True, UInt8, address_space=buf_addr_space],
        src_p: UnsafePointer[mut=False, Scalar[src_type]],
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
    ) -> None:
        "Copy the message to the output tensor. This function needs to be called by all threads in a warp."
        ...


@register_passable("trivial")
struct BF16TokenFormat[
    output_layout: Layout, //, _hid_dim: Int, _top_k: Int, _alignment: Int
](TokenFormat):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment

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
            String(Self.output_layout),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

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


@register_passable("trivial")
struct BlockwiseFP8TokenFormat[
    fp8_dtype: DType,
    scales_dtype: DType,
    output_layout: Layout,
    scales_layout: Layout,
    //,
    _hid_dim: Int,
    _top_k: Int,
    _alignment: Int,
](TokenFormat):
    comptime hid_dim = Self._hid_dim
    comptime top_k = Self._top_k
    comptime alignment = Self._alignment

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
            String(Self.output_layout),
            ", scales_layout = ",
            String(Self.scales_layout),
            ", hid_dim = ",
            String(Self.hid_dim),
            ", top_k = ",
            String(Self.top_k),
            ", alignment = ",
            String(Self.alignment),
            "]",
        )

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

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
        __comptime_assert (
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
    ) -> None:
        comptime src_width = simd_width_of[src_type]()
        comptime byte_width = src_width * size_of[Self.fp8_dtype]()

        comptime fp8_max = Scalar[Self.fp8_dtype].MAX_FINITE
        comptime fp8_max_t = Scalar[Self.fp8_dtype].MAX_FINITE.cast[
            Self.scales_dtype
        ]()

        comptime n_threads_per_group = Self.group_size // src_width
        __comptime_assert (
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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn dispatch_kernel[
    input_type: DType,
    num_threads: Int,
    input_tokens_layout: Layout,
    topk_ids_layout: Layout,
    n_sms: Int,
    n_aux_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    p2p_world_size: Int,
    token_fmt_type: TokenFormat,
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
    atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
    my_rank: Int32,
):
    """
    Dispatch tokens to experts on remote ranks based on the top-k expert IDs.
    This kernel utilizes the non-blocking SHMEM API if `use_shmem` is True, and
    would return immediately after initiating the communication. The
    communication is considered complete after calling the `dispatch_cb_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        topk_ids_layout: The layout of the top-k expert IDs.
        n_sms: The total number of SMs in the device.
        n_aux_sms: The number SMs that are used for counting the number of tokens
            for each expert, and for signaling the completion of the communication.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        max_tokens_per_rank: The maximum number of tokens per rank.
        p2p_world_size: Size of a High-speed GPU interconnect group.
        token_fmt_type: Type conforming to TokenFormat trait that defines the
            token encoding scheme.
        use_shmem: Whether to use the SHMEM API for the communication.

    Args:
        input_tokens: The input tokens to be dispatched.
        topk_ids: The top-k expert IDs for each token.
        send_buf_p: The pointer to the send buffer. The underlying buffer is
            of shape `(max_tokens_per_rank, msg_bytes)`. Need to be allocated
            using `shmem_alloc` if `use_shmem` is True.
        recv_buf_ptrs: An array of pointers to the receive buffers for each
            device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`. Need
            to be allocated using `shmem_alloc` if `use_shmem` is True.
        recv_count_ptrs: An array of pointers to the receive count buffers for
            each device in the p2p world. Each buffer is of shape
            `(n_local_experts, n_ranks)`. Need to be allocated using
            `shmem_alloc` if `use_shmem` is True.
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
    """

    comptime n_local_experts = n_experts // n_ranks
    comptime n_warps = num_threads // WARP_SIZE
    comptime n_comm_sms = n_sms - n_aux_sms
    __comptime_assert n_local_experts <= n_warps or n_ranks == p2p_world_size, (
        "EP dispatch: number of experts per rank must be less than or equal to "
        + String(n_warps)
    )

    comptime top_k = topk_ids.shape[1]()
    comptime hid_dim = input_tokens.shape[1]()
    comptime msg_bytes = token_fmt_type.msg_size()

    var send_buf_layout = RuntimeLayout[
        Layout.row_major(max_tokens_per_rank, msg_bytes),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    comptime recv_layout_static = Layout.row_major(
        n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes
    )
    var recv_buf_layout = RuntimeLayout[
        recv_layout_static,
        element_type = _get_layout_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
        linear_idx_type = _get_index_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    var tid = thread_idx.x
    var num_tokens = input_tokens.dim[0]()

    # The reserved counter is incremented once a warp is ready to send.
    # The finished counter is incremented once the token is sent.
    var expert_reserved_counter = atomic_counter
    var expert_finished_counter = atomic_counter + n_experts

    var my_p2p_world, my_p2p_rank = divmod(my_rank, p2p_world_size)

    # The auxiliary SMs are used for counting counting the number of tokens
    # that need to be sent to each expert. It also monitors the completion of
    # the communication for each expert.
    if block_idx.x < UInt(n_aux_sms):
        var expert_idx: Int32 = block_idx.x * UInt(n_warps) + warp_id()
        var expert_count: Int32 = 0

        if expert_idx < n_experts:
            for i in range(lane_id(), num_tokens * top_k, WARP_SIZE):
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

                var dst_rank = expert_idx // n_local_experts
                var dst_expert_local_idx = expert_idx % n_local_experts
                var signal_offset = recv_count_layout(
                    RtTuple_2(Int(dst_expert_local_idx), Int(my_rank))
                )

                ep_signal_completion[use_shmem](
                    my_rank,
                    dst_rank,
                    recv_count_ptrs,
                    signal_offset,
                    UInt64(expert_count),
                )

                expert_reserved_counter[expert_idx] = 0
                expert_finished_counter[expert_idx] = 0

    # All the other SMs are used for sending the tokens to the experts. A token will
    # first be copied to the send buffer (so the NIC can see it), and then be sent
    # to the remote device.
    else:
        for token_idx in range(
            block_idx.x - UInt(n_aux_sms), num_tokens, n_comm_sms
        ):
            # First, all threads in the block copy the input token to the send buffer.
            var curr_send_buf_ptr = send_buf_p + send_buf_layout(
                RtTuple_2(token_idx, 0)
            )
            var input_tensor_ptr = input_tokens.ptr + input_tokens._offset(
                token_idx, 0
            )
            token_fmt_type.copy_token_to_send_buf[
                input_type, UInt(num_threads)
            ](curr_send_buf_ptr, input_tensor_ptr)

            if tid < UInt(top_k):
                # Store all the top-k expert IDs in current token's message.
                # The remote device will use the expert ID to determine a token's
                # top-k id.
                # Cast the expert ID to a 16-bit integer to save space.
                var top_k_idx = topk_ids.load[width=1](token_idx, Int(tid))
                curr_send_buf_ptr.store[
                    width = size_of[UInt16](),
                    alignment = align_of[DType.uint16](),
                ](
                    token_fmt_type.topk_info_offset()
                    + Int(tid * UInt(size_of[UInt16]())),
                    bitcast[DType.uint8, size_of[UInt16]()](UInt16(top_k_idx)),
                )

                # Store the source token index in current token's message.
                if tid == 0:
                    curr_send_buf_ptr.store[
                        width = size_of[Int32](),
                        alignment = align_of[DType.int32](),
                    ](
                        token_fmt_type.src_info_offset(),
                        bitcast[DType.uint8, size_of[Int32]()](
                            Int32(token_idx)
                        ),
                    )

            barrier()

            # Try to copy the message to the target expert's recv_buf if the target device
            # is on the same node.
            for topk_idx in range(warp_id(), top_k, n_warps):
                var target_expert = topk_ids.load[width=1](token_idx, topk_idx)
                var dst_rank, dst_expert_local_idx = divmod(
                    target_expert, n_local_experts
                )
                var dst_p2p_world, dst_p2p_rank = divmod(
                    dst_rank, p2p_world_size
                )

                if my_p2p_world == dst_p2p_world:
                    var slot_idx: Int32 = 0
                    if lane_id() == 0:
                        slot_idx = Atomic.fetch_add(
                            expert_reserved_counter + target_expert, 1
                        )
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

                    block_memcpy[msg_bytes, WARP_SIZE](
                        dst_recv_buf_ptr,
                        curr_send_buf_ptr,
                        lane_id(),
                    )

                    syncwarp()

                    if lane_id() == 0:
                        _ = Atomic.fetch_add[ordering = Consistency.RELEASE](
                            expert_finished_counter + target_expert, 1
                        )

            # We set up `n_local_experts` Reliable Communications (RCs) for each remote
            # device. We would like to use the same RC for each expert. However, NVSHMEM
            # does not allow us to explicitly specify the RC for each transfer. Instead,
            # we set the environment variable `NVSHMEM_IBGDA_RC_MAP_BY=warp` so that the RC
            # is selected by the warp ID using round-robin. We can then control the RC
            # for each expert by using the correct warp.
            @parameter
            if use_shmem:
                var rc_map_offset: Int32 = (
                    block_idx.x * UInt(n_warps) + warp_id()
                ) % UInt(n_local_experts)

                var topk_idx = lane_id()
                if topk_idx < UInt(top_k) and warp_id() < UInt(n_local_experts):
                    var target_expert = topk_ids.load[width=1](
                        token_idx, Int(topk_idx)
                    )
                    var dst_rank, dst_expert_local_idx = divmod(
                        target_expert, n_local_experts
                    )
                    var dst_p2p_world = dst_rank // p2p_world_size
                    if (
                        rc_map_offset == dst_expert_local_idx
                        and my_p2p_world != dst_p2p_world
                    ):
                        var slot_idx = Atomic.fetch_add(
                            expert_reserved_counter + target_expert, 1
                        )
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
                            UInt(msg_bytes),
                            dst_rank,
                        )

                        _ = Atomic.fetch_add[ordering = Consistency.RELEASE](
                            expert_finished_counter + target_expert, 1
                        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn dispatch_cb_kernel[
    num_threads: Int,
    output_tokens_layout: Layout,
    row_offsets_layout: Layout,
    expert_ids_layout: Layout,
    src_info_layout: Layout,
    n_sms: Int,
    n_aux_sms: Int,
    n_experts: Int,
    n_ranks: Int,
    max_tokens_per_rank: Int,
    token_fmt_type: TokenFormat,
    expert_m_padding: Int = 0,
    fused_shared_expert: Bool = False,
    shared_expert_input_dtype: DType = DType.bfloat16,
](
    format_handler: token_fmt_type,
    row_offsets: LayoutTensor[DType.uint32, row_offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin],
    src_info: LayoutTensor[DType.int32, src_info_layout, MutAnyOrigin],
    recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
    atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
    my_rank: Int32,
    maybe_input_tokens: OptionalReg[
        LayoutTensor[
            shared_expert_input_dtype, Layout.row_major[2](), ImmutAnyOrigin
        ]
    ],
):
    """
    This kernel is called after the `dispatch_kernel` to complete the communication.
    It will keep polling the receive count buffer, and once the count is no longer
    MAX_FINITE, it can confirm that the communication is complete from a remote rank.

    The kernel will also aggregate the tokens from all the experts, and store them in
    the output tensor using a ragged representation.

    Parameters:
        num_threads: The number of threads in the block.
        output_tokens_layout: The layout of the output tokens.
        row_offsets_layout: The layout of the row offsets.
        expert_ids_layout: The layout of the expert IDs.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        n_aux_sms: The number of auxiliary SMs in the device.
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
        recv_buf_p: The pointer to the receive buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)`.
        recv_count_p: The pointer to the receive count buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks)`.
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
        maybe_input_tokens: The optional input tokens for the shared experts.
            If fused_shared_expert is True, this will be used to load the
            input tokens for the shared experts.
    """
    comptime n_local_experts = n_experts // n_ranks
    comptime n_warps = num_threads // WARP_SIZE
    comptime n_comm_sms = n_sms - n_aux_sms

    comptime top_k = token_fmt_type.top_k
    comptime hid_dim = token_fmt_type.hid_dim
    comptime msg_bytes = token_fmt_type.msg_size()
    __comptime_assert (
        n_local_experts <= n_warps
    ), "EP dispatch: local experts per device should be less than " + String(
        WARP_SIZE
    )

    comptime recv_layout_static = Layout.row_major(
        n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes
    )
    var recv_buf_layout = RuntimeLayout[
        recv_layout_static,
        element_type = _get_layout_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
        linear_idx_type = _get_index_type(
            recv_layout_static, AddressSpace.GENERIC
        ),
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    # The first SM is used for checking if any of a local expert has received
    # tokens from all the remote ranks. It will also calculate the offset where
    # the tokens start in the output tensor.
    if block_idx.x < UInt(n_aux_sms):
        var shared_mem = stack_allocation[
            1, DType.uint32, address_space = AddressSpace.SHARED
        ]()
        if thread_idx.x == 0:

            @parameter
            if fused_shared_expert:
                var input_tokens = maybe_input_tokens.value()
                var shared_expert_token_count = input_tokens.dim(0)

                @parameter
                if expert_m_padding != 0:
                    shared_expert_token_count = align_up(
                        Int(shared_expert_token_count), expert_m_padding
                    )

                # Place the shared expert's inputs before all routed experts' inputs.
                shared_mem[] = shared_expert_token_count | 0x01000000
                expert_ids[0] = 0
                row_offsets[0] = 0
                row_offsets[1] = shared_expert_token_count

            else:
                shared_mem[] = 0
        barrier()

        var local_expert_id = warp_id()
        if local_expert_id >= UInt(n_local_experts):
            return

        comptime scan_round = ceildiv(n_ranks, WARP_SIZE)
        var prefix_sum_arr = stack_allocation[
            scan_round, DType.uint32, address_space = AddressSpace.LOCAL
        ]()
        var local_expert_token_count: UInt32 = 0

        # We need to scan the receive count buffer for each rank to get the total
        # number of tokens for the local expert. Also, we calculate the prefix sum
        # to get the offset where each rank ends in the output tensor.
        @parameter
        for round_i in range(scan_round):
            var target_rank = lane_id() + UInt(round_i * WARP_SIZE)
            var expert_rank_offset = recv_count_layout(
                RtTuple_2(Int(local_expert_id), Int(target_rank))
            )

            if target_rank < UInt(n_ranks):
                var target_count_ptr = recv_count_p + expert_rank_offset
                var token_count = load_acquire[scope = Scope.SYSTEM](
                    target_count_ptr
                )
                while token_count == UInt64.MAX_FINITE:
                    token_count = load_acquire[scope = Scope.SYSTEM](
                        target_count_ptr
                    )

                prefix_sum_arr[round_i] = UInt32(token_count)
            else:
                prefix_sum_arr[round_i] = UInt32(0)
            syncwarp()
            prefix_sum_arr[round_i] = warp.prefix_sum(prefix_sum_arr[round_i])
            syncwarp()
            prefix_sum_arr[round_i] += local_expert_token_count
            syncwarp()
            local_expert_token_count = warp.shuffle_idx(
                prefix_sum_arr[round_i], WARP_SIZE - 1
            )

        local_expert_token_count = warp.shuffle_idx(
            prefix_sum_arr[scan_round - 1], (n_ranks - 1) % WARP_SIZE
        )

        @parameter
        if expert_m_padding != 0:
            local_expert_token_count = align_up(
                Int(local_expert_token_count), expert_m_padding
            )

        # Conduct a atomic add to get how many experts have already completed the
        # communication, and the offset where the previous expert end in the output
        # tensor.
        var expert_idx_and_offsets: UInt32 = 0
        if lane_id() == 0:
            expert_idx_and_offsets = Atomic.fetch_add(
                shared_mem, local_expert_token_count | 0x01000000
            )

        # It is unlikely a rank will receive more than 16777216 tokens.
        expert_idx_and_offsets = warp.broadcast(expert_idx_and_offsets)
        var expert_idx = expert_idx_and_offsets >> 24
        var prev_expert_offset = expert_idx_and_offsets & 0x00FFFFFF

        @parameter
        for round_i in range(scan_round):
            var target_rank = lane_id() + UInt(round_i * WARP_SIZE)
            var expert_rank_offset = recv_count_layout(
                RtTuple_2(Int(local_expert_id), Int(target_rank))
            )

            if target_rank < UInt(n_ranks):
                atomic_counter.store(
                    expert_rank_offset * 2,
                    Int32(
                        EP_DATA_READY_FLAG
                        + prev_expert_offset
                        + prefix_sum_arr[round_i]
                    ),
                )

        if lane_id() == 0:
            # The shared expert's inputs are placed before the routed experts' inputs,
            comptime expert_id_offset = 1 if fused_shared_expert else 0

            expert_ids[Int(expert_idx)] = (
                Int(local_expert_id) + expert_id_offset
            )
            row_offsets[Int(expert_idx) + 1] = (
                prev_expert_offset + local_expert_token_count
            )

            if expert_idx == 0:
                row_offsets[0] = 0

    # All the other SMs are used for copying the tokens to the output tensor.
    # The compute resources are partitioned into multiple work groups (wg), and
    # each work group is responsible for copying tokens for a single expert from
    # a remote rank.
    else:
        var sm_id = block_idx.x - UInt(n_aux_sms)

        # If we need to pack the shared expert's inputs, we do that before
        # waiting for the arrival of the routed experts' inputs.
        @parameter
        if fused_shared_expert:
            var smem_buf_p = stack_allocation[
                format_handler.token_size(),
                DType.uint8,
                alignment = simd_width_of[DType.uint8](),
                address_space = AddressSpace.SHARED,
            ]()
            var input_tokens = maybe_input_tokens.value()
            var shared_expert_token_count = input_tokens.dim(0)

            for token_idx in range(
                sm_id, shared_expert_token_count, n_comm_sms
            ):
                var input_tensor_p = input_tokens.ptr + token_idx * hid_dim
                format_handler.copy_token_to_send_buf[
                    shared_expert_input_dtype,
                    UInt(num_threads),
                    buf_addr_space = AddressSpace.SHARED,
                ](smem_buf_p, input_tensor_p)
                barrier()
                format_handler.copy_msg_to_output_tensor(smem_buf_p, token_idx)

        comptime n_wg_per_sm = ceildiv(n_experts, n_comm_sms)
        comptime wg_size = n_warps // n_wg_per_sm
        comptime wg_threads = wg_size * WARP_SIZE

        var wg_idx = warp_id() // UInt(wg_size)
        var global_wg_idx = sm_id * UInt(n_wg_per_sm) + wg_idx
        var warp_id_in_wg = warp_id() % UInt(wg_size)

        if wg_idx >= UInt(n_wg_per_sm) or global_wg_idx >= UInt(n_experts):
            return

        var local_expert_id = global_wg_idx % UInt(n_local_experts)
        var target_rank = global_wg_idx // UInt(n_local_experts)
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

        for token_idx in range(warp_id_in_wg, token_count, wg_size):
            var token_pos = Int(token_idx + output_offset)
            var recv_buf_ptr = recv_buf_p + recv_buf_layout(
                RtTuple_4(
                    Int(local_expert_id),
                    Int(target_rank),
                    Int(token_idx),
                    0,
                )
            )

            format_handler.copy_msg_to_output_tensor(recv_buf_ptr, token_pos)

            if lane_id() < UInt(top_k):
                # Load top-k expert IDs from the token's message.
                var src_topk_idx = bitcast[DType.uint16, 1](
                    recv_buf_ptr.load[width = size_of[UInt16]()](
                        token_fmt_type.topk_info_offset()
                        + Int(lane_id() * UInt(size_of[UInt16]())),
                    )
                )
                var global_expert_idx = (
                    my_rank * n_local_experts + local_expert_id
                )
                if global_expert_idx == Int32(src_topk_idx):
                    # Store the source token index and the top-k id.
                    var src_idx = bitcast[DType.int32, 1](
                        recv_buf_ptr.load[width = size_of[Int32]()](
                            token_fmt_type.src_info_offset()
                        )
                    )

                    src_info[token_pos, 0] = src_idx
                    src_info[token_pos, 1] = lane_id()

        barrier()
        if lane_id() == 0 and warp_id_in_wg == 0:
            recv_count_p.store(expert_rank_offset, UInt64.MAX_FINITE)
            offset_ptr.store(1, token_count)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn combine_kernel[
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
    atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
    my_rank: Int32,
    maybe_output_tokens: OptionalReg[
        LayoutTensor[input_type, Layout.row_major[2](), MutAnyOrigin]
    ],
):
    """
    Send tokens to the original rank based on the src_info tensor.
    This kernel utilizes the non-blocking SHMEM API, and would return immediately
    after initiating the communication. The communication is considered complete
    after calling the `combine_cb_kernel`.

    Parameters:
        input_type: The type of the input tokens.
        num_threads: The number of threads in the block.
        input_tokens_layout: The layout of the input tokens.
        src_info_layout: The layout of the source token info.
        n_sms: The total number of SMs in the device.
        top_k: The number of selected experts per token.
        n_experts: The total number of experts in the model.
        n_ranks: The number of all devices participating in the communication.
        msg_bytes: This is the total number of bytes we need to send for each token.
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
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
        maybe_output_tokens: The optional output for the shared experts.
            If fused_shared_expert is True, this will be used to store the
            output tokens for the shared experts.
    """
    comptime n_local_experts = n_experts // n_ranks
    comptime n_warps = num_threads // WARP_SIZE

    comptime src_simd_width = simd_width_of[input_type]()
    comptime byte_simd_width = simd_width_of[DType.uint8]()

    comptime hid_dim = input_tokens.shape[1]()

    __comptime_assert (
        msg_bytes == hid_dim * size_of[Scalar[input_type]]()
    ), "EP combine: input shape doesn't match message size."
    __comptime_assert (
        msg_bytes % byte_simd_width == 0
    ), "EP combine: message size must be divisible by " + String(
        byte_simd_width
    )

    comptime send_layout_static = Layout.row_major(
        n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes
    )
    var send_buf_layout = RuntimeLayout[
        send_layout_static,
        element_type = _get_layout_type(
            send_layout_static, AddressSpace.GENERIC
        ),
        linear_idx_type = _get_index_type(
            send_layout_static, AddressSpace.GENERIC
        ),
    ]()
    var recv_buf_layout = RuntimeLayout[
        Layout.row_major(max_tokens_per_rank, top_k, msg_bytes),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    var tid = Int(thread_idx.x)
    var sm_id = Int(block_idx.x)
    var my_p2p_world, my_p2p_rank = divmod(my_rank, p2p_world_size)

    # Each rank holds `n_local_experts` experts, and for each expert, it needs to
    # send back different tokens to `n_ranks` remote ranks. We use one block
    # per-expert-per-rank to send back the tokens.
    for global_idx in range(sm_id, n_experts, n_sms):
        var local_expert_id = global_idx % n_local_experts
        var target_rank = global_idx // n_local_experts
        var expert_rank_offset = recv_count_layout(
            RtTuple_2(Int(local_expert_id), Int(target_rank))
        )
        var dst_p2p_world, dst_p2p_rank = divmod(target_rank, p2p_world_size)

        # Info for where the tokens for the current expert and rank start and end
        # are stored in the atomic counter by the `dispatch_cb_kernel`.
        comptime DATA_READY_FLAG = 1024
        var token_end_count = atomic_counter.load[
            width=2,
            alignment = align_of[SIMD[DType.int32, 2]](),
            invariant=True,
        ](2 * expert_rank_offset)
        var token_end = token_end_count[0] - DATA_READY_FLAG
        var token_start = token_end - token_end_count[1]

        # If the target device is on the same node, we can directly copy the tokens
        # to the receive buffer, skipping the send buffer.
        if Int32(dst_p2p_world) == my_p2p_world:
            for token_idx in range(token_start, token_end):
                var src_token_info = src_info.aligned_load[2](Int(token_idx), 0)
                var src_idx = src_token_info[0]
                var src_topk_idx = src_token_info[1]

                var dst_recv_buf_ptr = recv_buf_ptrs[
                    dst_p2p_rank
                ] + recv_buf_layout(
                    RtTuple_3(Int(src_idx), Int(src_topk_idx), 0)
                )
                block_memcpy[hid_dim * size_of[input_type](), num_threads](
                    dst_recv_buf_ptr,
                    input_tokens.ptr_at_offset(
                        IndexList[2](Int(token_idx), 0)
                    ).bitcast[UInt8](),
                    UInt(tid),
                )

        # If the target device is on a different node, we need to send the tokens
        # to the target device using the SHMEM API.
        else:

            @parameter
            if use_shmem:
                # The tokens are sent back to the original rank using the same RC as the
                # one they come from.
                var rc_map_offset = (
                    sm_id * n_warps + Int(warp_id())
                ) % n_local_experts

                var n_rounds = ceildiv(token_end - token_start, n_warps)
                for round_i in range(n_rounds):
                    var token_idx = token_start + round_i * n_warps + warp_id()
                    if token_idx < token_end:
                        var curr_send_buf_ptr = send_buf_p + send_buf_layout(
                            RtTuple_2(Int(token_idx), 0)
                        )

                        # To use SHMEM API, we need to copy the tokens to the
                        # send buffer first.
                        block_memcpy[
                            hid_dim * size_of[input_type](), WARP_SIZE
                        ](
                            curr_send_buf_ptr,
                            input_tokens.ptr_at_offset(
                                IndexList[2](Int(token_idx), 0)
                            ).bitcast[UInt8](),
                            UInt(lane_id()),
                        )

                    barrier()

                    if (
                        warp_id() < UInt(n_local_experts)
                        and local_expert_id == rc_map_offset
                    ):
                        var token_idx = (
                            token_start + round_i * n_warps + lane_id()
                        )
                        if token_idx < token_end:
                            var src_token_info = src_info.aligned_load[2](
                                Int(token_idx), 0
                            )
                            var src_idx = src_token_info[0]
                            var src_topk_idx = src_token_info[1]

                            var curr_send_buf_ptr = (
                                send_buf_p
                                + send_buf_layout(RtTuple_2(Int(token_idx), 0))
                            )
                            var dst_recv_buf_ptr = recv_buf_ptrs[
                                my_p2p_rank
                            ] + recv_buf_layout(
                                RtTuple_3(Int(src_idx), Int(src_topk_idx), 0)
                            )

                            shmem_put_nbi[kind = SHMEMScope.default](
                                dst_recv_buf_ptr,
                                curr_send_buf_ptr,
                                UInt(msg_bytes),
                                target_rank,
                            )

        barrier()

        # Once all the tokens for the current expert and rank have been sent,
        # signal the completion of the communication.
        var rc_map_offset = (sm_id * n_warps + Int(warp_id())) % n_local_experts
        if (
            warp_id() < UInt(n_local_experts)
            and local_expert_id == rc_map_offset
        ):
            if lane_id() == 0:
                var signal_offset = my_rank * n_local_experts + local_expert_id

                ep_signal_completion[use_shmem](
                    my_rank,
                    target_rank,
                    recv_count_ptrs,
                    signal_offset,
                    UInt64(token_end - token_start),
                )

                atomic_counter.store[
                    width=2, alignment = align_of[SIMD[DType.int32, 2]]()
                ](expert_rank_offset * 2, 0)

    @parameter
    if fused_shared_expert:
        var output_tokens = maybe_output_tokens.value()
        var shared_expert_token_count = output_tokens.dim(0)

        for token_idx in range(sm_id, shared_expert_token_count, n_sms):
            var output_tokens_p = output_tokens.ptr + token_idx * hid_dim
            block_memcpy[hid_dim * size_of[input_type](), num_threads](
                output_tokens_p.bitcast[UInt8](),
                input_tokens.ptr_at_offset(
                    IndexList[2](Int(token_idx), 0)
                ).bitcast[UInt8](),
                UInt(tid),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn combine_cb_kernel[
    output_type: DType,
    num_threads: Int,
    output_tokens_layout: Layout,
    n_sms: Int,
    n_aux_sms: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    msg_bytes: Int,
    max_tokens_per_rank: Int,
    router_weights_wrapper: OptionalReg[
        fn (Int, Int) capturing -> Float32
    ] = None,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    output_tokens: LayoutTensor[
        output_type, output_tokens_layout, MutAnyOrigin
    ],
    recv_buf_p: UnsafePointer[UInt8, MutExternalOrigin],
    recv_count_p: UnsafePointer[UInt64, MutExternalOrigin],
    atomic_counter: UnsafePointer[Int32, MutExternalOrigin],
    my_rank: Int32,
):
    """
    This kernel is called after the `combine_kernel` to complete the communication.
    It will keep polling the receive count buffer, and once the count is no longer
    MAX_FINITE, it can confirm that the communication is complete from a remote rank.

    Parameters:
        output_type: The type of the output tokens.
        num_threads: The number of threads in the block.
        output_tokens_layout: The layout of the output tokens.
        n_sms: The total number of SMs in the device.
        n_aux_sms: The number of auxiliary SMs in the device.
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
        recv_count_p: The pointer to the receive count buffer. Need to be allocated using
            `shmem_alloc`. The underlying buffer is of shape
            `(n_local_experts, n_ranks)`.
        atomic_counter: The pointer to the atomic counter.
        my_rank: The rank of the current device.
    """

    comptime n_local_experts = n_experts // n_ranks
    comptime n_warps = num_threads // WARP_SIZE
    comptime n_red_sms = n_sms - n_aux_sms

    comptime dst_simd_width = simd_width_of[output_type]()
    comptime byte_simd_width = simd_width_of[DType.uint8]()

    comptime last_dim = 1 if router_weights_wrapper else 2
    comptime hid_dim = output_tokens_layout.shape[last_dim].value()

    __comptime_assert (
        msg_bytes == hid_dim * size_of[Scalar[output_type]]()
    ), "EP combine: output shape doesn't match message size."
    __comptime_assert (
        msg_bytes % byte_simd_width == 0
    ), "EP combine: message size must be divisible by " + String(
        byte_simd_width
    )

    var recv_buf_layout = RuntimeLayout[
        Layout.row_major(max_tokens_per_rank, top_k, msg_bytes),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    var recv_count_layout = RuntimeLayout[
        Layout.row_major(n_local_experts, n_ranks),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    comptime DATA_READY_FLAG = 1024
    comptime _align = align_of[SIMD[DType.uint8, byte_simd_width]]()

    # `num_tokens` is the total number of tokens before the EP communication. The
    # actual number of tokens we receive is `num_tokens * top_k`.
    var num_tokens = output_tokens.dim[0]()
    var tid = Int(thread_idx.x)
    var sm_id = Int(block_idx.x)

    # The first SM is used for checking if we have received tokens from all the
    # remote ranks.
    if sm_id < n_aux_sms:
        if tid < n_experts:
            var target_count_ptr = recv_count_p + tid
            while (
                load_acquire[scope = Scope.SYSTEM](target_count_ptr)
                == UInt64.MAX_FINITE
            ):
                pass

            target_count_ptr[] = UInt64.MAX_FINITE
        barrier()

        # Once all the tokens have been received, set flags for other SMs to copy
        # the tokens to the output tensor.
        if tid < n_red_sms:
            atomic_counter.store(n_aux_sms + tid, DATA_READY_FLAG)

    # All the other SMs are used for copying the tokens to the output tensor.
    else:
        if tid == 0:
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

        __comptime_assert (
            hid_dim % n_chunk_elems == 0
        ), "EP combine: hid_dim must be divisible by n_chunk_elems"

        # This will allow a single token to be processed by multiple blocks.
        # Reduce the latency when there is only a small number of tokens.
        var global_id = sm_id - n_aux_sms + Int(warp_id()) * n_red_sms

        for chunk_idx in range(
            global_id, num_tokens * n_chunks_per_tok, n_warps * n_red_sms
        ):
            var token_idx, chunk_idx_in_token = divmod(
                chunk_idx, n_chunks_per_tok
            )

            var accum = SIMD[DType.float32, dst_simd_width](0)
            var recv_chunk = SIMD[output_type, dst_simd_width](0)

            @parameter
            for topk_idx in range(top_k):
                var recv_buf_ptr = recv_buf_p + recv_buf_layout(
                    RtTuple_3(
                        token_idx,
                        topk_idx,
                        Int(chunk_idx_in_token * n_chunk_bytes),
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

                    var weight = router_weights_fn(token_idx, topk_idx)
                    accum += weight * recv_chunk.cast[DType.float32]()

                else:
                    # The output tensor is of shape `(num_tokens, top_k, hid_dim)`.
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
                            Int(token_idx),
                            Int(
                                chunk_idx_in_token * n_chunk_elems
                                + Int(lane_id()) * dst_simd_width
                            ),
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


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
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
    comptime input_dim = input_tensor.shape[1]()
    comptime output_dim = output_tensor.shape[1]()
    comptime simd_width = simd_width_of[input_dtype]()

    # This should also make sure the input and output tensors has static shape.
    __comptime_assert (
        input_dim == output_dim * 2
    ), "Input dimension must be twice the output dimension."
    __comptime_assert (
        output_dim % simd_width == 0
    ), "Output dimension must be divisible by the SIMD width."

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var gid = tid + bid * num_threads

    with PDL():
        var num_tokens = row_offsets[row_offsets.size() - 1]
        var num_elem = num_tokens * output_dim

        for i in range(gid, num_elem // simd_width, num_threads * num_sms):
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
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
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
    comptime input_dim = input_tensor.shape[1]()
    comptime output_dim = output_tensor.shape[1]()
    comptime simd_width = simd_width_of[input_dtype]()

    __comptime_assert (
        input_dim == output_dim * 2
    ), "Input dimension must be twice the output dimension."
    __comptime_assert (
        output_dim % simd_width == 0
    ), "Output dimension must be divisible by the SIMD width."

    comptime n_threads_per_group = group_size // simd_width
    __comptime_assert (
        WARP_SIZE % n_threads_per_group == 0
    ), "Each warp must process a multiple of quantization groups"
    comptime fp8_max_t = Scalar[fp8_dtype].MAX_FINITE.cast[accum_dtype]()

    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var gid = tid + bid * num_threads

    with PDL():
        var num_tokens = row_offsets[row_offsets.size() - 1]
        var num_elem = num_tokens * output_dim

        for i in range(gid, num_elem // simd_width, num_threads * num_sms):
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
