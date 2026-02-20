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

from io.io import _printf
from random import randint, randn, seed
from sys import (
    align_of,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    simd_width_of,
)

from algorithm import sync_parallelize
from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchmarkInfo,
    BenchId,
    BenchMetric,
    Report,
    ThroughputMeasure,
)
from comm.sync import can_enable_p2p
from gpu.host import DeviceBuffer, DeviceContext
from layout import UNKNOWN_VALUE, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from math import ceildiv
from shmem.ep import (
    ep_dispatch_async_kernel_api,
    ep_dispatch_wait_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EP_DATA_READY_FLAG,
    EPLocalSyncCounters,
    NVFP4TokenFormat,
    TokenFormat,
)
from testing import assert_almost_equal, assert_equal
from utils import IndexList

from linalg.fp4_utils import (
    E2M1_TO_FLOAT32,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
)
from gpu.host.info import B200


fn legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: UnsafePointer[Int32, MutExternalOrigin], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids + tok_id * top_k

        # The top-k ids for a token should be unique. If not, we will assign a
        # random id to the duplicate id.
        fn is_duplicate() -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if topk_ids_for_token[i] == topk_ids_for_token[j]:
                        return i
            return -1

        var duplicate_idx = is_duplicate()
        while duplicate_idx != -1:
            randint(topk_ids_for_token + duplicate_idx, 1, 0, n_experts - 1)
            duplicate_idx = is_duplicate()


trait DispatchTestT(ImplicitlyDestructible):
    """Trait to unify the test dispatch logic for different token formats."""

    comptime hidden_size: Int
    comptime top_k: Int
    comptime n_experts: Int
    comptime n_ranks: Int
    comptime n_slots: Int
    comptime n_tokens_per_rank: Int
    comptime TokenFormatType: TokenFormat

    fn __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        ...

    @always_inline
    fn get_token_handler(
        self, dev_idx: Int, slot_idx: Int, out result: Self.TokenFormatType
    ):
        ...

    @always_inline
    fn save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        ...

    @always_inline
    fn check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        ...


struct BF16DispatchTest[
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = Self.n_experts * Self.n_tokens_per_rank

    comptime output_layout = Layout.row_major(
        Self.n_tokens_per_rank * Self.n_experts, Self.hidden_size
    )
    comptime TokenFormatType = BF16TokenFormat[
        output_layout = Self.output_layout, Self.hidden_size, Self.top_k
    ]

    var device_output_bufs_list: List[DeviceBuffer[DType.bfloat16]]
    var host_output_bufs_list: List[
        UnsafePointer[Scalar[DType.bfloat16], MutExternalOrigin]
    ]

    fn __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[DType.bfloat16]](
            capacity=Self.n_ranks
        )
        self.host_output_bufs_list = List[
            UnsafePointer[Scalar[DType.bfloat16], MutExternalOrigin]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[DType.bfloat16](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )
            self.host_output_bufs_list.append(
                alloc[Scalar[DType.bfloat16]](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )

    fn __del__(deinit self):
        for i in range(Self.n_ranks):
            self.host_output_bufs_list[i].free()

    @always_inline
    fn get_token_handler(
        self, dev_idx: Int, slot_idx: Int, out result: Self.TokenFormatType
    ):
        var output_tensor = LayoutTensor[
            DType.bfloat16, Self.output_layout, MutAnyOrigin
        ](
            self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.hidden_size,
        )
        result = Self.TokenFormatType(output_tensor)

    @always_inline
    fn save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        for i in range(Self.n_ranks):
            list_of_ctx[i].enqueue_copy(
                self.host_output_bufs_list[i], self.device_output_bufs_list[i]
            )
            list_of_ctx[i].synchronize()

    @always_inline
    fn check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        device_val = self.host_output_bufs_list[dev_idx][
            slot_idx * Self.max_recv_num_tokens * Self.hidden_size
            + token_idx * Self.hidden_size
            + hid_dim_idx
        ]

        assert_equal(
            device_val,
            expected_val,
            "Output value mismatch for dev "
            + String(dev_idx)
            + " slot "
            + String(slot_idx)
            + " token "
            + String(token_idx)
            + " hid_dim "
            + String(hid_dim_idx),
        )


struct BlockwiseFP8DispatchTest[
    fp8_dtype: DType,
    scales_dtype: DType,
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime group_size = 128
    comptime max_recv_num_tokens = Self.n_experts * Self.n_tokens_per_rank

    comptime output_layout = Layout.row_major(
        Self.n_tokens_per_rank * Self.n_experts, Self.hidden_size
    )
    comptime output_scales_layout = Layout.row_major(
        Self.hidden_size // Self.group_size,
        Self.n_tokens_per_rank * Self.n_experts,
    )
    comptime TokenFormatType = BlockwiseFP8TokenFormat[
        fp8_dtype = Self.fp8_dtype,
        scales_dtype = Self.scales_dtype,
        output_layout = Self.output_layout,
        scales_layout = Self.output_scales_layout,
        Self.hidden_size,
        Self.top_k,
    ]

    var device_output_bufs_list: List[DeviceBuffer[Self.fp8_dtype]]
    var device_output_scales_bufs_list: List[DeviceBuffer[Self.scales_dtype]]
    var host_output_bufs_list: List[
        UnsafePointer[Scalar[Self.fp8_dtype], MutExternalOrigin]
    ]
    var host_output_scales_bufs_list: List[
        UnsafePointer[Scalar[Self.scales_dtype], MutExternalOrigin]
    ]

    fn __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[Self.fp8_dtype]](
            capacity=Self.n_ranks
        )
        self.device_output_scales_bufs_list = List[
            DeviceBuffer[Self.scales_dtype]
        ](capacity=Self.n_ranks)
        self.host_output_bufs_list = List[
            UnsafePointer[Scalar[Self.fp8_dtype], MutExternalOrigin]
        ](capacity=Self.n_ranks)
        self.host_output_scales_bufs_list = List[
            UnsafePointer[Scalar[Self.scales_dtype], MutExternalOrigin]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.fp8_dtype](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )
            self.device_output_scales_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.scales_dtype](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.hidden_size
                    // Self.group_size
                )
            )
            self.host_output_bufs_list.append(
                alloc[Scalar[Self.fp8_dtype]](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )
            self.host_output_scales_bufs_list.append(
                alloc[Scalar[Self.scales_dtype]](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.hidden_size
                    // Self.group_size
                )
            )

    fn __del__(deinit self):
        for i in range(Self.n_ranks):
            self.host_output_bufs_list[i].free()
            self.host_output_scales_bufs_list[i].free()

    @always_inline
    fn get_token_handler(
        self, dev_idx: Int, slot_idx: Int, out result: Self.TokenFormatType
    ):
        var output_tensor = LayoutTensor[
            Self.fp8_dtype, Self.output_layout, MutAnyOrigin
        ](
            self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.hidden_size,
        )
        var output_scales_tensor = LayoutTensor[
            Self.scales_dtype, Self.output_scales_layout, MutAnyOrigin
        ](
            self.device_output_scales_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx
            * Self.max_recv_num_tokens
            * Self.hidden_size
            // Self.group_size,
        )
        result = Self.TokenFormatType(output_tensor, output_scales_tensor)

    @always_inline
    fn save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        for i in range(Self.n_ranks):
            list_of_ctx[i].enqueue_copy(
                self.host_output_bufs_list[i], self.device_output_bufs_list[i]
            )
            list_of_ctx[i].enqueue_copy(
                self.host_output_scales_bufs_list[i],
                self.device_output_scales_bufs_list[i],
            )
            list_of_ctx[i].synchronize()

    @always_inline
    fn check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        var output_offset = (
            slot_idx * Self.max_recv_num_tokens * Self.hidden_size
            + token_idx * Self.hidden_size
            + hid_dim_idx
        )
        var scale_offset = (
            slot_idx
            * Self.max_recv_num_tokens
            * Self.hidden_size
            // Self.group_size
            + (hid_dim_idx // Self.group_size) * Self.max_recv_num_tokens
            + token_idx
        )
        var fp8_val = self.host_output_bufs_list[dev_idx][output_offset]
        var token_scale = self.host_output_scales_bufs_list[dev_idx][
            scale_offset
        ]
        var token_val = fp8_val.cast[Self.scales_dtype]() * token_scale

        assert_almost_equal(
            expected_val,
            token_val.cast[DType.bfloat16](),
            "Output value mismatch for dev "
            + String(dev_idx)
            + " slot "
            + String(slot_idx)
            + " token "
            + String(token_idx)
            + " hid_dim "
            + String(hid_dim_idx),
            rtol=1e-1,
            atol=1e-1,
        )


struct NVFP4DispatchTest[
    fp4_dtype: DType,
    scales_dtype: DType,
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = Self.n_experts * Self.n_tokens_per_rank
    comptime n_local_experts = Self.n_experts // Self.n_ranks

    comptime scales_padded_size = Self.n_local_experts * ceildiv(
        Self.max_recv_num_tokens * Self.n_ranks, SF_MN_GROUP_SIZE
    ) * SF_MN_GROUP_SIZE

    comptime uint8_last_dim = Self.hidden_size // 2

    comptime output_layout = Layout.row_major(
        Self.n_tokens_per_rank * Self.n_experts, Self.uint8_last_dim
    )
    comptime output_scales_layout = Layout.row_major(
        Self.scales_padded_size // SF_MN_GROUP_SIZE,
        ceildiv(Self.hidden_size, SF_ATOM_K * NVFP4_SF_VECTOR_SIZE),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime output_scales_offset_layout = Layout.row_major(
        Self.n_experts // Self.n_ranks,
    )

    comptime TokenFormatType = NVFP4TokenFormat[
        fp4_dtype = Self.fp4_dtype,
        scales_dtype = Self.scales_dtype,
        output_layout = Self.output_layout,
        scales_layout = Self.output_scales_layout,
        scales_offset_layout = Self.output_scales_offset_layout,
        Self.hidden_size,
        Self.top_k,
    ]

    var device_output_bufs_list: List[DeviceBuffer[Self.fp4_dtype]]
    var device_output_scales_bufs_list: List[DeviceBuffer[Self.scales_dtype]]
    var device_output_scales_offset_bufs_list: List[DeviceBuffer[DType.uint32]]
    var host_output_bufs_list: List[
        UnsafePointer[Scalar[Self.fp4_dtype], MutExternalOrigin]
    ]
    var host_output_scales_bufs_list: List[
        UnsafePointer[Scalar[Self.scales_dtype], MutExternalOrigin]
    ]
    var host_output_scales_offset_bufs_list: List[
        UnsafePointer[Scalar[DType.uint32], MutExternalOrigin]
    ]

    fn __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[Self.fp4_dtype]](
            capacity=Self.n_ranks
        )
        self.device_output_scales_bufs_list = List[
            DeviceBuffer[Self.scales_dtype]
        ](capacity=Self.n_ranks)
        self.device_output_scales_offset_bufs_list = List[
            DeviceBuffer[DType.uint32]
        ](capacity=Self.n_ranks)
        self.host_output_bufs_list = List[
            UnsafePointer[Scalar[Self.fp4_dtype], MutExternalOrigin]
        ](capacity=Self.n_ranks)
        self.host_output_scales_bufs_list = List[
            UnsafePointer[Scalar[Self.scales_dtype], MutExternalOrigin]
        ](capacity=Self.n_ranks)
        self.host_output_scales_offset_bufs_list = List[
            UnsafePointer[Scalar[DType.uint32], MutExternalOrigin]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.fp4_dtype](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.uint8_last_dim
                )
            )
            self.device_output_scales_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.scales_dtype](
                    Self.n_slots
                    * Self.scales_padded_size
                    * Self.hidden_size
                    // NVFP4_SF_VECTOR_SIZE
                )
            )
            self.device_output_scales_offset_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[DType.uint32](
                    Self.n_slots * (Self.n_experts // Self.n_ranks)
                )
            )
            self.host_output_bufs_list.append(
                alloc[Scalar[Self.fp4_dtype]](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.uint8_last_dim
                )
            )
            self.host_output_scales_bufs_list.append(
                alloc[Scalar[Self.scales_dtype]](
                    Self.n_slots
                    * Self.scales_padded_size
                    * Self.hidden_size
                    // NVFP4_SF_VECTOR_SIZE
                )
            )
            self.host_output_scales_offset_bufs_list.append(
                alloc[Scalar[DType.uint32]](
                    Self.n_slots * (Self.n_experts // Self.n_ranks)
                )
            )

    fn __del__(deinit self):
        for i in range(Self.n_ranks):
            self.host_output_bufs_list[i].free()
            self.host_output_scales_bufs_list[i].free()
            self.host_output_scales_offset_bufs_list[i].free()

    @always_inline
    fn get_token_handler(
        self, dev_idx: Int, slot_idx: Int, out result: Self.TokenFormatType
    ):
        var output_tensor = LayoutTensor[
            Self.fp4_dtype, Self.output_layout, MutAnyOrigin
        ](
            self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.uint8_last_dim,
        )
        var output_scales_tensor = LayoutTensor[
            Self.scales_dtype, Self.output_scales_layout, MutAnyOrigin
        ](
            self.device_output_scales_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx
            * Self.scales_padded_size
            * Self.hidden_size
            // NVFP4_SF_VECTOR_SIZE
        )
        var output_scales_offset_tensor = LayoutTensor[
            DType.uint32, Self.output_scales_offset_layout, MutAnyOrigin
        ](
            self.device_output_scales_offset_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * (Self.n_experts // Self.n_ranks)
        )

        result = Self.TokenFormatType(
            output_tensor, output_scales_tensor, output_scales_offset_tensor
        )

    @always_inline
    fn save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        for i in range(Self.n_ranks):
            list_of_ctx[i].enqueue_copy(
                self.host_output_bufs_list[i], self.device_output_bufs_list[i]
            )
            list_of_ctx[i].enqueue_copy(
                self.host_output_scales_bufs_list[i],
                self.device_output_scales_bufs_list[i],
            )
            list_of_ctx[i].enqueue_copy(
                self.host_output_scales_offset_bufs_list[i],
                self.device_output_scales_offset_bufs_list[i],
            )
            list_of_ctx[i].synchronize()

    @always_inline
    fn check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        var output_offset = (
            slot_idx * Self.max_recv_num_tokens * Self.uint8_last_dim
            + token_idx * Self.uint8_last_dim
            + (hid_dim_idx // 2)
        )

        var host_scales_tensor = LayoutTensor[
            Self.scales_dtype, Self.output_scales_layout, MutAnyOrigin
        ](
            self.host_output_scales_bufs_list[dev_idx]
            + slot_idx
            * Self.scales_padded_size
            * Self.hidden_size
            // NVFP4_SF_VECTOR_SIZE
        )

        var expert_start_index = token_idx - expert_token_idx
        var scales_block_id = (
            UInt32(expert_start_index // SF_MN_GROUP_SIZE)
            + self.host_output_scales_offset_bufs_list[dev_idx][
                slot_idx * (Self.n_experts // Self.n_ranks) + expert_idx
            ]
        )
        var _scales_tensor = type_of(host_scales_tensor)(
            host_scales_tensor.ptr_at_offset(
                IndexList[5](Int(scales_block_id), 0, 0, 0, 0)
            )
        )

        var uint8_val = self.host_output_bufs_list[dev_idx][output_offset]

        var token_scale = get_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE](
            _scales_tensor, expert_token_idx, hid_dim_idx
        )
        var token_val = (
            E2M1_TO_FLOAT32[
                Int(
                    (
                        uint8_val
                        >> Scalar[Self.fp4_dtype](((hid_dim_idx % 2) * 4))
                    )
                    & 0x0F
                )
            ]
            * token_scale.cast[DType.float32]()
        )

        assert_almost_equal(
            expected_val,
            token_val.cast[DType.bfloat16](),
            "Scaled by :"
            + String(token_scale)
            + "Output value mismatch for dev "
            + String(dev_idx)
            + " slot "
            + String(slot_idx)
            + " token "
            + String(token_idx)
            + " hid_dim "
            + String(hid_dim_idx),
            rtol=2.5e-1,
            atol=2.5e-1,
        )


fn test_dispatch_common[
    DispatchTestType: DispatchTestT,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime input_type = DType.bfloat16
    comptime hidden_size = DispatchTestType.hidden_size
    comptime top_k = DispatchTestType.top_k
    comptime n_experts = DispatchTestType.n_experts
    comptime n_ranks = DispatchTestType.n_ranks
    comptime n_slots = DispatchTestType.n_slots
    comptime n_tokens_per_rank = DispatchTestType.n_tokens_per_rank
    comptime token_fmt_type = DispatchTestType.TokenFormatType

    comptime msg_bytes = token_fmt_type.msg_size()
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank

    comptime num_bytes = msg_bytes * top_k * n_tokens_per_rank

    var dispatch_test = DispatchTestType(list_of_ctx)

    print(
        "Running ep_dispatch test:",
        token_fmt_type.get_type_name(),
        "hidden_size:",
        hidden_size,
        "top_k:",
        top_k,
        "n_experts:",
        n_experts,
        "n_ranks:",
        n_ranks,
        "n_tokens_per_rank:",
        n_tokens_per_rank,
    )

    # fmt: off
    var send_bufs_list = List[DeviceBuffer[DType.uint8]](capacity=n_ranks)
    var recv_bufs_list = List[DeviceBuffer[DType.uint8]](capacity=n_ranks)
    var recv_count_bufs_list = List[DeviceBuffer[DType.uint64]](capacity=n_ranks)
    var atomic_counters_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)

    var host_topk_ids_list = InlineArray[UnsafePointer[Int32, MutExternalOrigin], n_ranks](fill={})
    var host_input_tokens_list = InlineArray[UnsafePointer[Scalar[input_type], MutExternalOrigin], n_ranks](fill={})

    var device_topk_bufs_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)
    var device_input_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var device_row_offsets_bufs_list = List[DeviceBuffer[DType.uint32]](capacity=n_ranks)
    var device_expert_ids_bufs_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)
    var device_src_token_info_bufs_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)


    for i in range(n_ranks):
        var ctx = list_of_ctx[i]
        send_bufs_list.append(list_of_ctx[i].enqueue_create_buffer[DType.uint8](n_slots * n_tokens_per_rank * msg_bytes))
        recv_bufs_list.append(ctx.enqueue_create_buffer[DType.uint8](n_slots * max_recv_num_tokens * msg_bytes))
        recv_count_bufs_list.append(ctx.enqueue_create_buffer[DType.uint64](n_slots * n_experts))
        atomic_counters_list.append(ctx.enqueue_create_buffer[DType.int32](
            n_slots * EPLocalSyncCounters[n_experts].total_size()
        ))
        ctx.enqueue_memset(atomic_counters_list[i], Int32(0))
        ctx.enqueue_memset(recv_count_bufs_list[i], UInt64.MAX_FINITE)

        host_topk_ids_list[i] = alloc[Int32](n_slots * n_tokens_per_rank * top_k)
        host_input_tokens_list[i] = alloc[Scalar[input_type]](n_slots * n_tokens_per_rank * hidden_size)

        device_topk_bufs_list.append(ctx.enqueue_create_buffer[DType.int32](n_slots * n_tokens_per_rank * top_k))
        device_input_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * n_tokens_per_rank * hidden_size))
        device_row_offsets_bufs_list.append(ctx.enqueue_create_buffer[DType.uint32](n_slots * (n_local_experts + 1)))
        device_expert_ids_bufs_list.append(ctx.enqueue_create_buffer[DType.int32](n_slots * n_local_experts))
        device_src_token_info_bufs_list.append(ctx.enqueue_create_buffer[DType.int32](n_slots * max_recv_num_tokens * 2))
    # fmt: on

    comptime topk_ids_layout = Layout.row_major(UNKNOWN_VALUE, top_k)
    comptime input_tokens_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    comptime output_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, hidden_size
    )
    comptime row_offsets_layout = Layout.row_major(n_local_experts + 1)
    comptime expert_ids_layout = Layout.row_major(n_local_experts)
    comptime src_token_info_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, 2
    )
    comptime ptrs_layout = Layout.row_major(n_ranks)
    comptime counters_layout = Layout.row_major(
        EPLocalSyncCounters[n_experts].total_size()
    )

    # Initialize the inputs
    for dev_idx in range(n_ranks):
        var ctx = list_of_ctx[dev_idx]
        # Initialize the topk ids and input tokens using fixed seed,
        seed(dev_idx)
        randint(
            host_topk_ids_list[dev_idx],
            n_slots * n_tokens_per_rank * top_k,
            0,
            n_experts - 1,
        )

        # The topk ids for a token is the expert id it needs to be sent to.
        # Since a token won't be sent to the same expert multiple times, we
        # need to legalize the topk ids to make sure they are unique for
        # each token.
        legalize_topk_ids[n_experts, top_k](
            host_topk_ids_list[dev_idx], n_slots * n_tokens_per_rank
        )

        randn(
            host_input_tokens_list[dev_idx],
            n_slots * n_tokens_per_rank * hidden_size,
        )

        ctx.enqueue_copy(
            device_topk_bufs_list[dev_idx], host_topk_ids_list[dev_idx]
        )
        ctx.enqueue_copy(
            device_input_bufs_list[dev_idx], host_input_tokens_list[dev_idx]
        )

    # fmt: off
    var send_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)
    var recv_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)
    var recv_count_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)

    for slot_idx in range(n_slots):
        for dev_idx in range(n_ranks):
            var ptr_idx = slot_idx * n_ranks + dev_idx
            send_ptrs_inputs[ptr_idx] = UInt64(
                Int(send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * msg_bytes)
            )
            recv_ptrs_inputs[ptr_idx] = UInt64(
                Int(recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * msg_bytes)
            )
            recv_count_ptrs_inputs[ptr_idx] = UInt64(
                Int(recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts)
            )

    @always_inline
    @parameter
    fn get_send_ptrs_tensor(slot_idx: Int, out result: LayoutTensor[DType.uint64, ptrs_layout, ImmutAnyOrigin]) raises:
        return type_of(result)(send_ptrs_inputs + slot_idx * n_ranks)

    @always_inline
    @parameter
    fn get_recv_ptrs_tensor(slot_idx: Int, out result: LayoutTensor[DType.uint64, ptrs_layout, ImmutAnyOrigin]) raises:
        return type_of(result)(recv_ptrs_inputs + slot_idx * n_ranks)

    @always_inline
    @parameter
    fn get_recv_count_ptrs_tensor( slot_idx: Int, out result: LayoutTensor[DType.uint64, ptrs_layout, ImmutAnyOrigin]) raises:
        return type_of(result)(recv_count_ptrs_inputs + slot_idx * n_ranks)

    @always_inline
    @parameter
    fn get_atomic_counters_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.int32, counters_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            atomic_counters_list[dev_idx].unsafe_ptr() + slot_idx * EPLocalSyncCounters[n_experts].total_size()
        )

    @always_inline
    @parameter
    fn get_topk_ids_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.int32, topk_ids_layout, ImmutAnyOrigin]) raises:
        return type_of(result)(
            device_topk_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k,
            RuntimeLayout[topk_ids_layout].row_major(
                IndexList[2](n_tokens_per_rank, top_k)
            )
        )

    @always_inline
    @parameter
    fn get_input_tokens_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[input_type, input_tokens_layout, ImmutAnyOrigin]) raises:
        return type_of(result)(
            device_input_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * hidden_size,
            RuntimeLayout[input_tokens_layout].row_major(
                IndexList[2](n_tokens_per_rank, hidden_size)
            )
        )

    @always_inline
    @parameter
    fn get_row_offsets_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.uint32, row_offsets_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_row_offsets_bufs_list[dev_idx].unsafe_ptr() + slot_idx * (n_local_experts + 1),
            RuntimeLayout[row_offsets_layout].row_major(
                IndexList[1](n_local_experts + 1)
            )
        )

    @always_inline
    @parameter
    fn get_expert_ids_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_expert_ids_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_local_experts,
            RuntimeLayout[expert_ids_layout].row_major(
                IndexList[1](n_local_experts)
            )
        )

    @always_inline
    @parameter
    fn get_src_token_info_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.int32, src_token_info_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_src_token_info_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * 2,
            RuntimeLayout[src_token_info_layout].row_major(
                IndexList[2](max_recv_num_tokens, 2)
            )
        )
    # fmt: on

    @always_inline
    @parameter
    fn run_dispatch_async(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
            use_shmem=False,
        ](
            get_atomic_counters_tensor(dev_idx, slot_idx),
            get_input_tokens_tensor(dev_idx, slot_idx),
            get_topk_ids_tensor(dev_idx, slot_idx),
            get_send_ptrs_tensor(slot_idx),
            get_recv_ptrs_tensor(slot_idx),
            get_recv_count_ptrs_tensor(slot_idx),
            ctx,
        )

    @always_inline
    @parameter
    fn run_dispatch_async_wait(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        var format_handler = dispatch_test.get_token_handler(dev_idx, slot_idx)
        ep_dispatch_wait_kernel_api[
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
        ](
            format_handler,
            get_row_offsets_tensor(dev_idx, slot_idx),
            get_expert_ids_tensor(dev_idx, slot_idx),
            get_src_token_info_tensor(dev_idx, slot_idx),
            get_recv_ptrs_tensor(slot_idx),
            get_recv_count_ptrs_tensor(slot_idx),
            get_atomic_counters_tensor(dev_idx, slot_idx),
            ctx,
        )

    @always_inline
    @parameter
    fn run_e2e(dev_idx: Int, slot_idx: Int) raises:
        run_dispatch_async(dev_idx, slot_idx)
        run_dispatch_async_wait(dev_idx, slot_idx)

    @always_inline
    @parameter
    fn clean_up(dev_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_memset(atomic_counters_list[dev_idx], Int32(0))
        ctx.enqueue_memset(recv_count_bufs_list[dev_idx], UInt64.MAX_FINITE)

    # warm up by running once
    for dev_i in range(n_ranks):
        run_e2e(dev_i, 0)

    for dev_i in range(n_ranks):
        clean_up(dev_i)
        list_of_ctx[dev_i].synchronize()

    # Necessary to fill this InlineArray w/ default BenchmarkInfo
    # otherwise each thread attempts to free uninitialized BenchmarkInfo
    # when copying below
    var default_info = BenchmarkInfo(
        name="",
        result=Report(),
        measures=List[ThroughputMeasure](),
    )
    var results_b = InlineArray[BenchmarkInfo, n_ranks](fill=default_info)

    # First, bench the dispatch kernel overhead

    @parameter
    fn per_gpu_dispatch(i: Int) raises:
        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn call_fn(ctx: DeviceContext, cache_iter: Int) raises:
                var dev_id = Int(ctx.id())
                run_dispatch_async(dev_id, cache_iter)

            b.iter_custom[call_fn](list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function[bench_iter](
            BenchId("bench dispatch"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize[per_gpu_dispatch](n_ranks)

    var max_time = 0.0
    var max_loc = 0

    for i in range(n_ranks):
        var val = results_b[i].result.mean(unit="ms")
        if val > max_time:
            max_time = val
            max_loc = i

    var b_final = Bench()
    b_final.info_vec.append(results_b[max_loc].copy())
    b_final.dump_report()

    # Then, bench the dispatch_wait kernel overhead
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @parameter
    fn per_gpu_dispatch_wait(i: Int) raises:
        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn call_fn(ctx: DeviceContext, cache_iter: Int) raises:
                var dev_id = Int(ctx.id())
                run_dispatch_async_wait(dev_id, cache_iter)

            b.iter_custom[call_fn](list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function[bench_iter](
            BenchId("bench dispatch_wait"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize[per_gpu_dispatch_wait](n_ranks)

    max_time = 0.0
    max_loc = 0

    for i in range(n_ranks):
        var val = results_b[i].result.mean(unit="ms")
        if val > max_time:
            max_time = val
            max_loc = i

    b_final = Bench()
    b_final.info_vec.append(results_b[max_loc].copy())
    b_final.dump_report()

    # We don't enable e2e benchmarking by default because it would hang
    # if AsyncRT has less than n_ranks worker threads.
    comptime if bench_e2e:
        for dev_i in range(n_ranks):
            clean_up(dev_i)
            list_of_ctx[dev_i].synchronize()

        @parameter
        fn per_gpu_e2e(i: Int) raises:
            @parameter
            @always_inline
            fn bench_iter(mut b: Bencher) raises:
                @parameter
                @always_inline
                fn call_fn(ctx: DeviceContext, cache_iter: Int) raises:
                    var dev_id = Int(ctx.id())
                    run_dispatch_async(dev_id, cache_iter + 1)
                    run_dispatch_async_wait(dev_id, cache_iter + 1)

                b.iter_custom[call_fn](list_of_ctx[i])

            run_e2e(i, 0)
            list_of_ctx[i].synchronize()

            var bench_config = BenchConfig()
            bench_config.show_progress = False
            var b = Bench(bench_config^)
            b.bench_function[bench_iter](
                BenchId("bench dispatch e2e"),
                [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
                fixed_iterations=n_slots - 1,
            )
            results_b[i] = b.info_vec[0].copy()

        sync_parallelize[per_gpu_e2e](n_ranks)

        max_time = 0.0
        max_loc = 0

        for i in range(n_ranks):
            var val = results_b[i].result.mean(unit="ms")
            if val > max_time:
                max_time = val
                max_loc = i

        b_final = Bench()
        b_final.info_vec.append(results_b[max_loc].copy())
        b_final.dump_report()

    # Verify the results for each device and each slot
    print("Verifying results...")

    @parameter
    @always_inline
    fn verify_results(dev_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]

        # Allocate host buffers for copying device outputs
        var host_output = alloc[Scalar[input_type]](
            n_slots * max_recv_num_tokens * hidden_size
        )
        var host_row_offsets = alloc[UInt32](n_slots * (n_local_experts + 1))
        var host_expert_ids = alloc[Int32](n_slots * n_local_experts)
        var host_src_token_info = alloc[Int32](
            n_slots * max_recv_num_tokens * 2
        )
        var host_atomic_counter = alloc[Int32](
            n_slots * EPLocalSyncCounters[n_experts].total_size()
        )

        # Copy device outputs to host
        ctx.enqueue_copy(
            host_row_offsets, device_row_offsets_bufs_list[dev_idx]
        )
        ctx.enqueue_copy(host_expert_ids, device_expert_ids_bufs_list[dev_idx])
        ctx.enqueue_copy(
            host_src_token_info, device_src_token_info_bufs_list[dev_idx]
        )
        ctx.enqueue_copy(host_atomic_counter, atomic_counters_list[dev_idx])
        ctx.synchronize()

        # Check results for each slot
        for slot_idx in range(n_slots):
            # Get pointers to this slot's data
            var slot_expert_ids = host_expert_ids + slot_idx * n_local_experts
            var slot_src_token_info = (
                host_src_token_info + slot_idx * max_recv_num_tokens * 2
            )
            var slot_dispatch_wait_counter = EPLocalSyncCounters[n_experts](
                host_atomic_counter
                + slot_idx * EPLocalSyncCounters[n_experts].total_size()
            ).get_dispatch_wait_ptr()

            # Check if we have received the correct number of tokens
            var expert_start_idx = n_local_experts * dev_idx
            var expert_end_idx = expert_start_idx + n_local_experts
            var expected_tokens = 0
            var received_tokens = 0

            # Count expected tokens from all ranks for this slot
            for rank in range(n_ranks):
                var rank_topk_ids = (
                    host_topk_ids_list[rank]
                    + slot_idx * n_tokens_per_rank * top_k
                )
                for tok_idx in range(n_tokens_per_rank * top_k):
                    if (
                        expert_start_idx
                        <= Int(rank_topk_ids[tok_idx])
                        < expert_end_idx
                    ):
                        expected_tokens += 1

            # Check the output tokens
            for expert_idx in range(n_local_experts):
                var curr_local_expert = slot_expert_ids[expert_idx]
                var curr_expert = (
                    Int32(n_local_experts * dev_idx) + curr_local_expert
                )
                var expert_token_idx = 0

                for remote_rank in range(n_ranks):
                    var expert_rank_offset = curr_local_expert * Int32(
                        n_ranks
                    ) + Int32(remote_rank)
                    var token_end = (
                        slot_dispatch_wait_counter[2 * expert_rank_offset]
                        - EP_DATA_READY_FLAG
                    )
                    var num_tokens = slot_dispatch_wait_counter[
                        2 * expert_rank_offset + 1
                    ]
                    var token_start = token_end - num_tokens
                    received_tokens += Int(num_tokens)

                    for token_idx in range(token_start, token_end):
                        var remote_loc = slot_src_token_info[2 * token_idx]
                        var remote_topk_id = slot_src_token_info[
                            2 * token_idx + 1
                        ]

                        # Get the remote rank's topk_ids for this slot
                        var remote_rank_topk_ids = (
                            host_topk_ids_list[remote_rank]
                            + slot_idx * n_tokens_per_rank * top_k
                        )

                        # Check if curr_expert is in remote rank's topk_ids
                        assert_equal(
                            remote_rank_topk_ids[
                                remote_loc * Int32(top_k) + remote_topk_id
                            ],
                            curr_expert,
                            "Expert mismatch for dev "
                            + String(dev_idx)
                            + " slot "
                            + String(slot_idx)
                            + " token "
                            + String(token_idx),
                        )

                        # Get the remote rank's input tokens for this slot
                        var remote_rank_input_tokens = (
                            host_input_tokens_list[remote_rank]
                            + slot_idx * n_tokens_per_rank * hidden_size
                        )

                        # Check if the received token matches the remote rank's token
                        for i in range(hidden_size):
                            var remote_token_val = remote_rank_input_tokens[
                                remote_loc * Int32(hidden_size) + Int32(i)
                            ]
                            dispatch_test.check_output_val(
                                dev_idx,
                                slot_idx,
                                expert_idx,
                                expert_token_idx,
                                Int(token_idx),
                                i,
                                remote_token_val,
                            )

                        expert_token_idx += 1

            assert_equal(
                received_tokens,
                expected_tokens,
                "Received tokens mismatch for dev "
                + String(dev_idx)
                + " slot "
                + String(slot_idx),
            )

        # Free host buffers
        host_output.free()
        host_row_offsets.free()
        host_expert_ids.free()
        host_src_token_info.free()
        host_atomic_counter.free()

    dispatch_test.save_outputs_to_host(list_of_ctx)
    sync_parallelize[verify_results](n_ranks)
    print("All results verified successfully!")

    for dev_idx in range(n_ranks):
        host_topk_ids_list[dev_idx].free()
        host_input_tokens_list[dev_idx].free()


def test_dispatch_bf16[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]):
    comptime dispatch_test_type = BF16DispatchTest[
        hidden_size, top_k, n_experts, n_ranks, n_slots, n_tokens_per_rank
    ]
    test_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def test_dispatch_blockwise_fp8[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]):
    comptime dispatch_test_type = BlockwiseFP8DispatchTest[
        fp8_dtype = DType.float8_e4m3fn,
        scales_dtype = DType.float32,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    test_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def test_dispatch_nvfp4[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]):
    comptime dispatch_test_type = NVFP4DispatchTest[
        fp4_dtype = DType.uint8,
        scales_dtype = DType.float8_e4m3fn,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    test_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def main():
    comptime test_gpu_counts = (2, 4, 8)

    if can_enable_p2p():
        print("Enabled P2P Mem Access on all GPUs.")
    else:
        raise Error("Cannot enable P2P Mem Access!")

    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "Only NVIDIA and AMD GPUs are supported"
    comptime n_local_experts = 32 if has_nvidia_gpu_accelerator() else 16

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        test_dispatch_bf16[
            hidden_size=3584,  # equivalent to send 7168 FP8s.
            top_k=8,
            n_experts = num_gpus * n_local_experts,
            n_ranks=num_gpus,
            n_slots=3,
            n_tokens_per_rank=64,
            bench_e2e=False,
        ](ctx)

        test_dispatch_blockwise_fp8[
            hidden_size=7168,
            top_k=8,
            n_experts = num_gpus * n_local_experts,
            n_ranks=num_gpus,
            n_slots=3,
            n_tokens_per_rank=64,
            bench_e2e=False,
        ](ctx)

        comptime device_info = DeviceContext.default_device_info

        comptime if has_nvidia_gpu_accelerator() and device_info == B200:
            test_dispatch_nvfp4[
                hidden_size=7168,
                top_k=8,
                n_experts = num_gpus * n_local_experts,
                n_ranks=num_gpus,
                n_slots=3,
                n_tokens_per_rank=64,
                bench_e2e=False,
            ](ctx)
