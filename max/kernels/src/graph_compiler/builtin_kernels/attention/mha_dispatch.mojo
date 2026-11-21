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

"""The paged-ragged MHA entry point shared by the attention registrations."""

from std.collections import OptionalReg

from extensibility import Fused, Input, LoadAccess, TensorTuple
from layout import (
    Coord,
    DefaultEngine,
    RowMajorLayout,
    TensorEngine,
    TensorLayout,
    TileTensor,
)
from max.gpu.host import DeviceContext
from nn.attention.gpu.mha import MHADecodeDispatchMetadata
from nn.kv_cache import generic_get_paged_cache
from nn.kv_cache_ragged import (
    generic_flash_attention_kv_cache_ragged,
    generic_flash_attention_kv_cache_ragged_sink,
)


@inline(.always)
def _unmarshal_mha_decode_dispatch_metadata(
    mha_decode_dispatch_metadata: TileTensor[
        DType.int64, address_space=.GENERIC, ...
    ],
) -> MHADecodeDispatchMetadata:
    comptime assert mha_decode_dispatch_metadata.rank == 1
    # The metadata is read through its flat pointer; a non-pointer storage
    # policy aborts here at run time.
    var metadata_ptr = mha_decode_dispatch_metadata.ptr
    return MHADecodeDispatchMetadata(
        Int(metadata_ptr[0]),
        Int(metadata_ptr[1]),
        Int(metadata_ptr[2]),
        Int(metadata_ptr[3]),
    )


@inline(.always)
def _execute_mha_ragged_paged_scalar_args[
    q_dtype: DType,
    output_dtype: DType,
    MaskIns: TypeList[Trait=Input & Fused & LoadAccess, ...],
    //,
    mask_str: StaticString,
    sink: Bool = False,
    local_window_size: Int = -1,
    SinkLayoutType: TensorLayout = RowMajorLayout[*Coord[Int64].element_types],
    # `sink_weights` is optional, so its engine cannot be inferred.
    SinkEngine: TensorEngine = DefaultEngine[element_width=1],
    v_head_dim: Int = -1,
](
    output: TileTensor[mut=True, output_dtype, address_space=.GENERIC, ...],
    q: TileTensor[q_dtype, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[DType.uint32, address_space=.GENERIC, ...],
    kv_blocks: TileTensor[mut=True, address_space=.GENERIC, ...],
    page_stride: TileTensor[
        mut=False, DType.int64, address_space=.GENERIC, ...
    ],
    cache_lengths: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    kv_lookup_table: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    max_prompt_length: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    max_cache_length: TileTensor[
        mut=False, DType.uint32, address_space=.GENERIC, ...
    ],
    layer_idx: UInt32,
    scale: Float32,
    mha_decode_dispatch_metadata: TileTensor[
        DType.int64, address_space=.GENERIC, ...
    ],
    mask_operands: TensorTuple[*MaskIns],
    context: DeviceContext,
    sink_weights: OptionalReg[
        TileTensor[
            q_dtype, SinkLayoutType, ImmUntrackedOrigin, Engine=SinkEngine
        ]
    ] = None,
) raises:
    comptime assert output.rank == 3
    comptime assert q.rank == 3
    comptime assert input_row_offsets.rank == 1
    comptime assert kv_blocks.rank == 6
    var decode_dispatch_metadata = _unmarshal_mha_decode_dispatch_metadata(
        mha_decode_dispatch_metadata
    )
    var kv_collection = generic_get_paged_cache[v_head_size=v_head_dim](
        kv_blocks,
        page_stride,
        cache_lengths,
        kv_lookup_table,
        max_prompt_length,
        max_cache_length,
    )
    comptime if sink:
        comptime assert (
            len(MaskIns) == 0
        ), "sink attention takes no mask operands"
        generic_flash_attention_kv_cache_ragged_sink[
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            sink_weights.value(),
            decode_dispatch_metadata,
        )
    else:
        generic_flash_attention_kv_cache_ragged[
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q,
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            output,
            context,
            decode_dispatch_metadata,
            mask_operands,
        )
