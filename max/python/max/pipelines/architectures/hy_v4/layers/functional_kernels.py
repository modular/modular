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

"""``F.functional`` wrappers for the Hy4 kernels that ModuleV3 lacks."""

from collections.abc import Callable
from typing import Any

from max import functional as F
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.nn import _graph_ops as kernels
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kv_cache import PagedCacheValues
from max.tensor import Tensor


def _functional(op: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``op``, handing it graph values for its tensor arguments.

    ``F.functional`` leaves a tensor passed directly for the op to convert,
    but these kernels read ``.device`` and build typed ops before converting.
    """

    def call(*args: Any, **kwargs: Any) -> Any:
        return op(
            *(TensorValue(a) if isinstance(a, Tensor) else a for a in args),
            **kwargs,
        )

    call.__name__ = op.__name__
    return F.functional(call)


def _mla_fp8_index_top_k_physical(
    q: TensorValue,
    q_s: TensorValue,
    input_row_offsets: TensorValue,
    k_collection: PagedCacheValues,
    mla_collection: PagedCacheValues,
    layer_idx: TensorValue,
    top_k: int,
    quantization_granularity: int,
    mask_variant: MHAMaskVariant,
) -> TensorValue:
    """Selects each query row's top-k keys as physical rows of ``mla_collection``.

    :func:`max.nn._graph_ops.mla_fp8_index_top_k_physical` on the
    ``attention.mla.indexer.ragged.float8.paged.remap.filtered`` op, whose
    top-k keeps only the columns at or above a sampled bound for a prefill
    chunk's rows before the exact select.

    Args:
        q: FP8 ``[tokens, heads, head_dim]`` queries.
        q_s: float32 ``[tokens, heads]`` query scales.
        input_row_offsets: ``[batch + 1]`` ragged row offsets.
        k_collection: The FP8 indexer K cache, with scales.
        mla_collection: The MLA cache the rows index; not NVFP4.
        layer_idx: The layer's slot in ``k_collection``.
        top_k: Keys per row, at most 2048.
        quantization_granularity: The K cache's scale granularity.
        mask_variant: ``NULL_MASK`` or ``CAUSAL_MASK``.

    Returns:
        int32 ``[tokens, top_k]``, ``block_id * page_size + offset`` in
        ``mla_collection``, ``-1`` past a row's keys or on an unmapped page.

    Raises:
        ValueError: If an argument is out of range or ``mla_collection`` is
            NVFP4.
    """
    if mla_collection.kv_blocks.dtype == DType.uint8:
        raise ValueError("the filtered indexer op has no NVFP4 remap")
    kernels._validate_mla_fp8_index_top_k(
        q,
        q_s,
        input_row_offsets,
        k_collection,
        layer_idx,
        top_k,
        mask_variant,
        kpool=1,
    )
    rows_type = TensorType(DType.int32, [q.shape[0], top_k], q.device)
    # Without the logical store the kernel never touches that output.
    logical_type = TensorType(DType.int32, [0, top_k], q.device)
    return ops.inplace_custom(
        "attention.mla.indexer.ragged.float8.paged.remap.filtered",
        device=q.device,
        values=[
            q,
            q_s,
            input_row_offsets,
            *k_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
            mla_collection.kv_blocks,
            mla_collection.lookup_table,
        ],
        out_types=[logical_type, rows_type],
        parameters={
            "num_heads": int(q.shape[1]),
            "depth": int(q.shape[2]),
            "k": top_k,
            "quantization_granularity": quantization_granularity,
            "mask_str": kernels._mask_str(mask_variant),
            "write_logical": False,
        },
    )[1].tensor


mla_fp8_index_top_k_physical = _functional(_mla_fp8_index_top_k_physical)
rope_ragged = _functional(kernels.rope_ragged)
store_k_cache_ragged = _functional(kernels.store_k_cache_ragged)
store_k_scale_cache_ragged = _functional(kernels.store_k_scale_cache_ragged)
