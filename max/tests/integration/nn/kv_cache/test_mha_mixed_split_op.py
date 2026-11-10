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
"""Which split record ``flash_attention_ragged`` hands to ragged MHA.

The graphs are only built, never run. Every ragged MHA call takes the KV
leaf's declared record after its own operands, or a zeros constant when the
leaf has none (a draft forward drops it). The plain, sink and split-head-dim
calls split a mixed batch with it; the relative-logit call takes it too,
though the kernel doesn't split a mask that reads operands. Every MLA combined
op takes the same leaf record, and so do the indexer ops
(``attention.mla.indexer.ragged.float8.paged`` and its two remap twins).
"""

from __future__ import annotations

import dataclasses

import pytest
from max import mlir, tree
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn._graph_ops import (
    flash_attention_ragged,
    mla_fp8_index_top_k,
    mla_fp8_index_top_k_with_physical,
    mla_prefill_decode_graph,
)
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kv_cache import (
    MIXED_STEP_SPLIT_SIZE,
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    MLAKVCacheParams,
)
from max.nn.legacy.attention.multi_latent_attention import (
    LatentAttentionWithRope,
)
from max.nn.legacy.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.legacy.rotary_embedding import DeepseekYarnRotaryEmbedding
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.rope_scaling import DeepseekYarnRopeScalingParams
from test_common.mixed_step_split_ops import record_source

MHA_OP = "attention.mha.ragged.paged"
NUM_HEADS = 8
HEAD_DIM = 128
OP_NAMES = {
    "plain": MHA_OP,
    "split_head_dim": MHA_OP,
    "sink": "attention.mha.ragged.paged.sink_weights",
    "rel_logits": MHA_OP,
}


def _params(v_head_dim: int | None = None) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=2,
        head_dim=HEAD_DIM,
        v_head_dim=v_head_dim,
        num_layers=1,
        devices=[DeviceRef.GPU()],
        page_size=128,
    )


def _attention_op(
    call: str, *, with_record: bool, mask: MHAMaskVariant | None = None
) -> mlir.Operation:
    split_head_dims = call == "split_head_dim"
    params = _params(v_head_dim=64 if split_head_dims else None)
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    cache_inputs = tree.leaves(params.get_symbolic_inputs())
    extra_types: dict[str, TensorType] = {
        "sink": TensorType(DType.bfloat16, [NUM_HEADS], gpu),
        "rel_logits": TensorType(DType.bfloat16, ["total", NUM_HEADS, 16], gpu),
    }
    input_types = [
        *cache_inputs,
        TensorType(DType.bfloat16, ["total", NUM_HEADS, HEAD_DIM], gpu),
        TensorType(DType.uint32, ["offsets"], gpu),
        TensorType(DType.uint32, [], cpu),
        *([extra_types[call]] if call in extra_types else []),
    ]
    with Graph("mha_mixed_split_op", input_types=input_types) as graph:
        n = len(cache_inputs)
        cache = params.unflatten_kv_inputs(iter(graph.inputs[:n]))[0]
        if not with_record:
            cache = dataclasses.replace(cache, mixed_step_split=None)
        optional = graph.inputs[n + 3].tensor if call in extra_types else None
        out = flash_attention_ragged(
            params,
            graph.inputs[n].tensor,
            graph.inputs[n + 1].tensor,
            cache,
            graph.inputs[n + 2].tensor,
            mask or MHAMaskVariant.CAUSAL_MASK,
            scale=HEAD_DIM**-0.5,
            sink_weights=optional if call == "sink" else None,
            rel_logits=optional if call == "rel_logits" else None,
        )
        graph.output(out)
    ops = [
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
        and mlir.StringAttr(op.attributes["symbol"]).value.startswith(MHA_OP)
    ]
    assert len(ops) == 1
    assert mlir.StringAttr(ops[0].attributes["symbol"]).value == OP_NAMES[call]
    return ops[0]


@pytest.mark.parametrize(
    "mask",
    [MHAMaskVariant.CAUSAL_MASK, MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK],
)
@pytest.mark.parametrize("call", ["plain", "split_head_dim", "sink"])
def test_the_leaf_record_is_the_last_mha_operand(
    call: str, mask: MHAMaskVariant
) -> None:
    op = _attention_op(call, with_record=True, mask=mask)
    assert record_source(op) == "argument"
    last = [v for v in op.operands if "chain" not in str(v.type)][-1]
    assert f"[{MIXED_STEP_SPLIT_SIZE}], ui32" in str(last.type)


@pytest.mark.parametrize(
    "call", ["plain", "split_head_dim", "sink", "rel_logits"]
)
def test_a_leaf_without_a_record_sends_zeros(call: str) -> None:
    op = _attention_op(call, with_record=False)
    assert record_source(op) == "mo.constant"


def test_relative_logits_take_the_leaf_record() -> None:
    op = _attention_op("rel_logits", with_record=True)
    assert record_source(op) == "argument"


MLA_SPARSE_OPS = {
    "bf16": "attention.mla.graph.prefill.decode.paged.sparse",
    "fp8": "attention.mla.graph.prefill.decode.paged.fp8.sparse",
    "nvfp4": "attention.mla.graph.prefill.decode.paged.sparse.nvfp4",
}
MLA_DENSE_OPS = {
    "bf16_dense": "attention.mla.graph.prefill.decode.paged",
    "fp8_dense": "attention.mla.graph.prefill.decode.paged.fp8",
}
MLA_ARMS = [*MLA_SPARSE_OPS, *MLA_DENSE_OPS]
MLA_NUM_HEADS = 8
MLA_QK_NOPE_HEAD_DIM = 128
MLA_QK_ROPE_HEAD_DIM = 64
MLA_V_HEAD_DIM = 128
MLA_KV_LORA_RANK = 512
MLA_TOPK = 64


def _mla_quant_config() -> QuantConfig:
    """Builds the block-scaled FP8 weight config the FP8 weights need."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _mla_op(arm: str, *, with_record: bool) -> mlir.Operation:
    """Builds the MLA combined op for ``arm`` and returns its graph op."""
    sparse = arm in MLA_SPARSE_OPS
    quant_config = _mla_quant_config() if arm in ("fp8", "fp8_dense") else None
    kv_params = MLAKVCacheParams(
        dtype=DType.uint8 if arm == "nvfp4" else DType.bfloat16,
        head_dim=MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.GPU()],
        num_q_heads=MLA_NUM_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e4m3fn,
            quantization_granularity=16,
            fp8_tail_dim=MLA_QK_ROPE_HEAD_DIM,
        )
        if arm == "nvfp4"
        else None,
    )
    rope = DeepseekYarnRotaryEmbedding(
        dim=MLA_QK_ROPE_HEAD_DIM,
        n_heads=MLA_NUM_HEADS,
        theta=10000.0,
        max_seq_len=2048,
        scaling_params=DeepseekYarnRopeScalingParams(
            scaling_factor=40.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        ),
    )
    attn: LatentAttentionWithRope | LatentAttentionWithRopeFp8
    if quant_config is not None:
        attn = LatentAttentionWithRopeFp8(
            rope=rope,
            num_attention_heads=MLA_NUM_HEADS,
            num_key_value_heads=1,
            hidden_size=1024,
            kv_params=kv_params,
            quant_config=quant_config,
            devices=[DeviceRef.GPU()],
            graph_mode="auto",
            q_lora_rank=256,
            kv_lora_rank=MLA_KV_LORA_RANK,
            qk_nope_head_dim=MLA_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=MLA_QK_ROPE_HEAD_DIM,
            v_head_dim=MLA_V_HEAD_DIM,
            buffer_size=4096,
        )
    else:
        attn = LatentAttentionWithRope(
            rope=rope,
            num_attention_heads=MLA_NUM_HEADS,
            num_key_value_heads=1,
            hidden_size=1024,
            kv_params=kv_params,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            graph_mode="auto",
            q_lora_rank=256,
            kv_lora_rank=MLA_KV_LORA_RANK,
            qk_nope_head_dim=MLA_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=MLA_QK_ROPE_HEAD_DIM,
            v_head_dim=MLA_V_HEAD_DIM,
            buffer_size=4096,
        )

    gpu = DeviceRef.GPU()
    qk_head_dim = MLA_QK_NOPE_HEAD_DIM + MLA_QK_ROPE_HEAD_DIM
    cache_head_dim = MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM
    # An NVFP4 cache's index rows also carry the indexer's scale rows.
    row_width = 2 * MLA_TOPK if arm == "nvfp4" else MLA_TOPK
    index_types = (
        [TensorType(DType.int32, ["total_tokens", row_width], gpu)]
        if sparse
        else []
    )
    input_types = [
        TensorType(
            DType.bfloat16, ["total_tokens", MLA_NUM_HEADS, qk_head_dim], gpu
        ),
        TensorType(DType.bfloat16, ["total_tokens", cache_head_dim], gpu),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        *index_types,
        *tree.leaves(kv_params.get_symbolic_inputs()[0]),
    ]
    with Graph("mla_mixed_split_op", input_types=input_types) as graph:
        q, kv, input_row_offsets = (x.tensor for x in graph.inputs[:3])
        kv_collection = kv_params.unflatten_kv_inputs(
            iter(graph.inputs[3 + len(index_types) :])
        )[0]
        if not with_record:
            kv_collection = dataclasses.replace(
                kv_collection, mixed_step_split=None
            )
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        metadata = attn.create_mla_prefill_metadata(
            input_row_offsets, kv_collection
        )
        freqs_cis = ops.cast(rope.freqs_cis, q.dtype).to(q.device)
        w_k_scale: TensorValue | None = None
        w_uk_scale: TensorValue | None = None
        w_uv_scale: TensorValue | None = None
        if isinstance(attn, LatentAttentionWithRopeFp8):
            w_k, w_k_scale = attn.w_k
            w_uk, w_uk_scale = attn.w_uk
            w_uv, w_uv_scale = attn.w_uv
        else:
            w_k = attn.w_k
            w_uk = attn.w_uk
            w_uv = attn.w_uv
        out = mla_prefill_decode_graph(
            q,
            kv,
            input_row_offsets,
            freqs_cis,
            attn.kv_a_proj_layernorm,
            metadata.buffer_row_offsets,
            metadata.cache_offsets,
            metadata.buffer_lengths.to(DeviceRef.CPU()),
            w_k,
            w_uk,
            w_uv,
            kv_params,
            kv_collection,
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            MHAMaskVariant.CAUSAL_MASK,
            attn.scale,
            1e-6,
            MLA_V_HEAD_DIM,
            kv_collection.attention_dispatch_metadata,
            kv_collection.mla_num_partitions,
            sparse_indices=graph.inputs[3].tensor if sparse else None,
            physical_indices=True if sparse else None,
            sparse_indices_stride=row_width if sparse else None,
            quant_config=quant_config,
            w_k_scale=w_k_scale,
            w_uk_scale=w_uk_scale,
            w_uv_scale=w_uv_scale,
        )
        graph.output(out)
    op_name = MLA_SPARSE_OPS[arm] if sparse else MLA_DENSE_OPS[arm]
    ops_found = [
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
        and mlir.StringAttr(op.attributes["symbol"]).value == op_name
    ]
    assert len(ops_found) == 1
    return ops_found[0]


@pytest.mark.parametrize("arm", MLA_ARMS)
def test_the_mla_op_takes_the_leaf_record(arm: str) -> None:
    op = _mla_op(arm, with_record=True)
    assert record_source(op) == "argument"


@pytest.mark.parametrize("arm", MLA_ARMS)
def test_a_mla_leaf_without_a_record_sends_zeros(arm: str) -> None:
    op = _mla_op(arm, with_record=False)
    assert record_source(op) == "mo.constant"


INDEXER_OPS = {
    "paged": "attention.mla.indexer.ragged.float8.paged",
    "remap": "attention.mla.indexer.ragged.float8.paged.remap",
    "remap_nvfp4": "attention.mla.indexer.ragged.float8.paged.remap.nvfp4",
}
INDEXER_ARMS = list(INDEXER_OPS)
INDEXER_NUM_HEADS = 8
INDEXER_HEAD_DIM = 128
INDEXER_TOPK = 8
INDEXER_PAGE_SIZE = 128


def _indexer_params() -> MLAKVCacheParams:
    """Builds the FP8 indexer cache the op scores against."""
    return MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=INDEXER_HEAD_DIM,
        num_layers=1,
        page_size=INDEXER_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=INDEXER_NUM_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=INDEXER_HEAD_DIM,
        ),
    )


def _mla_index_params(dtype: DType) -> MLAKVCacheParams:
    """Builds the MLA cache the indexer's physical rows point into."""
    return MLAKVCacheParams(
        dtype=dtype,
        head_dim=MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM,
        num_layers=1,
        page_size=INDEXER_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=INDEXER_NUM_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e4m3fn,
            quantization_granularity=16,
            fp8_tail_dim=MLA_QK_ROPE_HEAD_DIM,
        )
        if dtype == DType.uint8
        else None,
    )


def _indexer_op(arm: str, *, with_record: bool) -> mlir.Operation:
    """Builds the indexer op for ``arm`` and returns its graph op."""
    gpu = DeviceRef.GPU()
    indexer_params = _indexer_params()
    mla_params = (
        None
        if arm == "paged"
        else _mla_index_params(
            DType.uint8 if arm == "remap_nvfp4" else DType.float8_e4m3fn
        )
    )
    indexer_inputs = tree.leaves(indexer_params.get_symbolic_inputs()[0])
    input_types = [
        TensorType(
            DType.float8_e4m3fn,
            ["total_tokens", INDEXER_NUM_HEADS, INDEXER_HEAD_DIM],
            gpu,
        ),
        TensorType(DType.float32, ["total_tokens", INDEXER_NUM_HEADS], gpu),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        *indexer_inputs,
        *(
            tree.leaves(mla_params.get_symbolic_inputs()[0])
            if mla_params is not None
            else []
        ),
    ]
    with Graph("mla_indexer_mixed_split_op", input_types=input_types) as graph:
        q, q_s, input_row_offsets = (x.tensor for x in graph.inputs[:3])
        flat = list(graph.inputs[3:])
        indexer_collection = indexer_params.unflatten_kv_inputs(
            iter(flat[: len(indexer_inputs)])
        )[0]
        if not with_record:
            indexer_collection = dataclasses.replace(
                indexer_collection, mixed_step_split=None
            )
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        if mla_params is None:
            out = mla_fp8_index_top_k(
                q,
                q_s,
                input_row_offsets,
                indexer_collection,
                layer_idx,
                INDEXER_TOPK,
                INDEXER_HEAD_DIM,
            )
        else:
            mla_collection = mla_params.unflatten_kv_inputs(
                iter(flat[len(indexer_inputs) :])
            )[0]
            _, out = mla_fp8_index_top_k_with_physical(
                q,
                q_s,
                input_row_offsets,
                indexer_collection,
                mla_collection,
                layer_idx,
                INDEXER_TOPK,
                INDEXER_HEAD_DIM,
            )
        graph.output(out)
    ops_found = [
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
        and mlir.StringAttr(op.attributes["symbol"]).value == INDEXER_OPS[arm]
    ]
    assert len(ops_found) == 1
    return ops_found[0]


@pytest.mark.parametrize("arm", INDEXER_ARMS)
def test_the_indexer_op_takes_the_leaf_record(arm: str) -> None:
    op = _indexer_op(arm, with_record=True)
    assert record_source(op) == "argument"


@pytest.mark.parametrize("arm", INDEXER_ARMS)
def test_an_indexer_leaf_without_a_record_sends_zeros(arm: str) -> None:
    op = _indexer_op(arm, with_record=False)
    assert record_source(op) == "mo.constant"
