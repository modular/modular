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
"""CPU graph-construction smoke test for the combined sparse MLA prefill op.

``test_mla_prefill_decode_graph_sparse_smoke`` builds the combined
``mla_prefill_decode_graph`` graph with prefill metadata and ``buffer_lengths``
and stops -- it never compiles or executes, so it needs no GPU and runs on a
CPU worker. The compile + execute coverage lives in the GPU tests
(``test_mla_sparse_prefill_graph_gpu`` and
``test_mla_sparse_prefill_graph_gpu``).
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from _mla_sparse_test_utils import paged_kv_from_flat_graph_inputs
from max import tree
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.attention.multi_latent_attention import (
    LatentAttentionWithRope,
    MLAPrefillMetadata,
)
from max.nn.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.kernels import (
    mla_decode_graph,
    mla_fp8_index_top_k_physical,
    mla_fp8_index_top_k_with_physical,
    mla_prefill_decode_graph,
)
from max.nn.kv_cache import KVCacheQuantizationConfig, MLAKVCacheParams
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)


def test_mla_prefill_decode_graph_sparse_smoke() -> None:
    """Build ``mla_prefill_decode_graph`` with prefill metadata and ``buffer_lengths``."""
    num_heads = 16
    topk = 8
    indices_stride = topk
    hidden_size = 1024
    q_lora_rank = 256
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    page_size = 128

    quant_config = QuantConfig(
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

    scaling_params = DeepseekYarnRopeScalingParams(
        scaling_factor=40.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
    rope = DeepseekYarnRotaryEmbedding(
        dim=qk_rope_head_dim,
        n_heads=num_heads,
        theta=10000.0,
        max_seq_len=2048,
        scaling_params=scaling_params,
    )

    kv_params = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=576,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
    )

    attn = LatentAttentionWithRopeFp8(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        hidden_size=hidden_size,
        kv_params=kv_params,
        quant_config=quant_config,
        devices=[DeviceRef.GPU()],
        graph_mode="auto",
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        buffer_size=4096,
    )

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    cache_head_dim = kv_lora_rank + qk_rope_head_dim

    q_type = TensorType(
        DType.bfloat16,
        ["total_tokens", num_heads, qk_head_dim],
        DeviceRef.GPU(),
    )
    kv_type = TensorType(
        DType.bfloat16,
        ["total_tokens", cache_head_dim],
        DeviceRef.GPU(),
    )
    row_off_type = TensorType(
        DType.uint32, ["row_offsets_len"], DeviceRef.GPU()
    )
    sparse_idx_type = TensorType(
        DType.int32,
        ["total_tokens", "max_topk"],
        DeviceRef.GPU(),
    )
    topk_len_type = TensorType(DType.int32, ["batch"], DeviceRef.GPU())
    sink_type = TensorType(DType.float32, ["batch"], DeviceRef.GPU())
    batch_ctx_type = TensorType(
        DType.int32, ["buf_len_chunks"], DeviceRef.GPU()
    )

    kv_sym = tree.leaves(kv_params.get_symbolic_inputs()[0])

    def construct() -> Graph:
        with Graph(
            "mla_prefill_decode_sparse_smoke",
            input_types=[
                q_type,
                kv_type,
                row_off_type,
                sparse_idx_type,
                topk_len_type,
                sink_type,
                batch_ctx_type,
                *kv_sym,
            ],
        ) as g:
            q = g.inputs[0].tensor
            kv = g.inputs[1].tensor
            input_row_offsets = g.inputs[2].tensor
            sparse_indices = g.inputs[3].tensor
            sparse_topk_lengths = g.inputs[4].tensor
            sparse_attn_sink = g.inputs[5].tensor
            batch_context_lengths = g.inputs[6].tensor

            kv_collection = paged_kv_from_flat_graph_inputs(
                kv_params, list(g.inputs[7:])
            )
            assert kv_collection.attention_dispatch_metadata is not None
            assert kv_collection.mla_num_partitions is not None
            scalar_args = kv_collection.attention_dispatch_metadata
            num_partitions_scalar = kv_collection.mla_num_partitions

            w_k, w_k_scale = attn.w_k
            w_uk, w_uk_scale = attn.w_uk
            w_uv, w_uv_scale = attn.w_uv

            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
            freqs_cis = ops.cast(rope.freqs_cis, q.dtype).to(q.device)

            mla_md = attn.create_mla_prefill_metadata(
                input_row_offsets, kv_collection
            )
            mla_md = MLAPrefillMetadata(
                buffer_row_offsets=mla_md.buffer_row_offsets,
                cache_offsets=mla_md.cache_offsets,
                buffer_lengths=batch_context_lengths,
            )

            out = mla_prefill_decode_graph(
                q,
                kv,
                input_row_offsets,
                freqs_cis,
                attn.kv_a_proj_layernorm,
                mla_md.buffer_row_offsets,
                mla_md.cache_offsets,
                mla_md.buffer_lengths.to(DeviceRef.CPU()),
                w_k,
                w_uk,
                w_uv,
                kv_params,
                kv_collection,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                attn.scale,
                1e-6,
                v_head_dim,
                scalar_args,
                num_partitions_scalar,
                w_k_scale=w_k_scale,
                w_uk_scale=w_uk_scale,
                w_uv_scale=w_uv_scale,
                quant_config=quant_config,
                sparse_indices=sparse_indices,
                physical_indices=False,
                sparse_topk_lengths=sparse_topk_lengths,
                sparse_attn_sink=sparse_attn_sink,
                sparse_indices_stride=indices_stride,
            )
            g.output(out)
        return g

    _ = attn.state_dict()
    _ = construct()


def test_mla_sparse_graphs_nvfp4_cache_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build GLM 5.3's sparse BF16-weight MLA ops over an NVFP4 latent cache.

    The plan reads only lengths, so it takes the collection without its scale
    operands; the sparse ops take them and route to their ``.nvfp4``
    registrations. Staging checks every op against its registration, so a
    missing or misordered operand fails here. The prefill op carries the
    convert-once bounds, which the environment overrides.
    """
    monkeypatch.setenv("MODULAR_MLA_NVFP4_PREFILL_CONVERT_RATIO", "37")
    monkeypatch.setenv("MODULAR_MLA_NVFP4_PREFILL_CONVERT_MAX_ROWS", "4096")
    num_heads = 8
    topk = 8
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128

    rope = DeepseekYarnRotaryEmbedding(
        dim=qk_rope_head_dim,
        n_heads=num_heads,
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
    kv_params = MLAKVCacheParams(
        dtype=DType.uint8,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e4m3fn,
            quantization_granularity=16,
            fp8_tail_dim=qk_rope_head_dim,
        ),
    )
    assert kv_params.is_nvfp4_kv_cache
    attn = LatentAttentionWithRope(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        hidden_size=1024,
        kv_params=kv_params,
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        graph_mode="auto",
        q_lora_rank=256,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        buffer_size=4096,
    )

    gpu = DeviceRef.GPU()
    input_types = [
        TensorType(
            DType.bfloat16,
            ["total_tokens", num_heads, qk_nope_head_dim + qk_rope_head_dim],
            gpu,
        ),
        TensorType(
            DType.bfloat16,
            ["total_tokens", kv_lora_rank + qk_rope_head_dim],
            gpu,
        ),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        TensorType(DType.int32, ["total_tokens", 2 * topk], gpu),
        *tree.leaves(kv_params.get_symbolic_inputs()[0]),
    ]

    with Graph("mla_sparse_nvfp4_construction", input_types=input_types) as g:
        q, kv, input_row_offsets, rows = (x.tensor for x in g.inputs[:4])
        kv_collection = paged_kv_from_flat_graph_inputs(
            kv_params, list(g.inputs[4:])
        )
        assert kv_collection.kv_scales is not None
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        common: dict[str, Any] = dict(
            kv_params=kv_params,
            kv_collection=kv_collection,
            layer_idx=ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=attn.scale,
            epsilon=1e-6,
            v_head_dim=v_head_dim,
            scalar_args=kv_collection.attention_dispatch_metadata,
            num_partitions_scalar=kv_collection.mla_num_partitions,
        )
        sparse: dict[str, Any] = dict(
            sparse_indices=rows,
            physical_indices=True,
            sparse_indices_stride=2 * topk,
        )
        freqs_cis = ops.cast(rope.freqs_cis, q.dtype).to(q.device)
        decode_args: dict[str, Any] = dict(
            q=q,
            kv=kv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_norm_gamma=attn.kv_a_proj_layernorm,
            w_uk=attn.w_uk,
            w_uv=attn.w_uv,
            **common,
        )

        # No dense or FP8-weight registration decodes NVFP4.
        with pytest.raises(ValueError, match="NVFP4 KV cache"):
            mla_decode_graph(**decode_args)

        metadata = attn.create_mla_prefill_metadata(
            input_row_offsets, kv_collection
        )
        prefill = mla_prefill_decode_graph(
            q,
            kv,
            input_row_offsets,
            freqs_cis,
            attn.kv_a_proj_layernorm,
            metadata.buffer_row_offsets,
            metadata.cache_offsets,
            metadata.buffer_lengths.to(DeviceRef.CPU()),
            attn.w_k,
            attn.w_uk,
            attn.w_uv,
            **common,
            **sparse,
        )
        decode = mla_decode_graph(**decode_args, **sparse)
        g.output(prefill, decode)

    ir = str(g)
    assert '"mo.mla.graph.prefill.decode.paged.sparse.nvfp4"' in ir
    assert '"mo.mla.graph.decode.paged.sparse.nvfp4"' in ir
    assert re.search(r"nvfp4_convert_ratio = 37\b", ir)
    assert re.search(r"nvfp4_convert_max_rows = 4096\b", ir)


def test_mla_indexer_nvfp4_cache_rows_carry_scale_rows() -> None:
    """The indexer over an NVFP4 MLA cache returns value rows then scale rows.

    An FP8 MLA cache keeps the ``[tokens, top_k]`` physical rows and the plain
    remap op; an NVFP4 one pages its scales apart, so its rows are twice as
    wide and come from the ``.nvfp4`` registration, which takes the scales'
    page table.
    """
    num_heads = 8
    index_head_dim = 128
    top_k = 8
    gpu = DeviceRef.GPU()

    def mla_params(dtype: DType) -> MLAKVCacheParams:
        return MLAKVCacheParams(
            dtype=dtype,
            head_dim=576,
            num_layers=1,
            page_size=128,
            devices=[gpu],
            num_q_heads=num_heads,
            kvcache_quant_config=KVCacheQuantizationConfig(
                scale_dtype=DType.float8_e4m3fn,
                quantization_granularity=16,
                fp8_tail_dim=64,
            )
            if dtype == DType.uint8
            else None,
        )

    indexer_params = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=index_head_dim,
        num_layers=1,
        page_size=128,
        devices=[gpu],
        num_q_heads=num_heads,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=index_head_dim
        ),
    )
    nvfp4_params = mla_params(DType.uint8)
    fp8_params = mla_params(DType.float8_e4m3fn)
    input_types = [
        TensorType(
            DType.float8_e4m3fn,
            ["total_tokens", num_heads, index_head_dim],
            gpu,
        ),
        TensorType(DType.float32, ["total_tokens", num_heads], gpu),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        *tree.leaves(indexer_params.get_symbolic_inputs()[0]),
        *tree.leaves(nvfp4_params.get_symbolic_inputs()[0]),
        *tree.leaves(fp8_params.get_symbolic_inputs()[0]),
    ]
    num_indexer_inputs = len(
        tree.leaves(indexer_params.get_symbolic_inputs()[0])
    )
    num_nvfp4_inputs = len(tree.leaves(nvfp4_params.get_symbolic_inputs()[0]))

    with Graph("mla_indexer_nvfp4_construction", input_types=input_types) as g:
        q, q_s, row_offsets = (x.tensor for x in g.inputs[:3])
        flat = list(g.inputs[3:])
        indexer_collection = paged_kv_from_flat_graph_inputs(
            indexer_params, flat[:num_indexer_inputs]
        )
        nvfp4_collection = paged_kv_from_flat_graph_inputs(
            nvfp4_params,
            flat[num_indexer_inputs : num_indexer_inputs + num_nvfp4_inputs],
        )
        fp8_collection = paged_kv_from_flat_graph_inputs(
            fp8_params, flat[num_indexer_inputs + num_nvfp4_inputs :]
        )
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        outs = []
        for mla_collection in (nvfp4_collection, fp8_collection):
            args = (
                q,
                q_s,
                row_offsets,
                indexer_collection,
                mla_collection,
                layer_idx,
                top_k,
                index_head_dim,
            )
            logical, physical = mla_fp8_index_top_k_with_physical(*args)
            only_physical = mla_fp8_index_top_k_physical(*args)
            outs.append((logical, physical, only_physical))
        g.output(*(t for out in outs for t in out))

    (_, nvfp4_rows, nvfp4_only), (_, fp8_rows, fp8_only) = outs
    assert nvfp4_rows.shape[1] == 2 * top_k == nvfp4_only.shape[1]
    assert fp8_rows.shape[1] == top_k == fp8_only.shape[1]
    ir = str(g)
    assert '"mo.mla.indexer.ragged.float8.paged.remap.nvfp4"' in ir
    assert '"mo.mla.indexer.ragged.float8.paged.remap"' in ir
