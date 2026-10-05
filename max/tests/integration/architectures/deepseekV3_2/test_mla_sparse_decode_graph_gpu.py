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
"""GPU integration tests for sparse MLA decode graphs.

``test_mla_decode_graph_sparse_smoke`` and ``test_mla_decode_graph_sparse_bf16_smoke``
build the sparse ``mla_decode_graph`` graph only (no execution).
``test_mla_decode_graph_sparse_omitted_operands_match`` checks that omitting
``topk_lengths`` / ``sink`` matches passing their no-op values.
``test_mla_decode_graph_sparse_multi_step_smoke`` runs prefill and decode through
:class:`SparseLatentAttentionWithRopeFp8` and :class:`PagedKVCacheManager`.
The prefill-routing tests live in ``test_mla_sparse_prefill_graph_gpu.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from _mla_sparse_test_utils import (
    paged_kv_from_flat_graph_inputs,
    random_weights,
)
from max import tree
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.attention.multi_latent_attention import LatentAttentionWithRope
from max.nn.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.kernels import mla_decode_graph
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
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
from max.pipelines.architectures.deepseekV3_2.layers.sparse_mla import (
    SparseLatentAttentionWithRopeFp8,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs
from torch.utils.dlpack import from_dlpack


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
def test_mla_decode_graph_sparse_smoke() -> None:
    """Build ``mla_decode_graph`` with sparse indices (decode graph_mode)."""
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
        graph_mode="decode",
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

    kv_sym = tree.leaves(kv_params.get_symbolic_inputs()[0])

    def construct() -> Graph:
        with Graph(
            "mla_decode_sparse_smoke",
            input_types=[
                q_type,
                kv_type,
                row_off_type,
                sparse_idx_type,
                topk_len_type,
                sink_type,
                *kv_sym,
            ],
        ) as g:
            q = g.inputs[0].tensor
            kv = g.inputs[1].tensor
            input_row_offsets = g.inputs[2].tensor
            sparse_indices = g.inputs[3].tensor
            sparse_topk_lengths = g.inputs[4].tensor
            sparse_attn_sink = g.inputs[5].tensor

            kv_collection = paged_kv_from_flat_graph_inputs(
                kv_params, list(g.inputs[6:])
            )
            assert kv_collection.attention_dispatch_metadata is not None
            assert kv_collection.mla_num_partitions is not None
            scalar_args = kv_collection.attention_dispatch_metadata
            num_partitions_scalar = kv_collection.mla_num_partitions

            w_uk, w_uk_scale = attn.w_uk
            w_uv, w_uv_scale = attn.w_uv

            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
            freqs_cis = ops.cast(rope.freqs_cis, q.dtype).to(q.device)

            out = mla_decode_graph(
                q,
                kv,
                input_row_offsets,
                freqs_cis,
                attn.kv_a_proj_layernorm,
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
                w_uk_scale=w_uk_scale,
                w_uv_scale=w_uv_scale,
                quant_config=quant_config,
                sparse_indices=sparse_indices,
                sparse_topk_lengths=sparse_topk_lengths,
                sparse_attn_sink=sparse_attn_sink,
                sparse_indices_stride=indices_stride,
            )
            g.output(out)
        return g

    _ = attn.state_dict()
    _ = construct()


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
def test_mla_decode_graph_sparse_bf16_smoke() -> None:
    """Build bf16 ``mla_decode_graph`` with sparse indices (decode graph_mode)."""
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
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
    )

    attn = LatentAttentionWithRope(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        hidden_size=hidden_size,
        kv_params=kv_params,
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        graph_mode="decode",
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

    kv_sym = tree.leaves(kv_params.get_symbolic_inputs()[0])

    def construct() -> Graph:
        with Graph(
            "mla_decode_sparse_bf16_smoke",
            input_types=[
                q_type,
                kv_type,
                row_off_type,
                sparse_idx_type,
                topk_len_type,
                sink_type,
                *kv_sym,
            ],
        ) as g:
            q = g.inputs[0].tensor
            kv = g.inputs[1].tensor
            input_row_offsets = g.inputs[2].tensor
            sparse_indices = g.inputs[3].tensor
            sparse_topk_lengths = g.inputs[4].tensor
            sparse_attn_sink = g.inputs[5].tensor

            kv_collection = paged_kv_from_flat_graph_inputs(
                kv_params, list(g.inputs[6:])
            )
            assert kv_collection.attention_dispatch_metadata is not None
            assert kv_collection.mla_num_partitions is not None
            scalar_args = kv_collection.attention_dispatch_metadata
            num_partitions_scalar = kv_collection.mla_num_partitions

            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
            freqs_cis = ops.cast(rope.freqs_cis, q.dtype).to(q.device)

            out = mla_decode_graph(
                q,
                kv,
                input_row_offsets,
                freqs_cis,
                attn.kv_a_proj_layernorm,
                attn.w_uk,
                attn.w_uv,
                kv_params,
                kv_collection,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                attn.scale,
                1e-6,
                v_head_dim,
                scalar_args,
                num_partitions_scalar,
                sparse_indices=sparse_indices,
                sparse_topk_lengths=sparse_topk_lengths,
                sparse_attn_sink=sparse_attn_sink,
                sparse_indices_stride=indices_stride,
            )
            g.output(out)
        return g

    _ = attn.state_dict()
    _ = construct()


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
@pytest.mark.parametrize(
    "topk",
    [
        8,  # one partition: the decode kernel applies the sink
        1024,  # split-K against a 4096 cache: the combine kernel applies it
    ],
)
@pytest.mark.parametrize(
    "sink",
    [
        -np.inf,
        -1.0e38,  # what GLM-5.x / DSv3.2 passed before omitting the sink
    ],
)
def test_mla_decode_graph_sparse_omitted_operands_match(
    topk: int, sink: float
) -> None:
    """Omitted ``topk_lengths`` / ``sink`` match their no-op explicit values.

    ``None`` makes the op build the kernels specialized on the operand's
    absence. Full-length top-k and a ``-inf`` or ``-1e38`` sink are exact
    no-ops, so the explicit graph must produce bit-identical output.
    """
    num_heads = 16
    cache_len = 4096
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
    kv_params = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
    )

    def make_layers() -> tuple[
        DeepseekYarnRotaryEmbedding, LatentAttentionWithRopeFp8
    ]:
        # Fresh layers per graph: weight values bind to the first graph.
        rope = DeepseekYarnRotaryEmbedding(
            dim=qk_rope_head_dim,
            n_heads=num_heads,
            theta=10000.0,
            max_seq_len=2 * cache_len,
            scaling_params=DeepseekYarnRopeScalingParams(
                scaling_factor=40.0,
                original_max_position_embeddings=4096,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=1.0,
            ),
        )
        attn = LatentAttentionWithRopeFp8(
            rope=rope,
            num_attention_heads=num_heads,
            num_key_value_heads=1,
            hidden_size=hidden_size,
            kv_params=kv_params,
            quant_config=quant_config,
            devices=[DeviceRef.GPU()],
            graph_mode="decode",
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            buffer_size=4096,
        )
        _ = attn.state_dict()
        return rope, attn

    gpu = DeviceRef.GPU()
    data_types = [
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
        TensorType(DType.int32, ["total_tokens", topk], gpu),
    ]
    explicit_types = [
        TensorType(DType.int32, ["total_tokens"], gpu),
        TensorType(DType.float32, [num_heads], gpu),
    ]
    kv_sym = tree.leaves(kv_params.get_symbolic_inputs()[0])

    def construct(explicit: bool) -> Graph:
        extra_types = explicit_types if explicit else []
        with Graph(
            f"mla_decode_sparse_{'explicit' if explicit else 'omitted'}",
            input_types=[*data_types, *extra_types, *kv_sym],
        ) as g:
            q, kv, input_row_offsets, sparse_indices = (
                x.tensor for x in g.inputs[:4]
            )
            topk_lengths = g.inputs[4].tensor if explicit else None
            attn_sink = g.inputs[5].tensor if explicit else None
            kv_collection = paged_kv_from_flat_graph_inputs(
                kv_params, list(g.inputs[4 + len(extra_types) :])
            )
            assert kv_collection.attention_dispatch_metadata is not None
            assert kv_collection.mla_num_partitions is not None
            rope, attn = make_layers()
            w_uk, w_uk_scale = attn.w_uk
            w_uv, w_uv_scale = attn.w_uv
            out = mla_decode_graph(
                q,
                kv,
                input_row_offsets,
                ops.cast(rope.freqs_cis, q.dtype).to(q.device),
                attn.kv_a_proj_layernorm,
                w_uk,
                w_uv,
                kv_params,
                kv_collection,
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                MHAMaskVariant.CAUSAL_MASK,
                attn.scale,
                1e-6,
                v_head_dim,
                kv_collection.attention_dispatch_metadata,
                kv_collection.mla_num_partitions,
                w_uk_scale=w_uk_scale,
                w_uv_scale=w_uv_scale,
                quant_config=quant_config,
                sparse_indices=sparse_indices,
                sparse_topk_lengths=topk_lengths,
                sparse_attn_sink=attn_sink,
                sparse_indices_stride=topk,
            )
            g.output(out)
        return g

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    weights = random_weights(make_layers()[1])
    omitted = session.load(construct(False), weights_registry=weights)
    explicit = session.load(construct(True), weights_registry=weights)

    kv_inputs = paged_kv_cache_inputs(kv_params, [1], cache_lengths=[cache_len])
    blocks = (
        torch.randn(tuple(kv_inputs.kv_blocks.shape), dtype=torch.float32) * 0.5
    ).to(torch.float8_e4m3fn)
    kv_inputs = dataclasses.replace(
        kv_inputs,
        kv_blocks=Buffer.from_dlpack(blocks.view(torch.uint8))
        .view(DType.float8_e4m3fn)
        .to(device),
    )

    rng = np.random.default_rng(0)
    q = Buffer.from_dlpack(
        (
            torch.randn(
                (1, num_heads, qk_nope_head_dim + qk_rope_head_dim),
                dtype=torch.float32,
            )
            * 0.1
        ).to(torch.bfloat16)
    ).to(device)
    kv = Buffer.from_dlpack(
        (
            torch.randn(
                (1, kv_lora_rank + qk_rope_head_dim), dtype=torch.float32
            )
            * 0.1
        ).to(torch.bfloat16)
    ).to(device)
    row_offsets = Buffer.from_numpy(np.array([0, 1], dtype=np.uint32)).to(
        device
    )
    indices = Buffer.from_numpy(
        rng.choice(cache_len + 1, size=(1, topk), replace=False).astype(
            np.int32
        )
    ).to(device)
    full_topk = Buffer.from_numpy(np.array([topk], dtype=np.int32)).to(device)
    no_sink = Buffer.from_numpy(np.full(num_heads, sink, dtype=np.float32)).to(
        device
    )

    def run(model: Model, *extra: Buffer) -> np.ndarray:
        out = model.execute(
            q, kv, row_offsets, indices, *extra, *tree.leaves(kv_inputs)
        )[0]
        return from_dlpack(out).cpu().view(torch.int16).numpy()

    # Both graphs write the same new latent row into the cache, so the second
    # run sees the state the first one left.
    want = run(omitted)
    got = run(explicit, full_topk, no_sink)
    assert np.array_equal(got, want)


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
def test_mla_decode_graph_sparse_multi_step_smoke() -> None:
    """E2E prefill then decode for :class:`SparseLatentAttentionWithRopeFp8` (toy shapes).

    Also guards the top-k pad-sentinel fix: asserts the lightning indexer's
    ``-1`` padding survives ``__call__`` into the returned/attention indices
    (i.e. is not rewritten to ``0``). This exercises the layer path
    (``__call__`` -> ``_mla_impl``), not just the standalone kernel op.
    """
    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])

    prefill_len = 8  # > MLA_DECODE_MAX_SEQ_LEN for fused prefill path
    num_heads = 16
    cache_len = 64
    topk = 8
    hidden_size = 1024
    q_lora_rank = 256
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    page_size = 128

    buffer_size = 4096
    total_num_pages = 32
    rope_max_seq_len = 2048

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
        max_seq_len=rope_max_seq_len,
        scaling_params=scaling_params,
    )

    index_head_dim = 128
    mla_kv_params = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.int8,
            quantization_granularity=32,
        ),
    )
    indexer_kv_params = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=index_head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=32,
        ),
    )
    multi_kv = MultiKVCacheParams.from_params(
        {"mla": mla_kv_params, "indexer": indexer_kv_params}
    )

    sparse_attn = SparseLatentAttentionWithRopeFp8(
        rope=rope,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        hidden_size=hidden_size,
        kv_params=mla_kv_params,
        quant_config=quant_config,
        devices=[DeviceRef.GPU()],
        graph_mode="auto",
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        buffer_size=buffer_size,
        index_topk=topk,
    )

    kv_manager = PagedKVCacheManager(
        params=multi_kv,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=32,
    )

    len_mla_kv = len(tree.leaves(mla_kv_params.get_symbolic_inputs()[0]))
    len_indexer_kv = len(
        tree.leaves(indexer_kv_params.get_symbolic_inputs()[0])
    )
    kv_sym = list(multi_kv.flattened_kv_inputs())
    hidden_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", hidden_size],
        DeviceRef.GPU(),
    )
    row_off_type = TensorType(
        DType.uint32, ["row_offsets_len"], DeviceRef.GPU()
    )

    def construct() -> Graph:
        with Graph(
            "mla_sparse_latent_prefill_then_decode_smoke",
            input_types=[
                hidden_type,
                row_off_type,
                *kv_sym,
            ],
        ) as g:
            hidden = g.inputs[0].tensor
            input_row_offsets = g.inputs[1].tensor
            mla_in = g.inputs[2 : 2 + len_mla_kv]
            idx_in = g.inputs[2 + len_mla_kv : 2 + len_mla_kv + len_indexer_kv]
            kv_mla = paged_kv_from_flat_graph_inputs(
                mla_kv_params, list(mla_in)
            )
            kv_idx = paged_kv_from_flat_graph_inputs(
                indexer_kv_params, list(idx_in)
            )
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
            freqs_cis = ops.cast(rope.freqs_cis, hidden.dtype).to(hidden.device)
            out, topk_out = sparse_attn(
                layer_idx,
                hidden,
                kv_mla,
                kv_idx,
                freqs_cis,
                input_row_offsets,
                None,
            )
            # Surface the returned top-k indices too, so the test can assert the
            # indexer's -1 pad sentinels survive __call__ (see the prefill check).
            g.output(out, topk_out)
        return g

    _ = sparse_attn.state_dict()
    graph = construct()
    weights = random_weights(sparse_attn)
    model = session.load(graph, weights_registry=weights)

    def _run_check(out_buf: Buffer, num_tokens: int) -> None:
        out_t = from_dlpack(out_buf).cpu()
        out_np = (
            out_t.float().numpy()
            if out_t.dtype == torch.bfloat16
            else out_t.numpy()
        )
        assert out_np.shape == (num_tokens, hidden_size)
        assert not np.isnan(out_np).any()
        assert np.all(np.isfinite(out_np))

    context = create_text_context(np.empty(cache_len))
    kv_manager.claim(context)
    batch = [context]

    kv_manager.alloc(context)
    kv_ri_pref = kv_manager.runtime_inputs([batch])
    assert isinstance(kv_ri_pref, dict)
    mla_pref = kv_ri_pref["mla"]
    idx_pref = kv_ri_pref["indexer"]
    assert (
        isinstance(mla_pref, tuple)
        and mla_pref
        and isinstance(mla_pref[0], KVCacheInputsPerDevice)
    )
    assert (
        isinstance(idx_pref, tuple)
        and idx_pref
        and isinstance(idx_pref[0], KVCacheInputsPerDevice)
    )

    t_pref = (
        torch.randn((prefill_len, hidden_size), dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    hidden_prefill = Buffer.from_dlpack(t_pref).to(device)
    row_prefill = Buffer.from_numpy(
        np.array([0, prefill_len], dtype=np.uint32)
    ).to(device)
    kv_list = tree.leaves(kv_ri_pref)
    pref_results = model.execute(hidden_prefill, row_prefill, *kv_list)
    out_pref = pref_results[0]
    _run_check(out_pref, prefill_len)

    # Early prefill positions have < index_topk causal keys, so the indexer
    # emits -1 pads; __call__ must not rewrite them to 0.
    topk_pref = from_dlpack(pref_results[1]).cpu().numpy()
    assert (topk_pref == -1).any(), (
        "indexer -1 top-k pad sentinels were not preserved through "
        "SparseLatentAttentionWithRopeFp8.__call__ (the -1->0 clobber must "
        "stay removed)"
    )

    for _ in range(prefill_len):
        context.update(42)
    for ctx in batch:
        kv_manager.step(ctx)

    kv_manager.alloc(context)
    kv_ri_dec = kv_manager.runtime_inputs([batch])
    assert isinstance(kv_ri_dec, dict)
    mla_dec = kv_ri_dec["mla"]
    idx_dec = kv_ri_dec["indexer"]
    assert (
        isinstance(mla_dec, tuple)
        and mla_dec
        and isinstance(mla_dec[0], KVCacheInputsPerDevice)
    )
    assert (
        isinstance(idx_dec, tuple)
        and idx_dec
        and isinstance(idx_dec[0], KVCacheInputsPerDevice)
    )

    t_dec = (torch.randn((1, hidden_size), dtype=torch.float32) * 0.02).to(
        torch.bfloat16
    )
    hidden_dec = Buffer.from_dlpack(t_dec).to(device)
    row_dec = Buffer.from_numpy(np.array([0, 1], dtype=np.uint32)).to(device)
    kv_dec = tree.leaves(kv_ri_dec)
    out_dec = model.execute(hidden_dec, row_dec, *kv_dec)[0]
    _run_check(out_dec, 1)

    context.update(42)
    for ctx in batch:
        kv_manager.step(ctx)
