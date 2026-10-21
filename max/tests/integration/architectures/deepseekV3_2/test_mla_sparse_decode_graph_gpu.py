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
``test_mla_decode_graph_sparse_kpool_row_bound`` checks that a k-pool row
layout masks the ``-1`` padding inside the scan, including for a reused row.
``test_mla_graph_sparse_plain_topk_selection_lag`` checks the same for a
reused plain top-k row, through the decode and the combined op.
``test_mla_decode_graph_sparse_bf16_gfx950_matches_dense`` checks the gfx950
gather arm against dense attention, and
``test_mla_decode_graph_sparse_bf16_gfx950_topk_lengths_bound`` checks that it
drops slots past ``topk_lengths``.
``test_mla_decode_graph_sparse_multi_step_smoke`` runs prefill and decode through
:class:`SparseLatentAttentionWithRopeFp8` and :class:`PagedKVCacheManager`.
The prefill-routing tests live in ``test_mla_sparse_prefill_graph_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Any

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
from max.nn.kernels import mla_decode_graph, mla_prefill_decode_graph
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
from test_common.graph_utils import is_b100_b200, is_gfx950
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
                physical_indices=False,
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

            args = (
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
            )
            # The row layout and lag describe sparse indices.
            with pytest.raises(ValueError, match="sparse_indices"):
                mla_decode_graph(*args, index_kpool=4, index_pool_topk=2)
            with pytest.raises(ValueError, match="sparse_indices"):
                mla_decode_graph(*args, selection_lag=1)
            # Each sparse call says which kind of index it passes.
            with pytest.raises(ValueError, match="physical_indices"):
                mla_decode_graph(
                    *args,
                    sparse_indices=sparse_indices,
                    sparse_indices_stride=indices_stride,
                )
            with pytest.raises(ValueError, match="physical_indices"):
                mla_decode_graph(*args, physical_indices=True)
            with pytest.raises(ValueError, match="index_share"):
                mla_decode_graph(
                    *args,
                    sparse_indices=sparse_indices,
                    sparse_indices_stride=indices_stride,
                    physical_indices=True,
                    index_share=True,
                )
            out = mla_decode_graph(
                *args,
                sparse_indices=sparse_indices,
                physical_indices=False,
                sparse_topk_lengths=sparse_topk_lengths,
                sparse_attn_sink=sparse_attn_sink,
                sparse_indices_stride=indices_stride,
                index_kpool=4,
                index_pool_topk=1,
                selection_lag=1,
            )
            g.output(out)
        return g

    _ = attn.state_dict()
    _ = construct()


_SPARSE_NUM_HEADS = 16
_SPARSE_CACHE_LEN = 4096
_SPARSE_PAGE_SIZE = 128


@dataclasses.dataclass(frozen=True)
class _Fp8DecodeVariant:
    """One fp8 sparse decode graph of a :func:`_sparse_decode_fixture`."""

    explicit: bool = False
    """Passes ``topk_lengths`` and ``sink`` inputs instead of omitting them."""
    index_share: bool = False
    selection_lag: int = 0


@dataclasses.dataclass
class _SparseDecodeFixture:
    """Fp8 sparse MLA decode graphs sharing weights, a cache and a query."""

    models: list[Model]
    kv_inputs: KVCacheInputsPerDevice[Buffer, Buffer]
    q: Buffer
    kv: Buffer
    row_offsets: Buffer

    def run(self, model: Model, indices: Buffer, *extra: Buffer) -> np.ndarray:
        # Each run writes the same new latent row, so later runs see the
        # state the first left.
        out = model.execute(
            self.q,
            self.kv,
            self.row_offsets,
            indices,
            *extra,
            *tree.leaves(self.kv_inputs),
        )[0]
        return from_dlpack(out).cpu().view(torch.int16).numpy()


def _sparse_decode_fixture(
    topk: int,
    variants: Sequence[_Fp8DecodeVariant],
    cache_len: int = _SPARSE_CACHE_LEN,
) -> _SparseDecodeFixture:
    """Builds one single-token decode graph per variant over one cache."""
    num_heads = _SPARSE_NUM_HEADS
    hidden_size = 1024
    q_lora_rank = 256
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128

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
        page_size=_SPARSE_PAGE_SIZE,
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
    kv_sym = tree.leaves(kv_params.get_symbolic_inputs()[0])

    explicit_types = [
        TensorType(DType.int32, ["total_tokens"], gpu),
        TensorType(DType.float32, [num_heads], gpu),
    ]

    def construct(variant: _Fp8DecodeVariant) -> Graph:
        extra_types = explicit_types if variant.explicit else []
        with Graph(
            "mla_decode_sparse",
            input_types=[*data_types, *extra_types, *kv_sym],
        ) as g:
            q, kv, input_row_offsets, sparse_indices = (
                x.tensor for x in g.inputs[:4]
            )
            topk_lengths = g.inputs[4].tensor if variant.explicit else None
            attn_sink = g.inputs[5].tensor if variant.explicit else None
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
                physical_indices=False,
                sparse_topk_lengths=topk_lengths,
                sparse_attn_sink=attn_sink,
                sparse_indices_stride=topk,
                index_share=variant.index_share,
                selection_lag=variant.selection_lag,
            )
            g.output(out)
        return g

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    weights = random_weights(make_layers()[1])
    models = [
        session.load(construct(v), weights_registry=weights) for v in variants
    ]

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
    return _SparseDecodeFixture(models, kv_inputs, q, kv, row_offsets)


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
    fx = _sparse_decode_fixture(
        topk, [_Fp8DecodeVariant(), _Fp8DecodeVariant(explicit=True)]
    )
    omitted, explicit = fx.models
    device = Accelerator(0)
    rng = np.random.default_rng(0)
    indices = Buffer.from_numpy(
        rng.choice(_SPARSE_CACHE_LEN + 1, size=(1, topk), replace=False).astype(
            np.int32
        )
    ).to(device)
    full_topk = Buffer.from_numpy(np.array([topk], dtype=np.int32)).to(device)
    no_sink = Buffer.from_numpy(
        np.full(_SPARSE_NUM_HEADS, sink, dtype=np.float32)
    ).to(device)

    want = fx.run(omitted, indices)
    got = fx.run(explicit, indices, full_topk, no_sink)
    assert np.array_equal(got, want)


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
def test_mla_decode_graph_sparse_page_placement_invariant() -> None:
    """Relocating the cache's pages leaves the sparse decode output unchanged.

    The kernels resolve each logical key position through the page table, so
    moving every page -- table entry and block contents together -- must give
    bit-identical output. The negative control moves only the table entries,
    which must change it.
    """
    topk = 64
    page_size = _SPARSE_PAGE_SIZE
    fx = _sparse_decode_fixture(topk, [_Fp8DecodeVariant()])
    (model,) = fx.models
    device = Accelerator(0)

    rng = np.random.default_rng(0)
    logical = rng.choice(
        _SPARSE_CACHE_LEN + 1, size=(1, topk), replace=False
    ).astype(np.int32)
    logical[0, -3:] = -1
    indices = Buffer.from_numpy(logical).to(device)

    # The first run writes the new latent row; every later run writes the same
    # row to the same logical position, so it is part of the moved state too.
    want = fx.run(model, indices)
    base = fx.kv_inputs

    lut = from_dlpack(base.lookup_table).cpu().numpy()
    num_pages = (_SPARSE_CACHE_LEN + 1 + page_size - 1) // page_size
    moved_lut = lut.copy()
    moved_lut[0, :num_pages] = lut[0, :num_pages][::-1]
    assert not np.array_equal(moved_lut, lut)
    blocks = from_dlpack(base.kv_blocks.view(DType.uint8)).cpu()
    moved_blocks = blocks.clone()
    for page in range(num_pages):
        moved_blocks[int(moved_lut[0, page])] = blocks[int(lut[0, page])]

    def with_cache(table: np.ndarray, kv_blocks: torch.Tensor) -> None:
        fx.kv_inputs = dataclasses.replace(
            base,
            lookup_table=Buffer.from_numpy(table).to(device),
            kv_blocks=Buffer.from_dlpack(kv_blocks)
            .view(DType.float8_e4m3fn)
            .to(device),
        )

    with_cache(moved_lut, moved_blocks)
    assert np.array_equal(fx.run(model, indices), want)
    with_cache(moved_lut, blocks)
    assert not np.array_equal(fx.run(model, indices), want)


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
def test_mla_decode_graph_sparse_index_share_single_token() -> None:
    """``index_share`` at one query token matches the per-position op.

    The draft steps that set ``index_share`` decode one token, where the
    shared-index fold does not launch and the reused row is bounded by
    ``selection_lag`` alone. The output must be bit-identical to the plain
    op with the same lag. The negative control drops the lag, which scores
    the row's ``-1`` padding.
    """
    width, cache_len, lag = 128, 100, 2
    fx = _sparse_decode_fixture(
        width,
        [
            _Fp8DecodeVariant(selection_lag=lag),
            _Fp8DecodeVariant(index_share=True, selection_lag=lag),
            _Fp8DecodeVariant(index_share=True),
        ],
        cache_len=cache_len,
    )
    plain, shared, unbounded = fx.models
    rng = np.random.default_rng(0)
    # A plain top-k row made for the token `lag` positions earlier: every key
    # it could see, in score order, then `-1` padding.
    valid = cache_len + 1 - lag
    row = np.full((1, width), -1, dtype=np.int32)
    row[0, :valid] = rng.permutation(valid)
    indices = Buffer.from_numpy(row).to(Accelerator(0))

    want = fx.run(plain, indices)
    assert np.array_equal(fx.run(shared, indices), want)
    assert not np.array_equal(fx.run(unbounded, indices), want)


@dataclasses.dataclass
class _Bf16SparseRowHarness:
    """Loads and runs one-token BF16 sparse MLA decodes over a shared cache.

    Every run writes the same new latent row into the cache, so each sees the
    state the first one left.
    """

    width: int
    cache_len: int
    load: Callable[..., Model]
    run: Callable[[Model, Buffer, int], np.ndarray]
    row: Callable[[int], Buffer]


def _bf16_sparse_row_harness(
    width: int, cache_len: int
) -> _Bf16SparseRowHarness:
    num_heads = 16
    hidden_size = 1024
    q_lora_rank = 256
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    page_size = 128

    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
    )

    def make_layers() -> tuple[
        DeepseekYarnRotaryEmbedding, LatentAttentionWithRope
    ]:
        # Fresh layers per graph: weight values bind to the first graph.
        rope = DeepseekYarnRotaryEmbedding(
            dim=qk_rope_head_dim,
            n_heads=num_heads,
            theta=10000.0,
            max_seq_len=2 * (cache_len + 1),
            scaling_params=DeepseekYarnRopeScalingParams(
                scaling_factor=40.0,
                original_max_position_embeddings=4096,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=1.0,
            ),
        )
        attn = LatentAttentionWithRope(
            rope=rope,
            num_attention_heads=num_heads,
            num_key_value_heads=1,
            hidden_size=hidden_size,
            kv_params=kv_params,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            graph_mode="auto",
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
        TensorType(DType.int32, ["total_tokens", width], gpu),
        TensorType(DType.int32, ["batch"], gpu),
        *tree.leaves(kv_params.get_symbolic_inputs()[0]),
    ]

    def construct(name: str, prefill_decode: bool, **layout: int) -> Graph:
        with Graph(name, input_types=input_types) as g:
            q, kv, input_row_offsets, sparse_indices, topk_lengths = (
                x.tensor for x in g.inputs[:5]
            )
            kv_collection = paged_kv_from_flat_graph_inputs(
                kv_params, list(g.inputs[5:])
            )
            assert kv_collection.attention_dispatch_metadata is not None
            assert kv_collection.mla_num_partitions is not None
            rope, attn = make_layers()
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
                sparse_indices=sparse_indices,
                physical_indices=False,
                sparse_topk_lengths=topk_lengths,
                sparse_indices_stride=width,
                **layout,
            )
            freqs_cis = ops.cast(rope.freqs_cis, q.dtype).to(q.device)
            if prefill_decode:
                metadata = attn.create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
                out = mla_prefill_decode_graph(
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
                )
            else:
                out = mla_decode_graph(
                    q,
                    kv,
                    input_row_offsets,
                    freqs_cis,
                    attn.kv_a_proj_layernorm,
                    attn.w_uk,
                    attn.w_uv,
                    **common,
                )
            g.output(out)
        return g

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    weights = random_weights(make_layers()[1])

    def load(name: str, prefill_decode: bool = False, **layout: int) -> Model:
        return session.load(
            construct(name, prefill_decode, **layout), weights_registry=weights
        )

    # Staged on first use, after every load: precompile_mefs records the loads
    # under virtual devices, which refuse device buffers.
    @functools.cache
    def inputs() -> tuple[Buffer, ...]:
        kv_inputs = paged_kv_cache_inputs(
            kv_params, [1], cache_lengths=[cache_len]
        )
        kv_inputs = dataclasses.replace(
            kv_inputs,
            kv_blocks=Buffer.from_dlpack(
                (torch.randn(tuple(kv_inputs.kv_blocks.shape)) * 0.5).to(
                    torch.bfloat16
                )
            ).to(device),
        )
        q = Buffer.from_dlpack(
            (
                torch.randn((1, num_heads, qk_nope_head_dim + qk_rope_head_dim))
                * 0.1
            ).to(torch.bfloat16)
        ).to(device)
        kv = Buffer.from_dlpack(
            (torch.randn((1, kv_lora_rank + qk_rope_head_dim)) * 0.1).to(
                torch.bfloat16
            )
        ).to(device)
        row_offsets = Buffer.from_numpy(np.array([0, 1], dtype=np.uint32)).to(
            device
        )
        return q, kv, row_offsets, *tree.leaves(kv_inputs)

    rng = np.random.default_rng(0)

    def row(valid: int) -> Buffer:
        indices = np.full((1, width), -1, dtype=np.int32)
        indices[0, :valid] = rng.choice(
            cache_len + 1, size=valid, replace=False
        )
        return Buffer.from_numpy(indices).to(device)

    def run(model: Model, indices: Buffer, topk: int) -> np.ndarray:
        lengths = Buffer.from_numpy(np.array([topk], dtype=np.int32)).to(device)
        q, kv, row_offsets, *kv_inputs = inputs()
        out = model.execute(q, kv, row_offsets, indices, lengths, *kv_inputs)[0]
        return from_dlpack(out).cpu().view(torch.int16).numpy()

    return _Bf16SparseRowHarness(width, cache_len, load, run, row)


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
def test_mla_decode_graph_sparse_kpool_row_bound() -> None:
    """A k-pool row layout masks the ``-1`` padding inside the scan.

    A k-pool indexer with ``kpool = 4`` and ``pool_topk = 2`` writes rows of
    ``4 * 2 + 3 = 11`` slots. A token that sees 4096 keys fills only 8 of them
    (both pools, no tail), so slots 8 to 10 are ``-1`` while still below the
    causal limit. With the layout, the BF16 op must match the plain op given
    ``topk_lengths = 8``; without it, the op scores the padding as zero keys.
    A row reused two positions later (``selection_lag = 2``) was computed for
    4094 keys and holds ``8 + 4094 % 4 = 10``.
    """
    kpool = 4
    pool_topk = 2
    h = _bf16_sparse_row_harness(
        width=kpool * pool_topk + kpool - 1, cache_len=4095
    )
    plain = h.load("plain")
    pooled = h.load("kpool", index_kpool=kpool, index_pool_topk=pool_topk)
    reused = h.load(
        "kpool_lag2",
        index_kpool=kpool,
        index_pool_topk=pool_topk,
        selection_lag=2,
    )

    fresh = h.row(8)
    want = h.run(plain, fresh, 8)
    assert np.array_equal(h.run(pooled, fresh, h.width), want)
    # Without the layout the padding is scored, so the output must move.
    assert not np.array_equal(h.run(plain, fresh, h.width), want)

    stale = h.row(10)
    assert np.array_equal(
        h.run(reused, stale, h.width), h.run(plain, stale, 10)
    )


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
@pytest.mark.parametrize("prefill_decode", [False, True])
def test_mla_graph_sparse_plain_topk_selection_lag(
    prefill_decode: bool,
) -> None:
    """A reused plain top-k row is bounded by the length made for its token.

    An MTP draft step reuses the selection step 0 made for the token
    ``selection_lag`` positions earlier. Below ``top_k`` keys that row holds
    one entry per key that token saw, so the slots between its length and
    this token's causal limit are ``-1`` and must not be scored. Both the
    decode op and the combined op's decode branch must take the lag.
    """
    lag = 2
    h = _bf16_sparse_row_harness(width=128, cache_len=100)
    plain = h.load("plain", prefill_decode)
    reused = h.load("lag", prefill_decode, selection_lag=lag)

    valid = h.cache_len + 1 - lag
    stale = h.row(valid)
    want = h.run(plain, stale, valid)
    assert np.array_equal(h.run(reused, stale, h.width), want)
    # Without the lag the padding below the causal limit is scored.
    assert not np.array_equal(h.run(plain, stale, h.width), want)


_GFX950_SPARSE_WIDTH = 320


@dataclasses.dataclass
class _Gfx950SparseHarness:
    """BF16 sparse MLA decode and a dense reference over one shared cache."""

    sparse: Model
    dense: Model
    run: Callable[..., np.ndarray]


@functools.cache
def _gfx950_sparse_harness() -> _Gfx950SparseHarness:
    num_heads = 16
    hidden_size = 1024
    q_lora_rank = 256
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    page_size = 128
    width = _GFX950_SPARSE_WIDTH
    cache_head_dim = kv_lora_rank + qk_rope_head_dim

    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=cache_head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
    )

    def make_layers() -> tuple[
        DeepseekYarnRotaryEmbedding, LatentAttentionWithRope
    ]:
        rope = DeepseekYarnRotaryEmbedding(
            dim=qk_rope_head_dim,
            n_heads=num_heads,
            theta=10000.0,
            max_seq_len=4096,
            scaling_params=DeepseekYarnRopeScalingParams(
                scaling_factor=40.0,
                original_max_position_embeddings=4096,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=1.0,
            ),
        )
        attn = LatentAttentionWithRope(
            rope=rope,
            num_attention_heads=num_heads,
            num_key_value_heads=1,
            hidden_size=hidden_size,
            kv_params=kv_params,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
            graph_mode="auto",
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
    base_types = [
        TensorType(
            DType.bfloat16,
            ["total_tokens", num_heads, qk_nope_head_dim + qk_rope_head_dim],
            gpu,
        ),
        TensorType(DType.bfloat16, ["total_tokens", cache_head_dim], gpu),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
    ]
    kv_types = tree.leaves(kv_params.get_symbolic_inputs()[0])
    sparse_types = [
        TensorType(DType.int32, ["total_tokens", width], gpu),
        TensorType(DType.int32, ["total_tokens"], gpu),
    ]

    def common(
        attn: LatentAttentionWithRope, kv_inputs: Sequence[Any]
    ) -> dict[str, Any]:
        kv_collection = paged_kv_from_flat_graph_inputs(
            kv_params, list(kv_inputs)
        )
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        return dict(
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

    def construct_sparse() -> Graph:
        with Graph(
            "gfx950_sparse", input_types=[*base_types, *kv_types, *sparse_types]
        ) as g:
            q, kv, row_offsets = (x.tensor for x in g.inputs[:3])
            indices, lengths = (x.tensor for x in g.inputs[-2:])
            rope, attn = make_layers()
            out = mla_decode_graph(
                q,
                kv,
                row_offsets,
                ops.cast(rope.freqs_cis, q.dtype).to(q.device),
                attn.kv_a_proj_layernorm,
                attn.w_uk,
                attn.w_uv,
                **common(attn, g.inputs[3:-2]),
                sparse_indices=indices,
                physical_indices=False,
                sparse_topk_lengths=lengths,
                sparse_indices_stride=width,
            )
            g.output(out)
        return g

    def construct_dense() -> Graph:
        with Graph("gfx950_dense", input_types=[*base_types, *kv_types]) as g:
            q, kv, row_offsets = (x.tensor for x in g.inputs[:3])
            rope, attn = make_layers()
            args = common(attn, g.inputs[3:])
            metadata = attn.create_mla_prefill_metadata(
                row_offsets, args["kv_collection"]
            )
            out = mla_prefill_decode_graph(
                q,
                kv,
                row_offsets,
                ops.cast(rope.freqs_cis, q.dtype).to(q.device),
                attn.kv_a_proj_layernorm,
                metadata.buffer_row_offsets,
                metadata.cache_offsets,
                metadata.buffer_lengths.to(DeviceRef.CPU()),
                attn.w_k,
                attn.w_uk,
                attn.w_uv,
                **args,
            )
            g.output(out)
        return g

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    weights = random_weights(make_layers()[1])
    sparse = session.load(construct_sparse(), weights_registry=weights)
    dense = session.load(construct_dense(), weights_registry=weights)

    def run(
        model: Model,
        cache_len: int,
        seq_len: int,
        *sparse_inputs: np.ndarray,
    ) -> np.ndarray:
        # Seeded per shape, so every model sees the same cache and query.
        gen = torch.Generator().manual_seed(cache_len * 4096 + seq_len)
        kv_inputs = paged_kv_cache_inputs(
            kv_params, [seq_len], cache_lengths=[cache_len]
        )
        # Reversed pages, so a remap that skips the page table reads the
        # wrong keys.
        lut = from_dlpack(kv_inputs.lookup_table).cpu().numpy()
        num_pages = -(-(cache_len + seq_len) // page_size)
        lut[0, :num_pages] = lut[0, :num_pages][::-1].copy()
        blocks = (
            torch.randn(tuple(kv_inputs.kv_blocks.shape), generator=gen) * 0.5
        )
        kv_inputs = dataclasses.replace(
            kv_inputs,
            lookup_table=Buffer.from_numpy(lut).to(device),
            kv_blocks=Buffer.from_dlpack(blocks.to(torch.bfloat16)).to(device),
        )
        q = torch.randn(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            generator=gen,
        )
        kv = torch.randn((seq_len, cache_head_dim), generator=gen)
        inputs = [
            Buffer.from_dlpack((q * 0.1).to(torch.bfloat16)).to(device),
            Buffer.from_dlpack((kv * 0.1).to(torch.bfloat16)).to(device),
            Buffer.from_numpy(np.array([0, seq_len], dtype=np.uint32)).to(
                device
            ),
            *tree.leaves(kv_inputs),
            *(Buffer.from_numpy(x).to(device) for x in sparse_inputs),
        ]
        out = model.execute(*inputs)[0]
        return from_dlpack(out).cpu().float().numpy()

    return _Gfx950SparseHarness(sparse, dense, run)


def _scattered_rows(
    rng: np.random.Generator, visible: Sequence[int], keep: float = 1.0
) -> np.ndarray:
    """Spreads each token's visible keys over a row with ``-1`` holes."""
    rows = np.full((len(visible), _GFX950_SPARSE_WIDTH), -1, dtype=np.int32)
    for t, n in enumerate(visible):
        keys = rng.permutation(n)[: max(1, int(n * keep))]
        cols = rng.choice(_GFX950_SPARSE_WIDTH, size=len(keys), replace=False)
        rows[t, cols] = keys
    return rows


@pytest.mark.skipif(
    not is_gfx950(), reason="Exercises the gfx950 sparse MLA gather arm."
)
@pytest.mark.parametrize(
    "cache_len, seq_len",
    [
        (200, 1),
        # Past one 256-token gather chunk, so the second chunk's offsets apply.
        (8, 290),
    ],
)
def test_mla_decode_graph_sparse_bf16_gfx950_matches_dense(
    cache_len: int, seq_len: int
) -> None:
    """Selecting every visible key matches dense causal attention.

    Each token selects all the keys it can see, in shuffled order and spread
    over a row with interior ``-1`` holes, as a k-pool selection leaves them.
    The sparse op takes logical positions through a reversed page table, so
    the remap, the compaction and the per-chunk offsets must all be right for
    it to match the dense op. Keeping only half the keys must not.
    """
    h = _gfx950_sparse_harness()
    rng = np.random.default_rng(seq_len)
    visible = [cache_len + t + 1 for t in range(seq_len)]
    lengths = np.full(seq_len, _GFX950_SPARSE_WIDTH, dtype=np.int32)

    want = h.run(h.dense, cache_len, seq_len)
    got = h.run(
        h.sparse,
        cache_len,
        seq_len,
        _scattered_rows(rng, visible),
        lengths,
    )
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-3)

    half = h.run(
        h.sparse,
        cache_len,
        seq_len,
        _scattered_rows(rng, visible, keep=0.5),
        lengths,
    )
    assert not np.allclose(half, want, rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(
    not is_gfx950(), reason="Exercises the gfx950 sparse MLA gather arm."
)
def test_mla_decode_graph_sparse_bf16_gfx950_topk_lengths_bound() -> None:
    """Slots past ``topk_lengths`` are dropped even when they hold keys.

    A row whose every slot is valid, read with a shorter ``topk_lengths``,
    must match the same row with those slots cleared to ``-1``.
    """
    h = _gfx950_sparse_harness()
    cache_len = 1000
    topk = 250
    rng = np.random.default_rng(0)
    row = rng.choice(
        cache_len + 1, size=(1, _GFX950_SPARSE_WIDTH), replace=False
    )
    row = row.astype(np.int32)
    cleared = row.copy()
    cleared[0, topk:] = -1

    got = h.run(h.sparse, cache_len, 1, row, np.array([topk], dtype=np.int32))
    want = h.run(
        h.sparse,
        cache_len,
        1,
        cleared,
        np.array([_GFX950_SPARSE_WIDTH], dtype=np.int32),
    )
    assert np.array_equal(got, want)


@pytest.mark.skipif(
    not (is_b100_b200() or is_gfx950()),
    reason="Sparse MLA decode runs on SM100 (B100/B200) and gfx950 (MI355).",
)
def test_mla_decode_graph_sparse_multi_step_smoke() -> None:
    """E2E prefill then decode for :class:`SparseLatentAttentionWithRopeFp8` (toy shapes).

    Also guards the top-k pad-sentinel fix: asserts the lightning indexer's
    ``-1`` padding survives ``__call__`` into the returned physical rows
    (i.e. is not rewritten to ``0``). This exercises the layer path
    (``__call__`` -> ``_mla_impl``), not just the standalone kernel op.

    A second cache layer reuses the returned rows the way a shared top-k layer
    or a later MTP draft step does. It caches the same latent rows as the
    first, so its output must match the first layer's exactly, in prefill and
    in decode.
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
        num_layers=2,
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
        num_layers=2,
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
            out, rows = sparse_attn(
                layer_idx,
                hidden,
                kv_mla,
                kv_idx,
                freqs_cis,
                input_row_offsets,
                None,
            )
            reused, _ = sparse_attn(
                ops.constant(1, DType.uint32, device=DeviceRef.CPU()),
                hidden,
                kv_mla,
                kv_idx,
                freqs_cis,
                input_row_offsets,
                None,
                prev_topk_indices=rows,
                reuse_prev_topk=True,
            )
            # Surface the returned rows too, so the test can assert the
            # indexer's -1 pad sentinels survive __call__ (see the prefill check).
            g.output(out, rows, reused)
        return g

    _ = sparse_attn.state_dict()
    graph = construct()
    weights = random_weights(sparse_attn)
    model = session.load(graph, weights_registry=weights)

    kv_manager = PagedKVCacheManager(
        params=multi_kv,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=32,
    )

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
    np.testing.assert_array_equal(
        from_dlpack(pref_results[0]).cpu().float().numpy(),
        from_dlpack(pref_results[2]).cpu().float().numpy(),
    )

    # Early prefill positions have < index_topk causal keys, so the indexer
    # emits -1 pads; __call__ must not rewrite them to 0.
    rows_pref = from_dlpack(pref_results[1]).cpu().numpy()
    assert (rows_pref == -1).any(), (
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
    dec_results = model.execute(hidden_dec, row_dec, *kv_dec)
    _run_check(dec_results[0], 1)
    np.testing.assert_array_equal(
        from_dlpack(dec_results[0]).cpu().float().numpy(),
        from_dlpack(dec_results[2]).cpu().float().numpy(),
    )

    context.update(42)
    for ctx in batch:
        kv_manager.step(ctx)
