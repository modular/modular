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
"""Sparse MLA decode through the HiSparse hot buffer, against a resident cache.

``test_mla_decode_graph_hisparse_hot_buffer_matches_resident`` decodes one
token per request over a paged FP8 latent cache twice: once with every page
resident, and once with request 0's first four pages holding only a pinned
host-mirror copy, their lookup-table entries nulled, and their selections
routed through :func:`hisparse_resolve` / :func:`hisparse_gather_group` into
a hot buffer leased from the same cache. Three steps re-pick selections that
overlap the previous step's by about half, so steps two and three hit rows the
first step gathered. At every step the two arms' attention outputs must be
bit-identical, and the resolved rows must equal the resident physical rows on
resident pages and point into the hot pages everywhere else.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
import torch
from _mla_sparse_test_utils import (
    paged_kv_from_flat_graph_inputs,
    random_weights,
)
from max import tree
from max.driver import Accelerator, Buffer, Usage, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    HiSparseParams,
    hisparse_gather_group,
    hisparse_resolve,
    mla_decode_graph,
)
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.legacy.attention.mask_config import MHAMaskVariant
from max.nn.legacy.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.legacy.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs
from torch.utils.dlpack import from_dlpack

_NUM_HEADS = 16
_HIDDEN_SIZE = 1024
_Q_LORA_RANK = 256
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOP_K = 128
_BATCH = 2
_CTX0 = 1000
_CTX1 = 300
_OFFLOAD_PAGES = 4
"""Request 0's leading logical pages, held only in the host mirror."""
_POOL_PAGES = 15
"""Pages the pool hands out; page 15 is the null page tail padding reads."""
_HOT_PAGES = (11, 12)
"""Request 0's leased hot pages, past every page the two requests own."""
_STATE_PAGE = 13
"""Request 0's leased HiSparse state page, zero-filled."""
_HOST_PAGE_BYTES = 1 * 1 * _PAGE_SIZE * 1 * (_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)
"""One latent page's data bytes: one layer, one head, 576 FP8 bytes a row."""
_MIRROR_BYTES = _OFFLOAD_PAGES * _HOST_PAGE_BYTES


def _quant_config() -> QuantConfig:
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


def _kv_params() -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=_NUM_HEADS,
        sparse_indices_stride=_TOP_K,
        sparse_physical_indices=True,
    )


def _make_layers(
    kv_params: MLAKVCacheParams, quant_config: QuantConfig
) -> tuple[DeepseekYarnRotaryEmbedding, LatentAttentionWithRopeFp8]:
    """Fresh layers per graph: weight values bind to the first graph."""
    rope = DeepseekYarnRotaryEmbedding(
        dim=_QK_ROPE_HEAD_DIM,
        n_heads=_NUM_HEADS,
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
    attn = LatentAttentionWithRopeFp8(
        rope=rope,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=1,
        hidden_size=_HIDDEN_SIZE,
        kv_params=kv_params,
        quant_config=quant_config,
        devices=[DeviceRef.GPU()],
        graph_mode="decode",
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        buffer_size=4096,
    )
    _ = attn.state_dict()
    return rope, attn


def _decode(
    q: Any,
    kv: Any,
    input_row_offsets: Any,
    kv_params: MLAKVCacheParams,
    kv_collection: Any,
    attn: LatentAttentionWithRopeFp8,
    rope: DeepseekYarnRotaryEmbedding,
    quant_config: QuantConfig,
    sparse_indices: Any,
) -> Any:
    """One sparse MLA decode over physical rows, as ``_mla_impl`` calls it."""
    assert kv_collection.attention_dispatch_metadata is not None
    assert kv_collection.mla_num_partitions is not None
    w_uk, w_uk_scale = attn.w_uk
    w_uv, w_uv_scale = attn.w_uv
    return mla_decode_graph(
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
        _V_HEAD_DIM,
        kv_collection.attention_dispatch_metadata,
        kv_collection.mla_num_partitions,
        w_uk_scale=w_uk_scale,
        w_uv_scale=w_uv_scale,
        quant_config=quant_config,
        sparse_indices=sparse_indices,
        physical_indices=True,
        sparse_indices_stride=_TOP_K,
    )


def _construct(
    name: str,
    hisparse: bool,
    kv_params: MLAKVCacheParams,
    quant_config: QuantConfig,
    params: HiSparseParams,
    kv_sym: list[Any],
    host_type: TensorType,
    hot_type: TensorType,
    mirror_type: TensorType,
) -> Graph:
    """Builds one decode graph: plain resident, or HiSparse-resolved."""
    gpu = DeviceRef.GPU()
    data_types = [
        TensorType(
            DType.bfloat16,
            [_BATCH, _NUM_HEADS, _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM],
            gpu,
        ),
        TensorType(
            DType.bfloat16,
            [_BATCH, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM],
            gpu,
        ),
        TensorType(DType.uint32, [_BATCH + 1], gpu),
        TensorType(DType.int32, [_BATCH, _TOP_K], gpu),
    ]
    # Arm B reads the logical positions beside the physical rows.
    logical_type = [TensorType(DType.int32, [_BATCH, _TOP_K], gpu)]
    hisparse_types = [host_type, hot_type, mirror_type]
    input_types = [
        *data_types,
        *(logical_type if hisparse else []),
        *kv_sym,
        *(hisparse_types if hisparse else []),
    ]
    with Graph(name, input_types=input_types) as g:
        q, kv, input_row_offsets, physical = (x.tensor for x in g.inputs[:4])
        n_data = 4 + int(hisparse)
        logical = g.inputs[4].tensor if hisparse else None
        n_kv = len(kv_sym)
        kv_collection = paged_kv_from_flat_graph_inputs(
            kv_params, list(g.inputs[n_data : n_data + n_kv])
        )
        rope, attn = _make_layers(kv_params, quant_config)
        if hisparse:
            assert logical is not None
            host_table, hot_table, host_mirror = (
                x.tensor for x in g.inputs[n_data + n_kv :]
            )
            resolved = hisparse_resolve(
                physical,
                logical,
                input_row_offsets,
                kv_collection,
                host_table,
                hot_table,
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                params,
            )
            hisparse_gather_group(
                kv_collection,
                hot_table,
                host_mirror,
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                ops.constant(1, DType.uint32, device=DeviceRef.CPU()),
                params,
            )
            out = _decode(
                q,
                kv,
                input_row_offsets,
                kv_params,
                kv_collection,
                attn,
                rope,
                quant_config,
                resolved,
            )
            g.output(out, resolved)
        else:
            out = _decode(
                q,
                kv,
                input_row_offsets,
                kv_params,
                kv_collection,
                attn,
                rope,
                quant_config,
                physical,
            )
            g.output(out)
    return g


def _physical(lut: np.ndarray, logical: np.ndarray) -> np.ndarray:
    """The indexer's remap: each position's row, null pages giving page 0."""
    rows = np.full(logical.shape, -1, dtype=np.int32)
    b, k = np.nonzero(logical >= 0)
    page = logical[b, k] // _PAGE_SIZE
    rows[b, k] = (
        lut[b, page] * _PAGE_SIZE + logical[b, k] % _PAGE_SIZE
    ).astype(np.int32)
    return rows


def _step_positions(
    rng: np.random.Generator, prev: tuple[np.ndarray, ...] | None
) -> tuple[np.ndarray, ...]:
    """Picks request 0's host-only and resident positions and request 1's.

    Request 0 takes half its selections from its offloaded pages and half
    from resident ones. From the second step on, about half of each group is
    kept from the previous step, so the hot buffer is hit, not just filled.
    """
    host_pool = np.arange(_OFFLOAD_PAGES * _PAGE_SIZE)
    resident_pool = np.arange(_OFFLOAD_PAGES * _PAGE_SIZE, _CTX0 + 1)
    req1_pool = np.arange(_CTX1 + 1)
    if prev is None:
        host = rng.choice(host_pool, 64, replace=False)
        resident = rng.choice(resident_pool, 64, replace=False)
        req1 = rng.choice(req1_pool, _TOP_K, replace=False)
    else:
        p_host, p_resident, p_req1 = prev

        def refresh(
            keep: np.ndarray, pool: np.ndarray, count: int
        ) -> np.ndarray:
            held = keep[: count // 2]
            fresh = rng.choice(
                np.setdiff1d(pool, held), count - len(held), replace=False
            )
            return np.concatenate([held, fresh])

        host = refresh(p_host, host_pool, 64)
        resident = refresh(p_resident, resident_pool, 64)
        req1 = refresh(p_req1, req1_pool, _TOP_K)
    return host, resident, req1


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Sparse MLA decode graph is only wired for NVIDIA GPUs.",
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="Sparse MLA decode kernel is SM100-class (B100/B200); skip elsewhere.",
)
def test_mla_decode_graph_hisparse_hot_buffer_matches_resident() -> None:
    """HiSparse-routed sparse decode is bit-identical to a resident decode.

    Both arms share one cache, one query and one set of selections; arm B
    reads request 0's offloaded pages through hot-buffer rows gathered from
    a pinned host mirror, arm A reads them in place. Three steps overlap
    their selections by about half, so the later steps hit the hot buffer.
    """
    kv_params = _kv_params()
    quant_config = _quant_config()
    params = HiSparseParams.for_top_k(
        top_k=_TOP_K, query_rows=1, page_size=_PAGE_SIZE, num_groups=1
    )
    kv_sym = tree.leaves(kv_params.get_symbolic_inputs()[0])
    lut_sym = kv_params.get_symbolic_inputs()[0].lookup_table
    assert isinstance(lut_sym, TensorType)
    host_type = TensorType(DType.int32, lut_sym.shape, lut_sym.device)
    hot_type = TensorType(
        DType.uint32, [lut_sym.shape[0], params.hot_pages + 1], lut_sym.device
    )
    mirror_type = TensorType(
        DType.uint8, [_MIRROR_BYTES], device=DeviceRef.CPU()
    )

    # Every graph is loaded before any device buffer is staged, so the
    # precompile record run compiles both under virtual devices.
    session = InferenceSession(devices=[Accelerator()])
    weights = random_weights(_make_layers(kv_params, quant_config)[1])
    resident_model = session.load(
        _construct(
            "mla_hisparse_resident_decode",
            False,
            kv_params,
            quant_config,
            params,
            kv_sym,
            host_type,
            hot_type,
            mirror_type,
        ),
        weights_registry=weights,
    )
    hisparse_model = session.load(
        _construct(
            "mla_hisparse_hot_decode",
            True,
            kv_params,
            quant_config,
            params,
            kv_sym,
            host_type,
            hot_type,
            mirror_type,
        ),
        weights_registry=weights,
    )

    device = Accelerator(0)
    kv_inputs = paged_kv_cache_inputs(
        kv_params,
        [1] * _BATCH,
        cache_lengths=[_CTX0, _CTX1],
        total_num_pages=_POOL_PAGES,
    )
    # Random FP8 latent bytes; the leased state page (and the null tail page
    # the LUT padding reads) must start zeroed.
    blocks = (
        torch.randn(tuple(kv_inputs.kv_blocks.shape), dtype=torch.float32) * 0.5
    ).to(torch.float8_e4m3fn)
    block_bytes = blocks.view(torch.uint8)
    block_bytes[_STATE_PAGE] = 0
    block_bytes[_POOL_PAGES] = 0
    kv_inputs = dataclasses.replace(
        kv_inputs,
        kv_blocks=Buffer.from_dlpack(block_bytes)
        .view(DType.float8_e4m3fn)
        .to(device),
    )

    # The host mirror holds request 0's offloaded pages, copied out of the
    # cache buffer after filling it, in pinned host memory the GPU reads.
    pool_bytes = (
        kv_inputs.kv_blocks.view(DType.uint8)
        .to_numpy()
        .reshape(_POOL_PAGES + 1, -1)
    )
    assert pool_bytes.shape[1] == _HOST_PAGE_BYTES
    mirror = Buffer(
        shape=[_MIRROR_BYTES],
        dtype=DType.uint8,
        device=device,
        usage=Usage.STAGING,
    )
    mirror.inplace_copy_from(
        Buffer.from_numpy(pool_bytes[:_OFFLOAD_PAGES].reshape(-1).copy())
    )

    # Arm A's lookup table maps every page; arm B nulls request 0's first
    # four, which is what marks them host-only.
    lut_resident = kv_inputs.lookup_table.to_numpy()
    lut_hisparse = lut_resident.copy()
    lut_hisparse[0, :_OFFLOAD_PAGES] = 0
    host_table = np.full(lut_resident.shape, -1, dtype=np.int32)
    host_table[0, :_OFFLOAD_PAGES] = np.arange(_OFFLOAD_PAGES)
    hot_table = np.zeros((_BATCH, params.hot_pages + 1), dtype=np.uint32)
    hot_table[0, : params.hot_pages] = _HOT_PAGES
    hot_table[0, params.hot_pages] = _STATE_PAGE

    leaves_resident = tree.leaves(kv_inputs)
    leaves_hisparse = tree.leaves(
        dataclasses.replace(
            kv_inputs,
            lookup_table=Buffer.from_numpy(lut_hisparse).to(device),
        )
    )
    host_table_buf = Buffer.from_numpy(host_table).to(device)
    hot_table_buf = Buffer.from_numpy(hot_table).to(device)

    q = Buffer.from_dlpack(
        (
            torch.randn(
                (_BATCH, _NUM_HEADS, _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
                dtype=torch.float32,
            )
            * 0.1
        ).to(torch.bfloat16)
    ).to(device)
    kv = Buffer.from_dlpack(
        (
            torch.randn(
                (_BATCH, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM), dtype=torch.float32
            )
            * 0.1
        ).to(torch.bfloat16)
    ).to(device)
    row_offsets = Buffer.from_numpy(np.arange(_BATCH + 1, dtype=np.uint32)).to(
        device
    )

    rng = np.random.default_rng(0)
    prev: tuple[np.ndarray, ...] | None = None
    for step in range(3):
        host, resident, req1 = _step_positions(rng, prev)
        prev = (host, resident, req1)
        logical = np.stack([np.concatenate([host, resident]), req1]).astype(
            np.int32
        )
        physical_a = _physical(lut_resident, logical)
        physical_b = _physical(lut_hisparse, logical)

        out_a = resident_model.execute(
            q,
            kv,
            row_offsets,
            Buffer.from_numpy(physical_a).to(device),
            *leaves_resident,
        )[0]
        out_b, resolved_buf = hisparse_model.execute(
            q,
            kv,
            row_offsets,
            Buffer.from_numpy(physical_b).to(device),
            Buffer.from_numpy(logical).to(device),
            *leaves_hisparse,
            host_table_buf,
            hot_table_buf,
            mirror,
        )
        resolved = resolved_buf.to_numpy()

        # The resolve leaves resident rows alone and points host-only rows
        # into the leased hot pages.
        page = np.where(logical >= 0, logical // _PAGE_SIZE, 0)
        is_resident = lut_hisparse[np.arange(_BATCH)[:, None], page] != 0
        assert np.array_equal(resolved[is_resident], physical_b[is_resident])
        hot_rows = resolved[~is_resident]
        assert hot_rows.size > 0, "no selection reached the hot buffer"
        hot_lo = _HOT_PAGES[0] * _PAGE_SIZE
        hot_hi = (_HOT_PAGES[-1] + 1) * _PAGE_SIZE
        assert np.all((hot_rows >= hot_lo) & (hot_rows < hot_hi))

        # The two arms' attention outputs must be bit-identical.
        a = from_dlpack(out_a).cpu()
        b = from_dlpack(out_b).cpu()
        max_abs_diff = np.abs(a.float().numpy() - b.float().numpy()).max()
        print(f"step {step}: max abs difference {max_abs_diff}")
        assert np.array_equal(
            a.view(torch.int16).numpy(), b.view(torch.int16).numpy()
        )
        assert max_abs_diff == 0.0
