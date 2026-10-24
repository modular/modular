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
"""Sparse BF16-weight MLA graph ops over an NVFP4 latent cache, on SM100.

The SM100 sparse kernels dequantize each NVFP4 element to ``e4m3(e2m1 * sf)``
in shared memory, so a graph over an NVFP4 cache must match the same graph over
an FP8 cache holding those bytes. That compares the ``.nvfp4`` registrations
end to end -- operand order, the scales' page table, the fused write -- without
a tolerance.

The scales are paged apart from the values, as the Jenga KV manager does, so
the NVFP4 graph's index rows are ``[value rows | scale rows]``: each selected
token's physical row, then its row in the scale pool, which the indexer
resolves through the scales lookup table. One case gives the scales a lookup
table that differs from the values'.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from _mla_sparse_test_utils import (
    paged_kv_from_flat_graph_inputs,
    random_weights,
)
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.attention.multi_latent_attention import LatentAttentionWithRope
from max.nn.kernels import (
    mla_decode_graph,
    mla_fp8_index_top_k_with_physical,
    mla_prefill_decode_graph,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
)
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs
from torch.utils.dlpack import from_dlpack

# GLM 5.3's per-device head count under TP8.
_NUM_HEADS = 8
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_PAGE_SIZE = 128
_CACHE_LEN = 300
_TOPK = 64

_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _kv_params(dtype: DType) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=dtype,
        head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=_NUM_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e4m3fn,
            quantization_granularity=16,
            fp8_tail_dim=_QK_ROPE_HEAD_DIM,
        )
        if dtype == DType.uint8
        else None,
    )


def _dequantize_nvfp4(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Returns the FP8 bytes the kernels expand ``packed`` and ``sf`` to.

    A row is the packed latent followed by the FP8 rope bytes, which the
    kernels pass through unchanged.
    """
    latent = packed[..., :-_QK_ROPE_HEAD_DIM]
    codes = torch.stack([latent & 0xF, latent >> 4], dim=-1).flatten(-2)
    magnitude = _E2M1_MAGNITUDES[(codes & 0x7).long()]
    signed = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    scale = sf.float().repeat_interleave(16, dim=-1)
    dequantized = (signed * scale).to(torch.float8_e4m3fn)
    return torch.cat(
        [dequantized.view(torch.uint8), packed[..., -_QK_ROPE_HEAD_DIM:]],
        dim=-1,
    ).view(torch.float8_e4m3fn)


def _rope(num_tokens: int) -> DeepseekYarnRotaryEmbedding:
    return DeepseekYarnRotaryEmbedding(
        dim=_QK_ROPE_HEAD_DIM,
        n_heads=_NUM_HEADS,
        theta=10000.0,
        max_seq_len=2 * (_CACHE_LEN + num_tokens),
        scaling_params=DeepseekYarnRopeScalingParams(
            scaling_factor=40.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        ),
    )


def _row_width(dtype: DType) -> int:
    """Slots per index row: an NVFP4 cache's rows also carry scale rows."""
    return 2 * _TOPK if dtype == DType.uint8 else _TOPK


def _build(dtype: DType, num_tokens: int, prefill_decode: bool) -> Graph:
    kv_params = _kv_params(dtype)
    rope = _rope(num_tokens)
    attn = _attention(kv_params, rope)
    gpu = DeviceRef.GPU()
    input_types = [
        TensorType(
            DType.bfloat16,
            ["total_tokens", _NUM_HEADS, _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM],
            gpu,
        ),
        TensorType(
            DType.bfloat16,
            ["total_tokens", _KV_LORA_RANK + _QK_ROPE_HEAD_DIM],
            gpu,
        ),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        TensorType(DType.int32, ["total_tokens", _row_width(dtype)], gpu),
        *tree.leaves(kv_params.get_symbolic_inputs()[0]),
    ]
    with Graph(f"mla_sparse_{dtype}", input_types=input_types) as g:
        q, kv, input_row_offsets, rows = (x.tensor for x in g.inputs[:4])
        kv_collection = paged_kv_from_flat_graph_inputs(
            kv_params, list(g.inputs[4:])
        )
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        common: dict[str, Any] = dict(
            kv_params=kv_params,
            kv_collection=kv_collection,
            layer_idx=ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=attn.scale,
            epsilon=1e-6,
            v_head_dim=_V_HEAD_DIM,
            scalar_args=kv_collection.attention_dispatch_metadata,
            num_partitions_scalar=kv_collection.mla_num_partitions,
            sparse_indices=rows,
            physical_indices=True,
            sparse_indices_stride=_row_width(dtype),
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


def _attention(
    kv_params: MLAKVCacheParams, rope: DeepseekYarnRotaryEmbedding
) -> LatentAttentionWithRope:
    attn = LatentAttentionWithRope(
        rope=rope,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=1,
        hidden_size=1024,
        kv_params=kv_params,
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        graph_mode="auto",
        q_lora_rank=256,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        buffer_size=4096,
    )
    _ = attn.state_dict()
    return attn


def _with_reversed_pages(
    inputs: KVCacheInputsPerDevice[Buffer, Buffer],
    device: Accelerator,
    *,
    permute_scales: bool = False,
) -> tuple[KVCacheInputsPerDevice[Buffer, Buffer], np.ndarray, np.ndarray]:
    """Maps the request's pages to block ids in reverse, so a token's block
    differs from its page index.

    The scales share that table, or with ``permute_scales`` get one of their
    own that sends each page to the next page's block.

    Returns:
        The inputs, the values' page table, and the scales'.
    """
    num_pages = int(inputs.kv_blocks.shape[0]) - 1
    table = np.full(
        tuple(int(d) for d in inputs.lookup_table.shape),
        num_pages,
        dtype=np.uint32,
    )
    table[0, :num_pages] = np.arange(num_pages)[::-1]
    lookup_table = Buffer.from_numpy(table).to(device)
    inputs.lookup_table = lookup_table
    scales_table = table.copy()
    if permute_scales:
        scales_table[0, :num_pages] = (np.arange(num_pages) + 1) % num_pages
    if inputs.kv_scales is not None:
        inputs.scales_lookup_table = Buffer.from_numpy(scales_table).to(device)
    return inputs, table, scales_table


@pytest.mark.skipif(
    not is_b100_b200(),
    reason="The NVFP4 sparse MLA kernels are SM100-class (B100/B200) only.",
)
@pytest.mark.parametrize("prefill_decode", [False, True])
@pytest.mark.parametrize("permute_scales", [False, True])
def test_mla_sparse_graph_nvfp4_matches_fp8_dequant(
    prefill_decode: bool, permute_scales: bool
) -> None:
    """The NVFP4 graph reads exactly what an FP8 cache of its bytes holds.

    The selected keys are all older than the step's own tokens, which each
    graph writes in its own format. ``prefill_decode`` runs 16 query tokens
    through the combined op's sparse prefill branch; otherwise one token
    decodes. ``permute_scales`` pages the scales apart from the values, so a
    kernel that took a token's scale row from its value row would read another
    token's scales.
    """
    num_tokens = 16 if prefill_decode else 1
    session = InferenceSession(devices=[Accelerator()])
    weights = random_weights(
        _attention(_kv_params(DType.bfloat16), _rope(num_tokens))
    )
    models: dict[DType, Model] = {
        dtype: session.load(
            _build(dtype, num_tokens, prefill_decode),
            weights_registry=weights,
        )
        for dtype in (DType.uint8, DType.float8_e4m3fn)
    }

    # Staged after every load: precompile_mefs records the loads under
    # virtual devices, which refuse device buffers.
    device = Accelerator(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    nvfp4, table, scales_table = _with_reversed_pages(
        paged_kv_cache_inputs(
            _kv_params(DType.uint8), [num_tokens], cache_lengths=[_CACHE_LEN]
        ),
        device,
        permute_scales=permute_scales,
    )
    fp8, _, _ = _with_reversed_pages(
        paged_kv_cache_inputs(
            _kv_params(DType.float8_e4m3fn),
            [num_tokens],
            cache_lengths=[_CACHE_LEN],
        ),
        device,
    )
    assert nvfp4.kv_scales is not None
    packed = torch.randint(
        0, 256, tuple(int(d) for d in nvfp4.kv_blocks.shape), dtype=torch.uint8
    )
    # The rope bytes are FP8 values (a random byte could be a NaN).
    packed[..., -_QK_ROPE_HEAD_DIM:] = (
        (torch.randn(packed[..., -_QK_ROPE_HEAD_DIM:].shape) * 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    # Power-of-two scales keep every `e2m1 * sf` exact in e4m3.
    sf = torch.pow(
        2.0,
        torch.randint(-3, 1, tuple(int(d) for d in nvfp4.kv_scales.shape)),
    ).to(torch.float8_e4m3fn)
    # The scales of the tokens in value block `table[0, p]` sit in scale block
    # `scales_table[0, p]`; the FP8 cache holds the dequantization of each
    # token's own.
    sf_of_value_block = sf.clone()
    for page in range(int(nvfp4.kv_blocks.shape[0]) - 1):
        sf_of_value_block[table[0, page]] = sf[scales_table[0, page]]
    nvfp4.kv_blocks = Buffer.from_dlpack(packed).to(device)
    nvfp4.kv_scales = (
        Buffer.from_dlpack(sf.view(torch.uint8))
        .view(DType.float8_e4m3fn)
        .to(device)
    )
    fp8.kv_blocks = (
        Buffer.from_dlpack(
            _dequantize_nvfp4(packed, sf_of_value_block).view(torch.uint8)
        )
        .view(DType.float8_e4m3fn)
        .to(device)
    )

    positions = rng.choice(_CACHE_LEN, size=_TOPK, replace=False)
    physical = (
        table[0, positions // _PAGE_SIZE].astype(np.int64) * _PAGE_SIZE
        + positions % _PAGE_SIZE
    ).astype(np.int32)
    # A scale page holds `num_layers * page_size` rows, and the cache has one
    # layer.
    scale_rows = (
        scales_table[0, positions // _PAGE_SIZE].astype(np.int64) * _PAGE_SIZE
        + positions % _PAGE_SIZE
    ).astype(np.int32)
    rows = {
        DType.float8_e4m3fn: Buffer.from_numpy(
            np.tile(physical, (num_tokens, 1))
        ).to(device),
        DType.uint8: Buffer.from_numpy(
            np.tile(np.concatenate([physical, scale_rows]), (num_tokens, 1))
        ).to(device),
    }
    q = Buffer.from_dlpack(
        (
            torch.randn(
                (
                    num_tokens,
                    _NUM_HEADS,
                    _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM,
                )
            )
            * 0.1
        ).to(torch.bfloat16)
    ).to(device)
    kv = Buffer.from_dlpack(
        (torch.randn((num_tokens, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM)) * 0.1).to(
            torch.bfloat16
        )
    ).to(device)
    row_offsets = Buffer.from_numpy(
        np.array([0, num_tokens], dtype=np.uint32)
    ).to(device)

    def run(
        dtype: DType, inputs: KVCacheInputsPerDevice[Buffer, Buffer]
    ) -> np.ndarray:
        out = models[dtype].execute(
            q, kv, row_offsets, rows[dtype], *tree.leaves(inputs)
        )[0]
        return from_dlpack(out).cpu().view(torch.int16).numpy()

    got = run(DType.uint8, nvfp4)
    want = run(DType.float8_e4m3fn, fp8)
    assert np.any(want != 0)
    np.testing.assert_array_equal(got, want)


@pytest.mark.skipif(
    not is_b100_b200(),
    reason="The NVFP4 sparse MLA kernels are SM100-class (B100/B200) only.",
)
def test_mla_indexer_nvfp4_rows_follow_the_scales_table() -> None:
    """The indexer over an NVFP4 MLA cache resolves scale rows through the
    cache's scales page table, apart from the values' physical rows."""
    num_tokens = 16
    index_head_dim = 128
    gpu = DeviceRef.GPU()
    indexer_params = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=index_head_dim,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[gpu],
        num_q_heads=_NUM_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=index_head_dim
        ),
    )
    mla_params = _kv_params(DType.uint8)
    num_indexer_inputs = len(
        tree.leaves(indexer_params.get_symbolic_inputs()[0])
    )
    input_types = [
        TensorType(
            DType.float8_e4m3fn,
            ["total_tokens", _NUM_HEADS, index_head_dim],
            gpu,
        ),
        TensorType(DType.float32, ["total_tokens", _NUM_HEADS], gpu),
        TensorType(DType.uint32, ["row_offsets_len"], gpu),
        *tree.leaves(indexer_params.get_symbolic_inputs()[0]),
        *tree.leaves(mla_params.get_symbolic_inputs()[0]),
    ]
    with Graph("mla_indexer_nvfp4", input_types=input_types) as g:
        q, q_s, row_offsets = (x.tensor for x in g.inputs[:3])
        flat = list(g.inputs[3:])
        logical, physical = mla_fp8_index_top_k_with_physical(
            q,
            q_s,
            row_offsets,
            paged_kv_from_flat_graph_inputs(
                indexer_params, flat[:num_indexer_inputs]
            ),
            paged_kv_from_flat_graph_inputs(
                mla_params, flat[num_indexer_inputs:]
            ),
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            _TOPK,
            index_head_dim,
        )
        g.output(logical, physical)
    model = InferenceSession(devices=[Accelerator()]).load(g)

    device = Accelerator(0)
    torch.manual_seed(0)
    indexer_inputs = paged_kv_cache_inputs(
        indexer_params, [num_tokens], cache_lengths=[_CACHE_LEN]
    )
    assert indexer_inputs.kv_scales is not None
    indexer_inputs.kv_blocks = (
        Buffer.from_dlpack(
            torch.randn(tuple(int(d) for d in indexer_inputs.kv_blocks.shape))
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        .view(DType.float8_e4m3fn)
        .to(device)
    )
    indexer_inputs.kv_scales = Buffer.from_dlpack(
        torch.rand(tuple(int(d) for d in indexer_inputs.kv_scales.shape)) + 0.5
    ).to(device)
    mla_inputs, table, scales_table = _with_reversed_pages(
        paged_kv_cache_inputs(
            mla_params, [num_tokens], cache_lengths=[_CACHE_LEN]
        ),
        device,
        permute_scales=True,
    )
    q_buf = (
        Buffer.from_dlpack(
            torch.randn((num_tokens, _NUM_HEADS, index_head_dim))
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        .view(DType.float8_e4m3fn)
        .to(device)
    )
    q_s_buf = Buffer.from_dlpack(torch.rand((num_tokens, _NUM_HEADS))).to(
        device
    )
    row_offsets_buf = Buffer.from_numpy(
        np.array([0, num_tokens], dtype=np.uint32)
    ).to(device)

    out_logical, out_physical = model.execute(
        q_buf,
        q_s_buf,
        row_offsets_buf,
        *tree.leaves(indexer_inputs),
        *tree.leaves(mla_inputs),
    )
    logical = from_dlpack(out_logical).cpu().numpy()
    physical = from_dlpack(out_physical).cpu().numpy()
    assert physical.shape == (num_tokens, 2 * _TOPK)

    # The scale pool's blocks are a layer's worth of pages apart.
    rows_per_block = _PAGE_SIZE
    page = np.clip(logical // _PAGE_SIZE, 0, table.shape[1] - 1)
    in_page = logical % _PAGE_SIZE
    live = logical >= 0
    want_values = np.where(
        live, table[0, page].astype(np.int64) * _PAGE_SIZE + in_page, -1
    )
    want_scales = np.where(
        live,
        scales_table[0, page].astype(np.int64) * rows_per_block + in_page,
        -1,
    )
    assert live.any()
    np.testing.assert_array_equal(physical[:, :_TOPK], want_values)
    np.testing.assert_array_equal(physical[:, _TOPK:], want_scales)
    # The scales' table differs from the values', so the halves differ.
    assert np.any(want_values[live] != want_scales[live])


@pytest.mark.skipif(
    not is_b100_b200(),
    reason="The NVFP4 sparse MLA kernels are SM100-class (B100/B200) only.",
)
def test_mla_sparse_prefill_nvfp4_convert_once_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse prefill converts an NVFP4 context to FP8 once when the gate
    allows, and then matches the NVFP4 kernel bit for bit.

    The conversion takes each row's scales through the cache's scales page
    table, while the NVFP4 kernel reads the index row's scale half. Index rows
    whose scale half names other tokens' rows tell the two apart: they change
    the NVFP4 kernel's output and not the conversion's. With 16 query tokens
    over 316 context rows (one 384-row page set), a ratio of 20 or a 256-row
    cap sends the step back to the NVFP4 kernel.
    """
    num_tokens = 16
    session = InferenceSession(devices=[Accelerator()])
    weights = random_weights(
        _attention(_kv_params(DType.bfloat16), _rope(num_tokens))
    )

    def load(ratio: int, max_rows: int) -> Model:
        monkeypatch.setenv(
            "MODULAR_MLA_NVFP4_PREFILL_CONVERT_RATIO", str(ratio)
        )
        monkeypatch.setenv(
            "MODULAR_MLA_NVFP4_PREFILL_CONVERT_MAX_ROWS", str(max_rows)
        )
        return session.load(
            _build(DType.uint8, num_tokens, prefill_decode=True),
            weights_registry=weights,
        )

    models = {
        "convert": load(150, 262144),
        "off": load(0, 262144),
        "low_ratio": load(20, 262144),
        "capped": load(150, 256),
    }

    device = Accelerator(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    inputs, table, scales_table = _with_reversed_pages(
        paged_kv_cache_inputs(
            _kv_params(DType.uint8), [num_tokens], cache_lengths=[_CACHE_LEN]
        ),
        device,
        permute_scales=True,
    )
    assert inputs.kv_scales is not None
    packed = torch.randint(
        0, 256, tuple(int(d) for d in inputs.kv_blocks.shape), dtype=torch.uint8
    )
    packed[..., -_QK_ROPE_HEAD_DIM:] = (
        (torch.randn(packed[..., -_QK_ROPE_HEAD_DIM:].shape) * 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    sf = torch.pow(
        2.0,
        torch.randint(-3, 1, tuple(int(d) for d in inputs.kv_scales.shape)),
    ).to(torch.float8_e4m3fn)
    inputs.kv_blocks = Buffer.from_dlpack(packed).to(device)
    inputs.kv_scales = (
        Buffer.from_dlpack(sf.view(torch.uint8))
        .view(DType.float8_e4m3fn)
        .to(device)
    )

    # Every graph writes the step's own rows identically, so the selection
    # can include them: this is a chunk on top of a 300-token prefix.
    positions = rng.choice(_CACHE_LEN + num_tokens, size=_TOPK, replace=False)
    value_rows = (
        table[0, positions // _PAGE_SIZE].astype(np.int64) * _PAGE_SIZE
        + positions % _PAGE_SIZE
    ).astype(np.int32)
    scale_rows = (
        scales_table[0, positions // _PAGE_SIZE].astype(np.int64) * _PAGE_SIZE
        + positions % _PAGE_SIZE
    ).astype(np.int32)

    def index_rows(scales: np.ndarray) -> Buffer:
        return Buffer.from_numpy(
            np.tile(np.concatenate([value_rows, scales]), (num_tokens, 1))
        ).to(device)

    rows = index_rows(scale_rows)
    probe = index_rows(np.roll(scale_rows, 1))
    q = Buffer.from_dlpack(
        (
            torch.randn(
                (num_tokens, _NUM_HEADS, _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)
            )
            * 0.1
        ).to(torch.bfloat16)
    ).to(device)
    kv = Buffer.from_dlpack(
        (torch.randn((num_tokens, _KV_LORA_RANK + _QK_ROPE_HEAD_DIM)) * 0.1).to(
            torch.bfloat16
        )
    ).to(device)
    row_offsets = Buffer.from_numpy(
        np.array([0, num_tokens], dtype=np.uint32)
    ).to(device)

    def run(name: str, index: Buffer) -> np.ndarray:
        out = models[name].execute(
            q, kv, row_offsets, index, *tree.leaves(inputs)
        )[0]
        return from_dlpack(out).cpu().view(torch.int16).numpy()

    want = run("off", rows)
    assert np.any(want != 0)
    np.testing.assert_array_equal(run("convert", rows), want)
    # The conversion ran: it ignores the index rows' scale half.
    np.testing.assert_array_equal(run("convert", probe), want)
    nvfp4_probe = run("off", probe)
    assert np.any(nvfp4_probe != want)
    # Past the ratio or the cap, the NVFP4 kernel runs.
    np.testing.assert_array_equal(run("low_ratio", probe), nvfp4_probe)
    np.testing.assert_array_equal(run("capped", probe), nvfp4_probe)
