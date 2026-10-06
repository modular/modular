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
"""Graph-op plumbing for MiniMax-M3's NVFP4 KV cache (B200 / SM100).

Two ops carry the format: ``mo.fused_qk_rms_norm_rope.ragged.paged.dual.nvfp4``
writes K/V into the packed cache, and ``mo.msa.attention.ragged.paged.nvfp4``
reads it. Kernel-level edge cases (ties, scale clamps, poisoned tails, bitwise
K) live in ``test_fused_dual_qk_rms_norm_rope_nvfp4.mojo`` and
``Kernels/test/msa/test_msa_sm100_d128_decode_q8kv4.mojo``; this test owns the
operand order, the scale operands and the routes, through the Python wrappers.

* The writer: V must land as exactly the bytes and scales MiniMax's quantizer
  gives on the host, and K must track a torch RMSNorm+RoPE reference.
* The reader: attention over the NVFP4 cache must equal, bit for bit, the FP8
  op over the same K/V dequantized on the host, on the decode, speculative
  decode and prefill routes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    fused_dual_qk_rms_norm_rope_nvfp4_ragged,
    msa_sparse_attention_ragged,
)
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    PagedCacheValues,
    packed_page_stride,
)
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs

_HEAD_DIM = 128
_PACKED = _HEAD_DIM // 2
_GROUP = 16
_SF_COLS = _HEAD_DIM // _GROUP
_PAGE_SIZE = 128
_TOPK = 16
_NUM_Q_HEADS = 16
_N_KV_HEADS = 1
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_NVFP4 = KVCacheQuantizationConfig(
    scale_dtype=DType.float8_e4m3fn, quantization_granularity=_GROUP
)

pytestmark = pytest.mark.skipif(
    not is_b100_b200(), reason="the NVFP4 KV cache runs on SM100 only"
)


def _e4m3(x: npt.NDArray[np.float32]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float8_e4m3fn)


def _nvfp4_quantize(
    x: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """MiniMax's quantizer on ``[..., 128]`` rows: ``(packed, scale bytes)``.

    Scale ``E4M3(max(amax, 1e-12) / 6)`` clamped to ``[2^-9, 448]``, true
    division, round half to even, zero magnitude as +0, element 2i in the low
    nibble.
    """
    lead = x.shape[:-1]
    g = x.reshape(*lead, _SF_COLS, _GROUP).astype(np.float32)
    amax = np.maximum(np.abs(g).max(-1), np.float32(1e-12))
    scale = (
        _e4m3(np.minimum(amax / np.float32(6.0), np.float32(448.0)))
        .float()
        .numpy()
    )
    scale = np.maximum(scale, np.float32(1.0 / 512.0))
    xs = g / scale[..., None]
    a = np.minimum(np.abs(xs), np.float32(6.0))
    mag = np.select(
        [a > 5.0, a >= 3.5, a > 2.5, a >= 1.75, a > 1.25, a >= 0.75, a > 0.25],
        [7, 6, 5, 4, 3, 2, 1],
        0,
    ).astype(np.uint8)
    code = mag | np.where((xs < 0) & (mag != 0), 8, 0).astype(np.uint8)
    code = code.reshape(*lead, _HEAD_DIM)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).astype(np.uint8)
    scale_bytes = _e4m3(scale).view(torch.uint8).numpy()
    return packed, scale_bytes


def _nvfp4_dequantize_fp8(
    packed: npt.NDArray[np.uint8], scale_bytes: npt.NDArray[np.uint8]
) -> torch.Tensor:
    """The kernels' dequant: ``e4m3(e2m1 * scale)``, one rounding."""
    lead = packed.shape[:-1]
    code = np.empty((*lead, _HEAD_DIM), dtype=np.uint8)
    code[..., 0::2] = packed & 0xF
    code[..., 1::2] = packed >> 4
    value = _E2M1[code & 7] * np.where(code & 8, -1.0, 1.0).astype(np.float32)
    scale = (
        torch.from_numpy(np.ascontiguousarray(scale_bytes))
        .view(torch.float8_e4m3fn)
        .float()
        .numpy()
    )
    value = value.reshape(*lead, _SF_COLS, _GROUP) * scale[..., None]
    return _e4m3(value.reshape(*lead, _HEAD_DIM))


def _nvfp4_params(n_kv_heads: int) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.uint8,
        n_kv_heads=n_kv_heads,
        head_dim=_HEAD_DIM,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        kvcache_quant_config=_NVFP4,
    )


def _bf16_params(n_kv_heads: int) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=n_kv_heads,
        head_dim=_HEAD_DIM,
        num_layers=1,
        page_size=_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _freqs_cis(max_seq: int) -> torch.Tensor:
    """Interleaved ``(cos, sin)`` RoPE table ``[max_seq, head_dim]``, fp32."""
    pos = torch.arange(max_seq, dtype=torch.float32)
    exponent = torch.arange(0, _HEAD_DIM, 2, dtype=torch.float32) / _HEAD_DIM
    angles = torch.outer(pos, 1.0 / (10000.0**exponent))
    freqs = torch.empty(max_seq, _HEAD_DIM, dtype=torch.float32)
    freqs[:, 0::2] = torch.cos(angles)
    freqs[:, 1::2] = torch.sin(angles)
    return freqs


def _norm_rope(
    x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm (unit gamma, no offset) then interleaved RoPE, in fp32."""
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    f = freqs[positions][:, None, :]
    cos, sin = f[..., 0::2], f[..., 1::2]
    re, im = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = re * cos - im * sin
    out[..., 1::2] = re * sin + im * cos
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a1, b1 = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.dot(a1, b1) / (a1.norm() * b1.norm()).clamp_min(1e-12))


def test_dual_nvfp4_writes_quantized_kv() -> None:
    """The writer stores MiniMax-quantized V exactly and normed+roped K."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    prompt_lens = [37, 1, 131]
    cache_lens = [5, 200, 0]
    tokens = sum(prompt_lens)
    main_heads, kv_heads, index_heads = 16, 1, 4
    eps = 1e-6

    main_params = _nvfp4_params(kv_heads)
    index_params = _bf16_params(1)
    gpu = DeviceRef.GPU()

    def staged(heads: int) -> TensorType:
        return TensorType(DType.bfloat16, [tokens, heads, _HEAD_DIM], gpu)

    gamma_type = TensorType(DType.bfloat16, [_HEAD_DIM], gpu)
    freqs = _freqs_cis(512)
    main_inputs = main_params.flattened_kv_inputs()
    with Graph(
        "dual_nvfp4_store",
        input_types=[
            staged(main_heads),
            staged(kv_heads),
            staged(kv_heads),
            staged(index_heads),
            staged(1),
            TensorType(DType.uint32, [len(prompt_lens) + 1], gpu),
            TensorType(DType.float32, list(freqs.shape), gpu),
            *([gamma_type] * 4),
            *main_inputs,
            *index_params.flattened_kv_inputs(),
        ],
    ) as graph:
        q, k, v, iq, ik, iro, fr, g0, g1, g2, g3, *kv = graph.inputs
        main_kv = main_params.unflatten_kv_inputs(iter(kv[: len(main_inputs)]))
        index_kv = index_params.unflatten_kv_inputs(
            iter(kv[len(main_inputs) :])
        )
        q_out, iq_out = fused_dual_qk_rms_norm_rope_nvfp4_ragged(
            main_params,
            index_params,
            q.tensor,
            k.tensor,
            v.tensor,
            iq.tensor,
            ik.tensor,
            iro.tensor,
            main_kv[0],
            index_kv[0],
            q_main_gamma=g0.tensor,
            k_main_gamma=g1.tensor,
            q_index_gamma=g2.tensor,
            k_index_gamma=g3.tensor,
            freqs_cis=fr.tensor,
            main_epsilon=eps,
            index_epsilon=eps,
            layer_idx=ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            weight_offset=0.0,
            interleaved=True,
        )
        graph.output(q_out, iq_out)
    model = session.load(graph)

    torch.manual_seed(0)
    q_h = torch.randn(tokens, main_heads, _HEAD_DIM).to(torch.bfloat16)
    k_h = torch.randn(tokens, kv_heads, _HEAD_DIM).to(torch.bfloat16)
    v_h = torch.randn(tokens, kv_heads, _HEAD_DIM).to(torch.bfloat16)
    # Plant the quantizer's edges in V: a zero row, a tiny row, an outlier.
    v_h[0] = 0
    v_h[1] *= 1e-3
    v_h[2, :, ::16] = 3000.0
    iq_h = torch.randn(tokens, index_heads, _HEAD_DIM).to(torch.bfloat16)
    ik_h = torch.randn(tokens, 1, _HEAD_DIM).to(torch.bfloat16)
    ones = torch.ones(_HEAD_DIM, dtype=torch.bfloat16)

    main_rt = paged_kv_cache_inputs(
        main_params, prompt_lens, cache_lengths=cache_lens, total_num_pages=8
    )
    index_rt = paged_kv_cache_inputs(
        index_params, prompt_lens, cache_lengths=cache_lens, total_num_pages=8
    )
    offsets = np.zeros(len(prompt_lens) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(prompt_lens)
    q_res, _ = model.execute(
        *(
            Buffer.from_dlpack(t).to(device)
            for t in (q_h, k_h, v_h, iq_h, ik_h)
        ),
        Buffer.from_numpy(offsets).to(device),
        Buffer.from_dlpack(freqs).to(device),
        *(Buffer.from_dlpack(ones).to(device) for _ in range(4)),
        *tree.leaves(main_rt),
        *tree.leaves(index_rt),
    )
    assert isinstance(q_res, Buffer)

    lut = main_rt.lookup_table.to_numpy()
    assert main_rt.kv_scales is not None
    blocks = main_rt.kv_blocks.to_numpy()
    scales = main_rt.kv_scales.view(DType.uint8).to_numpy()
    positions = []
    for bs, length in enumerate(prompt_lens):
        for t in range(length):
            pos = cache_lens[bs] + t
            positions.append(
                (int(lut[bs, pos // _PAGE_SIZE]), pos % _PAGE_SIZE)
            )
    k_packed = np.stack([blocks[b, 0, 0, p] for b, p in positions])
    k_sf = np.stack([scales[b, 0, 0, p] for b, p in positions])
    v_packed = np.stack([blocks[b, 1, 0, p] for b, p in positions])
    v_sf = np.stack([scales[b, 1, 0, p] for b, p in positions])

    want_v, want_v_sf = _nvfp4_quantize(v_h.float().numpy())
    np.testing.assert_array_equal(v_packed, want_v)
    np.testing.assert_array_equal(v_sf, want_v_sf)

    token_pos = torch.tensor(
        [cache_lens[b] + t for b, n in enumerate(prompt_lens) for t in range(n)]
    )
    k_ref = _norm_rope(k_h, token_pos, freqs, eps)
    k_got = _nvfp4_dequantize_fp8(k_packed, k_sf).float()
    assert torch.isfinite(k_got).all()
    assert _cosine(k_got, k_ref) > 0.99

    q_ref = _norm_rope(q_h, token_pos, freqs, eps)
    q_got = torch.from_dlpack(q_res).float().cpu()
    assert _cosine(q_got, q_ref) > 0.999


@pytest.fixture(scope="module")
def attention_model() -> tuple[Accelerator, Model]:
    """FP8 and NVFP4 attention in one graph with symbolic batch shapes, so
    every case runs on a single compile."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    gpu = DeviceRef.GPU()
    cpu = DeviceRef.CPU()

    def page_type(dtype: DType, width: int) -> BufferType:
        return BufferType(
            dtype, ["pages", 2, 1, _PAGE_SIZE, _N_KV_HEADS, width], gpu
        )

    lut_type = TensorType(DType.uint32, ["batch", "max_pages"], gpu)
    with Graph(
        "msa_nvfp4_vs_fp8",
        input_types=[
            TensorType(
                DType.float8_e4m3fn, ["rows", _NUM_Q_HEADS, _HEAD_DIM], gpu
            ),
            TensorType(DType.uint32, ["offsets"], gpu),
            TensorType(DType.uint32, ["offsets"], gpu),
            TensorType(DType.uint32, [1], cpu),
            page_type(DType.float8_e4m3fn, _HEAD_DIM),
            page_type(DType.uint8, _PACKED),
            page_type(DType.float8_e4m3fn, _SF_COLS),
            TensorType(DType.uint32, ["batch"], gpu),
            lut_type,
            lut_type,
            TensorType(DType.uint32, [1], cpu),
            TensorType(DType.uint32, [1], cpu),
            TensorType(DType.int64, [2], gpu),
            TensorType(DType.int32, [_N_KV_HEADS, "rows", _TOPK], gpu),
        ],
    ) as graph:
        (
            q,
            iro,
            cro,
            tcl,
            fp8_blocks,
            nvfp4_blocks,
            sf,
            cl,
            lut,
            sf_lut,
            mp,
            mc,
            sa,
            d,
        ) = graph.inputs
        fp8_kv = PagedCacheValues(
            kv_blocks=fp8_blocks.buffer,
            cache_lengths=cl.tensor,
            lookup_table=lut.tensor,
            max_prompt_length=mp.tensor,
            max_cache_length=mc.tensor,
            page_stride=packed_page_stride(fp8_blocks.buffer),
            attention_dispatch_metadata=sa.tensor,
        )
        nvfp4_kv = PagedCacheValues(
            kv_blocks=nvfp4_blocks.buffer,
            cache_lengths=cl.tensor,
            lookup_table=lut.tensor,
            max_prompt_length=mp.tensor,
            max_cache_length=mc.tensor,
            page_stride=packed_page_stride(nvfp4_blocks.buffer),
            kv_scales=sf.buffer,
            scales_page_stride=packed_page_stride(sf.buffer),
            scales_lookup_table=sf_lut.tensor,
            attention_dispatch_metadata=sa.tensor,
        )
        fp8_params = MHAKVCacheParams(
            dtype=DType.float8_e4m3fn,
            n_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_layers=1,
            page_size=_PAGE_SIZE,
            devices=[gpu],
        )
        outs = []
        for params, kv in ((fp8_params, fp8_kv), (_nvfp4_params(1), nvfp4_kv)):
            outs.append(
                msa_sparse_attention_ragged(
                    kv_params=params,
                    input=q.tensor,
                    input_row_offsets=iro.tensor,
                    cache_row_offsets=cro.tensor,
                    total_context_length=tcl.tensor,
                    kv_collection=kv,
                    layer_idx=ops.constant(0, DType.uint32, device=cpu),
                    block_indices=d.tensor,
                    group=_NUM_Q_HEADS // _N_KV_HEADS,
                    topk=_TOPK,
                    sparse_block_size=_PAGE_SIZE,
                    scale=float(_HEAD_DIM**-0.5),
                )
            )
        graph.output(*outs)
    return device, session.load(graph)


@pytest.mark.parametrize(
    "q_lens,cache_lens,extra_cache,separate_scales_lut",
    [
        # Decode with split-K, the trailing block whole.
        ([1] * 5, [2047] * 5, 0, False),
        # Decode with short contexts: most of the 16 selections are -1.
        ([1, 1, 1], [5, 130, 400], 0, False),
        # Graph-capture replay passes an aligned-up max_cache_length; the
        # rows past each request's keys must stay masked.
        ([1, 1, 1], [700, 2047, 33], 300, True),
        # Speculative decode at every draft width the op compiles.
        *[([n] * 3, [2043, 900, 61], 0, n % 2 == 0) for n in range(2, 9)],
        # Prefill on top of cached pages, page-aligned and mid-page.
        ([33, 200], [256, 61], 0, True),
        # Decode and prefill requests in one batch (the prefill route).
        ([1, 40, 1, 9], [3000, 0, 127, 500], 0, True),
    ],
)
def test_msa_attention_nvfp4_matches_fp8(
    q_lens: list[int],
    cache_lens: list[int],
    extra_cache: int,
    separate_scales_lut: bool,
    attention_model: tuple[Accelerator, Model],
) -> None:
    """Attention over NVFP4 K/V equals FP8 attention over the dequantized
    K/V, bit for bit.

    With ``separate_scales_lut`` the NVFP4 scales sit on pages in reverse
    order and are found through their own page table, as when the pool pages
    the scales leaf independently.
    """
    device, model = attention_model
    batch = len(q_lens)
    rows = sum(q_lens)
    keys = [c + q for c, q in zip(cache_lens, q_lens, strict=True)]
    pages = [-(-k // _PAGE_SIZE) for k in keys]
    max_pages = max(pages)
    total_pages = sum(pages)
    # One page past the pool backs the page table's sentinel entries.
    pool_pages = total_pages + 1

    torch.manual_seed(rows)
    kv_values = (
        torch.randn(pool_pages, 2, 1, _PAGE_SIZE, _N_KV_HEADS, _HEAD_DIM) * 0.5
    ).to(torch.bfloat16)
    packed, sf_bytes = _nvfp4_quantize(kv_values.float().numpy())
    fp8_values = _nvfp4_dequantize_fp8(packed, sf_bytes)
    q_h = (torch.randn(rows, _NUM_Q_HEADS, _HEAD_DIM) * 0.5).to(
        torch.float8_e4m3fn
    )

    lut_np = np.full((batch, max_pages), total_pages, dtype=np.uint32)
    first = 0
    for b, n in enumerate(pages):
        lut_np[b, :n] = np.arange(first, first + n, dtype=np.uint32)
        first += n
    sf_lut_np = lut_np.copy()
    sf_pages = sf_bytes
    if separate_scales_lut:
        # Page p's scales live on page total_pages - 1 - p.
        sf_lut_np = np.where(
            lut_np < total_pages, total_pages - 1 - lut_np, lut_np
        ).astype(np.uint32)
        sf_pages = sf_bytes.copy()
        sf_pages[:total_pages] = sf_bytes[:total_pages][::-1]

    # Each query row selects its request's blocks in order; -1 past them.
    d_idx = np.full((_N_KV_HEADS, rows, _TOPK), -1, dtype=np.int32)
    row = 0
    for b, n_q in enumerate(q_lens):
        for _ in range(n_q):
            n = min(pages[b], _TOPK)
            d_idx[:, row, :n] = np.arange(n, dtype=np.int32)
            row += 1

    iro_np = np.zeros(batch + 1, dtype=np.uint32)
    iro_np[1:] = np.cumsum(q_lens)
    cro_np = np.zeros(batch + 1, dtype=np.uint32)
    cro_np[1:] = np.cumsum(keys)
    ref, got = model.execute(
        Buffer.from_dlpack(q_h.view(torch.uint8))
        .view(DType.float8_e4m3fn)
        .to(device),
        Buffer.from_numpy(iro_np).to(device),
        Buffer.from_numpy(cro_np).to(device),
        Buffer.from_numpy(np.array([sum(keys)], dtype=np.uint32)),
        Buffer.from_dlpack(fp8_values.view(torch.uint8))
        .view(DType.float8_e4m3fn)
        .to(device),
        Buffer.from_numpy(packed).to(device),
        Buffer.from_numpy(np.ascontiguousarray(sf_pages))
        .view(DType.float8_e4m3fn)
        .to(device),
        Buffer.from_numpy(np.array(cache_lens, dtype=np.uint32)).to(device),
        Buffer.from_numpy(lut_np).to(device),
        Buffer.from_numpy(sf_lut_np).to(device),
        Buffer.from_numpy(np.array([max(q_lens)], dtype=np.uint32)),
        # The cache manager's post-write key count, plus any replay slack. The
        # NVFP4 decode reads scale rows up to this bound.
        Buffer.from_numpy(np.array([max(keys) + extra_cache], dtype=np.uint32)),
        Buffer.from_numpy(np.array([batch, max(keys)], dtype=np.int64)).to(
            device
        ),
        Buffer.from_numpy(d_idx).to(device),
    )
    assert isinstance(ref, Buffer)
    assert isinstance(got, Buffer)
    ref_bits = ref.view(DType.uint16).to_numpy()
    got_bits = got.view(DType.uint16).to_numpy()
    assert int(np.count_nonzero(ref_bits)) > ref_bits.size // 2
    np.testing.assert_array_equal(got_bits, ref_bits)


def test_nvfp4_ops_accept_an_empty_replica() -> None:
    """A data-parallel replica with no requests runs the writer and the
    attention op on zero rows and leaves the cache untouched."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    gpu = DeviceRef.GPU()
    cpu = DeviceRef.CPU()
    params = _nvfp4_params(1)
    index_params = _bf16_params(1)

    def staged(heads: int) -> TensorType:
        return TensorType(DType.bfloat16, [0, heads, _HEAD_DIM], gpu)

    gamma_type = TensorType(DType.bfloat16, [_HEAD_DIM], gpu)
    main_inputs = params.flattened_kv_inputs()
    with Graph(
        "nvfp4_empty_replica",
        input_types=[
            staged(_NUM_Q_HEADS),
            staged(1),
            staged(1),
            staged(4),
            staged(1),
            TensorType(DType.uint32, [1], gpu),
            TensorType(DType.uint32, [1], gpu),
            TensorType(DType.uint32, [1], cpu),
            TensorType(DType.float32, [512, _HEAD_DIM], gpu),
            gamma_type,
            TensorType(DType.int32, [_N_KV_HEADS, 0, _TOPK], gpu),
            *main_inputs,
            *index_params.flattened_kv_inputs(),
        ],
    ) as graph:
        q, k, v, iq, ik, iro, cro, tcl, fr, g, d, *kv = graph.inputs
        main_kv = params.unflatten_kv_inputs(iter(kv[: len(main_inputs)]))[0]
        index_kv = index_params.unflatten_kv_inputs(
            iter(kv[len(main_inputs) :])
        )[0]
        layer = ops.constant(0, DType.uint32, device=cpu)
        q8, _ = fused_dual_qk_rms_norm_rope_nvfp4_ragged(
            params,
            index_params,
            q.tensor,
            k.tensor,
            v.tensor,
            iq.tensor,
            ik.tensor,
            iro.tensor,
            main_kv,
            index_kv,
            q_main_gamma=g.tensor,
            k_main_gamma=g.tensor,
            q_index_gamma=g.tensor,
            k_index_gamma=g.tensor,
            freqs_cis=fr.tensor,
            main_epsilon=1e-6,
            index_epsilon=1e-6,
            layer_idx=layer,
            weight_offset=0.0,
            interleaved=True,
        )
        out = msa_sparse_attention_ragged(
            kv_params=params,
            input=q8,
            input_row_offsets=iro.tensor,
            cache_row_offsets=cro.tensor,
            total_context_length=tcl.tensor,
            kv_collection=main_kv,
            layer_idx=layer,
            block_indices=d.tensor,
            group=_NUM_Q_HEADS // _N_KV_HEADS,
            topk=_TOPK,
            sparse_block_size=_PAGE_SIZE,
            scale=float(_HEAD_DIM**-0.5),
        )
        graph.output(out)
    model = session.load(graph)

    # One cached page, but zero requests this step: an empty batch.
    main_rt = paged_kv_cache_inputs(params, [1], total_num_pages=2)
    index_rt = paged_kv_cache_inputs(index_params, [1], total_num_pages=2)
    empty_lut = Buffer.from_numpy(np.zeros((0, 1), dtype=np.uint32)).to(device)
    for rt in (main_rt, index_rt):
        rt.cache_lengths = Buffer.from_numpy(np.zeros(0, dtype=np.uint32)).to(
            device
        )
        rt.lookup_table = empty_lut
        rt.max_prompt_length = Buffer.from_numpy(np.zeros(1, dtype=np.uint32))
        rt.max_cache_length = Buffer.from_numpy(np.zeros(1, dtype=np.uint32))
        if rt.scales_lookup_table is not None:
            rt.scales_lookup_table = empty_lut
    blocks_before = main_rt.kv_blocks.to_numpy().copy()
    zero_offsets = np.zeros(1, dtype=np.uint32)

    def empty_rows(heads: int) -> Buffer:
        return Buffer.from_dlpack(
            torch.zeros(0, heads, _HEAD_DIM, dtype=torch.bfloat16)
        ).to(device)

    (out_buf,) = model.execute(
        *(empty_rows(h) for h in (_NUM_Q_HEADS, 1, 1, 4, 1)),
        Buffer.from_numpy(zero_offsets).to(device),
        Buffer.from_numpy(zero_offsets).to(device),
        Buffer.from_numpy(zero_offsets),
        Buffer.from_dlpack(_freqs_cis(512)).to(device),
        Buffer.from_dlpack(torch.ones(_HEAD_DIM, dtype=torch.bfloat16)).to(
            device
        ),
        Buffer.from_numpy(np.zeros((_N_KV_HEADS, 0, _TOPK), dtype=np.int32)).to(
            device
        ),
        *tree.leaves(main_rt),
        *tree.leaves(index_rt),
    )
    assert isinstance(out_buf, Buffer)
    assert tuple(out_buf.shape) == (0, _NUM_Q_HEADS, _HEAD_DIM)
    np.testing.assert_array_equal(main_rt.kv_blocks.to_numpy(), blocks_before)
