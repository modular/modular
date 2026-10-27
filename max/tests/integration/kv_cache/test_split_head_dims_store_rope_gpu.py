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
"""The store and RoPE graph ops on a cache whose rows pack a 192-wide key and
a 128-wide value.

The ops must name the value width to the kernels: from one plane alone they
would build an MLA cache, writing the value over the key and rotating the
wrong columns. Run through a compiled graph, the key must land in columns
0-191 rotated over its first 64, the value in columns 192-319, and the query
must come back rotated at its own width.
"""

from __future__ import annotations

import dataclasses

import ml_dtypes
import numpy as np
import pytest
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy.kernels import (
    fused_chunked_qkv_rope_store_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.legacy.kv_cache import MHAKVCacheParams, SplitKVHeadDims
from test_common.simple_kv_cache import paged_kv_cache_inputs

K_DIM = 192
V_DIM = 128
ROPE_DIM = 64
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
LAYER = 1
MAX_POS = 512


def _bf16(x: np.ndarray) -> np.ndarray:
    """Rounds to bfloat16, kept as float32 for the arithmetic."""
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def _to_device(x: np.ndarray, device: Accelerator) -> Buffer:
    bits = np.ascontiguousarray(x.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _from_device(buf: Buffer) -> np.ndarray:
    bits = buf.copy(device=CPU()).view(DType.uint16).to_numpy()
    return bits.view(ml_dtypes.bfloat16).astype(np.float32)


def _bits(buf: Buffer) -> np.ndarray:
    return buf.copy(device=CPU()).view(DType.uint16).to_numpy()


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)


def _rope(
    x: np.ndarray, positions: np.ndarray, freqs: np.ndarray
) -> np.ndarray:
    """NeoX RoPE over the first ``ROPE_DIM`` columns, in float32."""
    out = x.astype(np.float32)
    half = ROPE_DIM // 2
    cos = freqs[positions, 0::2][:, None, :]
    sin = freqs[positions, 1::2][:, None, :]
    first = out[..., :half].copy()
    second = out[..., half:ROPE_DIM].copy()
    out[..., :half] = first * cos - second * sin
    out[..., half:ROPE_DIM] = first * sin + second * cos
    return out


def test_split_cache_store_and_rope() -> None:
    device = Accelerator()
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=K_DIM,
        v_head_dim=V_DIM,
        num_layers=2,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    prompt_lens = [5, 130]
    cache_lens = [0, 3]
    total = sum(prompt_lens)
    runtime = paged_kv_cache_inputs(
        params, prompt_lens, cache_lengths=cache_lens
    )
    leaves, treedef = tree.flatten(runtime)
    gpu = DeviceRef.GPU()
    cache_types = [
        BufferType(b.dtype, b.shape, DeviceRef.from_device(b.device))
        if b is runtime.kv_blocks
        else TensorType(b.dtype, b.shape, DeviceRef.from_device(b.device))
        for b in leaves
    ]

    with Graph(
        "split_store_rope",
        input_types=[
            TensorType(DType.bfloat16, [total, NUM_KV_HEADS, K_DIM], gpu),
            TensorType(DType.bfloat16, [total, NUM_KV_HEADS, V_DIM], gpu),
            TensorType(DType.bfloat16, [total, NUM_Q_HEADS, K_DIM], gpu),
            TensorType(DType.uint32, [len(prompt_lens) + 1], gpu),
            TensorType(DType.float32, [MAX_POS, ROPE_DIM], gpu),
            *cache_types,
        ],
    ) as graph:
        k, v, q, offsets, freqs, *cache = graph.inputs
        kv = dataclasses.replace(
            tree.unflatten(treedef, cache),
            split_head_dims=SplitKVHeadDims(),
        )
        layer_idx = ops.constant(LAYER, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_ragged(
            kv, k.tensor, offsets.tensor, layer_idx, v_head_dim=V_DIM
        )
        store_v_cache_ragged(
            kv, v.tensor, offsets.tensor, layer_idx, v_head_dim=V_DIM
        )
        graph.output(
            fused_qk_ragged_rope(
                params,
                q.tensor,
                offsets.tensor,
                kv,
                freqs.tensor,
                layer_idx,
                interleaved=False,
            )
        )

    model = InferenceSession(devices=[device]).load(graph)

    rng = np.random.default_rng(0)
    k_np = _bf16(rng.standard_normal([total, NUM_KV_HEADS, K_DIM]))
    v_np = _bf16(rng.standard_normal([total, NUM_KV_HEADS, V_DIM]))
    q_np = _bf16(rng.standard_normal([total, NUM_Q_HEADS, K_DIM]))
    offsets_np = np.array([0, *np.cumsum(prompt_lens)], dtype=np.uint32)
    theta = 10000.0 ** (-np.arange(0, ROPE_DIM, 2) / ROPE_DIM)
    angles = np.arange(MAX_POS)[:, None] * theta[None, :]
    freqs_np = np.empty([MAX_POS, ROPE_DIM], dtype=np.float32)
    freqs_np[:, 0::2] = np.cos(angles)
    freqs_np[:, 1::2] = np.sin(angles)

    (q_out,) = model(
        _to_device(k_np, device),
        _to_device(v_np, device),
        _to_device(q_np, device),
        Buffer.from_numpy(offsets_np).to(device),
        Buffer.from_numpy(freqs_np).to(device),
        *leaves,
    )

    positions = np.concatenate(
        [
            np.arange(c, c + p)
            for c, p in zip(cache_lens, prompt_lens, strict=True)
        ]
    )
    blocks = _from_device(runtime.kv_blocks)
    lookup = runtime.lookup_table.to_numpy()
    rows = np.stack(
        [
            blocks[
                lookup[b, pos // params.page_size],
                0,
                LAYER,
                pos % params.page_size,
            ]
            for b, (c, p) in enumerate(
                zip(cache_lens, prompt_lens, strict=True)
            )
            for pos in range(c, c + p)
        ]
    )
    np.testing.assert_allclose(
        rows[..., :K_DIM],
        _bf16(_rope(k_np, positions, freqs_np)),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_array_equal(rows[..., K_DIM:], v_np)
    assert isinstance(q_out, Buffer)
    np.testing.assert_allclose(
        _from_device(q_out),
        _bf16(_rope(q_np, positions, freqs_np)),
        rtol=1e-2,
        atol=1e-2,
    )


@dataclasses.dataclass(frozen=True)
class _ChunkedLayout:
    """One device's share of a MiMo-V2.6-Flash attention group's chunked
    QKV projection."""

    chunks: int
    num_q_heads: int
    num_kv_heads: int
    chunk_padding: int

    @property
    def q_rows(self) -> int:
        return self.num_q_heads // self.chunks * K_DIM

    @property
    def k_rows(self) -> int:
        return self.num_kv_heads // self.chunks * K_DIM

    @property
    def v_rows(self) -> int:
        return self.num_kv_heads // self.chunks * V_DIM

    @property
    def width(self) -> int:
        return self.q_rows + self.k_rows + self.v_rows + self.chunk_padding


_CHUNKED_LAYOUTS = {
    # Full attention: 4 KV heads, chunks padded to whole 128-row FP8 blocks.
    "full-tp4": _ChunkedLayout(1, 16, 1, 64),
    "full-tp2": _ChunkedLayout(2, 32, 2, 64),
    # Sliding window: 8 KV heads, chunks already whole blocks.
    "swa-tp4": _ChunkedLayout(1, 16, 2, 0),
    "swa-tp2": _ChunkedLayout(2, 32, 4, 0),
}
_CHUNKED_MAX_POS = 2048
_VALUE_SCALE = 0.1


def _chunked_graph(
    params: MHAKVCacheParams, layout: _ChunkedLayout, *, fused: bool
) -> Graph:
    gpu = DeviceRef.GPU()
    with Graph(
        f"split_chunked_qkv_{'fused' if fused else 'unfused'}",
        input_types=[
            TensorType(
                DType.bfloat16, ["tokens", layout.chunks, layout.width], gpu
            ),
            TensorType(DType.uint32, ["offsets"], gpu),
            TensorType(DType.bfloat16, [_CHUNKED_MAX_POS, ROPE_DIM], gpu),
            *tree.leaves(params.get_symbolic_inputs()),
        ],
    ) as graph:
        qkv, offsets, freqs = (value.tensor for value in graph.inputs[:3])
        kv = params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        layer_idx = ops.constant(LAYER, DType.uint32, device=DeviceRef.CPU())
        if fused:
            q = fused_chunked_qkv_rope_store_ragged(
                qkv,
                offsets,
                freqs,
                kv,
                layer_idx,
                _VALUE_SCALE,
                num_q_heads=layout.num_q_heads,
                qk_head_dim=K_DIM,
                v_head_dim=V_DIM,
                chunk_padding=layout.chunk_padding,
            )
        else:
            # The unfused split path: the store ops at the heads' own widths,
            # then RoPE on Q and the cached K.
            tokens = qkv.shape[0]
            k_start = layout.q_rows
            v_start = k_start + layout.k_rows
            q = ops.reshape(
                qkv[:, :, :k_start], [tokens, layout.num_q_heads, K_DIM]
            )
            k = ops.reshape(
                qkv[:, :, k_start:v_start],
                [tokens, layout.num_kv_heads, K_DIM],
            )
            v = ops.reshape(
                qkv[:, :, v_start : v_start + layout.v_rows],
                [tokens, layout.num_kv_heads, V_DIM],
            )
            v = ops.cast(ops.cast(v, DType.float32) * _VALUE_SCALE, v.dtype)
            store_k_cache_ragged(kv, k, offsets, layer_idx, v_head_dim=V_DIM)
            store_v_cache_ragged(kv, v, offsets, layer_idx, v_head_dim=V_DIM)
            q = fused_qk_ragged_rope(
                params,
                q,
                offsets,
                kv,
                freqs,
                layer_idx,
                interleaved=False,
            )
        graph.output(q)
    return graph


@dataclasses.dataclass
class _ChunkedModels:
    device: Accelerator
    params: MHAKVCacheParams
    layout: _ChunkedLayout
    unfused: Model
    fused: Model


@pytest.fixture(scope="module", params=list(_CHUNKED_LAYOUTS))
def chunked_models(request: pytest.FixtureRequest) -> _ChunkedModels:
    layout = _CHUNKED_LAYOUTS[request.param]
    device = Accelerator()
    session = InferenceSession(devices=[device])
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=layout.num_kv_heads,
        head_dim=K_DIM,
        v_head_dim=V_DIM,
        num_layers=2,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    return _ChunkedModels(
        device,
        params,
        layout,
        session.load(_chunked_graph(params, layout, fused=False)),
        session.load(_chunked_graph(params, layout, fused=True)),
    )


@pytest.mark.parametrize(
    ("prompt_lens", "cache_lens"),
    [
        ([130, 7], [3, 250]),
        ([1, 1, 1, 1, 1], [0, 127, 128, 255, 1500]),
        ([8, 8, 8, 8], [5, 120, 128, 900]),
        ([1, 8, 37, 129], [129, 7, 0, 1000]),
    ],
    ids=["chunked-prefill", "decode", "verify", "ragged"],
)
def test_chunked_qkv_store_matches_the_unfused_split_path(
    prompt_lens: list[int],
    cache_lens: list[int],
    chunked_models: _ChunkedModels,
) -> None:
    """The fused chunked store leaves a split cache and the rotated queries
    bitwise as the store and RoPE ops do: the key rotated over its first 64
    columns in columns 0-191, the scaled value in 192-319, and nothing else
    in the cache touched."""
    device = chunked_models.device
    params = chunked_models.params
    layout = chunked_models.layout
    total = sum(prompt_lens)
    runtime = paged_kv_cache_inputs(
        params, prompt_lens, cache_lengths=cache_lens, total_num_pages=40
    )
    leaves = tree.leaves(runtime)

    rng = np.random.default_rng(len(prompt_lens))
    initial = _bf16(rng.standard_normal(runtime.kv_blocks.shape))
    qkv = _bf16(rng.standard_normal([total, layout.chunks, layout.width]))
    offsets = np.array([0, *np.cumsum(prompt_lens)], dtype=np.uint32)
    theta = 10000.0 ** (-np.arange(0, ROPE_DIM, 2) / ROPE_DIM)
    angles = np.arange(_CHUNKED_MAX_POS)[:, None] * theta[None, :]
    freqs = np.empty([_CHUNKED_MAX_POS, ROPE_DIM], dtype=np.float32)
    freqs[:, 0::2] = np.cos(angles)
    freqs[:, 1::2] = np.sin(angles)
    inputs = [
        _to_device(qkv, device),
        Buffer.from_numpy(offsets).to(device),
        _to_device(freqs, device),
    ]

    results = []
    for model in (chunked_models.unfused, chunked_models.fused):
        runtime.kv_blocks.inplace_copy_from(_to_device(initial, device))
        (q,) = model(*inputs, *leaves)
        assert isinstance(q, Buffer)
        results.append((_bits(q), _bits(runtime.kv_blocks)))
    (q_ref, blocks_ref), (q_out, blocks_out) = results

    assert q_out.shape == (total, layout.num_q_heads, K_DIM)
    np.testing.assert_array_equal(q_out, q_ref)
    np.testing.assert_array_equal(blocks_out, blocks_ref)

    # The comparison is not vacuous: each new row holds the scaled value
    # after the key, whose columns past the rotary ones are as projected.
    lookup = runtime.lookup_table.to_numpy()
    kv_per_chunk = layout.num_kv_heads // layout.chunks
    token = 0
    for b, (c, p) in enumerate(zip(cache_lens, prompt_lens, strict=True)):
        for pos in range(c, c + p):
            row = blocks_out[lookup[b, pos // 128], 0, LAYER, pos % 128]
            for head in range(layout.num_kv_heads):
                chunk, h = divmod(head, kv_per_chunk)
                k = layout.q_rows + h * K_DIM
                v = layout.q_rows + layout.k_rows + h * V_DIM
                np.testing.assert_array_equal(
                    row[head, ROPE_DIM:K_DIM],
                    _bf16_bits(qkv[token, chunk, k + ROPE_DIM : k + K_DIM]),
                )
                np.testing.assert_array_equal(
                    row[head, K_DIM:],
                    _bf16_bits(
                        qkv[token, chunk, v : v + V_DIM]
                        * np.float32(_VALUE_SCALE)
                    ),
                )
            token += 1
