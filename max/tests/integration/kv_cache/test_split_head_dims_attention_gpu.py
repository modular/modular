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
"""The ragged paged attention graph op on a cache whose rows pack a 192-wide
key and a 128-wide value.

Through a compiled graph that stores the key and value and then attends over
them, the op must read the key from columns 0-191 and the value from columns
192-319, and return a 128-wide output matching an fp32 reference. The decode
test builds its graph from the cache params' own graph inputs and runs a
prefill, a decode step and an 8-token verify step on one cache, each with the
partition count the params put in the dispatch metadata. The sliding-window
test does the same for a group of 8 Q heads per KV head with a 128-key window
and per-head sink logits, through the op's ``.sink_weights`` variant.
"""

from __future__ import annotations

import dataclasses
import itertools

import ml_dtypes
import numpy as np
import pytest
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import MHAKVCacheParams, SplitKVHeadDims
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs

K_DIM = 192
V_DIM = 128
NUM_KV_HEADS = 1
NUM_Q_HEADS = 16
LAYER = 1

pytestmark = pytest.mark.skipif(
    not is_b100_b200(), reason="a split KV cache's attention runs on SM100"
)


def _bf16(x: np.ndarray) -> np.ndarray:
    """Rounds to bfloat16, kept as float32 for the arithmetic."""
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def _to_device(x: np.ndarray, device: Accelerator) -> Buffer:
    bits = np.ascontiguousarray(x.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _from_device(buf: Buffer) -> np.ndarray:
    bits = buf.copy(device=CPU()).view(DType.uint16).to_numpy()
    return bits.view(ml_dtypes.bfloat16).astype(np.float32)


def _attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    window: int | None = None,
    sinks: np.ndarray | None = None,
) -> np.ndarray:
    """Causal grouped-query attention of the last ``len(q)`` positions of a
    sequence whose keys and values are ``k`` and ``v``, in float32.

    A query sees at most ``window`` keys, its own included, and ``sinks``
    holds one logit per Q head that joins the softmax with no value row.
    """
    group = q.shape[1] // k.shape[1]
    k = np.repeat(k, group, axis=1)
    v = np.repeat(v, group, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(K_DIM)
    n, keys = q.shape[0], k.shape[0]
    visible = np.tril(np.ones((n, keys), dtype=bool), k=keys - n)
    if window is not None:
        visible &= np.triu(
            np.ones((n, keys), dtype=bool), k=keys - n - window + 1
        )
    scores = np.where(visible, scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    if sinks is not None:
        row_max = np.maximum(row_max, sinks[:, None, None])
    weights = np.exp(scores - row_max)
    denom = weights.sum(axis=-1, keepdims=True)
    if sinks is not None:
        denom += np.exp(sinks[:, None, None] - row_max)
    return np.einsum("hqk,khd->qhd", weights / denom, v)


def test_split_cache_attention() -> None:
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
    total = sum(prompt_lens)
    runtime = paged_kv_cache_inputs(params, prompt_lens)
    leaves, treedef = tree.flatten(runtime)
    gpu = DeviceRef.GPU()
    cache_types = [
        BufferType(b.dtype, b.shape, DeviceRef.from_device(b.device))
        if b is runtime.kv_blocks
        else TensorType(b.dtype, b.shape, DeviceRef.from_device(b.device))
        for b in leaves
    ]

    with Graph(
        "split_attention",
        input_types=[
            TensorType(DType.bfloat16, [total, NUM_KV_HEADS, K_DIM], gpu),
            TensorType(DType.bfloat16, [total, NUM_KV_HEADS, V_DIM], gpu),
            TensorType(DType.bfloat16, [total, NUM_Q_HEADS, K_DIM], gpu),
            TensorType(DType.uint32, [len(prompt_lens) + 1], gpu),
            *cache_types,
        ],
    ) as graph:
        k, v, q, offsets, *cache = graph.inputs
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
            flash_attention_ragged(
                params,
                q.tensor,
                offsets.tensor,
                kv,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                scale=K_DIM**-0.5,
            )
        )

    model = InferenceSession(devices=[device]).load(graph)

    rng = np.random.default_rng(0)
    k_np = _bf16(rng.standard_normal([total, NUM_KV_HEADS, K_DIM]))
    v_np = _bf16(rng.standard_normal([total, NUM_KV_HEADS, V_DIM]))
    q_np = _bf16(rng.standard_normal([total, NUM_Q_HEADS, K_DIM]))
    offsets_np = np.array([0, *np.cumsum(prompt_lens)], dtype=np.uint32)

    (out,) = model(
        _to_device(k_np, device),
        _to_device(v_np, device),
        _to_device(q_np, device),
        Buffer.from_numpy(offsets_np).to(device),
        *leaves,
    )

    assert isinstance(out, Buffer)
    assert out.shape == (total, NUM_Q_HEADS, V_DIM)
    want = np.concatenate(
        [
            _attention(q_np[a:b], k_np[a:b], v_np[a:b])
            for a, b in itertools.pairwise(offsets_np)
        ]
    )
    np.testing.assert_allclose(_from_device(out), want, rtol=2e-2, atol=2e-2)


def test_split_cache_decode_through_graph_inputs() -> None:
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
    gpu = DeviceRef.GPU()
    cache_inputs = tree.leaves(params.get_symbolic_inputs())
    with Graph(
        "split_decode",
        input_types=[
            TensorType(DType.bfloat16, ["total", NUM_KV_HEADS, K_DIM], gpu),
            TensorType(DType.bfloat16, ["total", NUM_KV_HEADS, V_DIM], gpu),
            TensorType(DType.bfloat16, ["total", NUM_Q_HEADS, K_DIM], gpu),
            TensorType(DType.uint32, ["offsets"], gpu),
            *cache_inputs,
        ],
    ) as graph:
        k, v, q, offsets, *cache = graph.inputs
        (kv,) = params.unflatten_kv_inputs(iter(cache))
        layer_idx = ops.constant(LAYER, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_ragged(
            kv, k.tensor, offsets.tensor, layer_idx, v_head_dim=V_DIM
        )
        store_v_cache_ragged(
            kv, v.tensor, offsets.tensor, layer_idx, v_head_dim=V_DIM
        )
        graph.output(
            flash_attention_ragged(
                params,
                q.tensor,
                offsets.tensor,
                kv,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                scale=K_DIM**-0.5,
            )
        )
    model = InferenceSession(devices=[device]).load(graph)

    rng = np.random.default_rng(1)
    # A short and a 4,000-token sequence; each step's pages extend the last's,
    # so one page pool carries the cache through all three steps.
    lengths = [5, 4000]
    seq_k = [np.zeros([0, NUM_KV_HEADS, K_DIM], np.float32) for _ in lengths]
    seq_v = [np.zeros([0, NUM_KV_HEADS, V_DIM], np.float32) for _ in lengths]
    total_pages = 1 + 32
    blocks: Buffer | None = None
    for step, (prompt_lens, want_q_max) in enumerate(
        [(lengths, 4000), ([1, 1], 1), ([8, 8], 8)]
    ):
        cache_lengths = [len(x) for x in seq_k]
        runtime = paged_kv_cache_inputs(
            params,
            prompt_lens,
            cache_lengths=cache_lengths,
            total_num_pages=total_pages,
        )
        if blocks is not None:
            runtime = dataclasses.replace(runtime, kv_blocks=blocks)
        blocks = runtime.kv_blocks
        assert runtime.attention_dispatch_metadata is not None
        batch, q_max, num_partitions, _ = (
            runtime.attention_dispatch_metadata.to_numpy().tolist()
        )
        assert (batch, q_max) == (2, want_q_max)
        if step > 0:
            # The split-K route, not a single partition.
            assert num_partitions > 1, num_partitions

        total = sum(prompt_lens)
        k_np = _bf16(rng.standard_normal([total, NUM_KV_HEADS, K_DIM]))
        v_np = _bf16(rng.standard_normal([total, NUM_KV_HEADS, V_DIM]))
        q_np = _bf16(rng.standard_normal([total, NUM_Q_HEADS, K_DIM]))
        offsets_np = np.array([0, *np.cumsum(prompt_lens)], dtype=np.uint32)
        (out,) = model(
            _to_device(k_np, device),
            _to_device(v_np, device),
            _to_device(q_np, device),
            Buffer.from_numpy(offsets_np).to(device),
            *tree.leaves(runtime),
        )
        assert isinstance(out, Buffer)
        assert out.shape == (total, NUM_Q_HEADS, V_DIM)

        want = []
        for b, (start, end) in enumerate(itertools.pairwise(offsets_np)):
            seq_k[b] = np.concatenate([seq_k[b], k_np[start:end]])
            seq_v[b] = np.concatenate([seq_v[b], v_np[start:end]])
            want.append(_attention(q_np[start:end], seq_k[b], seq_v[b]))
        np.testing.assert_allclose(
            _from_device(out), np.concatenate(want), rtol=2e-2, atol=2e-2
        )


def test_split_cache_sliding_window_with_sinks() -> None:
    device = Accelerator()
    num_kv_heads, num_q_heads, window = 2, 16, 128
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=num_kv_heads,
        head_dim=K_DIM,
        v_head_dim=V_DIM,
        num_layers=2,
        page_size=128,
        devices=[DeviceRef.GPU()],
        window_size=window,
    )
    gpu = DeviceRef.GPU()
    cache_inputs = tree.leaves(params.get_symbolic_inputs())
    with Graph(
        "split_sliding_window",
        input_types=[
            TensorType(DType.bfloat16, ["total", num_kv_heads, K_DIM], gpu),
            TensorType(DType.bfloat16, ["total", num_kv_heads, V_DIM], gpu),
            TensorType(DType.bfloat16, ["total", num_q_heads, K_DIM], gpu),
            TensorType(DType.uint32, ["offsets"], gpu),
            TensorType(DType.bfloat16, [num_q_heads], gpu),
            *cache_inputs,
        ],
    ) as graph:
        k, v, q, offsets, sinks, *cache = graph.inputs
        (kv,) = params.unflatten_kv_inputs(iter(cache))
        layer_idx = ops.constant(LAYER, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_ragged(
            kv, k.tensor, offsets.tensor, layer_idx, v_head_dim=V_DIM
        )
        store_v_cache_ragged(
            kv, v.tensor, offsets.tensor, layer_idx, v_head_dim=V_DIM
        )
        graph.output(
            flash_attention_ragged(
                params,
                q.tensor,
                offsets.tensor,
                kv,
                layer_idx,
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK,
                scale=K_DIM**-0.5,
                local_window_size=window,
                sink_weights=sinks.tensor,
            )
        )
    model = InferenceSession(devices=[device]).load(graph)

    rng = np.random.default_rng(2)
    sinks_np = _bf16(rng.uniform(-4, 4, [num_q_heads]))
    # A prefill of a short and a 4,000-token sequence, a decode step, an
    # 8-token verify step, then chunks longer than the window.
    lengths = [5, 4000]
    seq_k = [np.zeros([0, num_kv_heads, K_DIM], np.float32) for _ in lengths]
    seq_v = [np.zeros([0, num_kv_heads, V_DIM], np.float32) for _ in lengths]
    # ``paged_kv_cache_inputs`` numbers each request's pages after the
    # previous request's, so the short sequence's second page would move the
    # long one's. Each sequence keeps a fixed page range instead: pages 0-1
    # for the 144-token one, 2-34 for the 4,209-token one.
    first_page = [0, 2]
    total_pages = 2 + 33
    blocks: Buffer | None = None
    for prompt_lens in [lengths, [1, 1], [8, 8], [130, 200]]:
        cache_lengths = [len(x) for x in seq_k]
        runtime = paged_kv_cache_inputs(
            params,
            prompt_lens,
            cache_lengths=cache_lengths,
            total_num_pages=total_pages,
        )
        lut = np.full(runtime.lookup_table.shape, total_pages, np.uint32)
        for b, (cached, new) in enumerate(
            zip(cache_lengths, prompt_lens, strict=True)
        ):
            num_pages = -(-(cached + new) // params.page_size)
            lut[b, :num_pages] = np.arange(num_pages) + first_page[b]
        runtime = dataclasses.replace(
            runtime, lookup_table=Buffer.from_numpy(lut).to(device)
        )
        if blocks is not None:
            runtime = dataclasses.replace(runtime, kv_blocks=blocks)
        blocks = runtime.kv_blocks
        assert runtime.attention_dispatch_metadata is not None
        _, _, num_partitions, _ = (
            runtime.attention_dispatch_metadata.to_numpy().tolist()
        )
        # The window, not the 4,000-key cache, sizes the partitions.
        assert num_partitions == 1, num_partitions

        total = sum(prompt_lens)
        k_np = _bf16(rng.standard_normal([total, num_kv_heads, K_DIM]))
        v_np = _bf16(rng.standard_normal([total, num_kv_heads, V_DIM]))
        q_np = _bf16(rng.standard_normal([total, num_q_heads, K_DIM]))
        offsets_np = np.array([0, *np.cumsum(prompt_lens)], dtype=np.uint32)
        (out,) = model(
            _to_device(k_np, device),
            _to_device(v_np, device),
            _to_device(q_np, device),
            Buffer.from_numpy(offsets_np).to(device),
            _to_device(sinks_np, device),
            *tree.leaves(runtime),
        )
        assert isinstance(out, Buffer)
        assert out.shape == (total, num_q_heads, V_DIM)

        want = []
        for b, (start, end) in enumerate(itertools.pairwise(offsets_np)):
            seq_k[b] = np.concatenate([seq_k[b], k_np[start:end]])
            seq_v[b] = np.concatenate([seq_v[b], v_np[start:end]])
            want.append(
                _attention(
                    q_np[start:end], seq_k[b], seq_v[b], window, sinks_np
                )
            )
        np.testing.assert_allclose(
            _from_device(out), np.concatenate(want), rtol=2e-2, atol=2e-2
        )
