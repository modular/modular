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
"""Executes the compressor and indexer graph-op bindings on a B200.

The cache-op suite runs the registered CSA2 store and attention ops; these
tests drive the two bindings it does not reach -- the compressor's pooling
ops and the indexer's MXFP4 ops -- through the same ModuleV3 call sites
the trunk uses (:data:`compress_pool_op` and the :mod:`index_ops`
wrappers), so no graph-op binding of the attention half first executes in
a later PR.

The compressor case runs one request through a 5-row prefill and a 1-row
decode, the smallest forwards that exercise every ratio-2 path: closed
pairs pool on their closing rows -- per-channel softmax over the pair's
scores weighting its KV rows, one RMSNorm, one bf16 cast, checked against
a float64 oracle -- rows that close no pair stay zero, the prefill's last
two raw rows save into the state ring at their positions' slots, and the
decode at position 5 pools its straddled pair against the row the ring
saved at position 4 (the source carry), then saves its own row at its
position's slot.

The indexer case stores four one-hot keys through the real k-store and
selects top-1 with two queries. One-hot keys make the MXFP4 quantization
exact, so the oracle needs no quantized-dot model: the query matching a
visible key holds its request's only positive score and must be picked,
and the query matching a key past its request's cached length -- the
causal bound -- must never be picked.

The module cases drive ``Compressor.forward`` and ``Indexer.forward``
themselves (not just their op bindings): the compressor takes the
state-ring arguments the same way at both ratios -- dropping them at
ratio 1, pooling through the paged ring at ratio 2 -- and the indexer
runs a full forward as the owner (storing its compressed record and
index key, then selecting its match) and as a reindex layer, once
building its RoPE table internally and once taking the caller-staged
table the attention contract already exposes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from max import functional as F
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    STATE_RING_ROWS,
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layer_resolver import (
    resolve,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    RaggedRows,
    position_rope_table,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.compressor import (
    Compressor,
    compress_pool_op,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.index_ops import (
    indexer_k_store,
    indexer_q_rope_quant,
    indexer_select,
    indexer_select_block_topk,
    indexer_select_masked,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.indexer import (
    Indexer,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.norm import EPS
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh
from runtime_config import text_config

HEAD = 512
"""The compressor's latent width, the only one the pooling ops support."""
IDENTITY_ROPE_ROWS = 8
"""Position rows of the identity rotation table (all cosines one)."""


def _hf_config() -> dict[str, Any]:
    return {
        "text_config": {
            "vocab_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "head_dim": HEAD,
            "sliding_window": 128,
            "compress_ratios": [0, 2, 1, 0],
            "kv_source_layer_ids": [1, 2],
            "index_head_dim": 128,
            "engram_num_embeddings": [2],
            "engram_head_dim": 256,
            "dspark_target_layer_ids": [],
            "hidden_size": 512,
            "moe_intermediate_size": 128,
            "q_lora_rank": 64,
            "o_lora_rank": 64,
            "o_groups": 2,
            "n_routed_experts": 4,
            "index_source_layer_ids": [1, 2],
            "candidate_source_layer_id": 2,
            "index_n_heads": 4,
            "index_topk": 8,
            "candidate_topk_blocks": 4,
            "hc_mult": 2,
            "engram_layer_ids": [0],
        }
    }


def _shape(key: str) -> tuple[int, ...]:
    """The interface's per-block shape of leaf ``key``."""
    config = V41Config.from_hf(_hf_config())
    params = build_kv_params(
        config,
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.GPU()],
    )
    child = params.children[key]
    assert isinstance(child, KVCacheParams), key
    return tuple(int(d) for d in child.shape_per_block)


def _dev(x: np.ndarray) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(x)).to(Accelerator())


def _host(x: np.ndarray) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(x))


def _leaf(
    blocks: Tensor,
    lut: np.ndarray,
    cache_lengths: np.ndarray,
    capacity: int,
) -> PagedCacheValues:
    """A leaf over device blocks, host metadata in the ops' dtypes."""
    return PagedCacheValues(
        kv_blocks=blocks,
        cache_lengths=_dev(cache_lengths),
        lookup_table=_dev(lut),
        max_prompt_length=_host(np.array([capacity], dtype=np.uint32)),
        max_cache_length=_host(np.array([capacity], dtype=np.uint32)),
        page_stride=_host(np.array([-1], dtype=np.int64)),
    )


def _pair(
    kv0: np.ndarray, kv1: np.ndarray, s0: np.ndarray, s1: np.ndarray
) -> np.ndarray:
    """The pooling op's pair math: per-channel softmax over the scores."""
    m = np.maximum(s0, s1)
    e0, e1 = np.exp(s0 - m), np.exp(s1 - m)
    z = e0 + e1
    return (e0 / z) * kv0 + (e1 / z) * kv1


def _rms(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The pooling op's normalization: fp32 RMSNorm, one bf16 cast."""
    var = np.mean(x * x, axis=-1, keepdims=True)
    return weight * x * (1.0 / np.sqrt(var + EPS))


def test_compress_pools_each_token() -> None:
    """Ratio 1: every row's KV projection RMS-normalizes to its latent."""
    rng = np.random.default_rng(61)
    rows = 3
    kv = (rng.standard_normal((rows, HEAD)) * 2).astype(np.float32)
    weight = (rng.standard_normal(HEAD) * 0.5 + 1).astype(np.float32)
    latent = compress_pool_op(
        _dev(kv),
        _dev(np.arange(rows, dtype=np.int32)),
        _dev(np.array([0, rows], dtype=np.uint32)),
        _dev(weight),
        _dev(np.zeros(rows, dtype=np.int32)),
        _dev(np.arange(rows, dtype=np.int32)),
        _dev(np.array([rows], dtype=np.int32)),
        ratio=1,
        epsilon=EPS,
    )
    got = latent.cast(DType.float32).to_numpy()
    want = _rms(kv.astype(np.float64), weight.astype(np.float64))
    # One bf16 cast: a 2**-7 relative window covers a single rounding.
    np.testing.assert_allclose(got, want, rtol=2.0**-7, atol=2.0**-7)


def test_compress_paged_pools_pairs_and_carries_the_ring() -> None:
    """Ratio 2: pair pooling, the ring save, and the straddled carry."""
    state_shape = _shape("c2a_state")
    assert state_shape[2] == STATE_RING_ROWS and state_shape[4] == 2 * HEAD, (
        state_shape
    )
    rng = np.random.default_rng(29)
    state_blocks = _dev(np.zeros((1, *state_shape), dtype=np.float32))
    state = _leaf(
        state_blocks,
        np.array([[0]], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        capacity=8,
    )
    pre = (rng.standard_normal((5, 2 * HEAD)) * 2).astype(np.float32)
    dec = (rng.standard_normal((1, 2 * HEAD)) * 2).astype(np.float32)
    weight = (rng.standard_normal(HEAD) * 0.5 + 1).astype(np.float32)
    layer = F.constant(0, DType.uint32, device=CPU())

    def pool(raw: np.ndarray, positions: list[int]) -> np.ndarray:
        rows = raw.shape[0]
        return (
            compress_pool_op(
                _dev(raw),
                _dev(np.asarray(positions, dtype=np.int32)),
                _dev(np.array([0, rows], dtype=np.uint32)),
                _dev(weight),
                _dev(np.zeros(rows, dtype=np.int32)),
                _dev(np.arange(rows, dtype=np.int32)),
                _dev(np.array([rows], dtype=np.int32)),
                ratio=2,
                epsilon=EPS,
                state=state,
                layer_idx=layer,
            )
            .cast(DType.float32)
            .to_numpy()
        )

    # Prefill: positions 0..4. Pairs close at 1 and 3; row 4 stays open.
    latent = pool(pre, [0, 1, 2, 3, 4])
    kv = pre[:, :HEAD].astype(np.float64)
    score = pre[:, HEAD:].astype(np.float64)
    want = np.zeros((5, HEAD))
    for close in (1, 3):
        want[close] = _rms(
            _pair(kv[close - 1], kv[close], score[close - 1], score[close]),
            weight.astype(np.float64),
        )
    np.testing.assert_allclose(latent, want, rtol=2.0**-7, atol=2.0**-7)
    assert not latent[[0, 2, 4]].any(), "rows that close no pair are zero"

    # The ring save: the prefill's last two raw rows, at their positions'
    # slots -- position 4 in slot 4, position 3 in slot 3.
    ring = state_blocks.to_numpy().reshape(STATE_RING_ROWS, 2 * HEAD)
    np.testing.assert_array_equal(ring[4], pre[4])
    np.testing.assert_array_equal(ring[3], pre[3])

    # Decode at position 5: the straddled pair pools against the row the
    # ring holds at position 4, then the row saves into its own slot.
    latent = pool(dec, [5])
    want = _rms(
        _pair(
            pre[4, :HEAD].astype(np.float64),
            dec[0, :HEAD].astype(np.float64),
            pre[4, HEAD:].astype(np.float64),
            dec[0, HEAD:].astype(np.float64),
        ),
        weight.astype(np.float64),
    )
    np.testing.assert_allclose(latent[0], want, rtol=2.0**-7, atol=2.0**-7)
    ring = state_blocks.to_numpy().reshape(STATE_RING_ROWS, 2 * HEAD)
    np.testing.assert_array_equal(ring[5], dec[0])
    np.testing.assert_array_equal(ring[4], pre[4])


def test_indexer_stores_and_selects_the_matching_visible_key() -> None:
    """The k-store's records and the causal top-1 over them."""
    idx_shape = _shape("idx_c1a")
    keys_per_page = idx_shape[2]
    # Two requests, one block each; both blocks hold the same four keys,
    # so a key past the causal bound exists and scores when read.
    blocks = 2
    idx_blocks = _dev(np.zeros((blocks, *idx_shape), dtype=np.uint8))
    # Request 0 sits at position 4 (four keys cached), request 1 at
    # position 2 (two cached): request 1's causal window holds keys 0..2.
    leaf = _leaf(
        idx_blocks,
        np.array([[0], [1]], dtype=np.uint32),
        np.array([4, 2], dtype=np.uint32),
        capacity=IDENTITY_ROPE_ROWS,
    )
    layer = F.constant(0, DType.uint32, device=CPU())
    # An identity rotation: cosines one, sines zero, so stored records and
    # packed queries equal their unrotated values.
    table = np.concatenate(
        [np.ones((IDENTITY_ROPE_ROWS, 32)), np.zeros((IDENTITY_ROPE_ROWS, 32))],
        axis=1,
    ).astype(np.float32)
    keys = np.zeros((4, 128), dtype=np.float32)
    for j in range(4):
        keys[j, 32 * j] = 4.0  # one-hot: each key's block of 32 is exact
    store_keys = np.tile(keys, (blocks, 1))
    store_slots = np.concatenate(
        [np.arange(4) + block * keys_per_page for block in range(blocks)]
    )
    indexer_k_store(
        leaf,
        _dev(store_keys).cast(DType.bfloat16),
        _dev(np.tile(np.arange(4, dtype=np.int64), blocks)),
        _dev(table),
        _dev(np.ones(128, dtype=np.float32)).cast(DType.bfloat16),
        _dev(store_slots.astype(np.int64)),
        layer,
        ratio=1,
        epsilon=EPS,
    )
    # The scorer runs 32 heads of 128 values; every head carries the same
    # query, so the match is still the only positive score.
    heads = 32
    queries = np.tile(keys[[1, 3]], (1, heads))
    q = indexer_q_rope_quant(
        _dev(queries).cast(DType.bfloat16),
        _dev(np.ones((2, heads), dtype=np.float32)).cast(DType.bfloat16),
        _dev(np.array([4, 2], dtype=np.int64)),
        _dev(table),
        n_heads=heads,
        weight_scale=1.0,
    )
    picks = indexer_select(
        leaf,
        q,
        _dev(np.array([0, 1, 2], dtype=np.uint32)),
        layer,
        top_k=1,
        ratio=1,
    ).to_numpy()
    # Each query is its match's only positive score, so top-1 is exact:
    # the visible match is picked. Request 1's match, key 3, sits past
    # its causal window (positions 0..2) and must never be picked even
    # though the block holds it and it would score.
    assert picks[0, 0] == 1
    assert picks[1, 0] != 3

    # The candidate-block select: blocks of two entries score their
    # visible entries' maximum, so the match's block is the one strictly
    # positive block and must be kept; which other blocks join it is
    # unspecified.
    q_pos = _dev(np.array([4, 2], dtype=np.int32))
    _, kept = indexer_select_block_topk(
        leaf,
        q,
        _dev(np.array([0, 1, 2], dtype=np.uint32)),
        q_pos,
        layer,
        top_k=1,
        ratio=1,
        block_size=2,
        topk_blocks=2,
    )
    kept_np = kept.to_numpy()
    assert 0 in kept_np[0]

    # The masked select reads only the kept blocks: request 0 keeps the
    # match's block and still picks it; request 1's kept blocks exclude
    # its match's, so the match must not be picked.
    masked = indexer_select_masked(
        leaf,
        q,
        _dev(np.array([0, 1, 2], dtype=np.uint32)),
        _dev(np.array([[0, -1], [0, -1]], dtype=np.int32)),
        layer,
        top_k=1,
        ratio=1,
        block_size=2,
    ).to_numpy()
    assert masked[0, 0] == 1
    assert masked[1, 0] != 3


def _model_config(text: SimpleNamespace) -> DeepseekV41Config:
    """The resolved model config of a synthetic text layout."""
    kv = build_kv_params(
        V41Config.from_hf({"text_config": vars(text)}),
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.GPU()],
    )
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.GPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )


def _mesh() -> DeviceMesh:
    return DeviceMesh((Accelerator(),), (1,), ("tp",))


def _e8m0(raw: np.ndarray) -> Tensor:
    """A UE8M0 scale tensor, reinterpreted from its exponent bytes."""
    return Tensor.from_dlpack(
        Buffer.from_numpy(np.ascontiguousarray(raw, dtype=np.uint8)).view(
            DType.float8_e8m0fnu
        )
    ).to(Accelerator())


def _ragged_rows(offsets: np.ndarray, positions: np.ndarray) -> RaggedRows:
    """``RaggedRows`` for one row per request, lengths from the offsets."""
    batch = len(offsets) - 1
    bid = np.repeat(np.arange(batch, dtype=np.int32), np.diff(offsets))
    local = (
        np.arange(int(offsets[-1]), dtype=np.int32)
        - offsets[:-1].astype(np.int32)[bid]
    )
    lengths = np.diff(offsets).astype(np.int32)
    return RaggedRows(
        offsets=_dev(offsets.astype(np.uint32)),
        bid=_dev(bid),
        local=_dev(local),
        positions=_dev(positions.astype(np.int32)),
        lengths=_dev(lengths),
    )


def test_compress_ratio_one_drops_the_state_ring_arguments() -> None:
    """A caller passing the ring leaf and index at every layer is safe.

    ``Compressor.forward`` drops ``state`` at ratio 1; before the layer
    index dropped with it, the same uniform call crashed the ratio-1
    owner at graph build ("ratio 1 takes no state ring"). The pooled
    latent still matches the float64 oracle.
    """
    config = _model_config(text_config())
    with default_device(Accelerator()):
        compressor = Compressor(config, 1, _mesh())
    rng = np.random.default_rng(17)
    rows = 3
    kv = (rng.standard_normal((rows, HEAD)) * 2).astype(np.float32)
    state = _leaf(
        _dev(np.zeros((1, *_shape("c2a_state")), dtype=np.float32)),
        np.array([[0]], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        capacity=8,
    )
    latent, valid = compressor(
        _dev(kv),
        _ragged_rows(np.array([0, rows]), np.arange(rows, dtype=np.int64)),
        state,
        F.constant(0, DType.uint32, device=CPU()),
    )
    got = latent.cast(DType.float32).to_numpy()
    want = _rms(kv.astype(np.float64), np.ones(HEAD))
    np.testing.assert_allclose(got, want, rtol=2.0**-7, atol=2.0**-7)
    assert valid.to_numpy().all()


def test_compress_ratio_two_pools_and_saves_the_ring() -> None:
    """A ratio-2 forward keeps the ring leaf and pools through it.

    The block hands every compressor the ring leaf and layer index the
    same way; the ratio-2 owner keeps them where its ratio-1 counterpart
    drops them. The chunk starts at position 4, so the wrapper's
    positions differ from its local row indices and a wrapper that fed
    the pooling op the wrong one pools the wrong pairs. The last two raw
    rows save into the ring, at their positions' own slots.
    """
    config = _model_config(text_config())
    with default_device(Accelerator()):
        compressor = Compressor(config, 2, _mesh())
    rng = np.random.default_rng(23)
    rows = 5
    raw = (rng.standard_normal((rows, 2 * HEAD)) * 2).astype(np.float32)
    state_blocks = _dev(np.zeros((1, *_shape("c2a_state")), dtype=np.float32))
    state = _leaf(
        state_blocks,
        np.array([[0]], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        capacity=8,
    )
    latent, valid = compressor(
        _dev(raw),
        _ragged_rows(
            np.array([0, rows]), np.arange(4, 4 + rows, dtype=np.int64)
        ),
        state,
        F.constant(0, DType.uint32, device=CPU()),
    )
    got = latent.cast(DType.float32).to_numpy()
    kv = raw[:, :HEAD].astype(np.float64)
    score = raw[:, HEAD:].astype(np.float64)
    want = np.zeros((rows, HEAD))
    # Positions 4..8: the pairs (4, 5) and (6, 7) close on rows 1 and 3.
    for close in (1, 3):
        want[close] = _rms(
            _pair(kv[close - 1], kv[close], score[close - 1], score[close]),
            np.ones(HEAD),
        )
    np.testing.assert_allclose(got, want, rtol=2.0**-7, atol=2.0**-7)
    assert not got[[0, 2, 4]].any(), "rows that close no pair are zero"
    np.testing.assert_array_equal(
        valid.to_numpy(), [False, True, False, True, False]
    )
    # The ring save: the last two raw rows, at their positions' slots --
    # position 8 in slot 0, position 7 in slot 7.
    ring = state_blocks.to_numpy().reshape(-1, 2 * HEAD)
    np.testing.assert_array_equal(ring[0], raw[4])
    np.testing.assert_array_equal(ring[7], raw[3])


def test_indexer_forward_consumes_a_staged_rope_table() -> None:
    """A reindex forward selects the same entries from a staged table.

    The keys and queries rotate through the layer's own YaRN table, so
    each row's match survives the rotation (a pair's inner product is
    rotation-invariant, and the matching key rotates at the same
    position as its query): the match holds its request's only large
    score, and a table that disagreed with the stored keys' rotation
    would break row 1's match, whose channel sits inside the rotated
    tail. Running once with the table the forward builds and once with
    the caller-staged table the attention contract exposes, the
    selections must agree.
    """
    text = text_config(
        compress_ratios=[1, 1],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 1],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_topk=4,
        # The MXFP4 kernels require a multiple of 8 heads.
        index_n_heads=32,
    )
    config = _model_config(text)
    layer_info = resolve(config)[1]
    with default_device(Accelerator()):
        indexer = Indexer(
            config,
            _mesh(),
            "reindex_r1",
            layer_info=layer_info,
            rope=RopeParams(64, 32.0, 1.0, 4096),
            max_position=IDENTITY_ROPE_ROWS,
        )
    # Every head carries the same matches: row 0's query one-hot below
    # the rotated tail (key 2's channel, never rotated), row 1's inside
    # it (key 4's channel, rotated with its key at the same position).
    heads = config.index_n_heads
    wq = np.zeros((heads * 128, 64), np.float32)
    for head in range(heads):
        wq[head * 128 + 32, 0] = 4.0
        wq[head * 128 + 64, 1] = 4.0
    indexer.wq_b.weight = _dev(wq).cast(DType.float8_e4m3fn)
    indexer.wq_b.scale = _e8m0(np.full((heads * 128 // 32, 64 // 32), 127))
    indexer.weights_proj.weight = _dev(np.ones((heads, 64), np.float32)).cast(
        DType.bfloat16
    )

    idx_shape = _shape("idx_c1a")
    keys_per_page = int(idx_shape[2])
    n_keys = 8
    blocks = 2
    idx_blocks = _dev(np.zeros((blocks, *idx_shape), dtype=np.uint8))
    # Both requests sit at position 4, so each sees keys 0..4; the pool
    # keeps both blocks, so nothing is masked away.
    leaf = _leaf(
        idx_blocks,
        np.array([[0], [1]], dtype=np.uint32),
        np.array([4, 4], dtype=np.uint32),
        capacity=IDENTITY_ROPE_ROWS,
    )
    layer = F.constant(0, DType.uint32, device=CPU())
    keys = np.zeros((n_keys, 128), dtype=np.float32)
    for j in range(n_keys):
        keys[j, 16 * j] = 4.0
    store_slots = np.concatenate(
        [np.arange(n_keys) + block * keys_per_page for block in range(blocks)]
    )
    yarn = position_rope_table(
        indexer.rope.inv_freqs(),
        IDENTITY_ROPE_ROWS,
        n_keys,
        _dev(np.zeros(1, dtype=np.uint32)),
    )
    indexer_k_store(
        leaf,
        _dev(np.tile(keys, (blocks, 1))).cast(DType.bfloat16),
        _dev(np.tile(np.arange(n_keys, dtype=np.int64), blocks)),
        yarn,
        _dev(np.ones(128, dtype=np.float32)).cast(DType.bfloat16),
        _dev(store_slots.astype(np.int64)),
        layer,
        ratio=1,
        epsilon=EPS,
    )

    qr = np.zeros((2, 64), np.float32)
    qr[0, 0] = 1.0
    qr[1, 1] = 1.0
    x = _dev(np.ones((2, 64), np.float32)).cast(DType.bfloat16)
    rows = _ragged_rows(np.array([0, 1, 2]), np.array([4, 4], dtype=np.int64))
    pool = _dev(np.array([[0, 1], [0, 1]], dtype=np.int32))

    def forward(staged: bool) -> np.ndarray:
        picks, out_pool = indexer(
            x,
            _dev(qr).cast(DType.bfloat16),
            None,
            None,
            rows,
            {"idx_source": leaf},
            {"idx_source": layer},
            pool=pool,
            rope_table=(
                position_rope_table(
                    indexer.rope.inv_freqs(),
                    IDENTITY_ROPE_ROWS,
                    2,
                    x,
                )
                if staged
                else None
            ),
        )
        assert out_pool is None
        return picks.to_numpy()

    built = forward(staged=False)
    from_caller = forward(staged=True)
    # Four picks per row over five visible keys: the match is the only
    # large score, so it is always among them; which zero-score keys
    # join it is unspecified.
    assert 2 in built[0]
    assert 4 in built[1]
    np.testing.assert_array_equal(from_caller, built)


def test_indexer_owner_forward_stores_and_selects() -> None:
    """An owner forward stores both leaves and selects its own match.

    The candidate source runs the store side the reindex case leaves to
    the op bindings: its compressed record lands in the ratio-1 leaf and
    its index key -- projected from the latent, normalized, rotated and
    quantized by the k-store -- in the index leaf, both through the
    wrapper's own slot lists. Each row's match is the key it just stored
    at its own position (a pair's inner product is rotation-invariant),
    so the selection must pick it and the kept-block pool must hold the
    match's block.
    """
    text = text_config(
        compress_ratios=[1, 1],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 1],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_topk=4,
        # The MXFP4 kernels require a multiple of 8 heads.
        index_n_heads=32,
    )
    config = _model_config(text)
    layer_info = resolve(config)[0]
    with default_device(Accelerator()):
        indexer = Indexer(
            config,
            _mesh(),
            "full_r1_cand",
            layer_info=layer_info,
            rope=RopeParams(64, 32.0, 1.0, 4096),
            max_position=IDENTITY_ROPE_ROWS,
        )
    heads = config.index_n_heads
    wq = np.zeros((heads * 128, 64), np.float32)
    for head in range(heads):
        wq[head * 128 + 32, 0] = 4.0
        wq[head * 128 + 64, 1] = 4.0
    indexer.wq_b.weight = _dev(wq).cast(DType.float8_e4m3fn)
    indexer.wq_b.scale = _e8m0(np.full((heads * 128 // 32, 64 // 32), 127))
    indexer.weights_proj.weight = _dev(np.ones((heads, 64), np.float32)).cast(
        DType.bfloat16
    )
    # The key projection reads one latent channel per key channel, so
    # each row's latent is one-hot at its match's channel.
    wk = np.zeros((128, HEAD), np.float32)
    for channel in range(128):
        wk[channel, 4 * channel] = 1.0
    indexer.wk.weight = _dev(wk).cast(DType.bfloat16)

    idx_blocks = _dev(np.zeros((2, *_shape("idx_c1a")), dtype=np.uint8))
    comp_blocks = _dev(np.zeros((2, *_shape("c1a")), dtype=np.uint8))
    # Both requests sit at position 4, so each row's match is the key it
    # stores at entry 4 of its own block.
    idx_leaf = _leaf(
        idx_blocks,
        np.array([[0], [1]], dtype=np.uint32),
        np.array([4, 4], dtype=np.uint32),
        capacity=IDENTITY_ROPE_ROWS,
    )
    comp_leaf = _leaf(
        comp_blocks,
        np.array([[0], [1]], dtype=np.uint32),
        np.array([4, 4], dtype=np.uint32),
        capacity=IDENTITY_ROPE_ROWS,
    )
    layer = F.constant(0, DType.uint32, device=CPU())
    latent = np.zeros((2, HEAD), np.float32)
    latent[0, 4 * 32] = 4.0  # key channel 32, below the rotated tail
    latent[1, 4 * 64] = 4.0  # key channel 64, inside it
    x = _dev(np.ones((2, 64), np.float32)).cast(DType.bfloat16)
    qr = np.zeros((2, 64), np.float32)
    qr[0, 0] = 1.0
    qr[1, 1] = 1.0
    rows = _ragged_rows(np.array([0, 1, 2]), np.array([4, 4], dtype=np.int64))

    picks, pool = indexer(
        x,
        _dev(qr).cast(DType.bfloat16),
        _dev(latent).cast(DType.bfloat16),
        _dev(np.ones(2, dtype=np.bool_)),
        rows,
        {"comp": comp_leaf, "idx": idx_leaf},
        {"comp": layer, "idx": layer},
    )
    assert pool is not None
    picks_np = picks.to_numpy()
    # Four picks per row over five visible entries: the match is each
    # row's only positive score, so it is always among them.
    assert 4 in picks_np[0]
    assert 4 in picks_np[1]
    # The candidate pool keeps the match's block (entry 4, blocks of 4).
    pool_np = pool.to_numpy()
    assert 1 in pool_np[0]
    assert 1 in pool_np[1]
    # Both stores ran through the wrapper's own slot lists.
    assert idx_blocks.to_numpy().any(), "the index-key store wrote nothing"
    assert comp_blocks.to_numpy().any(), "the compressed store wrote nothing"
