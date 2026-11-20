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
"""GLM-5.3-Flash's k-pooled DSA indexer against the pinned torch reference.

The indexer under test is the one the model runs, :class:`PooledIndexer`,
compiled with one sparse layer's paged FP8 pooled-key cache and k-pool tail
ring as its inputs. Every schedule leaves two things to compare against the
reference: the selection of each query row, and the pooled keys the cache
holds once every request has finished.

Every width is the checkpoint's --- ``index_head_dim`` 128, ``index_kpool``
4, ``index_n_heads`` 32, ``q_lora_rank`` 1536, ``hidden_size`` 4096 --- but
the budget is not. The checkpoint's ``index_topk`` of 2048 selects every pool
of any sequence shorter than 8192 tokens, so a top-k at that budget ranks
nothing; 512 over 1024 tokens leaves two pools competing for every slot.
Nothing about the pooling depends on the budget's value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn import Module
from max.nn._graph_ops import (
    mla_kpool_compress,
    mla_kpool_ring_close,
    mla_kpool_seed_tail,
)
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    MultiKVCacheParams,
    padded_lut_cols,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.architectures.glm5_next.layers.indexer import (
    KPoolTail,
    PooledIndexer,
)
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig,
)
from max.pipelines.architectures.glm5_next.state_cache import (
    TAIL_RING_DTYPE,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import BufferLayout, TensorLayout
from torch.utils.dlpack import from_dlpack
from torch_reference.glm5_next_indexer import (
    Glm5NextIndexerRefConfig,
    Glm5NextIndexerReference,
)

HIDDEN_SIZE = 4096
Q_LORA_RANK = 1536
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_KPOOL = 4
INDEX_TOPK = 512
SEQ_LEN = 1024
PAGE_SIZE = 128
SEED = 20260826

POOLS_SELECTED = INDEX_TOPK // INDEX_KPOOL
SELECTION_WIDTH = INDEX_TOPK + INDEX_KPOOL - 1
POOLED_HEAD_DIM = INDEX_HEAD_DIM // INDEX_KPOOL

# e4m3 carries three mantissa bits, so a value rounds by at most 2^-4 of
# itself, and the pool's absmax scale bounds that by 2^-4 of the largest
# member. The slack covers the bf16 rounding the two sides do differently
# before quantizing.
_POOLED_REL_ERROR = 0.07

# The reference runs on CPU: it shares the GPU with the MAX graph under test,
# and its peak tensor is the score matrix, a fraction of a second of CPU
# matmul at these shapes.
_REFERENCE_DEVICE = "cpu"


def _weights() -> dict[str, torch.Tensor]:
    """Random weights shared by both implementations.

    ``index_kpool_compress_ape`` is zero-initialized in the reference, so it is
    given a *non-zero* value here on purpose: a zero embedding cannot
    distinguish an implementation that reads it from one that drops it.
    """
    generator = torch.Generator().manual_seed(SEED)

    def randn(*shape: int, scale: float) -> torch.Tensor:
        return (
            torch.randn(*shape, generator=generator, dtype=torch.float32)
            * scale
        ).to(torch.bfloat16)

    return {
        "wq_b.weight": randn(
            INDEX_N_HEADS * INDEX_HEAD_DIM,
            Q_LORA_RANK,
            scale=1.0 / math.sqrt(Q_LORA_RANK),
        ),
        "wk.weight": randn(
            INDEX_HEAD_DIM, HIDDEN_SIZE, scale=1.0 / math.sqrt(HIDDEN_SIZE)
        ),
        "k_norm.weight": randn(INDEX_HEAD_DIM, scale=0.1) + 1.0,
        "k_norm.bias": randn(INDEX_HEAD_DIM, scale=0.05),
        "weights_proj.weight": randn(
            INDEX_N_HEADS, HIDDEN_SIZE, scale=1.0 / math.sqrt(HIDDEN_SIZE)
        ),
        "index_kpool_compress_gate.weight": randn(
            INDEX_HEAD_DIM, HIDDEN_SIZE, scale=1.0 / math.sqrt(HIDDEN_SIZE)
        ),
        "index_kpool_compress_ape": randn(
            INDEX_KPOOL, INDEX_HEAD_DIM, scale=0.5
        ),
    }


def _activations(total_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED + 1)
    x = (
        torch.randn(
            total_tokens, HIDDEN_SIZE, generator=generator, dtype=torch.float32
        )
        / math.sqrt(HIDDEN_SIZE)
    ).to(torch.bfloat16)
    qr = (
        torch.randn(
            total_tokens, Q_LORA_RANK, generator=generator, dtype=torch.float32
        )
        / math.sqrt(Q_LORA_RANK)
    ).to(torch.bfloat16)
    return x, qr


def _bf16_to_device(tensor: torch.Tensor, device: Accelerator) -> Buffer:
    return (
        Buffer.from_numpy(tensor.contiguous().view(torch.float16).numpy())
        .view(DType.bfloat16)
        .to(device)
    )


# ------------------------------------------------------------------ reference


@dataclass(frozen=True)
class _Reference:
    """One sequence through the reference, scored and pooled whole."""

    selection: torch.Tensor
    """``[seq_len, SELECTION_WIDTH]`` int32 positions, ``-1`` unused."""
    pooled: torch.Tensor
    """``[seq_len // INDEX_KPOOL, INDEX_HEAD_DIM]`` float32 keys of the
    sequence's complete pools."""


def _reference(
    weights: dict[str, torch.Tensor], x: torch.Tensor, qr: torch.Tensor
) -> _Reference:
    module = Glm5NextIndexerReference(
        Glm5NextIndexerRefConfig(
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            index_n_heads=INDEX_N_HEADS,
            index_head_dim=INDEX_HEAD_DIM,
            index_topk=INDEX_TOPK,
            index_kpool=INDEX_KPOOL,
        )
    )
    module.load_state_dict(
        {
            "wq_b.weight": weights["wq_b.weight"],
            "wk.weight": weights["wk.weight"],
            "k_norm.weight": weights["k_norm.weight"],
            "k_norm.bias": weights["k_norm.bias"],
            "weights_proj.weight": weights["weights_proj.weight"],
            "index_kpool_compress_gate": weights[
                "index_kpool_compress_gate.weight"
            ],
            "index_kpool_compress_ape": weights["index_kpool_compress_ape"],
        },
        strict=True,
    )
    module = module.to(device=_REFERENCE_DEVICE, dtype=torch.bfloat16).eval()
    seq_len = x.shape[0]
    x, qr = x[None].to(_REFERENCE_DEVICE), qr[None].to(_REFERENCE_DEVICE)
    selection = module(
        x, qr, torch.ones(1, seq_len, dtype=torch.bool, device=x.device)
    )[0]
    with torch.no_grad():
        packed = torch.cat(
            [
                module.k_norm(module.wk(x)),
                torch.nn.functional.linear(x, module.index_kpool_compress_gate),
                torch.ones(1, seq_len, 1, dtype=torch.bfloat16),
            ],
            dim=-1,
        )
        pooled, _, valid = module.get_pooled_states(packed)
    complete = seq_len // INDEX_KPOOL
    assert int(valid.sum()) == complete, (
        "the reference should mark exactly the complete pools valid"
    )
    return _Reference(
        selection.cpu(), pooled[0, :complete].to(torch.float32).cpu()
    )


# ------------------------------------------------------------------------ MAX


def _config() -> Glm5NextConfig:
    devices = [DeviceRef.GPU()]
    mla = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=devices,
        num_q_heads=8,
    )
    return Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": mla, "indexer": _cache_params()}
        ),
        devices=devices,
        max_seq_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        q_lora_rank=Q_LORA_RANK,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        index_kpool=INDEX_KPOOL,
    )


def _cache_params() -> MLAKVCacheParams:
    """The indexer leaf ``hybrid_layout.construct_kv_params`` declares.

    One pooled key spans ``index_kpool`` FP8 rows of ``index_head_dim /
    index_kpool`` values, with one float32 scale per row.
    """
    return MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=POOLED_HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=INDEX_N_HEADS,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=POOLED_HEAD_DIM,
        ),
    )


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


class _SparseLayerIndexer(Module[..., Tensor]):
    """The indexer as ``Glm5NextTextModel`` calls it for one sparse layer."""

    def __init__(self) -> None:
        self.indexer = PooledIndexer(_config(), _quant_config())

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        input_row_offsets: Tensor,
        tail_pool: Tensor,
        tail_rows: Tensor,
        kv_blocks: Tensor,
        kv_scales: Tensor,
        cache_lengths: Tensor,
        lookup_table: Tensor,
        max_prompt_length: Tensor,
        max_cache_length: Tensor,
        page_stride: Tensor,
    ) -> Tensor:
        # The scales share the values' pages, as every non-pooled-page cache
        # does, so they share the stride and the lookup table too.
        cache = PagedCacheValues(
            kv_blocks=kv_blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            page_stride=page_stride,
            kv_scales=kv_scales,
            scales_page_stride=page_stride,
            scales_lookup_table=lookup_table,
        )
        return self.indexer(
            x,
            qr,
            input_row_offsets,
            cache,
            F.constant(0, DType.uint32, device=CPU()),
            KPoolTail(
                pool=tail_pool,
                rows=tail_rows,
                layer=F.constant(0, DType.int64, device=CPU()),
            ),
        )


# Ring slots by request: neither the identity nor dense, so a ring addressed
# by batch row rather than by slot lands on another request's pool.
_RING_SLOTS = (3, 0)
_RING_ROWS = 4

_Indexer = CompiledCallable[..., Tensor]


def _compile(weights: dict[str, torch.Tensor]) -> _Indexer:
    device = Accelerator()
    params = _cache_params()
    with default_device(device), default_dtype(DType.bfloat16):
        layer = _SparseLayerIndexer()
    return layer.compile(
        TensorLayout(DType.bfloat16, ["total_tokens", HIDDEN_SIZE], device),
        TensorLayout(DType.bfloat16, ["total_tokens", Q_LORA_RANK], device),
        TensorLayout(DType.uint32, ["batch_plus_one"], device),
        BufferLayout(
            TAIL_RING_DTYPE,
            [_RING_ROWS, 2, INDEX_KPOOL, INDEX_HEAD_DIM],
            device,
        ),
        TensorLayout(DType.uint32, [1, "batch"], device),
        BufferLayout(params.dtype, ["pages", *params.shape_per_block], device),
        BufferLayout(
            DType.float32, ["pages", *params.shape_per_scale_block], device
        ),
        TensorLayout(DType.uint32, ["batch"], device),
        TensorLayout(DType.uint32, ["batch", "lut_cols"], device),
        TensorLayout(DType.uint32, [1], CPU()),
        TensorLayout(DType.uint32, [1], CPU()),
        TensorLayout(DType.int64, [1], CPU()),
        weights={f"indexer.{name}": value for name, value in weights.items()},
    )


@pytest.fixture(scope="module")
def indexer() -> _Indexer:
    return _compile(_weights())


def _host(array: np.ndarray) -> Tensor:
    return Tensor(storage=Buffer.from_numpy(np.ascontiguousarray(array)))


def _device(array: np.ndarray, device: Accelerator) -> Tensor:
    return Tensor(
        storage=Buffer.from_numpy(np.ascontiguousarray(array)).to(device)
    )


@dataclass(frozen=True)
class _Served:
    """What a schedule left behind, per request."""

    selections: list[torch.Tensor]
    """``[length, SELECTION_WIDTH]`` int32, each row from the call that
    brought its token."""
    pooled: list[torch.Tensor]
    """``[pools spanned, INDEX_HEAD_DIM]`` float32 dequantized cache rows,
    one per pool the request's tokens reach, complete or not."""


def _batch_rows(
    per_request: Sequence[torch.Tensor],
    spans: Sequence[tuple[int, range]],
    device: Accelerator,
) -> Tensor:
    """Packs each active request's rows for this call, in batch order."""
    return Tensor(
        storage=_bf16_to_device(
            torch.cat(
                [
                    per_request[request][span.start : span.stop]
                    for request, span in spans
                ]
            ),
            device,
        )
    )


def _serve(
    indexer: _Indexer,
    xs: Sequence[torch.Tensor],
    qrs: Sequence[torch.Tensor],
    schedule: Sequence[tuple[int, ...]],
) -> _Served:
    """Runs the requests through ``schedule``, a token count per request per
    call, against one cache and one ring that persist across calls.

    A request is in a call's batch only when it brings tokens, so its batch
    row moves between calls while its pages and its ring slot stay put.
    """
    device = Accelerator()
    params = _cache_params()
    lengths = [x.shape[0] for x in xs]
    pages = [math.ceil(length / PAGE_SIZE) for length in lengths]
    first_page = np.concatenate([[0], np.cumsum(pages)])
    null_page = int(first_page[-1])
    lut = np.full(
        (len(lengths), padded_lut_cols(max(pages))), null_page, dtype=np.uint32
    )
    for request, count in enumerate(pages):
        lut[request, :count] = np.arange(
            first_page[request], first_page[request] + count
        )

    # One page past the requests' backs the null page the padding names.
    blocks = Buffer.zeros(
        shape=[null_page + 1, *params.shape_per_block],
        dtype=params.dtype,
        device=device,
    )
    scales = Buffer.zeros(
        shape=[null_page + 1, *params.shape_per_scale_block],
        dtype=DType.float32,
        device=device,
    )
    ring = Buffer.zeros(
        shape=[_RING_ROWS, 2, INDEX_KPOOL, INDEX_HEAD_DIM],
        dtype=TAIL_RING_DTYPE,
        device=device,
    )
    blocks_t, scales_t, ring_t = (
        Tensor(storage=b) for b in (blocks, scales, ring)
    )
    page_stride = _host(np.array([-1], dtype=np.int64))

    selections = [
        torch.full((length, SELECTION_WIDTH), -2, dtype=torch.int32)
        for length in lengths
    ]
    positions = [0] * len(lengths)
    for counts in schedule:
        spans = [
            (request, range(positions[request], positions[request] + count))
            for request, count in enumerate(counts)
            if count
        ]
        active = [request for request, _ in spans]
        offsets = np.concatenate(
            [[0], np.cumsum([len(span) for _, span in spans])]
        ).astype(np.uint32)
        result = indexer(
            _batch_rows(xs, spans, device),
            _batch_rows(qrs, spans, device),
            _device(offsets, device),
            ring_t,
            _device(
                np.array([[_RING_SLOTS[r] for r in active]], dtype=np.uint32),
                device,
            ),
            blocks_t,
            scales_t,
            _device(
                np.array([span.start for _, span in spans], dtype=np.uint32),
                device,
            ),
            _device(lut[active], device),
            _host(np.array([max(counts)], dtype=np.uint32)),
            _host(
                np.array([max(span.stop for _, span in spans)], dtype=np.uint32)
            ),
            page_stride,
        )
        rows = torch.from_numpy(result.to_numpy().copy())
        for i, (request, span) in enumerate(spans):
            selections[request][span.start : span.stop] = rows[
                offsets[i] : offsets[i + 1]
            ]
            positions[request] = span.stop

    assert positions == lengths, "the schedule must finish every request"
    return _Served(selections, _read_pooled(blocks, scales, lut, lengths))


def _read_pooled(
    blocks: Buffer, scales: Buffer, lut: np.ndarray, lengths: Sequence[int]
) -> list[torch.Tensor]:
    """Dequantizes each request's pooled keys out of its pages.

    Pool ``p`` sits at the request's token slots ``[p * kpool, (p + 1) *
    kpool)``, one FP8 row of ``POOLED_HEAD_DIM`` per slot, each row with the
    pool's scale.
    """
    values = (
        torch.from_numpy(blocks.view(DType.uint8).to(CPU()).to_numpy())
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    row_scales = torch.from_numpy(scales.to(CPU()).to_numpy())
    pooled = []
    for request, length in enumerate(lengths):
        slots = torch.arange(math.ceil(length / INDEX_KPOOL) * INDEX_KPOOL)
        page = torch.from_numpy(lut[request].astype(np.int64))[
            slots // PAGE_SIZE
        ]
        offset = slots % PAGE_SIZE
        rows = values[page, 0, 0, offset, 0] * row_scales[page, 0, 0, offset, 0]
        scale = row_scales[page, 0, 0, offset, 0, 0].reshape(-1, INDEX_KPOOL)
        assert torch.equal(scale, scale[:, :1].expand_as(scale)), (
            f"request {request}: a pool's scale differs between its rows"
        )
        pooled.append(rows.reshape(-1, INDEX_HEAD_DIM))
    return pooled


# ------------------------------------------------------------------- helpers


def _index_sets(indices: torch.Tensor) -> list[set[int]]:
    return [{int(v) for v in row.tolist() if v >= 0} for row in indices]


@dataclass(frozen=True)
class Agreement:
    """How closely two selections agree, as sets of token positions.

    ``whole_pools_only`` is the load-bearing field, not the exact-row fraction.
    Once a query's budget saturates, two pools at the top-k boundary can be
    separated by less than the FP8 scorer's rounding step, so which one wins
    is arbitrary. What is *not* arbitrary is the granularity of the
    disagreement: a tie swaps whole aligned pools, so every position in the
    symmetric difference comes with its ``index_kpool - 1`` siblings. A bug in
    the expansion, the masking or the tail breaks that alignment, which is why
    it is the assertion rather than a threshold on how many rows differ.
    """

    exact_rows: float
    mean_jaccard: float
    first_mismatch: int
    worst_pool_swaps: int
    whole_pools_only: bool

    def __str__(self) -> str:
        return (
            f"exact rows={self.exact_rows:.4f} "
            f"mean Jaccard={self.mean_jaccard:.6f} "
            f"first mismatch row={self.first_mismatch} "
            f"worst pools swapped={self.worst_pool_swaps} "
            f"whole pools only={self.whole_pools_only}"
        )


def _set_agreement(lhs: torch.Tensor, rhs: torch.Tensor) -> Agreement:
    left, right = _index_sets(lhs), _index_sets(rhs)
    assert len(left) == len(right)
    exact = 0
    jaccard = 0.0
    first_bad = -1
    worst = 0
    aligned = True
    for row, (a, b) in enumerate(zip(left, right, strict=True)):
        if a == b:
            exact += 1
        elif first_bad < 0:
            first_bad = row
        union = a | b
        jaccard += 1.0 if not union else len(a & b) / len(union)
        difference = a ^ b
        worst = max(worst, len(difference) // INDEX_KPOOL)
        for position in difference:
            pool = position // INDEX_KPOOL
            members = {
                pool * INDEX_KPOOL + offset for offset in range(INDEX_KPOOL)
            }
            if not members <= difference:
                aligned = False
    return Agreement(
        exact / len(left), jaccard / len(left), first_bad, worst, aligned
    )


def _pooled_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The worst pool's error, relative to that pool's largest value."""
    error = (actual - expected).abs().amax(dim=-1)
    return float((error / expected.abs().amax(dim=-1)).max())


def _assert_matches_reference(
    served: _Served, references: Sequence[_Reference], label: str
) -> None:
    for request, reference in enumerate(references):
        rows = served.selections[request]
        length = rows.shape[0]
        name = f"[{label}] request {request} (length {length})"

        for position, row in enumerate(rows.tolist()):
            picked = [v for v in row if v >= 0]
            assert len(picked) == len(set(picked)), (
                f"{name} position {position} selected a position twice"
            )
            assert max(picked) <= position, (
                f"{name} position {position} selected a future token"
            )
            start = (position + 1) // INDEX_KPOOL * INDEX_KPOOL
            assert set(range(start, position + 1)) <= set(picked), (
                f"{name} position {position} dropped its incomplete pool"
            )

        agreement = _set_agreement(rows, reference.selection)
        print(f"{name} selection: {agreement}")
        # Until a query sees more complete pools than it may keep, it keeps
        # all of them, so a score cannot reorder anything and any
        # disagreement is a logic bug.
        unsaturated = min(POOLS_SELECTED * INDEX_KPOOL + INDEX_KPOOL, length)
        assert (
            _set_agreement(
                rows[:unsaturated], reference.selection[:unsaturated]
            ).exact_rows
            == 1.0
        ), f"{name}: an unsaturated row differs from the reference"
        assert agreement.whole_pools_only
        assert agreement.mean_jaccard >= 0.98

        complete = length // INDEX_KPOOL
        pooled = served.pooled[request]
        error = _pooled_error(pooled[:complete], reference.pooled)
        cos = torch.nn.functional.cosine_similarity(
            pooled[:complete].reshape(-1), reference.pooled.reshape(-1), dim=0
        ).item()
        print(
            f"{name} pooled cache: worst pool error={error:.4f} of its "
            f"absmax, cos={cos:.6f}"
        )
        assert error <= _POOLED_REL_ERROR
        assert not pooled[complete:].any(), (
            f"{name}: an incomplete pool reached the cache"
        )


# --------------------------------------------------------------------- tests

_LENGTHS = (SEQ_LEN, SEQ_LEN - 3)


@pytest.fixture(scope="module")
def requests() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    x, qr = _activations(sum(_LENGTHS))
    bounds = list(pairwise([0, *np.cumsum(_LENGTHS).tolist()]))
    return [x[a:b] for a, b in bounds], [qr[a:b] for a, b in bounds]


@pytest.fixture(scope="module")
def references(
    requests: tuple[list[torch.Tensor], list[torch.Tensor]],
) -> list[_Reference]:
    weights = _weights()
    return [_reference(weights, x, qr) for x, qr in zip(*requests, strict=True)]


def _decode_rest(
    prefix: Sequence[tuple[int, ...]], lengths: Sequence[int]
) -> list[tuple[int, ...]]:
    """``prefix``, then one token per request per call until all are done."""
    remaining = [
        length - sum(call[request] for call in prefix)
        for request, length in enumerate(lengths)
    ]
    calls = list(prefix)
    while any(remaining):
        calls.append(tuple(1 if r else 0 for r in remaining))
        remaining = [max(r - 1, 0) for r in remaining]
    return calls


_SCHEDULES: dict[str, list[tuple[int, ...]]] = {
    "one prefill": [_LENGTHS],
    # The second chunk of the first request starts mid-pool and both
    # requests decode across pool boundaries, so pools close from the ring
    # while the batch mixes a multi-token request with a decode one.
    "chunked prefill and decode": _decode_rest(
        [(509, 1013), (510, 1)], _LENGTHS
    ),
}


@pytest.mark.parametrize("schedule", list(_SCHEDULES), ids=list(_SCHEDULES))
def test_indexer_matches_reference(
    indexer: _Indexer,
    requests: tuple[list[torch.Tensor], list[torch.Tensor]],
    references: list[_Reference],
    schedule: str,
) -> None:
    """Selections and the pooled cache, however the tokens arrive.

    The cache check holds the pool compression to the reference to within FP8
    rounding: the channel-wise softmax over the four members, and the ape
    added to the gate logits before it. Get either wrong and long-context
    retrieval degrades without anything crashing.
    """
    xs, qrs = requests
    served = _serve(indexer, xs, qrs, _SCHEDULES[schedule])
    _assert_matches_reference(served, references, schedule)


def test_ape_is_load_bearing(
    requests: tuple[list[torch.Tensor], list[torch.Tensor]],
    references: list[_Reference],
) -> None:
    """Without the ape, the pooled cache misses the reference's tolerance.

    The checkpoint ships ``index_kpool_compress_ape`` zero-initialized, which
    is exactly why an implementation that silently drops it passes every
    smoke test; this shows the cache check above would catch one.
    """
    weights = _weights()
    weights["index_kpool_compress_ape"] = torch.zeros_like(
        weights["index_kpool_compress_ape"]
    )
    xs, qrs = requests
    served = _serve(_compile(weights), xs[:1], qrs[:1], [(SEQ_LEN,)])
    error = _pooled_error(served.pooled[0], references[0].pooled)
    print(f"ape ablation: worst pool error={error:.4f} of its absmax")
    assert error > 2 * _POOLED_REL_ERROR, "the ape is not reaching the pools"


# --------------------------------------- tail ring versus one-shot compression

# Short and ragged: this differential is about *when* a pool's members arrive,
# not how many there are. The slots are deliberately neither the identity nor
# dense, so a ring addressed by batch row rather than by ring slot lands on
# another request's in-progress pool. No length is a multiple of
# ``index_kpool``, so every request also ends mid-pool.
_TAIL_LENGTHS = (13, 9, 11)
_TAIL_SLOTS = (3, 0, 5)
_TAIL_MAX_SLOTS = 6
_TAIL_RING_SHAPE = [_TAIL_MAX_SLOTS, 2, INDEX_KPOOL, INDEX_HEAD_DIM]


@dataclass(frozen=True)
class _TailRig:
    """All the pooled-key writers, compiled once, over one set of operands.

    ``general_model`` (``ring_close`` + ``compress`` + ``seed_tail``, run
    unconditionally, every call) is what :class:`PooledIndexer` runs, for
    prefill, decode and any mix of the two in one call. ``oracle_model`` is the
    independent one-shot compression it is judged against.
    """

    device: Accelerator
    general_model: Model
    oracle_model: Model
    keys: torch.Tensor
    gate: torch.Tensor
    ape: np.ndarray


@pytest.fixture(scope="module")
def tail_rig() -> _TailRig:
    """Compiles the tail-ring writers and the one-shot oracle writer.

    All three take the keys and the gate scores as *inputs* rather than
    projecting them from hidden states. Left to each graph, ``wk`` and the
    gate round differently at different row counts, and a decode arm that
    feeds one token at a time would be compared against an oracle whose
    operands differ in the last bit --- which measures the matmul, not the
    pooling.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])

    rows_type = TensorType(
        DType.bfloat16, ["rows", INDEX_HEAD_DIM], DeviceRef.GPU()
    )
    ape_type = TensorType(
        DType.float32, [INDEX_KPOOL, INDEX_HEAD_DIM], DeviceRef.GPU()
    )
    offsets_type = TensorType(DType.uint32, ["batch_plus_one"], DeviceRef.GPU())
    batch_type = TensorType(DType.uint32, ["batch"], DeviceRef.GPU())
    tail_buffer_type = BufferType(
        DType.bfloat16, _TAIL_RING_SHAPE, DeviceRef.GPU()
    )

    # One-shot oracle: `compress` alone, from an empty cache, no ring.
    with Graph(
        "kpool_compress_oracle",
        input_types=(
            rows_type,
            rows_type,
            ape_type,
            offsets_type,
            offsets_type,
            batch_type,
        ),
    ) as compress_graph:
        compress_graph.output(
            mla_kpool_compress(
                compress_graph.inputs[0].tensor,
                compress_graph.inputs[1].tensor,
                compress_graph.inputs[2].tensor,
                compress_graph.inputs[3].tensor,
                compress_graph.inputs[4].tensor,
                compress_graph.inputs[5].tensor,
                INDEX_KPOOL,
            )
        )

    # `ring_close` + `compress` + `seed_tail`, unconditionally: what
    # `PooledIndexer` always runs, regardless of alignment or
    # per-request token count. `pool_row_offsets` is supplied rather than
    # computed here, mirroring `select_pooled`'s own arithmetic, which the
    # caller (`_run_general_writer`) reproduces on the host.
    with Graph(
        "kpool_general",
        input_types=(
            rows_type,  # keys
            rows_type,  # gate
            ape_type,
            offsets_type,  # input_row_offsets
            offsets_type,  # pool_row_offsets
            batch_type,  # cache_lengths
            batch_type,  # slot_idx
            tail_buffer_type,
        ),
    ) as general_graph:
        k_in = general_graph.inputs[0].tensor
        gate_in = general_graph.inputs[1].tensor
        iro_in = general_graph.inputs[3].tensor
        clen_in = general_graph.inputs[5].tensor
        ring_pooled, ring_closed_pool = mla_kpool_ring_close(
            general_graph.inputs[7].buffer,
            k_in,
            gate_in,
            general_graph.inputs[2].tensor,
            iro_in,
            clen_in,
            general_graph.inputs[6].tensor,
            INDEX_KPOOL,
        )
        compress_pooled = mla_kpool_compress(
            k_in,
            gate_in,
            general_graph.inputs[2].tensor,
            iro_in,
            general_graph.inputs[4].tensor,
            clen_in,
            INDEX_KPOOL,
        )
        # `seed_tail` last: it reuses ring slots `ring_close` just read, and
        # correctness depends on `ring_close` observing them first -- see
        # `select_pooled`'s comment on this same ordering.
        mla_kpool_seed_tail(
            general_graph.inputs[7].buffer,
            k_in,
            gate_in,
            iro_in,
            clen_in,
            general_graph.inputs[6].tensor,
            INDEX_KPOOL,
        )
        general_graph.output(ring_pooled, ring_closed_pool, compress_pooled)

    generator = torch.Generator().manual_seed(SEED + 8)
    total = sum(_TAIL_LENGTHS)
    keys_host = torch.randn(
        total, INDEX_HEAD_DIM, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    # Gate logits and the position table span several units so the pool softmax
    # actually discriminates between members: a near-uniform softmax weights
    # every member the same however the writer reached them, and would agree
    # with one that mixed them up.
    gate_host = (
        torch.randn(
            total, INDEX_HEAD_DIM, generator=generator, dtype=torch.float32
        )
        * 4.0
    ).to(torch.bfloat16)
    ape_host = (
        torch.randn(
            INDEX_KPOOL,
            INDEX_HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        * 2.0
    )

    return _TailRig(
        device=device,
        general_model=session.load(general_graph),
        oracle_model=session.load(compress_graph),
        keys=keys_host,
        gate=gate_host,
        ape=ape_host.numpy(),
    )


def _tail_schedule(prefill: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Token counts per request per call: one prefill chunk, then one a step.

    ``prefill`` equal to the lengths gives a single call; all zeros gives pure
    single-token decode. A request drops out of the batch when it is done,
    which is what a real batch does and what makes the ring's slot -- rather
    than the row -- the only stable address.
    """
    remaining = [n - p for n, p in zip(_TAIL_LENGTHS, prefill, strict=True)]
    calls = [list(prefill)]
    while any(remaining):
        calls.append([1 if r else 0 for r in remaining])
        remaining = [max(r - 1, 0) for r in remaining]
    return [tuple(call) for call in calls if any(call)]


def _run_general_writer(
    rig: _TailRig, schedule: Sequence[tuple[int, ...]]
) -> torch.Tensor:
    """Runs ``ring_close`` + ``compress`` + ``seed_tail`` on an explicit,
    caller-chosen schedule, on every call, as :class:`PooledIndexer` does.

    ``schedule`` is not tied to a prefill/decode split: each call can bring any
    per-request token count, so one call can hold requests at different points
    in their pool (some mid-pool from a previous call, some fresh) that bring
    different numbers of new tokens.

    Args:
        rig: The compiled writers and shared operands.
        schedule: Token counts per request per call. A zero entry marks that
            request inactive for that call. ``sum`` across calls, per
            request, must equal that request's entry in ``_TAIL_LENGTHS``.

    Returns:
        ``[batch, max_pools, index_head_dim]`` bf16, one row per pool this
        schedule closes for that request.
    """
    max_pools = max(_TAIL_LENGTHS) // INDEX_KPOOL
    pooled_out = torch.zeros(
        len(_TAIL_LENGTHS), max_pools, INDEX_HEAD_DIM, dtype=torch.bfloat16
    )
    closes = np.zeros((len(_TAIL_LENGTHS), max_pools), dtype=np.int32)

    ring = _bf16_to_device(
        torch.zeros(_TAIL_RING_SHAPE, dtype=torch.bfloat16), rig.device
    )
    ape = Buffer.from_numpy(rig.ape).to(rig.device)
    starts = np.concatenate([[0], np.cumsum(_TAIL_LENGTHS)])
    positions = [0] * len(_TAIL_LENGTHS)

    for counts in schedule:
        active = [b for b, count in enumerate(counts) if count]
        rows = np.concatenate(
            [
                np.arange(
                    starts[b] + positions[b],
                    starts[b] + positions[b] + counts[b],
                )
                for b in active
            ]
        )
        cache_lengths = np.array(
            [positions[b] for b in active], dtype=np.uint32
        )
        slot_idx = np.array([_TAIL_SLOTS[b] for b in active], dtype=np.uint32)
        n = np.array([counts[b] for b in active], dtype=np.int64)

        offsets = np.zeros(len(active) + 1, dtype=np.uint32)
        offsets[1:] = np.cumsum(n)

        # Same arithmetic as `select_pooled`: pools this call
        # completes from new tokens alone, excluding whatever `ring_close`
        # closes -- floor of the post-call position minus ceil of the
        # pre-call one.
        cache_len64 = cache_lengths.astype(np.int64)
        full_pools = np.maximum(
            (cache_len64 + n) // INDEX_KPOOL
            - (cache_len64 + INDEX_KPOOL - 1) // INDEX_KPOOL,
            0,
        )
        pool_row_offsets = np.zeros(len(active) + 1, dtype=np.uint32)
        pool_row_offsets[1:] = np.cumsum(full_pools)

        ring_pooled, ring_closed, compress_pooled = rig.general_model.execute(
            _bf16_to_device(rig.keys[rows], rig.device),
            _bf16_to_device(rig.gate[rows], rig.device),
            ape,
            Buffer.from_numpy(offsets).to(rig.device),
            Buffer.from_numpy(pool_row_offsets).to(rig.device),
            Buffer.from_numpy(cache_lengths).to(rig.device),
            Buffer.from_numpy(slot_idx).to(rig.device),
            ring,
        )
        ring_pooled_host = from_dlpack(ring_pooled.to(CPU()))
        ring_closed_host = from_dlpack(ring_closed.to(CPU()))
        compress_pooled_host = from_dlpack(compress_pooled.to(CPU()))

        for i, b in enumerate(active):
            closed = int(ring_closed_host[i])
            if closed >= 0:
                pooled_out[b, closed] = ring_pooled_host[i]
                closes[b, closed] += 1

            # The interior pools `compress` built start right after whatever
            # `ring_close` just closed (or at the request's own cache
            # boundary, when nothing was pending): `(cache_len + kpool - 1)
            # // kpool` is that boundary either way.
            base_pool = int(cache_lengths[i]) + INDEX_KPOOL - 1
            base_pool //= INDEX_KPOOL
            for p in range(int(full_pools[i])):
                pooled_out[b, base_pool + p] = compress_pooled_host[
                    int(pool_row_offsets[i]) + p
                ]
                closes[b, base_pool + p] += 1

        for b in active:
            positions[b] += counts[b]

    for b, length in enumerate(_TAIL_LENGTHS):
        expected = np.zeros(max_pools, dtype=np.int32)
        expected[: length // INDEX_KPOOL] = 1
        assert np.array_equal(closes[b], expected), (
            f"request {b} closed pools {closes[b].tolist()}, expected "
            f"{expected.tolist()}"
        )
    return pooled_out


def _run_compress_oracle(rig: _TailRig) -> torch.Tensor:
    """The whole batch through ``mla_kpool_compress`` in one shot, from an
    empty cache."""
    offsets = np.zeros(len(_TAIL_LENGTHS) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(_TAIL_LENGTHS)
    pool_offsets = np.zeros(len(_TAIL_LENGTHS) + 1, dtype=np.uint32)
    pool_offsets[1:] = np.cumsum(
        [length // INDEX_KPOOL for length in _TAIL_LENGTHS]
    )
    cache_lengths = np.zeros(len(_TAIL_LENGTHS), dtype=np.uint32)
    (pooled,) = rig.oracle_model.execute(
        _bf16_to_device(rig.keys, rig.device),
        _bf16_to_device(rig.gate, rig.device),
        Buffer.from_numpy(rig.ape).to(rig.device),
        Buffer.from_numpy(offsets).to(rig.device),
        Buffer.from_numpy(pool_offsets).to(rig.device),
        Buffer.from_numpy(cache_lengths).to(rig.device),
    )
    flat = from_dlpack(pooled.to(CPU()))

    max_pools = max(_TAIL_LENGTHS) // INDEX_KPOOL
    expected = torch.zeros(
        len(_TAIL_LENGTHS), max_pools, INDEX_HEAD_DIM, dtype=torch.bfloat16
    )
    for b, length in enumerate(_TAIL_LENGTHS):
        pools = length // INDEX_KPOOL
        expected[b, :pools] = flat[
            int(pool_offsets[b]) : int(pool_offsets[b]) + pools
        ]
    return expected


def _assert_bit_identical(
    actual: torch.Tensor, expected: torch.Tensor, label: str
) -> None:
    """Bit for bit, not within a tolerance.

    Both writers run the same shared ``pool_channel`` over the same operands,
    so any difference at all is a difference in *which* members were gathered
    rather than a rounding difference --- and a tolerance would hide exactly
    the off-by-one member these tests look for.
    """
    mismatches = int(
        (actual.view(torch.uint16) != expected.view(torch.uint16)).sum().item()
    )
    detail = ""
    if mismatches:
        b, pool, channel = (
            int(v) for v in (actual != expected).nonzero()[0].tolist()
        )
        detail = (
            f"; first at request {b} pool {pool} channel {channel}: "
            f"{float(actual[b, pool, channel])} vs "
            f"{float(expected[b, pool, channel])}"
        )
    print(
        f"[{label}] mismatching bf16 values="
        f"{mismatches}/{actual.numel()}{detail}"
    )
    assert mismatches == 0, (
        f"{label}: the tail ring and the one-shot compression disagree{detail}"
    )


def test_one_shot_compression_matches_reference(tail_rig: _TailRig) -> None:
    """The oracle the writers are judged against, against the reference.

    ``mla_kpool_compress`` is the pooling every writer shares, so this holds
    it to the reference's ``get_pooled_states`` in bf16, finer than the FP8
    cache check can resolve.
    """
    actual = _run_compress_oracle(tail_rig)
    reference = Glm5NextIndexerReference(
        Glm5NextIndexerRefConfig(
            index_head_dim=INDEX_HEAD_DIM, index_kpool=INDEX_KPOOL
        )
    )
    expected = torch.zeros_like(actual)
    start = 0
    with torch.no_grad():
        reference.index_kpool_compress_ape.copy_(torch.from_numpy(tail_rig.ape))
        for request, length in enumerate(_TAIL_LENGTHS):
            rows = slice(start, start + length)
            start += length
            packed = torch.cat(
                [
                    tail_rig.keys[rows],
                    tail_rig.gate[rows],
                    torch.ones(length, 1, dtype=torch.bfloat16),
                ],
                dim=-1,
            )
            pooled, _, _ = reference.get_pooled_states(packed[None])
            complete = length // INDEX_KPOOL
            expected[request, :complete] = pooled[0, :complete]

    actual_f, expected_f = actual.to(torch.float32), expected.to(torch.float32)
    max_abs = (actual_f - expected_f).abs().max().item()
    # One bf16 ULP at the largest value present is 2^(exponent - 7). Both
    # sides round each of the four weighted terms, so the bound is in ULPs
    # rather than picked; dropping a member's weight or the ape moves values
    # by whole percent.
    largest = expected_f.abs().max().item()
    ulp = 2.0 ** (math.floor(math.log2(largest)) - 7)
    print(
        f"one-shot compression vs reference: max|diff|={max_abs:.3e} "
        f"({max_abs / ulp:.1f} bf16 ULP at |max|={largest:.3f})"
    )
    assert max_abs <= 4 * ulp


def test_decode_pools_match_one_shot_compression(tail_rig: _TailRig) -> None:
    """Pools closed one decode token at a time match one-shot compression.

    A pool whose members arrive on ``index_kpool`` separate decode steps has to
    come out identical to one compressed from all its members at once ---
    otherwise the cache the scorer reads during decode is not the cache a
    prefill would have written, and nothing downstream would notice.
    """
    _assert_bit_identical(
        _run_general_writer(tail_rig, _tail_schedule((0, 0, 0))),
        _run_compress_oracle(tail_rig),
        "one token per step",
    )


def test_prefill_pools_match_one_shot_compression(tail_rig: _TailRig) -> None:
    """The same writer with every token of a request in one call.

    This is the production prefill shape: the ring is seeded and several of a
    request's pools close in the same launch, so the rows sharing a ring slot
    are in flight together.
    """
    _assert_bit_identical(
        _run_general_writer(tail_rig, _tail_schedule(_TAIL_LENGTHS)),
        _run_compress_oracle(tail_rig),
        "single prefill call",
    )


def test_chunked_prefill_pools_match_one_shot_compression(
    tail_rig: _TailRig,
) -> None:
    """A prefill that stops mid-pool, then decode.

    The chunk boundaries are deliberately off the pool boundaries, so the first
    pool each request closes after the chunk spans both calls --- the case a
    writer that pooled only within one call cannot get right.
    """
    _assert_bit_identical(
        _run_general_writer(tail_rig, _tail_schedule((6, 3, 9))),
        _run_compress_oracle(tail_rig),
        "chunked prefill then decode",
    )


def test_mixed_ragged_batch_matches_one_shot_compression(
    tail_rig: _TailRig,
) -> None:
    """A ragged call whose requests differ in pool remainder and token count.

    Every request enters the second call holding a different remainder mod
    ``index_kpool`` in the ring and brings a different multi-token count, none
    of them ``1``, so the writer must address each request's ring slot and
    token range on its own. A writer that chose one path for the whole batch,
    or that took one new token per request, reads and writes out of bounds on
    a call shaped like this.
    """
    _assert_bit_identical(
        _run_general_writer(tail_rig, [(6, 3, 9), (7, 6, 2)]),
        _run_compress_oracle(tail_rig),
        "mixed ragged batch",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-q"]))
