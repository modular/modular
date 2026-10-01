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
"""GLM-5.3-Flash k-pooled DSA indexer against the pinned torch reference.

Every dimension the pooling works in is held at its real checkpoint value ---
``index_head_dim`` 128, ``index_kpool`` 4, ``index_n_heads`` 32,
``q_lora_rank`` 1536, ``hidden_size`` 4096. Only counts move, and they move
along two axes at once because the two properties worth testing pull against
each other:

* the real ``index_topk`` = 2048 is what produces the 2051-wide selection, and
  it needs at least 2048 tokens before the indexer selects anything at all;
* real selection *pressure* needs many more pools than the 512 a query keeps,
  and the dense scorer's ``[tokens, heads, pools]`` tensor grows as that
  product --- past a few hundred megabytes MAX's intermediate allocator refuses
  it.

So :data:`CASES` runs the checkpoint's ``index_topk`` at the shortest length
that yields the real 2051 width, plus a reduced budget at a 2:1
pool-to-slot ratio. Nothing about the pooling logic depends on the budget's
value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from graph_op_reference import (
    Glm5NextIndexer,
    indexer_traffic_bytes_per_token,
)
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kernels import (
    mla_kpool_compress,
    mla_kpool_ring_close,
    mla_kpool_seed_tail,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
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
SEED = 20260826

# The reference runs on CPU: it shares the GPU with the MAX graph under test,
# and its peak tensor is the same score matrix, which is a fraction of a second
# of CPU matmul at these shapes.
_REFERENCE_DEVICE = "cpu"


@dataclass(frozen=True)
class Case:
    """One (budget, length) pair, named by what it is there to prove."""

    name: str
    index_topk: int
    seq_len: int

    @property
    def max_pools(self) -> int:
        return math.ceil(self.seq_len / INDEX_KPOOL)

    @property
    def selection_width(self) -> int:
        return self.index_topk + INDEX_KPOOL - 1


CASES = (
    # The checkpoint's budget, at the shortest length that reaches the real
    # 2051-wide selection. 2052 is not a multiple of `index_kpool`, so the last
    # pool is partial and the tail has to carry it.
    Case("topk2048", index_topk=2048, seq_len=2052),
    # Two pools competing for every slot, so the top-k actually discriminates.
    Case("topk512", index_topk=512, seq_len=1024),
)


@pytest.fixture(params=CASES, ids=[c.name for c in CASES])
def case(request: pytest.FixtureRequest) -> Case:
    return request.param


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


def _row_offsets(lengths: tuple[int, ...]) -> torch.Tensor:
    offsets = np.zeros(len(lengths) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(lengths)
    return torch.from_numpy(offsets)


# ------------------------------------------------------------------ reference


def _reference_module(
    case: Case, weights: dict[str, torch.Tensor]
) -> Glm5NextIndexerReference:
    module = Glm5NextIndexerReference(
        Glm5NextIndexerRefConfig(
            hidden_size=HIDDEN_SIZE,
            q_lora_rank=Q_LORA_RANK,
            index_n_heads=INDEX_N_HEADS,
            index_head_dim=INDEX_HEAD_DIM,
            index_topk=case.index_topk,
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
    return module.to(device=_REFERENCE_DEVICE, dtype=torch.bfloat16).eval()


def _run_reference(
    case: Case,
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    qr: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    module = _reference_module(case, weights)
    return module(
        x.to(_REFERENCE_DEVICE),
        qr.to(_REFERENCE_DEVICE),
        attention_mask.to(_REFERENCE_DEVICE),
    ).cpu()


# ------------------------------------------------------------------------ MAX


def _max_indexer(case: Case) -> Glm5NextIndexer:
    return Glm5NextIndexer(
        hidden_size=HIDDEN_SIZE,
        q_lora_rank=Q_LORA_RANK,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=case.index_topk,
        index_kpool=INDEX_KPOOL,
        devices=[DeviceRef.GPU()],
    )


def _bf16_to_device(tensor: torch.Tensor, device: Accelerator) -> Buffer:
    return (
        Buffer.from_numpy(tensor.contiguous().view(torch.float16).numpy())
        .view(DType.bfloat16)
        .to(device)
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


def _run_max_selection(
    case: Case,
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    qr: torch.Tensor,
    row_offsets: torch.Tensor,
    max_pools: int,
) -> torch.Tensor:
    """Runs the dense selection path."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    indexer = _max_indexer(case)
    indexer.load_state_dict(weights, strict=True)

    batch = int(row_offsets.shape[0]) - 1
    # The batch dimension is static here: the dense scorer's `q @ pooled^T` has
    # `batch * max_pools` columns and the multistage GEMM will not split a
    # dynamic N. Only this verification path cares --- the fused pooled scorer
    # reads a paged cache and never forms this matmul.
    input_types = [
        TensorType(
            DType.bfloat16, ["total_tokens", HIDDEN_SIZE], DeviceRef.GPU()
        ),
        TensorType(
            DType.bfloat16, ["total_tokens", Q_LORA_RANK], DeviceRef.GPU()
        ),
        TensorType(DType.uint32, [batch + 1], DeviceRef.GPU()),
    ]

    with Graph(
        "Glm5NextIndexerSelection", input_types=tuple(input_types)
    ) as graph:
        graph.output(
            indexer(
                graph.inputs[0].tensor,
                graph.inputs[1].tensor,
                graph.inputs[2].tensor,
                max_pools=max_pools,
            )
        )

    compiled = session.load(graph, weights_registry=indexer.state_dict())
    args: list[Buffer] = [
        _bf16_to_device(x, device),
        _bf16_to_device(qr, device),
        Buffer.from_numpy(row_offsets.numpy()).to(device),
    ]
    return from_dlpack(compiled.execute(*args)[0]).cpu()


def _run_max_pool_compress(
    case: Case,
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    row_offsets: torch.Tensor,
    max_pools: int,
    *,
    quantize: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Returns the pooled keys, and their FP8 cache rows when asked."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    indexer = _max_indexer(case)
    indexer.load_state_dict(weights, strict=True)

    with Graph(
        "Glm5NextIndexerPoolCompress",
        input_types=(
            TensorType(
                DType.bfloat16, ["total_tokens", HIDDEN_SIZE], DeviceRef.GPU()
            ),
            TensorType(
                DType.uint32, [int(row_offsets.shape[0])], DeviceRef.GPU()
            ),
        ),
    ) as graph:
        grid = indexer.pool_grid(graph.inputs[1].tensor, max_pools)
        keys, gate = indexer.keys_and_gate(graph.inputs[0].tensor)
        pooled = indexer.compress_pools(keys, gate, grid)
        outputs = [pooled]
        if quantize:
            rows, scales = indexer.quantize_pooled_keys(pooled, _quant_config())
            # BF16 represents every e4m3 value exactly and there is no
            # FP8-to-integer cast, so this reads the stored bytes losslessly.
            outputs += [rows.cast(DType.bfloat16), scales]
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=indexer.state_dict())
    results = compiled.execute(
        _bf16_to_device(x, device),
        Buffer.from_numpy(row_offsets.numpy()).to(device),
    )
    return tuple(from_dlpack(result).cpu() for result in results)


# ------------------------------------------------------------------- helpers


def _index_sets(indices: torch.Tensor) -> list[set[int]]:
    return [{int(v) for v in row.tolist() if v >= 0} for row in indices]


@dataclass(frozen=True)
class Agreement:
    """How closely two selections agree, as sets of token positions.

    ``whole_pools_only`` is the load-bearing field, not the exact-row fraction.
    Once a query's budget saturates, two pools at the top-k boundary can be
    separated by less than a float32 rounding step --- and the two
    implementations reach those scores through differently shaped matmuls ---
    so which one wins is arbitrary. What is *not* arbitrary is the granularity
    of the disagreement: a tie swaps whole aligned pools, so every position in
    the symmetric difference comes with its ``index_kpool - 1`` siblings. A bug
    in the expansion, the masking or the tail breaks that alignment, which is
    why it is the assertion rather than a threshold on how many rows differ.
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


# --------------------------------------------------------------------- tests


def test_pool_compression_matches_reference(case: Case) -> None:
    """The learned pooled key, on real head dimensions.

    Channel-wise softmax over the four members, the ape added to the gate
    logits before it, and the weights cast back to bf16 before the weighted
    sum --- get any of the three wrong and long-context retrieval degrades
    without anything crashing.
    """
    weights = _weights()
    x, _ = _activations(case.seq_len)

    max_pooled = _run_max_pool_compress(
        case, weights, x, _row_offsets((case.seq_len,)), case.max_pools
    )[0]

    reference = _reference_module(case, weights)
    packed = torch.cat(
        [
            reference.k_norm(reference.wk(x[None])),
            torch.nn.functional.linear(
                x[None], reference.index_kpool_compress_gate
            ),
            torch.ones(1, case.seq_len, 1, dtype=torch.bfloat16),
        ],
        dim=-1,
    )
    ref_pooled, _, ref_valid = reference.get_pooled_states(packed)

    complete = case.seq_len // INDEX_KPOOL
    assert int(ref_valid.sum()) == complete, (
        "the reference should mark exactly the complete pools valid"
    )
    max_f = max_pooled[0, :complete].to(torch.float32)
    ref_f = ref_pooled[0, :complete].to(torch.float32)
    max_abs = (max_f - ref_f).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        max_f.reshape(-1), ref_f.reshape(-1), dim=0
    ).item()
    # bfloat16 carries 7 explicit mantissa bits, so one ULP at the largest
    # value present is 2^(exponent - 7). Both sides compute the same four-term
    # weighted sum and each term rounds, so the tolerance is stated in ULPs
    # rather than picked: 4 is the loosest value that still fails if a member's
    # weight or the ape is dropped (which moves values by whole percent).
    largest = ref_f.abs().max().item()
    ulp = 2.0 ** (math.floor(math.log2(largest)) - 7)
    print(
        f"[{case.name}] pooled key over {complete} pools: "
        f"max|diff|={max_abs:.3e} ({max_abs / ulp:.1f} bf16 ULP at "
        f"|max|={largest:.3f}) cos={cos:.9f}"
    )
    assert max_abs <= 4 * ulp
    assert cos >= 1.0 - 5e-6


def test_ape_is_load_bearing() -> None:
    """Zeroing ``index_kpool_compress_ape`` must move the pooled keys.

    The checkpoint ships it zero-initialized, which is exactly why an
    implementation that silently drops it passes every smoke test.
    """
    case = CASES[1]
    weights = _weights()
    x, _ = _activations(case.seq_len)
    offsets = _row_offsets((case.seq_len,))

    with_ape = _run_max_pool_compress(
        case, weights, x, offsets, case.max_pools
    )[0]
    zeroed = dict(weights)
    zeroed["index_kpool_compress_ape"] = torch.zeros_like(
        weights["index_kpool_compress_ape"]
    )
    without_ape = _run_max_pool_compress(
        case, zeroed, x, offsets, case.max_pools
    )[0]

    delta = (
        (with_ape.to(torch.float32) - without_ape.to(torch.float32))
        .abs()
        .max()
        .item()
    )
    print(f"[{case.name}] ape ablation: max|diff|={delta:.3e}")
    assert delta > 1e-3, "the ape is not reaching the pool softmax"


def test_selection_matches_reference() -> None:
    """Selected token sets against the reference, plain and left-padded.

    Only at `topk512`. The other case's budget is 2051 slots over 2052
    tokens, so it selects everything and the top-k ranks nothing -- it
    reported an exact match on every row while this case reports 0.94, which
    is the difference between a test that can fail and one that cannot.

    The padded half is here rather than in its own test because it asserts
    against the same MAX selection: the reference derives pool boundaries
    from ``first_key`` while MAX's ragged layout has none, so left-padding
    the reference and shifting its answer back has to agree with MAX exactly
    as well as the unpadded one does. Any gap between the two is `first_key`
    handling, and if they disagreed every cached pooled key would be wrong
    the moment padding changed.
    """
    case = CASES[1]
    pad = 7
    weights = _weights()
    x, qr = _activations(case.seq_len)

    actual = _run_max_selection(
        case, weights, x, qr, _row_offsets((case.seq_len,)), case.max_pools
    )
    reference = _run_reference(
        case,
        weights,
        x[None],
        qr[None],
        torch.ones(1, case.seq_len, dtype=torch.bool),
    )[0]

    assert actual.shape == (case.seq_len, case.selection_width)
    assert actual.dtype == torch.int32
    agreement = _set_agreement(actual, reference)
    print(f"[{case.name}] selection: {agreement}")
    # Rows that still fit inside the budget must match exactly: every visible
    # pool is selected there, so a score tie cannot reorder anything and any
    # disagreement is a logic bug.
    unsaturated = min(case.index_topk, case.seq_len)
    assert (
        _set_agreement(actual[:unsaturated], reference[:unsaturated]).exact_rows
        == 1.0
    )
    assert agreement.whole_pools_only
    assert agreement.mean_jaccard >= 0.99

    padded_x = torch.cat([torch.zeros_like(x[:pad]), x])
    padded_qr = torch.cat([torch.zeros_like(qr[:pad]), qr])
    mask = torch.ones(1, case.seq_len + pad, dtype=torch.bool)
    mask[0, :pad] = False
    padded = _run_reference(
        case, weights, padded_x[None], padded_qr[None], mask
    )[0][pad:]
    shifted = torch.where(padded < 0, padded, padded - pad)

    padded_agreement = _set_agreement(actual, shifted)
    print(f"[{case.name}] left-pad {pad}: {padded_agreement}")
    # The padded and unpadded references must agree with MAX *equally* well.
    # The residual disagreement they share is top-k tie-breaking.
    assert padded_agreement.exact_rows == agreement.exact_rows
    assert padded_agreement.first_mismatch == agreement.first_mismatch
    assert padded_agreement.whole_pools_only


def test_ragged_batch_matches_per_sequence_reference() -> None:
    """A ragged batch whose lengths are not multiples of ``index_kpool``.

    Each sequence is also run through the reference on its own, so this checks
    both that the pooling is per-sequence and that no selection escapes into a
    neighbour's tokens.
    """
    case = CASES[1]
    lengths = (case.seq_len, case.seq_len - 3)
    weights = _weights()
    x, qr = _activations(sum(lengths))
    offsets = _row_offsets(lengths)
    max_pools = math.ceil(max(lengths) / INDEX_KPOOL)

    actual = _run_max_selection(case, weights, x, qr, offsets, max_pools)

    for seq, length in enumerate(lengths):
        begin, end = int(offsets[seq]), int(offsets[seq + 1])
        reference = _run_reference(
            case,
            weights,
            x[begin:end][None],
            qr[begin:end][None],
            torch.ones(1, length, dtype=torch.bool),
        )[0]
        rows = actual[begin:end]
        agreement = _set_agreement(rows, reference)
        print(f"[{case.name}] ragged seq {seq} (len {length}): {agreement}")
        assert agreement.whole_pools_only
        assert agreement.mean_jaccard >= 0.99
        assert int(rows.max()) < length, "a selection escaped its own sequence"


def test_tail_is_the_current_incomplete_pool(case: Case) -> None:
    """The tail's start is the visible count rounded down to a pool boundary.

    Not a fixed offset, and never past the query's own position --- the two
    ways to get this wrong either let attention read a future token or drop a
    just-written one.
    """
    weights = _weights()
    x, qr = _activations(case.seq_len)
    actual = _run_max_selection(
        case, weights, x, qr, _row_offsets((case.seq_len,)), case.max_pools
    )

    tail = actual[:, -(INDEX_KPOOL - 1) :]
    for position in range(case.seq_len):
        start = ((position + 1) // INDEX_KPOOL) * INDEX_KPOOL
        expected = [
            start + offset
            for offset in range(INDEX_KPOOL - 1)
            if start + offset <= position
        ]
        got = [int(v) for v in tail[position].tolist() if v >= 0]
        assert got == expected, f"position {position}: tail {got} != {expected}"
        assert int(actual[position].max()) <= position, (
            f"position {position} selected a future token"
        )
    print(f"[{case.name}] tail verified for all {case.seq_len} positions")


def test_rebuild_versus_cached_pooled_keys() -> None:
    """The differential the whole lane rests on.

    Only at `topk512`. `prefix` is rounded down to a pool boundary, so this
    never exercises the partial final pool that separates the two cases, and
    `test_pool_compression_matches_reference` covers that at both lengths.

    A pooled key is immutable once its ``index_kpool`` tokens exist, which is
    what lets it be written once and read forever. This runs the compression
    twice --- over a prefix and over the full sequence --- and requires the
    pools they share to be bit-identical. If the shorter run's arithmetic
    differed at all (a shape-dependent reduction order, say), a pooled cache
    would silently diverge from a rebuild and nothing cheap would notice.
    """
    case = CASES[1]
    weights = _weights()
    x, _ = _activations(case.seq_len)
    prefix = (case.seq_len // 2) // INDEX_KPOOL * INDEX_KPOOL
    prefix_pools = prefix // INDEX_KPOOL

    rebuilt = _run_max_pool_compress(
        case, weights, x, _row_offsets((case.seq_len,)), case.max_pools
    )[0]
    from_prefix = _run_max_pool_compress(
        case, weights, x[:prefix], _row_offsets((prefix,)), prefix_pools
    )[0]

    shared = rebuilt[0, :prefix_pools].view(torch.uint16)
    cached_shared = from_prefix[0].view(torch.uint16)
    mismatches = int((shared != cached_shared).sum().item())
    print(
        f"[{case.name}] pool immutability: {prefix_pools} shared pools, "
        f"mismatching bf16 values={mismatches}/{shared.numel()}"
    )
    assert mismatches == 0, (
        "pooled keys built over a prefix differ from the same pools built over "
        "the full sequence, so they cannot be cached"
    )


def test_quantized_pooled_cache_rows() -> None:
    """The pooled cache's FP8 row layout and its per-pool scale.

    One 128-value pooled key spans ``index_kpool`` cache rows of
    ``index_head_dim / index_kpool`` values, which is byte-identical to a
    compacted pooled cache; the scale is per *pool* and replicated across those
    rows so the cached arm quantizes exactly what a rebuild would.
    """
    case = CASES[1]
    weights = _weights()
    x, _ = _activations(case.seq_len)
    max_pools = case.max_pools

    pooled, rows, scales = _run_max_pool_compress(
        case,
        weights,
        x,
        _row_offsets((case.seq_len,)),
        max_pools,
        quantize=True,
    )
    indexer = _max_indexer(case)

    assert rows.shape == (
        max_pools * INDEX_KPOOL,
        1,
        indexer.pooled_cache_head_dim,
    )
    assert scales.shape == (max_pools * INDEX_KPOOL, 1, 1)
    per_pool = scales.reshape(max_pools, INDEX_KPOOL)
    assert bool(torch.equal(per_pool, per_pool[:, :1].expand_as(per_pool))), (
        "a pool's scale must be identical in every row it occupies"
    )

    keys = rows.reshape(max_pools, INDEX_HEAD_DIM).to(torch.float32)
    dequantized = keys * per_pool[:, :1].to(torch.float32)
    reference = pooled[0].to(torch.float32)
    rel = (
        (dequantized - reference).abs().max()
        / reference.abs().max().clamp(min=1e-6)
    ).item()
    print(f"[{case.name}] pooled FP8 round trip: max relative error={rel:.4f}")
    # e4m3 carries three mantissa bits, so 2^-4 = 6.25% is the per-value
    # ceiling against a per-row absmax scale.
    assert rel <= 0.07


def test_traffic_arithmetic() -> None:
    """The numbers that make the pooled cache mandatory rather than nice.

    At 1M context over 11 sparse layers on an 8 TB/s part: rebuilding pools
    every step reads 4.5 GB per decoded token and caching them reads 0.4 GB,
    with GLM-5.2's per-token indexer between at 1.5 GB. That ordering is why a
    faithful port of this model's indexer is slower than the model it replaces.
    """
    kv_len = 1 << 20
    layers = 11

    def traffic(kpool: int, pooled_cache: bool) -> int:
        return indexer_traffic_bytes_per_token(
            num_sparse_layers=layers,
            kv_len=kv_len,
            index_head_dim=INDEX_HEAD_DIM,
            index_kpool=kpool,
            pooled_cache=pooled_cache,
        )

    rebuild = traffic(INDEX_KPOOL, False)
    cached = traffic(INDEX_KPOOL, True)
    glm52 = traffic(1, True)
    bandwidth = 8e12
    for label, value in (
        ("rebuild pools each step", rebuild),
        ("cached pooled keys", cached),
        ("GLM-5.2 per-token", glm52),
    ):
        print(
            f"{label:>24}: {value / 1e9:6.2f} GB/token "
            f"-> {value / bandwidth * 1e6:6.1f} us at 8 TB/s"
        )
    assert rebuild / glm52 > 2.5
    assert glm52 / cached > 3.5


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
    unconditionally, every call) is what :meth:`Indexer._select_pooled` runs.
    ``oracle_model`` is the independent one-shot compression it is judged
    against.

    There is no separate prefill or decode writer. :meth:`_select_pooled`
    used to pick between them with a batch-wide ``ops.cond`` predicate, which
    could route a ragged multi-token request onto a rectangular
    one-token-per-request kernel; every schedule below now drives the one
    writer that replaced them.
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
    # `Indexer._select_pooled` now always runs, regardless of alignment or
    # per-request token count. `pool_row_offsets` is supplied rather than
    # computed here, mirroring `_select_pooled`'s own arithmetic, which the
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
        # `Indexer._select_pooled`'s comment on this same ordering.
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
    caller-chosen schedule -- unconditionally, on *every* call, exactly as
    :meth:`Indexer._select_pooled` now always does.

    ``schedule`` is not tied to a fixed prefill/decode split: each call can
    bring any per-request token count, so this exercises the shape the
    retired ``ops.cond`` predicate could not -- a call where different requests are at different
    points in their pool (some mid-pool from a previous call, some fresh),
    and bring different numbers of new tokens.

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

        # Same arithmetic as `Indexer._select_pooled`: pools this call
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
    the off-by-one member this lane exists to prevent.
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


def test_decode_pools_match_one_shot_compression(tail_rig: _TailRig) -> None:
    """The gate this whole slice rests on.

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
    """The shape that used to crash: a ragged, mixed-alignment call.

    A batch-wide predicate picking one writer for the whole call cannot serve
    a call where requests differ in both how much of a pool the ring is
    already holding (a different remainder mod ``index_kpool`` each) and how
    many new tokens they bring (also different, and none of them ``1``) --
    every request here enters the second call at a different pending-pool
    remainder and completes with a different, multi-token count in the same
    call. The old rectangular ``tail_update`` writer took exactly one new
    token per request; a call shaped like this is what made it read and
    write out of bounds.
    """
    _assert_bit_identical(
        _run_general_writer(tail_rig, [(6, 3, 9), (7, 6, 2)]),
        _run_compress_oracle(tail_rig),
        "mixed ragged batch",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-q"]))
