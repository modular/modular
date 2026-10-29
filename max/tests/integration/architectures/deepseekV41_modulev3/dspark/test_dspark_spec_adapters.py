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
"""CPU tests of the DSpark block spec-decode adapters.

The block driver is graph-value typed and knows no ``TensorAdapter``, so the
adapters convert at their own boundary and must work in both worlds -- these
tests drive them with eager single-device tensors, which the conversions pass
through unchanged.

The trunk's fused kernels are GPU-only, so the verify path is pinned against
a duck-typed trunk that returns marked streams: what is under test is the
adapter's own math -- the K+1 verify rows, the leaf and extras assembly, the
taps payload -- never the trunk. The drafter's block forward is likewise
GPU-bound, so its position and window-start arithmetic are pinned directly
at the helpers ``forward_block`` calls.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dspark_test_support import TESTDATA
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.kv_cache import PagedCacheValues as UpstreamPagedCacheValues
from max.pipelines.architectures.deepseekV41_modulev3 import spec_adapters
from max.pipelines.architectures.deepseekV41_modulev3.dspark_drafter import (
    DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
    extra_input_names,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41_modulev3.spec_adapters import (
    WINDOW_LEAF,
    DeepseekV41BlockTarget,
    DSparkDeepseekV41Proposer,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative._tensor_compat import as_tensors
from max.pipelines.speculative.block_driver import (
    ArgmaxDraftSampler,
    BlockBatch,
)
from max.tensor import Tensor, default_device
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

HIDDEN = 128
HEADS = 2
HEAD_DIM = 64
Q_LORA = 64
O_GROUPS = 2
O_LORA = 32
INTER = 64
EXPERTS = 8
VOCAB = 97
MARKOV = 16
STAGES = 3
BLOCK = 2
WINDOW = 8
BATCH = 3

_MESH = DeviceMesh((CPU(),), (1,), ("tp",))


# ``_positions`` reads everything off its arguments, so an uninitialized
def _small_config(vocab: int = VOCAB) -> DeepseekV41Config:
    """The drafter's config at this module's shrunk TP1 widths.

    ``vocab`` overrides the vocabulary for the TP4 cases, whose head
    shards by it (see ``_small_tp4_config``).
    """
    text = SimpleNamespace(
        **json.load(open(TESTDATA / "config_full.json"))["text_config"]
    )
    kv_params = build_kv_params(
        V41Config.from_hf({"text_config": vars(text)}),
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.CPU()],
    )
    return dataclasses.replace(
        DeepseekV41Config(
            dtype=DType.bfloat16,
            kv_params=kv_params,
            devices=[DeviceRef.CPU()],
            max_seq_len=64,
            **DeepseekV41Config.resolve_text_fields(text),
        ),
        vocab_size=vocab,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        q_lora_rank=Q_LORA,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA,
        moe_intermediate_size=INTER,
        sliding_window=WINDOW,
        dspark_n_routed_experts=EXPERTS,
        dspark_num_experts_per_tok=2,
        dspark_markov_rank=MARKOV,
        dspark_noise_token_id=96,
        dspark_block_size=BLOCK,
    )


def _small_tp4_config() -> DeepseekV41Config:
    """The TP4 case's config: the shrunk widths valid on a 4-rank mesh.

    The TP1 defaults stay untouched for every other test; only the
    three sampler/head TP4 cases build the drafter on the mesh, and
    they pass this config. Every constraint checked in one pass:

    - ``wo_b``'s in_dim ``o_groups * o_lora_rank`` (2 * 64) and the
      shared expert's ``moe_intermediate_size`` (128) divide
      ``32 * TP``, the whole 32-element scale blocks
      ``MXFP8Linear.shard`` demands (the MoE fixture in
      ``test_dspark.py`` widens its K the same way).
    - ``wq_b``'s rows ``heads * head_dim`` (2 * 64 = 128) and
      ``wo_a``'s rows ``o_groups * o_lora_rank`` (128) divide the mesh
      for their ``Sharded(0)`` placements, as ``attn_sink`` does per
      head.
    - The target head shards by vocabulary, exactly production's
      ``ColumnParallelLinear`` over ``[vocab, hidden]`` (the trunk's
      own placement, which ``drafter.logits``' allgather gathers), so
      the vocabulary divides the mesh (128 / 4) and the drafter's
      Markov head follows ``config.vocab_size`` to the same width.
    - ``main_proj``'s in_dim ``hidden * n_taps`` (384) divides 32 and
      never shards; the routed experts stay replicated under this
      port's TP.
    """
    return dataclasses.replace(
        _small_config(_TP4_VOCAB),
        o_lora_rank=64,
        moe_intermediate_size=128,
    )


def _rope() -> RopeParams:
    text = SimpleNamespace(
        **json.load(open(TESTDATA / "config_full.json"))["text_config"]
    )
    return RopeParams.from_text_config(text)


def _v41_config() -> V41Config:
    """The trunk's frozen interface view, at the fixture's table extents.

    Every field is the real config's except the Engram table row counts,
    the interface's only unbounded extents (384M rows per table in the
    checkpoint -- 196 GB of ``torch.zeros`` the fixture must never build).
    The adapter under test reads each table's name, never its rows, so the
    shrunk extents change nothing it observes.
    """
    config = V41Config.from_hf(json.load(open(TESTDATA / "config_full.json")))
    return dataclasses.replace(config, engram_num_embeddings=(4, 4))


def _to_max(t: torch.Tensor) -> Tensor:
    return Tensor(t, device=CPU())


def _torch(t: Any) -> torch.Tensor:
    """Reads a single-device or mesh tensor back as torch."""
    shard = t.local_shards[0] if t.is_distributed else t
    return torch.from_dlpack(shard.to(CPU()))


def _eager() -> AbstractContextManager[Any]:
    """An eager realization context spanning one adapter call.

    The adapters convert at their own boundary -- ``from_upstream`` in,
    ``as_graph_values`` out -- and every one of those conversions needs an
    active realization context to realize through. Production always has
    one (the trace's, or the eager executor's); a bare test body does not,
    so each adapter call runs inside this. Entering the context also makes
    its graph ambient, so the fixture tensors built inside it can lower
    through ``F.constant``.
    """
    return ensure_context()


# --------------------------------------------------------------------------- #
# Fixtures: the driver's batch, the upstream caches, a duck-typed trunk
# --------------------------------------------------------------------------- #


def _graph_typed(value: Tensor) -> TensorValue:
    """``value`` in a graph-typed slot of the driver or an adapter.

    The same eager/graph duality :func:`_upstream_kv` crosses: the eager run
    carries a realized tensor where the API declares a graph value.
    """
    return cast(TensorValue, value)


def _upstream_kv(cache_lengths: list[int]) -> UpstreamPagedCacheValues:
    """One device's paged cache values, as the driver hands them over.

    The driver's records are graph-typed, but this eager run carries
    realized tensors in the same slots -- the duality the adapters convert
    at their boundary -- so the record is built in its eager
    parameterization and cast onto the graph-typed alias once, here.
    """
    n = len(cache_lengths)
    return cast(
        UpstreamPagedCacheValues,
        KVCacheInputsPerDevice[Tensor, Tensor](
            kv_blocks=_to_max(
                torch.zeros(4, n, 2, HEAD_DIM, dtype=torch.bfloat16)
            ),
            cache_lengths=_to_max(
                torch.tensor(cache_lengths, dtype=torch.int32)
            ),
            lookup_table=_to_max(
                torch.arange(4 * n, dtype=torch.int32).reshape(n, 4)
            ),
            max_prompt_length=_to_max(torch.tensor([n], dtype=torch.uint32)),
            max_cache_length=_to_max(torch.tensor([64], dtype=torch.uint32)),
            page_stride=_to_max(torch.tensor([1], dtype=torch.int64)),
        ),
    )


def _extras() -> dict[str, Tensor]:
    """The trunk's extras with the shapes the interface declares."""
    config = _v41_config()
    batch = 2
    raw = {
        "swa_begin": torch.zeros(batch, dtype=torch.int32),
        "global_write_floor": torch.zeros(batch, dtype=torch.int32),
        "image_token_mask": torch.zeros(9, dtype=torch.uint8),
        "engram_lookback": torch.zeros(batch, 3, dtype=torch.int32),
        "engram_lookback_dead": torch.zeros(batch, 3, dtype=torch.uint8),
        # The graph reads only the extent of the decoder's row count.
        "decoder_rows": torch.zeros(9, dtype=torch.uint8),
    }
    for ordinal, rows in enumerate(config.engram_num_embeddings):
        raw[f"engram_table_{ordinal}_weight"] = torch.zeros(
            rows, 256, dtype=torch.uint8
        )
        raw[f"engram_table_{ordinal}_scale"] = torch.zeros(
            rows, 8, dtype=torch.uint8
        )
    raw["engram_token_map"] = torch.zeros(config.vocab_size, dtype=torch.int32)
    return {name: _to_max(raw[name]) for name in extra_input_names(config)}


def _kv_params_stub() -> Any:
    """A kv params tree whose children name the trunk's leaves."""
    tree = SimpleNamespace()
    tree.children = {WINDOW_LEAF: object(), "global": object()}
    return tree


def _batch(
    *,
    lengths: list[int],
    extra: dict[str, Tensor] | None = None,
) -> BlockBatch:
    """A single-device BlockBatch over requests of the given lengths.

    The driver's tensor slots are graph-typed, but this eager run carries
    realized tensors in them -- the same duality ``_upstream_kv`` casts
    across -- so the tensor fields go through :func:`_graph_typed`.
    """
    total = sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    tokens = _graph_typed(_to_max(torch.arange(total, dtype=torch.int64)))
    merged_offsets = _graph_typed(
        _to_max(torch.tensor(offsets, dtype=torch.uint32))
    )
    kv = [_upstream_kv(lengths)]
    return BlockBatch(
        tokens=tokens,
        input_row_offsets=merged_offsets,
        draft_tokens=_graph_typed(
            _to_max(torch.zeros(len(lengths), BLOCK, dtype=torch.int64))
        ),
        kv_collections=kv,
        draft_kv_collections=kv,
        passthrough_kv={"global": kv},
        return_n_logits=_graph_typed(
            _to_max(torch.tensor([BLOCK + 1], dtype=torch.int64))
        ),
        devices=[DeviceRef.CPU()],
        signal_buffers=[],
        ep_inputs=None,
        merged_tokens=tokens,
        merged_offsets=merged_offsets,
        merged_offsets_per_dev=[merged_offsets],
        host_merged_offsets=None,
        data_parallel_splits=None,
        data_parallel_degree=1,
        batch_context_lengths=[],
        vision_embeddings=[],
        vision_scatter_indices=[],
        extra=_extras() if extra is None else extra,
        num_accepted=None,
        draft_slot_ids=None,
        draft_block_offsets=None,
        block_size=BLOCK,
    )


class _RecordingTarget:
    """A duck-typed trunk: records the assembly, returns marked streams.

    ``logits_all_rows`` is the identity, so the test sees exactly the rows
    the adapter selected; ``hidden_states`` marks each merged row with its
    own index.
    """

    def __init__(self, total: int) -> None:
        self.mesh = _MESH
        self.dp = 1
        self.config = _v41_config()
        self.interface_config = self.config
        self.head = SimpleNamespace(weight=None)
        marks = torch.arange(total, dtype=torch.int32)
        self.marked = _to_max(
            torch.stack([marks] * HIDDEN, dim=1).to(torch.bfloat16)
        )
        self.taps = _to_max(torch.zeros(total, HIDDEN * STAGES))
        self.positions = _to_max(torch.arange(total, dtype=torch.int32) + 100)
        self.assembled: tuple[Any, ...] | None = None
        self.draft_width: Tensor | None = None

    def batch_from_parts(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv: dict[str, PagedCacheValues],
        extras: dict[str, Tensor],
        *,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        comm_buffers: Any = None,
        draft_width: Tensor | None = None,
    ) -> Any:
        self.assembled = (tokens, input_row_offsets, kv, extras)
        self.draft_width = draft_width
        return SimpleNamespace(
            row_offsets=input_row_offsets, positions=self.positions
        )

    def embed_stream(self, b: Any) -> tuple[Tensor, Tensor]:
        return self.marked, self.marked

    def run_trunk_with_taps(
        self, b: Any, stream: Tensor, pre: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return stream, pre, self.taps

    def hidden_states(self, stream: Tensor, pre: Tensor) -> Tensor:
        return self.marked

    def logits_all_rows(self, hidden: Tensor) -> Tensor:
        return hidden


def _target(total: int) -> Any:
    return cast(Any, _RecordingTarget(total))


# --------------------------------------------------------------------------- #
# The verify path
# --------------------------------------------------------------------------- #


def test_verify_gathers_the_last_k_plus_one_rows_per_request() -> None:
    """The accept reads one flat request-major block of K+1 logit rows.

    Row ``j`` of request ``r`` must be the merged row the accept's step
    ``j`` scores: the committed token at ``j == 0``, then the drafts, then
    the bonus row. Anything else verifies the wrong tokens behind
    well-shaped output.
    """
    lengths = [3, 6]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(cast(Any, target), _kv_params_stub())
    with _eager():
        verified = adapter(_batch(lengths=lengths))
        got = as_tensors(verified.logits)

    # Request 0's last 3 rows are 0, 1, 2; request 1's are 6, 7, 8. The
    # marked hidden carries its row index in every column. Reading back
    # realizes, so it sits outside the still-open eager context.
    got = _torch(got)
    assert tuple(got.shape) == (2 * (BLOCK + 1), HIDDEN)
    rows = got[:, 0].to(torch.int32)
    assert torch.equal(rows, torch.tensor([0, 1, 2, 6, 7, 8]))
    assert bool((got == got[:, :1]).all())


def test_verify_carries_the_taps_with_the_rows_that_produced_them() -> None:
    """The proposer must store context at the verify forward's own rows.

    The payload is the taps plus the batch's row offsets and positions --
    the store lands at exactly the positions the verify ran at, never
    re-derived.
    """
    lengths = [4, 2]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(cast(Any, target), _kv_params_stub())
    with _eager():
        verified = adapter(_batch(lengths=lengths))
    assert target.assembled is not None
    taps, row_offsets, positions = verified.hidden
    assert taps is target.taps
    assert positions is target.positions
    assert row_offsets is target.assembled[1]


def test_verify_assembles_the_leaves_and_extras() -> None:
    """The window leaf comes from the driver's primary, the rest pass through."""
    lengths = [3, 3]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(cast(Any, target), _kv_params_stub())
    with _eager():
        adapter(_batch(lengths=lengths))
    assert target.assembled is not None
    tokens, _offsets, kv, extras = target.assembled
    assert set(kv) == {WINDOW_LEAF, "global"}
    assert torch.equal(
        _torch(kv[WINDOW_LEAF].cache_lengths),
        torch.tensor(lengths, dtype=torch.int32),
    )
    assert torch.equal(_torch(tokens), torch.arange(6, dtype=torch.int64))
    assert list(extras) == extra_input_names(target.interface_config)


def test_verify_hands_the_batch_its_draft_width() -> None:
    """The batch carries the step's draft rows so the narrowing can widen.

    A verify chunk merges ``K`` placeholder rows into every request and
    commits only the real rows, so the decoder's kept-row bound must widen
    by exactly the width this step appended -- read off the ``draft_tokens``
    input's own shape, never a second engine-fed scalar the two could
    disagree on. Without a spec decoder there is nothing to widen; the
    batch's width is ``None`` there.
    """
    target = _RecordingTarget(6)
    adapter = DeepseekV41BlockTarget(cast(Any, target), _kv_params_stub())
    with _eager():
        adapter(_batch(lengths=[3, 3]))
    assert target.draft_width is not None
    assert int(_torch(target.draft_width)) == BLOCK


def test_verify_fails_loud_on_a_missing_extra() -> None:
    """A silently-defaulted extra would feed the trunk garbage masks."""
    lengths = [2]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(cast(Any, target), _kv_params_stub())
    batch = _batch(lengths=lengths)
    starved = {
        name: value
        for name, value in batch.extra.items()
        if name != "engram_lookback_dead"
    }
    with pytest.raises(ValueError, match="engram_lookback_dead"), _eager():
        adapter(dataclasses.replace(batch, extra=starved))


# --------------------------------------------------------------------------- #
# The block arithmetic (the helpers forward_block runs)
# --------------------------------------------------------------------------- #


def test_block_positions_start_at_the_post_commit_length() -> None:
    """Block row ``j`` of request ``r`` sits at position ``L_r + j``.

    The block cache's lengths sit past the committed tokens, and the anchor
    IS that length's own position -- the commit covered it. Starting
    anywhere else mis-rotates every block row against its stored context.
    """
    lengths = [5, 9]
    block_offsets = _to_max(
        torch.tensor([0, BLOCK, 2 * BLOCK], dtype=torch.uint32)
    )
    positions = spec_adapters.block_row_positions(
        block_offsets,
        _to_max(torch.tensor(lengths, dtype=torch.int32)),
        Dim(2 * BLOCK),
    )
    assert torch.equal(
        _torch(positions), torch.tensor([5, 6, 9, 10], dtype=torch.int32)
    )


class _BlockRecordingDrafter:
    """A drafter stand-in that records the proposer's block operands."""

    def __init__(self) -> None:
        self.mesh = _MESH
        self.data_parallel_degree = 1
        self.block: dict[str, Any] = {}

    def windows(self, swa: Any, starts: Any) -> str:
        self.block["swa"] = swa
        self.block["starts"] = starts
        return "windows"

    def forward_block(
        self, ids: Any, positions: Any, windows: Any, embed: Any
    ) -> str:
        self.block.update(ids=ids, positions=positions, windows=windows)
        self.block["embed"] = embed
        return "block-out"


def test_forward_block_runs_the_proposer_path_with_real_positions() -> None:
    """``forward_block`` itself computes the block rows' positions.

    Regression: the proposer reached for the trunk's removed
    ``Attention._positions`` helper here and died with an AttributeError
    before drafting anything; the positions now come from the trunk's real
    ``absolute_positions`` arithmetic, on the actual proposer/block-row
    path -- not a restatement of the helper's formula.

    The lengths sit past the committed tokens (the anchor is the length's
    own position), request 1 drafts a zero-length block, and the two rows
    past the last offset are padded rows pinned to the last request so
    their gathers stay in bounds.
    """
    config = _small_config()
    drafter = _BlockRecordingDrafter()
    embed = object()
    target = SimpleNamespace(embed=embed)
    proposer = DSparkDeepseekV41Proposer(
        cast(Any, drafter), cast(Any, target), config
    )
    lengths = [5, 9]
    # Request 1 drafts a zero-length block (its end equals request 0's),
    # and the ids carry two padded rows past the last offset.
    block_offsets = _to_max(torch.tensor([0, BLOCK, BLOCK], dtype=torch.uint32))
    ids = _to_max(torch.arange(2 * BLOCK, dtype=torch.int32))
    with _eager():
        out = proposer.forward_block(
            _batch(lengths=lengths),
            cast(Any, [ids]),
            cast(Any, [block_offsets]),
            [_upstream_kv(lengths)],
        )
    assert out == "block-out"
    assert drafter.block["embed"] is embed
    assert drafter.block["windows"] == "windows"
    assert torch.equal(
        _torch(drafter.block["starts"]),
        torch.tensor(lengths, dtype=torch.int32),
    )
    assert torch.equal(_torch(drafter.block["ids"]), _torch(ids))
    # Rows 0,1 of request 0 sit at 5,6; request 1 drafts nothing; the
    # padded rows 2,3 pin to it and read its post-commit length: 9+0, 9+1.
    assert torch.equal(
        _torch(drafter.block["positions"]),
        torch.tensor([5, 6, 9, 10], dtype=torch.int32),
    )


def test_window_starts_are_the_block_cache_lengths() -> None:
    """The window reads everything below the block's start, never it."""
    with _eager():
        swa = PagedCacheValues.from_upstream(
            [_upstream_kv([5, 9])], DeviceMapping.replicated(_MESH)
        )
        starts = spec_adapters.window_starts(swa)
    assert torch.equal(_torch(starts), torch.tensor([5, 9], dtype=torch.int32))


# --------------------------------------------------------------------------- #
# The proposer
# --------------------------------------------------------------------------- #


class _PickySampler:
    """Picks one column past each step's argmax, provably never the argmax.

    Pure tensor ops, like a real ``DraftSampler``: a host read here would
    realize mid-draft, which a traced (or lazily built) draft never does.
    """

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.calls = 0

    def sample_next(self, step_logits: Tensor, position: int) -> Tensor:
        self.calls += 1
        argmax = F.squeeze(F.argmax(step_logits, axis=-1), axis=-1)
        return F.cast((argmax + (1 + position)) % self.vocab, DType.int32)


def test_head_drafts_through_the_driver_sampler() -> None:
    """The head must bridge the driver's per-position sampler intact.

    A head that samples greedily instead looks identical on argmax
    proposals and silently drops the sampled mode's distributions.
    """
    config = _small_config()
    head_weight = _to_max(torch.randn(VOCAB, HIDDEN).to(torch.bfloat16))
    target = SimpleNamespace(head=SimpleNamespace(weight=head_weight))
    hidden = _to_max(torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16))
    anchors = _to_max(torch.randint(0, VOCAB, (BATCH,), dtype=torch.int32))
    accepted = SimpleNamespace(next_tokens=anchors)
    sampler = _PickySampler(VOCAB)
    # The serving model builds the drafter under the mesh's default device
    # (``_instantiate_module``'s ``default_device(mesh)``), so the test does
    # too -- but OUTSIDE the eager context, bare like test_dspark.py: the
    # drafter's E8M0 scale constants then realize through the interpreter
    # in their own graphs, instead of joining the context's deferred graph
    # where the head's rms_norm (the op the interpreter refuses, so the
    # one that forces the compiled path) would carry them into the MLIR
    # compile and trip the known E8M0 APInt assert.
    with default_device(_MESH):
        drafter = DSparkDrafter(config, _MESH, rope=_rope())
        proposer = DSparkDeepseekV41Proposer(drafter, cast(Any, target), config)
        batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        got = as_tensors(
            proposer.head(
                batch,
                hidden,
                cast(Any, accepted),
                cast(Any, sampler),
            )
        )

        # The same draft the drafter draws with the same sampler semantics.
        logits = drafter.logits(hidden, head_weight)
        want, _ = drafter.draft(
            logits,
            anchors,
            sampler=lambda step, position: sampler.sample_next(step, position),
        )
        greedy, _ = drafter.draft(logits, anchors)

    # The reads realize, so they sit outside the context.
    torch.testing.assert_close(_torch(got), _torch(want).to(torch.int32))
    assert not torch.equal(_torch(got), _torch(greedy).to(torch.int32))
    # One call per block position, twice: once through the head, once
    # through the reference draft.
    assert sampler.calls == 2 * BLOCK


def test_head_returns_the_block_width_per_request() -> None:
    config = _small_config()
    head_weight = _to_max(torch.randn(VOCAB, HIDDEN).to(torch.bfloat16))
    target = SimpleNamespace(head=SimpleNamespace(weight=head_weight))
    hidden = _to_max(torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16))
    anchors = _to_max(torch.randint(0, VOCAB, (BATCH,), dtype=torch.int32))
    # Constructed bare outside the context, like the sampler test above:
    # the drafter's E8M0 constants must stay out of the head's graph.
    with default_device(_MESH):
        drafter = DSparkDrafter(config, _MESH, rope=_rope())
        proposer = DSparkDeepseekV41Proposer(drafter, cast(Any, target), config)
        batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        drafts = as_tensors(
            proposer.head(
                batch,
                hidden,
                cast(Any, SimpleNamespace(next_tokens=anchors)),
                cast(Any, _PickySampler(VOCAB)),
            )
        )
    assert tuple(drafts.shape) == (BATCH, BLOCK)
    assert drafts.dtype == DType.int32


# --------------------------------------------------------------------------- #
# The greedy head on the mesh: the production sampler at TP1 and virtual TP4
# --------------------------------------------------------------------------- #


_TP4_MESH = DeviceMesh((CPU(),) * 4, (4,), ("tp",))
"""A virtual TP4 mesh: the one CPU device repeated four times, like the
simulated CPU meshes of the distributed tensor suites, so the head's mesh
staging and its SPMD greedy actually execute."""


_TP4_VOCAB = 128
"""The TP4-valid vocabulary: divisible by the mesh, since the target
head shards by it (see ``_small_tp4_config`` for the full constraint
walk-through)."""


def _drafter_on(
    mesh: DeviceMesh, config: DeepseekV41Config, *, live_markov: bool
) -> DSparkDrafter:
    """A drafter whose modules live on ``mesh``.

    Constructed bare outside any realization context, like the sampler test
    above: the drafter's E8M0 constants must stay out of the head's graph.
    With ``live_markov`` both Markov head weights get small nonzero values
    -- the zero-init head would zero every step's bigram bias, leaving each
    argmax independent of the token drawn before it, the one thing a
    serial greedy draft must not get for free.
    """
    with default_device(mesh):
        drafter = DSparkDrafter(config, mesh, rope=_rope())
        if live_markov:
            drafter.markov_head.embed.weight = Tensor(
                torch.randn(config.vocab_size, MARKOV).to(torch.bfloat16),
                device=mesh,
            )
            drafter.markov_head.head.weight = Tensor(
                torch.randn(config.vocab_size, MARKOV).to(torch.bfloat16),
                device=mesh,
            )
    return drafter


def _head_weight_on(mesh: DeviceMesh, vocab: int) -> Tensor:
    """The target LM head at its production TP placement.

    Production shards the head by vocabulary (the trunk's
    ``ColumnParallelLinear`` over ``[vocab, hidden]``, which
    ``drafter.logits``' allgather gathers), so the TP4 fixture shards
    it the same way; TP1 keeps it whole.
    """
    weight = Tensor(torch.randn(vocab, HIDDEN).to(torch.bfloat16), device=mesh)
    if mesh.num_devices > 1:
        weight = weight.to(DeviceMapping(mesh, (Sharded(0),)))
    return weight


@pytest.mark.parametrize("tp", [1, 4])
def test_head_greedy_drafts_the_reference_tokens(tp: int) -> None:
    """The production greedy head drafts the reference tokens, on the mesh.

    The driver hands the head ``ArgmaxDraftSampler()``; at tensor
    parallelism that sampler's ``ops.argmax`` bridge rejects the
    mesh-replicated step logits, so the head must draw its greedy natively
    (``F.argmax``'s SPMD reduction). Whatever the width, the drafts must be
    exactly torch's argmax of each step's biased logits -- the greedy
    contract -- mapped straight through the full vocab, with every rank
    agreeing.
    """
    mesh = _MESH if tp == 1 else _TP4_MESH
    vocab = VOCAB if tp == 1 else _TP4_VOCAB
    config = _small_config(vocab) if tp == 1 else _small_tp4_config()
    head_weight = _head_weight_on(mesh, vocab)
    target = SimpleNamespace(head=SimpleNamespace(weight=head_weight))
    hidden = _to_max(torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16))
    anchors = _to_max(torch.randint(0, vocab, (BATCH,), dtype=torch.int32))
    drafter = _drafter_on(mesh, config, live_markov=True)
    proposer = DSparkDeepseekV41Proposer(drafter, cast(Any, target), config)
    batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        got = as_tensors(
            proposer.head(
                batch,
                hidden,
                cast(Any, SimpleNamespace(next_tokens=anchors)),
                ArgmaxDraftSampler(),
            )
        )
        # The reference: the drafter's own greedy over the same staged
        # logits the head drafted from, whose biased step logits argmaxed
        # in torch are the reference selection.
        staged = F.distributed_broadcast(
            drafter.logits(
                F.distributed_broadcast(as_tensors(hidden), mesh),
                head_weight,
            ),
            mesh,
        )
        want, biased = drafter.draft(
            staged, F.distributed_broadcast(as_tensors(anchors), mesh)
        )

    got_t = _torch(got).to(torch.int64)
    assert tuple(got_t.shape) == (BATCH, BLOCK)
    assert got.dtype == DType.int32
    biased_t = _torch(biased)
    assert tuple(biased_t.shape) == (BATCH, BLOCK, vocab)
    ref = torch.stack(
        [biased_t[:, i].argmax(dim=-1) for i in range(BLOCK)], dim=1
    )
    torch.testing.assert_close(_torch(want).to(torch.int64), ref)
    torch.testing.assert_close(got_t, ref)
    # Replicated greedy: every rank drew the same full-vocab ids.
    for shard in want.local_shards if want.is_distributed else [want]:
        assert torch.equal(_torch(shard).to(torch.int64), ref)


def test_the_mesh_greedy_picks_full_vocab_columns_on_every_rank() -> None:
    """Each rank's pick is the full vocab's column, padding included.

    ``drafter.draft`` with no sampler is the exact greedy the head invokes
    for ``ArgmaxDraftSampler``, and its step logits are replicated
    whole-vocab (the head's allgather), so every rank's argmax column must
    be the same global token id -- never a shard-local index -- including
    when the winner is the padding column, and when the row is a tie,
    which CPU resolves to the first index (the documented argmax contract).
    """
    config = _small_tp4_config()
    # Six block rows, request-major at BLOCK columns each. The fourth row
    # peaks equally at columns 10 and 90; the first row peaks at the
    # padding column (the new TP4 max), the last at the configured noise
    # token id, so both global ids are covered as winners.
    picks = [_TP4_VOCAB - 1, 0, 13, 10, 42, 96]
    raw = torch.zeros(BATCH * BLOCK, _TP4_VOCAB, dtype=torch.float32)
    for row, column in enumerate(picks):
        raw[row, column] = 1.0
    raw[3, 90] = 1.0
    anchors = _to_max(torch.zeros(BATCH, dtype=torch.int32))
    # The zero-init Markov head keeps every step's bias exactly zero, so
    # each row's argmax is its own logits column.
    drafter = _drafter_on(_TP4_MESH, config, live_markov=False)

    with _eager():
        drafts, _ = drafter.draft(
            F.distributed_broadcast(_to_max(raw), _TP4_MESH),
            F.distributed_broadcast(anchors, _TP4_MESH),
        )

    assert drafts.is_distributed
    assert drafts.dtype == DType.int32
    want = torch.tensor(picks, dtype=torch.int64).reshape(BATCH, BLOCK)
    for shard in drafts.local_shards:
        assert torch.equal(_torch(shard).to(torch.int64), want)
    assert tuple(_torch(drafts).shape) == (BATCH, BLOCK)


class _LegacyGreedySubclass(ArgmaxDraftSampler):
    """A greedy subclass with its own ``sample_next``: the legacy body.

    Exact-type dispatch must treat it as its own sampler and bridge it,
    invoking this override -- an ``isinstance`` check would silently
    swap in the native SPMD greedy instead.
    """

    def __init__(self) -> None:
        self.calls = 0

    def sample_next(
        self,
        logits: TensorValue,
        position: int,
        token_ids: TensorValue | None = None,
    ) -> TensorValue:
        self.calls += 1
        return ops.squeeze(ops.argmax(logits, axis=-1), axis=-1)


@pytest.mark.parametrize("tp", [1, 4])
def test_head_still_bridges_a_greedy_subclass_override(tp: int) -> None:
    """A subclass overriding ``sample_next`` is its own sampler, never
    bypassed.

    Only the exact greedy class takes the native SPMD route, so a
    subclass must go through the bridge with its override invoked. At
    TP1 the override's legacy argmax drafts the reference tokens; at TP4
    it is invoked and hits the very conversion error the native route
    exists to route around -- an ``isinstance`` dispatch would swallow
    the override into a silently different greedy here.
    """
    mesh = _MESH if tp == 1 else _TP4_MESH
    vocab = VOCAB if tp == 1 else _TP4_VOCAB
    config = _small_config(vocab) if tp == 1 else _small_tp4_config()
    head_weight = _head_weight_on(mesh, vocab)
    target = SimpleNamespace(head=SimpleNamespace(weight=head_weight))
    hidden = _to_max(torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16))
    anchors = _to_max(torch.randint(0, vocab, (BATCH,), dtype=torch.int32))
    drafter = _drafter_on(mesh, config, live_markov=True)
    proposer = DSparkDeepseekV41Proposer(drafter, cast(Any, target), config)
    batch = _batch(lengths=[BLOCK] * BATCH)
    sampler = _LegacyGreedySubclass()

    if tp == 1:
        with _eager():
            got = as_tensors(
                proposer.head(
                    batch,
                    hidden,
                    cast(Any, SimpleNamespace(next_tokens=anchors)),
                    sampler,
                )
            )
            # The reference the override's own argmax must reproduce.
            staged = F.distributed_broadcast(
                drafter.logits(
                    F.distributed_broadcast(as_tensors(hidden), mesh),
                    head_weight,
                ),
                mesh,
            )
            want, _ = drafter.draft(
                staged, F.distributed_broadcast(as_tensors(anchors), mesh)
            )
        assert sampler.calls == BLOCK
        torch.testing.assert_close(
            _torch(got).to(torch.int64), _torch(want).to(torch.int64)
        )
    else:
        with _eager(), pytest.raises(ValueError, match="sharded tensor"):
            proposer.head(
                batch,
                hidden,
                cast(Any, SimpleNamespace(next_tokens=anchors)),
                sampler,
            )
        # The override ran and its own legacy conversion refused the
        # replicated step logits; the native greedy would not have raised.
        assert sampler.calls == 1


def test_the_legacy_argmax_path_still_rejects_mesh_logits() -> None:
    """Negative control: the TensorValue conversion is not weakened.

    Routed through the production bridge -- the ModuleV3-to-legacy path
    that accepts the mesh tensor and reaches the legacy sampler's
    conversion -- a replicated TP4 tensor must still be refused loudly,
    so a regression that quietly lets distributed tensors through the
    graph-value conversion fails here.
    """
    bridge = spec_adapters._sample_next(ArgmaxDraftSampler())
    with _eager():
        sharded = F.distributed_broadcast(
            _to_max(torch.randn(BATCH, VOCAB)), _TP4_MESH
        )
        assert sharded.is_distributed
        with pytest.raises(ValueError, match="single-device Tensor"):
            bridge(sharded, 0)


def test_the_legacy_argmax_sampler_still_serves_single_device_values() -> None:
    """TP1/legacy behavior is unchanged: the sampler still works.

    A genuinely single-device ``TensorValue`` is the legacy sampler's
    own world -- the distributed guard never fires -- so it keeps
    serving the graph-value callers it always did, picking each row's
    argmax column.
    """
    raw = torch.randn(BATCH, VOCAB)
    raw[0] = 0.0  # a tie row: the first index, per the CPU contract
    with _eager():
        logits = ops.constant(
            raw.numpy(), DType.float32, device=DeviceRef.CPU()
        )
        got = as_tensors(ArgmaxDraftSampler().sample_next(logits, 0))

    assert tuple(got.shape) == (BATCH,)
    got_t = _torch(got)
    assert got_t[0] == 0
    assert torch.equal(got_t[1:].to(torch.int64), raw[1:].argmax(dim=-1))


def test_the_proposer_declares_the_block_contract() -> None:
    """Every block position drafts, anchor included; the block is fixed."""
    config = _small_config()
    drafter = DSparkDrafter(config, _MESH, rope=_rope())
    proposer = DSparkDeepseekV41Proposer(drafter, _target(0), config)
    assert proposer.block_size == BLOCK
    assert proposer.mask_token_id == 96
    assert proposer.samples_from_anchor is True
    assert proposer.supports_zero_draft_rows is False


def test_the_proposer_needs_a_noise_token() -> None:
    config = _small_config()
    drafter = DSparkDrafter(config, _MESH, rope=_rope())
    with pytest.raises(ValueError, match="dspark_noise_token_id"):
        DSparkDeepseekV41Proposer(
            drafter,
            _target(0),
            dataclasses.replace(config, dspark_noise_token_id=None),
        )


def test_embed_block_passes_the_ids_through() -> None:
    """The drafter embeds through the target's own embedding."""
    config = _small_config()
    drafter = DSparkDrafter(config, _MESH, rope=_rope())
    proposer = DSparkDeepseekV41Proposer(drafter, _target(0), config)
    ids = _graph_typed(_to_max(torch.arange(2 * BLOCK, dtype=torch.int32)))
    got = proposer.embed_block(_batch(lengths=[1]), ids)
    assert got == [ids]


def test_materialize_stores_from_the_payload_and_context_cache() -> None:
    """The store reads the verify payload, never re-derived rows.

    The store's base is the payload's own per-request origin -- each
    request's first payload row's position -- not the context cache's
    staged lengths.
    """
    config = _small_config()
    taps = _to_max(torch.zeros(4, HIDDEN * STAGES))
    rows = _to_max(torch.tensor([0, 2, 4], dtype=torch.uint32))
    positions = _to_max(torch.arange(4, dtype=torch.int32))

    class _RecordingDrafter:
        def __init__(self) -> None:
            self.mesh = _MESH
            self.data_parallel_degree = 1
            self.stored: tuple[Any, ...] | None = None

        def store_context(
            self,
            swa: PagedCacheValues,
            taps_in: Tensor,
            row_offsets: Tensor,
            positions_in: Tensor,
        ) -> None:
            self.stored = (swa, taps_in, row_offsets, positions_in)

    recording = _RecordingDrafter()
    proposer = DSparkDeepseekV41Proposer(
        cast(Any, recording), _target(0), config
    )
    with _eager():
        proposer.materialize(
            _batch(lengths=[2, 2]),
            (taps, rows, positions),
            [_upstream_kv([2, 2])],
        )
    assert recording.stored is not None
    swa, taps_in, rows_in, positions_in = recording.stored
    assert taps_in is taps
    assert rows_in is rows
    assert positions_in is positions
    assert torch.equal(
        _torch(swa.cache_lengths), torch.tensor([0, 2], dtype=torch.int32)
    )


def test_materialize_bases_the_store_at_the_retained_rows_origin() -> None:
    """A narrowed verify's kept rows store at their true positions.

    Request 0 ran a 200-row prefill chunk past a length-100 context and
    the narrowing kept the last 128 rows: payload positions 172..299, the
    72 dropped rows advanced away from the staged length 100. Request 1
    ran a short chunk off a length-300 context and kept all of it.
    Basing the store at the staged lengths would write request 0's kept
    rows 72 slots too early -- a rotation of the drafter's window ring
    that the window read, which resolves true positions, would misread
    until the decode steps' stores had overwritten it.
    """
    config = _small_config()
    kept = [128, 50]
    rows = _to_max(
        torch.tensor([0, kept[0], kept[0] + kept[1]], dtype=torch.uint32)
    )
    positions = _to_max(
        torch.cat(
            [
                torch.arange(172, 172 + kept[0], dtype=torch.int32),
                torch.arange(300, 300 + kept[1], dtype=torch.int32),
            ]
        )
    )
    taps = _to_max(torch.zeros(kept[0] + kept[1], HIDDEN * STAGES))

    class _RecordingDrafter:
        def __init__(self) -> None:
            self.mesh = _MESH
            self.data_parallel_degree = 1
            self.stored: tuple[Any, ...] | None = None

        def store_context(
            self,
            swa: PagedCacheValues,
            taps_in: Tensor,
            row_offsets: Tensor,
            positions_in: Tensor,
        ) -> None:
            self.stored = (swa, taps_in, row_offsets, positions_in)

    recording = _RecordingDrafter()
    proposer = DSparkDeepseekV41Proposer(
        cast(Any, recording), _target(0), config
    )
    with _eager():
        proposer.materialize(
            _batch(lengths=kept),
            (taps, rows, positions),
            # The engine-staged pre-iteration lengths: unshifted by the
            # rows the narrowing dropped.
            [_upstream_kv([100, 300])],
        )
    assert recording.stored is not None
    swa, *_ = recording.stored
    assert torch.equal(
        _torch(swa.cache_lengths),
        torch.tensor([172, 300], dtype=torch.int32),
    )


def test_context_store_origins_are_the_staged_lengths_unshifted() -> None:
    """No dropped rows: the origin is the staged length itself.

    Decode steps and chunks no longer than the window drop nothing, so
    the payload's first-row positions are the pre-iteration lengths and
    the fix is the identity there.
    """
    with _eager():
        origins = spec_adapters.context_store_origins(
            _to_max(torch.tensor([0, 6, 9], dtype=torch.uint32)),
            _to_max(torch.arange(4, 13, dtype=torch.int32)),
            DType.int32,
        )
    assert torch.equal(
        _torch(origins), torch.tensor([4, 10], dtype=torch.int32)
    )
