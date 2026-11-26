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
"""CPU tests for DSpark speculative-decoding adapters."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from dspark_test_support import TESTDATA
from max import functional as F
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Dim,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MultiKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.nn.kv_cache import PagedCacheValues as UpstreamPagedCacheValues
from max.pipelines.architectures.deepseekV41_modulev3 import (
    deepseekV41 as trunk,
)
from max.pipelines.architectures.deepseekV41_modulev3 import spec_adapters
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    DeepseekV41,
)
from max.pipelines.architectures.deepseekV41_modulev3.dspark_drafter import (
    DSparkDrafter,
    _lowest_argmax,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
    extra_input_names,
    extra_inputs,
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
from max.pipelines.lib.role_contracts import VisionInputs
from max.pipelines.speculative import driver as spec_driver
from max.pipelines.speculative._tensor_compat import as_tensors
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
)
from max.pipelines.speculative.draft_scores import top_token_logit
from max.tensor import Tensor, default_device
from max.tensor.realization_context import (
    EagerRealizationContext,
    current_realization_context,
    ensure_context,
)
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated, Sharded

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
DEPTH = 3

_MESH = DeviceMesh((CPU(),), (1,), ("tp",))


def test_trunk_tap_captures_the_layer_entry_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hc, hidden, rows = 4, 3, 2
    tap_layer_id = 7
    entry = np.arange(rows * hc * hidden, dtype=np.float32).reshape(
        rows, hc, hidden
    )
    entry += 1
    carried = np.full((rows, hc, hidden), 101.0, np.float32)
    pre = np.full_like(carried, 202.0)
    post_attention = np.full((rows, hidden), -303.0, np.float32)
    comb = np.zeros((rows, hc, hc), np.float32)
    row_values = SimpleNamespace(offsets=None, positions=None)

    class Layer:
        layer_id = tap_layer_id
        group = "tap"
        plan = None

        def __call__(self, *args):
            return (
                carried,
                pre,
                post_attention,
                post_attention,
                comb,
                entry,
                [],
            )

    batch = SimpleNamespace(
        rows=row_values,
        data_parallel_splits=None,
        input_row_offsets_i64=None,
        replay=None,
        swa_begin=None,
        global_write_floor=None,
        image_token_mask=None,
        kv={},
        ep=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(hc_mult=hc, engram_layer_ids=()),
        dp=1,
        decoder_start=None,
        layers=[Layer()],
        _with_live_positions=lambda rows_arg: rows_arg,
    )
    monkeypatch.setattr(
        trunk,
        "mhc_identity_writeback",
        lambda stream: (post_attention, post_attention, comb),
    )
    monkeypatch.setattr(
        trunk, "mhc_post", lambda y, stream, post, coeffs, hc_mult: stream
    )
    monkeypatch.setattr(trunk, "leaf_values", lambda *_: ())
    monkeypatch.setattr(trunk, "slot_constants", lambda *_: ())
    monkeypatch.setattr(
        trunk,
        "_hc_mean",
        lambda stream: Tensor.from_dlpack(
            np.ascontiguousarray(stream.mean(axis=1))
        ),
    )

    run_trunk = cast(
        Callable[..., tuple[Tensor, Tensor, dict[int, Tensor] | None]],
        DeepseekV41._run_trunk,
    )
    _, _, taps = run_trunk(model, batch, carried, pre, (tap_layer_id,))

    assert taps is not None
    tap = taps[tap_layer_id].to_numpy()
    expected = entry.mean(axis=1)
    np.testing.assert_array_equal(tap, expected)
    assert not np.array_equal(tap, carried.mean(axis=1))
    expanded_post = np.broadcast_to(post_attention[:, None, :], entry.shape)
    assert not np.array_equal(tap, expanded_post.mean(axis=1))


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
    extra: Mapping[str, Tensor | Buffer] | None = None,
    vision: tuple[Tensor, Tensor] | None = None,
    verify_lengths: list[int] | None = None,
) -> BlockBatch:
    """A single-device BlockBatch over requests of the given lengths.

    The driver's tensor slots are graph-typed, but this eager run carries
    realized tensors in them -- the same duality ``_upstream_kv`` casts
    across -- so the tensor fields go through :func:`_graph_typed`.
    """
    if extra is None:
        extra = _extras()
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
        vision_embeddings=[] if vision is None else [_graph_typed(vision[0])],
        vision_scatter_indices=(
            [] if vision is None else [_graph_typed(vision[1])]
        ),
        extra=extra,
        num_accepted=None,
        draft_slot_ids=None,
        draft_block_offsets=None,
        verify_lengths=(
            _graph_typed(
                _to_max(torch.tensor(verify_lengths, dtype=torch.int32))
            )
            if verify_lengths is not None
            else None
        ),
        block_size=BLOCK,
    )


_DP = 2
_DP_MESH = DeviceMesh((CPU(), CPU()), (_DP,), ("dp",))
_DP_ROWS = DeviceMapping(_DP_MESH, (Sharded(0),))


def _dp_batch(
    *,
    lengths: list[int],
    verify_lengths: list[int],
) -> BlockBatch:
    """A two-replica BlockBatch: one request per rank, in its local frame.

    The host mirror carries the merged offsets the adapter's DP branch
    requires, and each rank's row inputs are its own replica's rows, as
    the trunk's split leaves them.
    """
    assert len(lengths) == _DP
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    # One request per rank: the split is the request boundary itself.
    splits = list(range(len(lengths) + 1))
    kv = [_upstream_kv(lengths[splits[r] : splits[r + 1]]) for r in range(_DP)]
    host_mirror = _to_max(
        torch.tensor(
            [o + i * BLOCK for i, o in enumerate(offsets)],
            dtype=torch.int64,
        )
    )
    return dataclasses.replace(
        _batch(lengths=lengths, verify_lengths=verify_lengths),
        kv_collections=kv,
        draft_kv_collections=kv,
        passthrough_kv={"global": kv},
        devices=[DeviceRef.from_device(d) for d in _DP_MESH.devices],
        host_merged_offsets=_graph_typed(host_mirror),
        data_parallel_splits=_graph_typed(
            _to_max(torch.tensor(splits, dtype=torch.int32))
        ),
        data_parallel_degree=_DP,
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
        self.vision: VisionInputs[Tensor] | None = None
        self.history: Any = None
        self.ep_config = None

    def ep_buffers(self, values: Sequence[Tensor]) -> None:
        assert not values

    def batch_from_parts(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv: dict[str, PagedCacheValues],
        extras: dict[str, Tensor],
        *,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        ep: Any = None,
        draft_width: Tensor | None = None,
        history: Any = None,
        vision: VisionInputs[Tensor] | None = None,
    ) -> Any:
        self.assembled = (tokens, input_row_offsets, kv, extras)
        self.history = history
        self.draft_width = draft_width
        self.vision = vision
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


class _RecordingDPTarget(_RecordingTarget):
    """The recording trunk over two CPU replicas.

    Each rank's hidden rows are its own replica's slice of the marked
    window, in its local row frame, as the trunk's DP split leaves them;
    the joined logits come back off rank 0, as the allgathered head
    reads them.
    """

    def __init__(self, lengths: list[int]) -> None:
        total = sum(lengths)
        super().__init__(total)
        self.mesh = _DP_MESH
        self.dp = _DP
        marks = torch.arange(total, dtype=torch.int32)
        marked = torch.stack([marks] * HIDDEN, dim=1).to(torch.bfloat16)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        self._offsets = offsets
        shards = []
        for r in range(_DP):
            rows = marked[offsets[r] : offsets[r + 1]]
            shards.append(_to_max(rows.clone()))
        self.marked = Tensor.from_shard_values(shards, _DP_ROWS)
        self.taps = Tensor.from_shard_values(
            [
                _to_max(
                    torch.zeros(offsets[r + 1] - offsets[r], HIDDEN * STAGES)
                )
                for r in range(_DP)
            ],
            _DP_ROWS,
        )
        self.positions = Tensor.from_shard_values(
            [
                _to_max(
                    torch.arange(offsets[r + 1] - offsets[r], dtype=torch.int32)
                    + 100
                )
                for r in range(_DP)
            ],
            _DP_ROWS,
        )

    def batch_from_parts(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv: dict[str, PagedCacheValues],
        extras: dict[str, Tensor],
        *,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        ep: Any = None,
        draft_width: Tensor | None = None,
        history: Any = None,
        vision: VisionInputs[Tensor] | None = None,
    ) -> Any:
        self.assembled = (tokens, input_row_offsets, kv, extras)
        self.draft_width = draft_width
        self.history = history
        self.vision = vision
        assert data_parallel_splits is not None
        # The host knows the split; reading the device offsets back here
        # would realize inside the still-open eager context.
        shards = []
        for r in range(_DP):
            local = [o - self._offsets[r] for o in self._offsets[r : r + 2]]
            shards.append(_to_max(torch.tensor(local, dtype=torch.uint32)))
        return SimpleNamespace(
            row_offsets=Tensor.from_shard_values(shards, _DP_ROWS),
            positions=self.positions,
        )

    def logits_all_rows(self, hidden: Tensor) -> Tensor:
        return hidden.local_shards[0]


class _TP4RecordingTarget(_RecordingTarget):
    """The recording trunk over a TP mesh, its rows replicated.

    The trunk's hidden rows live on the mesh, as production's do, and its
    head returns one device's logits, as production's allgathered head does.
    """

    def __init__(self, total: int, marks: torch.Tensor) -> None:
        super().__init__(total)
        self.mesh = _TP4_MESH
        self.marked = Tensor.from_shard_values(
            [_to_max(torch.stack([marks] * HIDDEN, dim=1).to(torch.bfloat16))]
            * _TP4_MESH.num_devices,
            DeviceMapping(_TP4_MESH, (Replicated(),)),
        )

    def logits_all_rows(self, hidden: Tensor) -> Tensor:
        return hidden.local_shards[0]


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
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
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


def test_verify_gathers_each_rows_verified_prefix() -> None:
    """A per-row verify reads only the rows the row's drafts occupy.

    Row ``r`` verifies ``k_r`` drafts, so its window ends with the anchor
    and those ``k_r`` rows; the slots past ``k_r`` repeat the last real
    row, never the next request's window.
    """
    lengths = [3, 6]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
    with _eager():
        verified = adapter(_batch(lengths=lengths, verify_lengths=[0, 2]))
        got = as_tensors(verified.logits)

    got = _torch(got)
    assert tuple(got.shape) == (2 * (BLOCK + 1), HIDDEN)
    rows = got[:, 0].to(torch.int32)
    # Request 0 verifies nothing: its anchor row repeats. Request 1
    # verifies 2 of its 3 rows and pads with the last one.
    assert torch.equal(rows, torch.tensor([2, 2, 2, 6, 7, 8]))
    assert bool((got == got[:, :1]).all())


def test_verify_at_full_width_gathers_the_uniform_rows() -> None:
    """Per-row verify at full width equals the uniform last-K+1 gather."""
    lengths = [3, 6]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
    with _eager():
        verified = adapter(
            _batch(lengths=lengths, verify_lengths=[BLOCK, BLOCK])
        )
        got = as_tensors(verified.logits)

    got = _torch(got)
    rows = got[:, 0].to(torch.int32)
    assert torch.equal(rows, torch.tensor([0, 1, 2, 6, 7, 8]))


def test_dp_verify_gathers_each_replicas_prefix_in_its_local_frame() -> None:
    """Each replica gathers its own rows from its rebased offsets.

    The global verify lengths are split per rank, and the joined block
    holds the replicas' rows in request order.
    """
    lengths = [3, 6]
    with _eager():
        adapter = DeepseekV41BlockTarget(
            cast(DeepseekV41, _RecordingDPTarget(lengths)), _kv_params_stub()
        )
        verified = adapter(_dp_batch(lengths=lengths, verify_lengths=[1, 2]))
        got = as_tensors(verified.logits)

    got = _torch(got)
    assert tuple(got.shape) == (2 * (BLOCK + 1), HIDDEN)
    rows = got[:, 0].to(torch.int32)
    # Replica 0's local rows 1, 2, 2 are global 1, 2, 2; replica 1's
    # local 3, 4, 5 are global 6, 7, 8.
    assert torch.equal(rows, torch.tensor([1, 2, 2, 6, 7, 8]))


def test_verify_carries_the_taps_with_the_rows_that_produced_them() -> None:
    """The proposer must store context at the verify forward's own rows.

    The payload is the taps plus the batch's row offsets and positions --
    the store lands at exactly the positions the verify ran at, never
    re-derived.
    """
    lengths = [4, 2]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
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
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
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
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
    with _eager():
        adapter(_batch(lengths=[3, 3]))
    assert target.draft_width is not None
    assert int(_torch(target.draft_width)) == BLOCK


def test_verify_hands_the_trunk_the_steps_image_rows() -> None:
    """The verify merges a step's image span rows, as the trunk alone does.

    The rows reach the trunk replicated over its mesh. A step without image
    inputs hands none, so the trunk skips the merge. Built lazily: nothing
    is realized.
    """
    lengths = [3, 3]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
    with F.lazy():
        rows = Tensor.zeros([2, HIDDEN], dtype=DType.bfloat16, device=CPU())
        indices = Tensor.zeros([2], dtype=DType.int32, device=CPU())
        adapter(_batch(lengths=lengths, vision=(rows, indices)))
        vision = target.vision
        adapter(_batch(lengths=lengths))
    assert vision is not None
    assert [int(d) for d in vision.embeddings.shape] == [2, HIDDEN]
    assert [int(d) for d in vision.scatter_indices.shape] == [2]
    for value in (vision.embeddings, vision.scatter_indices):
        assert value.mesh.devices == _MESH.devices
        assert list(value.mapping.placements) == [Replicated()]
    assert target.vision is None


def test_verify_fails_loud_on_a_missing_extra() -> None:
    """A silently-defaulted extra would feed the trunk garbage masks."""
    lengths = [2]
    target = _RecordingTarget(sum(lengths))
    adapter = DeepseekV41BlockTarget(
        cast(DeepseekV41, target), _kv_params_stub()
    )
    batch = _batch(lengths=lengths)
    starved = {
        name: value
        for name, value in batch.extra.items()
        if name != "engram_lookback_dead"
    }
    with pytest.raises(ValueError, match="engram_lookback_dead"), _eager():
        adapter(dataclasses.replace(batch, extra=starved))


def test_verify_records_the_window_only_when_the_export_carries_the_history() -> (
    None
):
    """The overlap export's verify owes the driver a lookback record.

    The default export records nothing (the host window needs no history);
    the history export's verify returns a roll-forward the driver runs once
    the acceptance verdict exists, which is the only point the accepted
    drafts the window ends with are known.
    """
    lengths = [2, 2]
    adapter = DeepseekV41BlockTarget(_target(sum(lengths)), _kv_params_stub())
    with _eager():
        default = adapter(_batch(lengths=lengths))
    assert default.roll_forward is None

    config = dataclasses.replace(
        _v41_config(), engram_lookback_history=True, artifact_export=True
    )
    batch = 2
    extras: dict[str, Tensor | Buffer] = {**_extras()}
    del extras["engram_lookback"]
    extras["engram_lookback_ids"] = _to_max(
        torch.zeros(batch, 3, dtype=torch.int64)
    )
    extras["engram_lookback_source"] = _to_max(
        torch.full((batch,), -1, dtype=torch.int64)
    )
    assert set(extras) == set(extra_input_names(config))
    target = _target(sum(lengths))
    target.config = config
    target.interface_config = config
    # The history is a recurrent leaf of the cache tree, beside the paged ones.
    kv_params = _kv_params_stub()
    kv_params.children["lookback_history"] = object()
    # The eager run carries realized tensors in the graph-typed slots.
    history: Any = RecurrentStateInputsPerDevice(
        leaves=(
            RecurrentLeafInputs(
                pool=cast(Any, _to_max(torch.zeros(4, 3, dtype=torch.int64))),
                live_row_ids=_graph_typed(
                    _to_max(torch.tensor([[0, 1]], dtype=torch.uint32))
                ),
            ),
        )
    )
    with_history = _batch(lengths=lengths, extra=extras)
    with_history = dataclasses.replace(
        with_history,
        passthrough_kv={
            **with_history.passthrough_kv,
            "lookback_history": [history],
        },
    )
    adapter = DeepseekV41BlockTarget(target, kv_params)
    with _eager():
        recording = adapter(with_history)
    assert recording.roll_forward is not None
    assert target.history is not None


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
        cast(Any, drafter), cast(DeepseekV41, target), config
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
        proposer = DSparkDeepseekV41Proposer(
            drafter, cast(DeepseekV41, target), config
        )
        batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        got = as_tensors(
            proposer.head(
                batch,
                hidden,
                cast(Accepted, accepted),
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
        proposer = DSparkDeepseekV41Proposer(
            drafter, cast(DeepseekV41, target), config
        )
        batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        drafts = as_tensors(
            proposer.head(
                batch,
                hidden,
                cast(Accepted, SimpleNamespace(next_tokens=anchors)),
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
    proposer = DSparkDeepseekV41Proposer(
        drafter, cast(DeepseekV41, target), config
    )
    batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        got = as_tensors(
            proposer.head(
                batch,
                hidden,
                cast(Accepted, SimpleNamespace(next_tokens=anchors)),
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


def _vocab_parallel_drafts(
    drafter: DSparkDrafter,
    head_weight: Tensor,
    hidden: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    with _eager():
        got = drafter.draft_vocab_parallel(
            F.distributed_broadcast(_to_max(hidden), _TP4_MESH),
            head_weight,
            F.distributed_broadcast(_to_max(anchors), _TP4_MESH),
        )
    assert not isinstance(got, tuple)
    assert got.dtype == DType.int32
    drafts = _torch(got).to(torch.int64)
    assert tuple(drafts.shape) == (BATCH, BLOCK)
    return drafts


def test_vocab_parallel_head_drafts_the_joined_heads_tokens() -> None:
    config = _small_tp4_config()
    head_weight = _head_weight_on(_TP4_MESH, _TP4_VOCAB)
    hidden = torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16)
    anchors = torch.randint(0, _TP4_VOCAB, (BATCH,), dtype=torch.int32)
    drafter = _drafter_on(_TP4_MESH, config, live_markov=True)

    got = _vocab_parallel_drafts(drafter, head_weight, hidden, anchors)

    with _eager():
        want, _ = drafter.draft(
            F.distributed_broadcast(
                drafter.logits(
                    F.distributed_broadcast(_to_max(hidden), _TP4_MESH),
                    head_weight,
                ),
                _TP4_MESH,
            ),
            F.distributed_broadcast(_to_max(anchors), _TP4_MESH),
        )
    assert torch.equal(got, _torch(want).to(torch.int64))


def test_vocab_parallel_head_breaks_cross_shard_ties_to_the_lowest_index() -> (
    None
):
    config = _small_tp4_config()
    weight = torch.empty(_TP4_VOCAB, HIDDEN).uniform_(-0.5, 0.5)
    weight[7] = weight[71] = 1.0
    head_weight = Tensor(weight.to(torch.bfloat16), device=_TP4_MESH).to(
        DeviceMapping(_TP4_MESH, (Sharded(0),))
    )
    hidden = torch.ones(BATCH * BLOCK, HIDDEN).to(torch.bfloat16)
    anchors = torch.zeros(BATCH, dtype=torch.int32)
    drafter = _drafter_on(_TP4_MESH, config, live_markov=False)

    got = _vocab_parallel_drafts(drafter, head_weight, hidden, anchors)

    assert torch.equal(got, torch.full((BATCH, BLOCK), 7, dtype=torch.int64))


def test_the_mesh_greedy_picks_full_vocab_columns_on_every_rank() -> None:
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
    subclass must go through the bridge with its override invoked -- an
    ``isinstance`` dispatch would swallow the override into a silently
    different greedy. The override is a graph-value sampler like the
    driver's sampled one, so on the TP4 mesh the bridge must hand it rank
    0's single-device logits, not the replicated mesh tensor its
    conversion refuses, and stage its picks back onto the mesh.
    """
    mesh = _MESH if tp == 1 else _TP4_MESH
    vocab = VOCAB if tp == 1 else _TP4_VOCAB
    config = _small_config(vocab) if tp == 1 else _small_tp4_config()
    head_weight = _head_weight_on(mesh, vocab)
    target = SimpleNamespace(head=SimpleNamespace(weight=head_weight))
    hidden = _to_max(torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16))
    anchors = _to_max(torch.randint(0, vocab, (BATCH,), dtype=torch.int32))
    drafter = _drafter_on(mesh, config, live_markov=True)
    proposer = DSparkDeepseekV41Proposer(
        drafter, cast(DeepseekV41, target), config
    )
    batch = _batch(lengths=[BLOCK] * BATCH)
    sampler = _LegacyGreedySubclass()

    with _eager():
        got = as_tensors(
            proposer.head(
                batch,
                hidden,
                cast(Accepted, SimpleNamespace(next_tokens=anchors)),
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


def test_the_legacy_argmax_path_still_rejects_mesh_logits() -> None:
    """Negative control: the TensorValue conversion is not weakened.

    The bridge hands a graph-value sampler rank 0's single-device copy;
    the sampler's own conversion must still refuse a replicated TP4
    tensor loudly, so a regression that quietly lets distributed tensors
    through the graph-value conversion fails here.
    """
    with _eager():
        sharded = F.distributed_broadcast(
            _to_max(torch.randn(BATCH, VOCAB)), _TP4_MESH
        )
        assert sharded.is_distributed
        with pytest.raises(ValueError, match="single-device Tensor"):
            ArgmaxDraftSampler().sample_next(cast(Any, sharded), 0)


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


def test_replica_proposals_enable_replica_verify_logits() -> None:
    """The q and target-logit ownership must select the same replica path."""
    with F.lazy():
        config = _small_config()
        drafter = DSparkDrafter(config, _MESH, rope=_rope())
        target = _target(0)
        assert isinstance(config.kv_params, MultiKVCacheParams)
        verify = DeepseekV41BlockTarget(target, config.kv_params)
        proposer = spec_adapters.block_proposer(
            verify,
            drafter,
            config,
            cast(
                Any,
                SimpleNamespace(
                    draft_probs_layout="replica",
                    draft_head_parallelism="replica",
                    per_row_verify=False,
                    draft_score_source=None,
                ),
            ),
        )
    assert verify.replica_logits
    assert proposer.per_replica_head
    assert proposer.per_replica_probs


def test_the_replica_head_refuses_head_scores() -> None:
    """The per-replica head mounts no confidence head, so a per-row boot
    asking for head scores fails at construction, not at serve time."""
    with F.lazy():
        config = _small_config()
        drafter = DSparkDrafter(config, _MESH, rope=_rope())
        target = _target(0)
        with pytest.raises(ValueError, match="head-free only"):
            spec_adapters.DSparkDeepseekV41Proposer(
                drafter,
                target,
                config,
                per_replica_head=True,
                scores=True,
                head_scores=True,
            )


def test_replica_proposals_do_not_use_the_joined_draft_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with F.lazy():
        config = _small_config()
        drafter = DSparkDrafter(config, _MESH, rope=_rope())
        proposer = DSparkDeepseekV41Proposer(
            drafter,
            _target(0),
            config,
            per_replica_head=True,
            per_replica_probs=True,
        )
    proposer.dp = 8
    hidden, expected = object(), object()
    monkeypatch.setattr(spec_adapters, "as_tensors", lambda value: value)
    monkeypatch.setattr(
        proposer,
        "_replica_head",
        lambda batch, value, accepted, sampler: (
            expected if value is hidden else None
        ),
    )
    assert (
        proposer.head(
            cast(Any, None), cast(Any, hidden), cast(Any, None), cast(Any, None)
        )
        is expected
    )


@pytest.mark.parametrize("degree", [4, 8])
def test_raw_trailing_tree_preserves_declared_order(degree: int) -> None:
    """The fused named signature preserves each actual context/resource leaf."""
    config = _small_config()
    config.raw_checkpoint_weights = True
    config.dspark_trailing_inputs = extra_inputs(
        _v41_config(),
        [DeviceRef.GPU(rank) for rank in range(degree)],
        degree,
        verify_rows=True,
        artifact=True,
    )
    named = spec_adapters.trailing_input_tree(config)
    assert named is not None
    assert tree.leaves(named) == spec_adapters.trailing_input_types(config)
    assert {cast(Any, group).kind for group in named.values()} == {
        "batch_context",
        "static_buffers",
    }
    context = named["context_0"]
    constants = named["constants_1"]
    assert isinstance(context, Mapping) and isinstance(constants, Mapping)
    assert list(context)[:2] == ["window_start", "write_floor"]
    assert list(constants) == [
        f"engram_table_{ordinal}_{part}"
        for ordinal in range(2)
        for part in ("weight", "scale")
    ]
    assert spec_adapters.ADAPTER.trailing_input_tree(config) is not None


def test_legacy_trailing_signature_remains_positional() -> None:
    assert spec_adapters.trailing_input_tree(_small_config()) is None


def test_raw_trailing_tree_refuses_unnamed_history() -> None:
    config = _small_config()
    config.raw_checkpoint_weights = True
    config.dspark_trailing_inputs = [
        (
            "engram_device_history",
            TensorType(DType.int32, [1], DeviceRef.GPU(0)),
        )
    ]
    with pytest.raises(ValueError, match="no artifact input producer"):
        spec_adapters.trailing_input_tree(config)


def test_text_only_artifact_omits_vision_inputs_but_preserves_bare_abi() -> (
    None
):
    config = _small_config()
    config.vision = None
    assert spec_adapters.ADAPTER.takes_vision_inputs(config)
    config.raw_checkpoint_weights = True
    assert not spec_adapters.ADAPTER.takes_vision_inputs(config)
    config.vision = cast(Any, object())
    assert spec_adapters.ADAPTER.takes_vision_inputs(config)


def test_head_free_scores_fill_the_member_without_the_head() -> None:
    """Head-free scores need no mounted head: the proposer returns the
    drafts beside the logit(q_max) of the same biased logits that drew
    them, so the scores stay aligned with the drafts position for
    position."""
    torch.manual_seed(0)
    config = _small_config()
    head_weight = _to_max(torch.randn(VOCAB, HIDDEN).to(torch.bfloat16))
    target = SimpleNamespace(head=SimpleNamespace(weight=head_weight))
    hidden = _to_max(torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16))
    anchors = _to_max(torch.randint(0, VOCAB, (BATCH,), dtype=torch.int32))
    accepted = SimpleNamespace(next_tokens=anchors)
    with default_device(_MESH):
        drafter = DSparkDrafter(config, _MESH, rope=_rope())
        proposer = DSparkDeepseekV41Proposer(
            drafter, cast(DeepseekV41, target), config, scores=True
        )
    assert proposer.confidence_outputs is True
    batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        headed = proposer.head(
            batch, hidden, cast(Accepted, accepted), ArgmaxDraftSampler()
        )
        assert isinstance(headed, tuple)
        drafts, scores = headed
        drafts = as_tensors(drafts)
        scores = as_tensors(scores)
        logits = drafter.logits(hidden, head_weight)
        _, biased = drafter.draft(logits, anchors)

    assert tuple(drafts.shape) == (BATCH, BLOCK)
    assert tuple(scores.shape) == (BATCH, BLOCK)
    assert scores.dtype == DType.float32
    # The float64 reference of the same biased logits: gap = L_max - lse,
    # score = gap - log1p(-exp(gap)).
    wide = _torch(biased).to(torch.float64)
    gap = wide.max(dim=-1).values - torch.logsumexp(wide, dim=-1)
    want = gap - torch.log1p(-torch.exp(gap))
    torch.testing.assert_close(
        _torch(scores), want.to(torch.float32), rtol=1e-4, atol=1e-4
    )


def test_head_free_scores_take_the_vocab_parallel_route() -> None:
    """The head-free source rides the vocab-parallel route: the scores
    equal the joined route's ``top_token_logit`` over the same biased
    logits, and the drafts are unchanged with scores on versus off."""
    torch.manual_seed(0)
    config = _small_tp4_config()
    head_weight = _head_weight_on(_TP4_MESH, _TP4_VOCAB)
    hidden = torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16)
    anchors = torch.randint(0, _TP4_VOCAB, (BATCH,), dtype=torch.int32)
    target = SimpleNamespace(head=SimpleNamespace(weight=head_weight))
    with default_device(_TP4_MESH):
        drafter = DSparkDrafter(config, _TP4_MESH, rope=_rope())
        proposer = DSparkDeepseekV41Proposer(
            drafter, cast(DeepseekV41, target), config, scores=True
        )
    proposer.vocab_parallel_draft = True
    accepted = SimpleNamespace(next_tokens=_to_max(anchors))
    batch = _batch(lengths=[BLOCK] * BATCH)

    with _eager():
        block_hs = F.distributed_broadcast(_to_max(hidden), _TP4_MESH)
        headed = proposer.head(
            batch, block_hs, cast(Accepted, accepted), ArgmaxDraftSampler()
        )
        assert isinstance(headed, tuple)
        drafts, scores = headed
        drafts = as_tensors(drafts)
        scores = as_tensors(scores)
        # The joined route's own reference: the same biased logits the
        # joined head materializes, scored by the same helper.
        logits = F.distributed_broadcast(
            drafter.logits(block_hs, head_weight), _TP4_MESH
        )
        want_drafts, biased = drafter.draft(
            logits, F.distributed_broadcast(_to_max(anchors), _TP4_MESH)
        )
        reference = top_token_logit(biased)
        # The drafts are unchanged with scores on versus off.
        bare = drafter.draft_vocab_parallel(
            block_hs,
            head_weight,
            F.distributed_broadcast(_to_max(anchors), _TP4_MESH),
        )

    assert tuple(drafts.shape) == (BATCH, BLOCK)
    assert tuple(scores.shape) == (BATCH, BLOCK)
    assert scores.dtype == DType.float32
    # The route's drafts are int32; the joined head's are int64 --
    # torch.equal compares values across the promotion.
    assert torch.equal(_torch(drafts), _torch(want_drafts).to(torch.int64))
    assert torch.equal(_torch(drafts), _torch(bare).to(torch.int64))
    torch.testing.assert_close(
        _torch(scores), _torch(reference), rtol=1e-5, atol=1e-5
    )
    # An independent float64 reference over the same tensor: the
    # in-context assert above cannot see a helper-internal error.
    wide = _torch(biased).to(torch.float64)
    gap = wide.max(dim=-1).values - torch.logsumexp(wide, dim=-1)
    want = gap - torch.log1p(-torch.exp(gap))
    torch.testing.assert_close(
        _torch(scores), want.to(torch.float32), rtol=1e-4, atol=1e-4
    )


def _excluded_mass(logits: Tensor) -> Tensor:
    """The unfused excluded mass: ``sum(exp(l_j - max))`` minus one top."""
    logits = F.cast(logits, DType.float32)
    top = F.max(logits, axis=-1)
    shifted = F.exp(logits - top)
    is_top = F.equal(logits, top)
    background = F.sum(
        shifted * F.cast(F.logical_not(is_top), DType.float32), axis=-1
    )
    ties = F.sum(F.cast(is_top, DType.float32), axis=-1)
    # Reduce the tie term before the add: a float32 sum near one would
    # absorb the small deficit the excluded sum keeps.
    return F.squeeze(background + (ties - 1.0), axis=-1)


def _combine_shard_logit(
    maxima: Tensor, indices: Tensor, excluded: Tensor, top_index: Tensor
) -> Tensor:
    """The unfused per-position combine over ``[shards, b]`` pieces."""
    maxima = F.cast(maxima, DType.float32)
    l_top = F.squeeze(F.max(maxima, axis=0), axis=0)
    rescaled = F.exp(maxima - l_top)
    mass = F.sum(excluded * rescaled, axis=0)
    is_top_shard = F.equal(indices, F.cast(top_index, indices.dtype))
    restored = F.sum(
        rescaled * F.cast(F.logical_not(is_top_shard), DType.float32),
        axis=0,
    )
    return F.squeeze(-F.log(mass + restored), axis=0)


def test_vocab_parallel_scores_match_the_unfused_pieces() -> None:
    """The fused branch's drafts and scores equal the unfused ops on
    the same shard logits: one fused launch replaces the max, argmax
    and excluded-mass reductions per position per rank."""
    torch.manual_seed(0)
    config = _small_tp4_config()
    head_weight = _head_weight_on(_TP4_MESH, _TP4_VOCAB)
    hidden = torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16)
    anchors = torch.randint(0, _TP4_VOCAB, (BATCH,), dtype=torch.int32)
    with default_device(_TP4_MESH):
        drafter = DSparkDrafter(config, _TP4_MESH, rope=_rope())

    with _eager():
        block_hs = F.distributed_broadcast(_to_max(hidden), _TP4_MESH)
        mesh_anchors = F.distributed_broadcast(_to_max(anchors), _TP4_MESH)
        drawn = drafter.draft_vocab_parallel(
            block_hs, head_weight, mesh_anchors, with_scores=True
        )
        assert isinstance(drawn, tuple)
        fused, fused_scores = drawn
        # The unfused reference: the ops the branch ran before the
        # fused kernel, over each rank's own shard of the same biased
        # logits.
        logits = (drafter.norm(block_hs) @ head_weight.T).local_shards
        embeds = drafter.markov_head.embed.weight.local_shards
        markov = drafter.markov_head.head.weight.local_shards
        widths = [int(shard.shape[-1]) for shard in logits]
        starts = [sum(widths[:rank]) for rank in range(len(widths))]
        steps = [
            F.reshape(
                F.rebind(shard.cast(DType.float32), [BATCH * BLOCK, w]),
                [BATCH, BLOCK, w],
            )
            for shard, w in zip(logits, widths, strict=True)
        ]
        current = [
            anchor.cast(DType.int32) for anchor in mesh_anchors.local_shards
        ]
        want_drafts = []
        want_scores = []
        for i in range(BLOCK):
            columns = []
            for rank, step_logits in enumerate(steps):
                step = (
                    step_logits[:, i]
                    + F.gather(embeds[rank], current[rank], axis=0).cast(
                        DType.float32
                    )
                    @ markov[rank][starts[rank] : starts[rank] + widths[rank]]
                    .cast(DType.float32)
                    .T
                )
                index = (F.argmax(step, axis=-1) + starts[rank]).cast(
                    DType.float32
                )
                columns.append(
                    F.concat(
                        [
                            F.max(step, axis=-1),
                            index,
                            F.unsqueeze(_excluded_mass(step), -1),
                        ],
                        axis=-1,
                    )
                )
            gathered = F.allgather(
                Tensor.from_shard_values(
                    [F.unsqueeze(col, 0) for col in columns],
                    DeviceMapping(_TP4_MESH, (Sharded(0),)),
                ),
                tensor_axis=0,
            )
            joined = gathered.local_shards[0]
            top, _ = _lowest_argmax(joined, config.vocab_size)
            want_drafts.append(top)
            want_scores.append(
                _combine_shard_logit(
                    joined[:, :, 0],
                    joined[:, :, 1],
                    joined[:, :, 2],
                    top,
                )
            )
            current = [
                _lowest_argmax(shard, config.vocab_size)[0]
                for shard in gathered.local_shards
            ]

    assert torch.equal(
        _torch(fused),
        torch.stack([_torch(d) for d in want_drafts], dim=1).to(torch.int64),
    )
    # The one batched combine must reproduce the per-position combines
    # exactly, not within a tolerance.
    assert torch.equal(
        _torch(fused_scores),
        torch.stack([_torch(s) for s in want_scores], dim=1),
    )


def test_vocab_parallel_scores_combine_once_for_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The with_scores path runs one combine over the stacked pieces
    after the loop, not one per position: the realized graph holds one
    shard-axis reduction set for the whole block, and adds no
    transpose over the plain path."""
    config = _small_tp4_config()
    head_weight = _head_weight_on(_TP4_MESH, _TP4_VOCAB)
    hidden = torch.randn(BATCH * BLOCK, HIDDEN).to(torch.bfloat16)
    anchors = torch.randint(0, _TP4_VOCAB, (BATCH,), dtype=torch.int32)
    with default_device(_TP4_MESH):
        drafter = DSparkDrafter(config, _TP4_MESH, rope=_rope())

    with _eager():
        block_hs = F.distributed_broadcast(_to_max(hidden), _TP4_MESH)
        mesh_anchors = F.distributed_broadcast(_to_max(anchors), _TP4_MESH)
        squeezes: list[object] = []
        real_squeeze = F.squeeze

        def counting_squeeze(*args: object, **kwargs: object) -> Tensor:
            squeezes.append(args)
            return real_squeeze(*args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(F, "squeeze", counting_squeeze)
            bare = drafter.draft_vocab_parallel(
                block_hs, head_weight, mesh_anchors
            )
        ctx = current_realization_context()
        assert isinstance(ctx, EagerRealizationContext)
        # The eager graph accumulates, so the plain call's ops come out
        # of the delta the scored call adds to this snapshot.
        plain_trace = str(ctx.graph)
        drawn = drafter.draft_vocab_parallel(
            block_hs, head_weight, mesh_anchors, with_scores=True
        )
        assert isinstance(drawn, tuple)
        drafts, scores = drawn
        scored_trace = str(ctx.graph)

    # The drafts are unchanged with scores on versus off.
    assert torch.equal(_torch(drafts), _torch(bare).to(torch.int64))
    assert tuple(_torch(scores).shape) == (BATCH, BLOCK)
    # The combine's own ops: one exp and one log for the whole block,
    # where the per-position path ran one per position.
    assert scored_trace.count("mo.exp") - plain_trace.count("mo.exp") == 1
    assert scored_trace.count("mo.log") - plain_trace.count("mo.log") == 1
    # The combine moves no axis of its own: no permute reaches the
    # graph. The with_scores branch's extra operand transposes over the
    # plain call come from the fused pieces op taking no matmul
    # epilogue; the old per-position path carried the same set.
    assert "mo.permute" not in scored_trace
    # Main's plain path squeezes only the drafted ids, once per shard and
    # step; the scored path squeezes the step maxima outside the argmax.
    assert len(squeezes) == BLOCK * len(_TP4_MESH.devices), len(squeezes)


def test_a_tp4_per_row_verify_places_the_lengths_on_the_mesh() -> None:
    """A TP mesh with dp == 1 still broadcasts the verify lengths.

    The per-row gather's body runs on every device, so lengths that
    stay on one device are unreadable on the others; the shared
    placement covers the dp == 1 case the inline broadcast skipped.
    """
    lengths = [3, 6]
    total = sum(lengths)
    marks = torch.arange(total, dtype=torch.int32)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    verify_lengths = _to_max(torch.tensor([0, 2], dtype=torch.int32))
    return_n_logits = _to_max(torch.tensor([BLOCK + 1], dtype=torch.int64))
    with _eager():
        replicated = DeviceMapping(_TP4_MESH, (Replicated(),))
        hidden = Tensor.from_shard_values(
            [_to_max(torch.stack([marks] * HIDDEN, dim=1).to(torch.bfloat16))]
            * _TP4_MESH.num_devices,
            replicated,
        )
        row_offsets = Tensor.from_shard_values(
            [_to_max(torch.tensor(offsets, dtype=torch.uint32))]
            * _TP4_MESH.num_devices,
            replicated,
        )
        var_h, _ = spec_driver.gather_verify_prefix(
            hidden,
            row_offsets,
            verify_lengths,
            None,
            return_n_logits,
            _TP4_MESH,
            1,
        )
        got = as_tensors(var_h)

        # The adapter's own dp == 1 branch must place the lengths too:
        # the trunk's mesh is the gather's, and a host tensor only reads
        # on an all-CPU test mesh -- on devices it would be unreadable.
        target = _TP4RecordingTarget(total, marks)
        kv = [_upstream_kv(lengths) for _ in range(_TP4_MESH.num_devices)]
        batch = dataclasses.replace(
            _batch(lengths=lengths, verify_lengths=[0, 2]),
            kv_collections=kv,
            draft_kv_collections=kv,
            passthrough_kv={"global": kv},
            devices=[DeviceRef.from_device(d) for d in _TP4_MESH.devices],
        )
        seen: list[Tensor] = []
        real_gather = spec_driver.gather_prefix_rows

        def record(
            h: Tensor,
            input_row_offsets: Tensor,
            verify_lengths: Tensor,
            return_n_logits: Tensor,
            mesh: DeviceMesh,
            data_parallel_degree: int = 1,
        ) -> tuple[Tensor, Tensor]:
            seen.append(verify_lengths)
            return real_gather(
                h,
                input_row_offsets,
                verify_lengths,
                return_n_logits,
                mesh,
                data_parallel_degree,
            )

        spec_driver.gather_prefix_rows = record
        try:
            DeepseekV41BlockTarget(
                cast(DeepseekV41, target), _kv_params_stub()
            )(batch)
        finally:
            spec_driver.gather_prefix_rows = real_gather
        assert seen[0].is_distributed
        assert seen[0].mesh == _TP4_MESH

    got = _torch(got)
    rows = got[:, 0].to(torch.int32)
    assert torch.equal(rows, torch.tensor([2, 2, 2, 6, 7, 8]))
