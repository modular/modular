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
"""The DSpark drafter's paged block: its leaf's arity and its dense parity.

A serving step holds no dense context pool -- the taps of earlier steps are
gone -- so the drafter reads its own paged latent cache: each layer's window is
*gathered* by slot (``window_reads`` then ``gather_window_rows``) and the
block's own rows are multiplied against the query separately, so only their
scores ever meet. That is the same key set the dense entry masks
out of the pool it is handed, so the two entries must draft the same tokens from
the same context, and that is what the gate here measures.

Five things are pinned, all on the CPU:

* the leaf's arity. A speculative leaf carries two slots a plain one does not
  (the draft dispatch metadata and its split-K count), so its record is two
  fields wider than the plain one's. The engine's slot map pins the drafter's
  record at that width and refuses the boot without them, so
  ``build_dspark_kv_params`` derives them from the pipeline's speculative
  configuration rather than leaving each caller to remember -- and the width it
  records is the block's trained ``depth``, whatever
  ``--num-speculative-tokens`` says.
* the window's slot arithmetic: the last ``window`` slots up to the anchor,
  floored at zero and never past the anchor, in a graph with a symbolic batch.
* the same arithmetic with an anchor clamped to the leaf's own length, so an
  off-by-N caller cannot gather an unallocated page.
* the two entries agree, at an anchor of zero, inside the window, past it (where
  the window truncates), for two requests of different lengths in one batch,
  and across a second step that only writes the rows it verified.
* the write places each row at its own position, so a step's rows are the ones
  the next step reads.

The parity runs in float32: the two entries differ only in how the window is
materialized, so a real difference is a bug and not rounding. The leaf the
parity builds declares no speculative method, because its slots are never read
here -- the arity test covers the served shape. ``dspark_paged_fixtures.py``
holds the scenario walk both this file and the GPU gate drive.
"""

from __future__ import annotations

import numpy as np
import pytest
from dspark_paged_fixtures import (
    SHORT_SCENARIOS,
    TOLERANCE,
    Compiled,
    PagedStep,
    Parity,
    Scenario,
    check,
    dense_input_types,
    paged_input_types,
    parity,
    window_scenario,
)
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.pipelines.architectures.mistral_large4_modulev3.dspark import (
    DSparkDrafter,
    DSparkDrafterConfig,
    build_dspark_kv_params,
    dspark_speculative_slots,
    window_start,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.dspark import (
    window_reads,
    window_slots,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    MistralLarge4Yarn,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh

HIDDEN = 8
HEADS = 2
NOPE = 4
ROPE = 4
V_HEAD = 4
LATENT = 6
MARKOV_RANK = 4
VOCAB = 12
DEPTH = 3
WINDOW = 4
"""Shorter than the long scenario's context, so the window truncates there."""

SCENARIOS = [*SHORT_SCENARIOS, window_scenario(WINDOW)]
"""Every window shape a serving step can produce."""

PLAIN_RECORD_FIELDS = 9
"""One device's record for a plain paged leaf.

The engine's map pins the eight-slot MLA base
(``InputSlotMap::MLA_KV_GROUP_WIDTH``), and this export declares the
mixed-step split record last, which that map grows every group by
(``with_kv_mixed_step_split``).
"""

SPECULATIVE_RECORD_FIELDS = PLAIN_RECORD_FIELDS + 2
"""The same record for a speculative leaf: the draft dispatch metadata and its
split-K count sit between the base and the split record."""


def _yarn() -> MistralLarge4Yarn:
    """The drafter's rope geometry, in a shape small enough to compile fast."""
    return MistralLarge4Yarn(
        rope_theta=10000.0,
        factor=2048.0,
        original_max_position_embeddings=512,
        beta_fast=32.0,
        beta_slow=1.0,
        attention_factor=1.0,
        lerope=False,
        lerope_alpha_dim=ROPE // 2,
    )


def _config() -> DSparkDrafterConfig:
    """A miniature drafter: one layer, three rows, a four-slot window."""
    return DSparkDrafterConfig(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_HEAD,
        kv_lora_rank=LATENT,
        intermediate_size=2 * HIDDEN,
        num_layers=1,
        depth=DEPTH,
        window=WINDOW,
        target_layers=(1,),
        markov_rank=MARKOV_RANK,
        vocab_size=VOCAB,
        rms_norm_eps=1e-5,
        softcap=30.0,
        yarn=_yarn(),
    )


def _rand(shape: tuple[int, ...], seed: int, scale: float = 0.2) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(
        np.float32
    )


def _weights(config: DSparkDrafterConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter of the miniature drafter, with its shape."""
    attention = {
        "wq.weight": (HEADS * (NOPE + ROPE), HIDDEN),
        "wkv_a_with_mqa.weight": (LATENT + ROPE, HIDDEN),
        "kv_a_norm.weight": (LATENT,),
        "wkv_b.weight": (HEADS * (NOPE + V_HEAD), LATENT),
        "wo.weight": (HIDDEN, HEADS * V_HEAD),
    }
    shapes: dict[str, tuple[int, ...]] = {
        "backbone.target_projection.weight": (HIDDEN, config.tap_width),
        "backbone.target_norm.weight": (HIDDEN,),
        "backbone.final_norm.weight": (HIDDEN,),
        "backbone.mask_embedding": (DEPTH - 1, HIDDEN),
        "autoregressive_head.embedding.weight": (VOCAB, MARKOV_RANK),
        "autoregressive_head.output.weight": (VOCAB, MARKOV_RANK),
        "autoregressive_head.joint_proj.weight": (
            3 * MARKOV_RANK,
            HIDDEN + 2 * MARKOV_RANK,
        ),
        "autoregressive_head.joint_proj.bias": (3 * MARKOV_RANK,),
    }
    for name, shape in attention.items():
        shapes[f"backbone.layers.0.attention.{name}"] = shape
    shapes["backbone.layers.0.attention_norm.weight"] = (HIDDEN,)
    shapes["backbone.layers.0.ffn_norm.weight"] = (HIDDEN,)
    shapes["backbone.layers.0.feed_forward.w1.weight"] = (2 * HIDDEN, HIDDEN)
    shapes["backbone.layers.0.feed_forward.w3.weight"] = (2 * HIDDEN, HIDDEN)
    shapes["backbone.layers.0.feed_forward.w2.weight"] = (HIDDEN, 2 * HIDDEN)
    return shapes


class _WindowSlots(Module[..., tuple[Tensor, Tensor]]):
    """Traces the window's slot arithmetic on a symbolic batch."""

    def forward(self, anchor_positions: Tensor) -> tuple[Tensor, Tensor]:
        """Returns each block's window slots and which of them are real."""
        return window_slots(anchor_positions, WINDOW)


class _WindowReads(Module[..., tuple[Tensor, Tensor, Tensor]]):
    """Traces a step's window addressing, with the anchor clamped."""

    def forward(
        self, anchor_positions: Tensor, cache_lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the clamped anchor, the slots, and which are real."""
        reads = window_reads(anchor_positions, cache_lengths, WINDOW, DEPTH)
        return reads.anchor, reads.slots, reads.keep


def _tensor(values: np.ndarray) -> Tensor:
    """A CPU tensor of ``values``, in its own numpy dtype."""
    return Tensor(storage=Buffer.from_numpy(np.ascontiguousarray(values)))


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    """One CPU session for every graph in this file."""
    return InferenceSession(devices=[CPU()])


@pytest.fixture(scope="module")
def compiled(session: InferenceSession) -> Compiled:
    """Both entries of one miniature drafter, compiled once.

    A CPU compile of the drafter costs about a minute and the paged entry adds
    the two cache kernels, so every scenario below runs on these two graphs: the
    dense one is the entry the parity gate uses, the paged one is serving's.
    """
    config = _config()
    cpu = DeviceRef.CPU()
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    kv_params = build_dspark_kv_params(
        KVCacheConfig(kv_cache_page_size=128),
        devices=[cpu],
        config=config,
        # The leaf stores the latents in the activation dtype, so a float32
        # engine reads a float32 cache: this file gates arithmetic, not the
        # bf16 rounding a served leaf carries.
        dtype=DType.float32,
    )
    assert kv_params.dtype == DType.float32
    with default_device(CPU()), default_dtype(DType.float32):
        drafter = DSparkDrafter(config)
        weights = {
            name: _rand(shape, seed=100 + index)
            for index, (name, shape) in enumerate(_weights(config).items())
        }
        dense = drafter.compile(
            *dense_input_types(config, DType.float32, cpu),
            session=session,
            weights=weights,
        )
        paged = PagedStep(drafter, kv_params, mesh).compile(
            *paged_input_types(config, kv_params, DType.float32, cpu),
            session=session,
            weights={
                f"drafter.{name}": value for name, value in weights.items()
            },
        )
    return Compiled(
        dense=dense,
        paged=paged,
        config=config,
        kv_params=kv_params,
        embed_weight=_tensor(_rand((VOCAB, HIDDEN), seed=6)),
        head_weight=_tensor(_rand((VOCAB, HIDDEN), seed=7)),
        place=_tensor,
    )


def test_the_drafter_leaf_declares_the_two_spec_only_slots() -> None:
    """A speculative leaf is two fields wider than a plain one.

    The two extra fields are the draft dispatch metadata and its split-K count,
    in the record positions the engine's slot map pins: right after the
    verify-width dispatch metadata for the first, last for the second. A leaf
    without them is refused at boot with an arity mismatch.

    Both widths sit past the eight-slot MLA base the engine's slot map pins
    (``InputSlotMap::MLA_KV_GROUP_WIDTH``): the mixed-step split record every
    paged leaf now declares last is the ninth, and the spec pair makes eleven.
    """
    config = _config()
    device = DeviceRef.GPU(0)
    leaf = build_dspark_kv_params(
        KVCacheConfig(kv_cache_page_size=128),
        devices=[device],
        config=config,
        speculative=SpeculativeConfig(
            speculative_method="dspark", num_speculative_tokens=3
        ),
    )
    fields = list(leaf.flattened_kv_inputs())
    assert len(fields) == SPECULATIVE_RECORD_FIELDS, (
        f"{len(fields)} fields per device; the drafter's record is the eight-slot"
        f" MLA base plus the split record and the spec pair --"
        f" {SPECULATIVE_RECORD_FIELDS}"
    )
    # Record order: blocks, lengths, lookup table, the two length scalars, the
    # page stride, then the verify-width dispatch pair and the draft-width one,
    # then the split record.
    assert [field.dtype for field in fields] == [
        DType.bfloat16,
        DType.uint32,
        DType.uint32,
        DType.uint32,
        DType.uint32,
        DType.int64,
        DType.int64,
        DType.int64,
        DType.int64,
        DType.int64,
        DType.uint32,
    ]
    # The dispatch metadata is device-resident, the split-K counts and the split
    # record are host scalars, exactly as the target's leaf declares them.
    assert [field.device for field in fields[6:]] == [
        device,
        device,
        DeviceRef.CPU(),
        DeviceRef.CPU(),
        DeviceRef.CPU(),
    ]
    assert all(field.rank == 1 for field in fields[5:])
    plain = build_dspark_kv_params(
        KVCacheConfig(kv_cache_page_size=128), devices=[device], config=config
    )
    assert len(list(plain.flattened_kv_inputs())) == PLAIN_RECORD_FIELDS
    # The serving dtype: bf16, which is what the engine's sidecar pins.
    assert plain.dtype == DType.bfloat16
    assert leaf.dtype == DType.bfloat16


def test_the_spec_slots_come_from_the_pipeline_configuration() -> None:
    """The leaf's width is the block's, whatever the flag says.

    The draft dispatch is sized at the block's trained width -- the block always
    runs that many rows, whatever a step verifies -- so an unset flag and one
    that names the block both give the block's count, and anything else is
    refused rather than sizing the drafter's record differently from the
    target's.
    """
    config = _config()
    assert dspark_speculative_slots(None, config) == (None, 0)
    # Unset: the block's width.
    unset = SpeculativeConfig(speculative_method="dspark")
    assert unset.num_speculative_tokens is None
    assert dspark_speculative_slots(unset, config) == ("dspark", DEPTH)
    # Naming the block's width is the same answer.
    named = SpeculativeConfig(
        speculative_method="dspark", num_speculative_tokens=DEPTH
    )
    assert dspark_speculative_slots(named, config) == ("dspark", DEPTH)
    # A narrower step is a different block, and is refused: the drafter samples
    # its anchor slot, so the verify width is the block's plus one.
    with pytest.raises(ValueError, match="drafts exactly"):
        dspark_speculative_slots(
            SpeculativeConfig(
                speculative_method="dspark", num_speculative_tokens=1
            ),
            config,
        )
    leaf = build_dspark_kv_params(
        KVCacheConfig(kv_cache_page_size=128),
        devices=[DeviceRef.GPU(0)],
        config=config,
        speculative=unset,
    )
    assert leaf.num_draft_tokens == config.num_drafts == DEPTH
    with pytest.raises(ValueError, match="dspark"):
        build_dspark_kv_params(
            KVCacheConfig(kv_cache_page_size=128),
            devices=[DeviceRef.GPU(0)],
            config=config,
            speculative=SpeculativeConfig(speculative_method="eagle"),
        )


def test_window_slots_cover_the_window_and_repeat_a_short_context(
    session: InferenceSession,
) -> None:
    """The slots are the window's positions, floored at zero and clamped.

    A context shorter than the window gathers its own anchor row to fill the
    rectangle and masks the repeats; a longer one reads exactly the last
    ``window`` slots. Neither ever names a slot past the anchor, which is a row
    the request has not written, nor one before zero. The batch is symbolic, as
    it is under the block driver.
    """
    device = DeviceRef.CPU()
    run = _WindowSlots().compile(
        TensorType(DType.int32, ["batch"], device=device),
        session=session,
        weights={},
    )
    anchors = np.array([0, 1, 2, 3, 8, 2047, 6000], dtype=np.int32)
    slots_out, keep_out = run(_tensor(anchors))
    got = slots_out.to_numpy()
    keep = keep_out.to_numpy()
    assert got.shape == (anchors.shape[0], WINDOW)
    assert keep.shape == got.shape
    for row, anchor in enumerate(anchors.tolist()):
        start = window_start(anchor, WINDOW)
        want = [min(max(start, 0) + index, anchor) for index in range(WINDOW)]
        np.testing.assert_array_equal(got[row], want)
        assert all(0 <= slot <= anchor for slot in want)
        # Exactly the in-window slots are kept, and they are a prefix of the
        # row, so a short context masks a suffix rather than a hole.
        np.testing.assert_array_equal(
            keep[row], np.arange(WINDOW) < min(WINDOW, anchor + 1)
        )
        if anchor + 1 >= WINDOW:
            # A context at least as long as the window reads exactly the last
            # ``window`` slots, with nothing repeated.
            assert want == list(range(anchor + 1 - WINDOW, anchor + 1))
            assert keep[row].all()


def test_window_reads_clamps_an_anchor_past_the_leaf(
    session: InferenceSession,
) -> None:
    """An anchor past the last written row reads that row, not a page.

    The gather resolves a slot through the request's lookup table and guards
    nothing, so an anchor one past the leaf would name an unallocated page and
    copy whatever bytes are there. The caller's anchor is derived from the
    leaf's lengths rather than read from them, so the clamp lives where the
    slots are made: every slot a request reads is one it has written.
    """
    device = DeviceRef.CPU()
    run = _WindowReads().compile(
        TensorType(DType.int32, ["batch"], device=device),
        TensorType(DType.uint32, ["batch"], device=device),
        session=session,
        weights={},
    )
    anchors = np.array([0, 3, 7, 2047, 40], dtype=np.int32)
    lengths = np.array([1, 4, 4, 2048, 4], dtype=np.uint32)
    anchor_out, slots_out, _keep = run(_tensor(anchors), _tensor(lengths))
    clamped = anchor_out.to_numpy()
    slots = slots_out.to_numpy()
    np.testing.assert_array_equal(
        clamped,
        np.minimum(np.maximum(anchors, 0), lengths.astype(np.int32) - 1),
    )
    for row, last in enumerate(clamped.tolist()):
        assert last <= int(lengths[row]) - 1, (row, last)
        assert int(slots[row].min()) >= 0, (row, slots[row])
        assert int(slots[row].max()) <= last, (row, slots[row])


@pytest.mark.parametrize(
    "scenario", SCENARIOS, ids=[scenario.name for scenario in SCENARIOS]
)
def test_the_paged_block_matches_the_dense_block(
    compiled: Compiled, scenario: Scenario
) -> None:
    """The two entries draft the same tokens from the same context.

    A request's window is materialized by slot in one entry and masked out of
    the pool in the other, so the logits may differ by float ordering -- but the
    draft tokens, which is what a reader would see, must not.
    """
    config = compiled.config
    taps = [
        _rand((length, config.tap_width), seed=200 + index, scale=0.5)
        for index, length in enumerate(scenario.lengths)
    ]
    results: list[Parity] = parity(
        compiled, scenario, taps, anchor_ids=[3 + i for i in range(len(taps))]
    )
    assert len(results) == len(scenario.steps)
    failures = check(results, logits=TOLERANCE)
    assert not failures, "the paged block disagrees with the dense one: " + (
        "; ".join(failures)
    )
