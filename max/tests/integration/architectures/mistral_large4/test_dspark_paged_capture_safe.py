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
"""The DSpark drafter's paged block graph must stay capturable.

Device-graph capture records one graph per device stream, so two things break
it and neither is visible from the numbers: a cross-device ``mo.transfer``
lowers to ``mgp.buffer.device_to_device`` wrapped in ``mgp.device_wait`` on both
sides, which CUDA refuses to merge into a capture sequence
(``CUDA_ERROR_STREAM_CAPTURE_MERGE``), and a device-to-host readback lowers to a
blocking ``mgp.sync``. The paged block runs inside the fused speculative graph,
so it has to stage like the rest of that graph does: every rank holds the same
cache and the same rows, and nothing is shipped between them.

What is pinned here, on virtual B200s with no compile and no execution:

* the block stages at eight ranks with the drafter's paged leaf as its cache,
  with one gather and one store per drafter layer;
* no tensor crosses devices and none is read back to the host;
* the batch is symbolic, so one staged graph serves every batch size -- the
  growth the block driver's dynamic shapes require;
* the cache leaf inside that graph is the speculative one, eleven fields per
  device;
* the same graph behind the target's *sharded* embedding and head -- the shape
  the fused graph actually builds, where the vocabulary is split over the eight
  ranks -- stages too, and reaches the drafter only through the collectives its
  own layers use: the embedding's all-reduce, the head's all-gather, and two
  all-reduces per layer, for the tensor-parallel FFN's row-parallel down
  projection and attention's row-parallel out projection.

The module under test is the same phase pair the export's proposer drives
(:meth:`~DSparkDrafter.write_context` then
:meth:`~DSparkDrafter.forward_block_paged`), reached through the same
``PagedCacheValues.from_upstream`` the ML4 text model uses.
"""

from __future__ import annotations

import re

import pytest
from dspark_paged_fixtures import PagedStep, ShardedPagedStep
from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.mistral_large4_modulev3.dspark import (
    DSparkDrafter,
    DSparkDrafterConfig,
    build_dspark_kv_params,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    MistralLarge4Yarn,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative import SpeculativeConfig
from max.tensor import default_device, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    TensorLayout,
)

DEVICES = 8
"""The deployment's rank count: one node of B200s."""

HIDDEN = 8
HEADS = 8
"""Eight query heads: the leaf refuses a head count its tensor-parallel degree
of eight does not divide, which is the deployment's own 64-over-8 shape."""
NOPE = 4
ROPE = 4
V_HEAD = 4
LATENT = 6
MARKOV_RANK = 4
VOCAB = 12
DEPTH = 3
WINDOW = 4
"""A miniature drafter: the trace reads shapes, not numbers, so it stays small."""

LEAF_RECORD_FIELDS = 11
"""One device's record for this graph's speculative leaf.

The eight-slot MLA base the engine's map pins (``MLA_KV_GROUP_WIDTH``), the
draft dispatch metadata and its split-K count, and the mixed-step split record
every paged leaf on this export declares last (``with_kv_mixed_step_split``).
"""

CROSS_DEVICE = re.compile(
    r"mo\.transfer\[[^\n]*, gpu:(\d+)> to <\"gpu\", (\d+)>"
)
"""A transfer whose source and destination are both GPUs."""

GATHER = "kv_cache.gather_rows.ragged.paged"
"""The op that copies a request's window out of one layer of the leaf."""

STORE = "kv_cache.store.paged.ragged"
"""The op that writes a step's context latents into one layer of the leaf."""

ALL_REDUCE = "mo.distributed.allreduce.sum"
"""The residual collectives: the embedding's slice summed, and two per layer
for the tensor-parallel FFN's down projection and attention's out projection."""

ALL_GATHER = "mo.distributed.allgather"
"""The head's collective: the ranks' vocabulary columns, gathered back."""


def _yarn() -> MistralLarge4Yarn:
    """The drafter's rope geometry, in a shape small enough to trace fast."""
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


def _stage(
    step: type[PagedStep | ShardedPagedStep],
) -> tuple[str, KVCacheParams]:
    """Traces one step type's paged block graph on virtual B200s.

    ``PagedStep`` takes the borrowed tables as whole weights, ``ShardedPagedStep``
    holds the target's vocab-parallel modules instead, so their input lists
    differ by those two arguments.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(DEVICES)
    try:
        devices = [DeviceRef.GPU(rank) for rank in range(DEVICES)]
        mesh = DeviceMesh(
            tuple(Accelerator(rank) for rank in range(DEVICES)),
            (DEVICES,),
            ("tp",),
        )
        mapping = DeviceMapping(mesh, (Replicated(),))
        config = _config()
        kv_params = build_dspark_kv_params(
            KVCacheConfig(kv_cache_page_size=128),
            devices=devices,
            config=config,
            speculative=SpeculativeConfig(
                speculative_method="dspark", num_speculative_tokens=DEPTH
            ),
        )
        borrowed = (
            [
                TensorLayout(DType.bfloat16, [VOCAB, HIDDEN], mapping),
                TensorLayout(DType.bfloat16, [VOCAB, HIDDEN], mapping),
            ]
            if step is PagedStep
            else []
        )
        input_types = [
            TensorLayout(DType.bfloat16, ["rows", config.tap_width], mapping),
            TensorLayout(DType.int32, ["rows"], mapping),
            TensorLayout(DType.uint32, ["offsets_len"], mapping),
            TensorLayout(DType.int32, ["batch"], mapping),
            TensorLayout(DType.int32, ["batch"], mapping),
            *borrowed,
            *kv_params.flattened_kv_inputs(),
        ]
        with default_device(mesh), default_dtype(DType.bfloat16), F.lazy():
            module = step(DSparkDrafter(config), kv_params, mesh)
            return str(module.trace(*input_types)), kv_params
    finally:
        set_virtual_device_count(0)


@pytest.fixture(scope="module")
def staged() -> tuple[str, KVCacheParams]:
    """The paged block graph's trace text, and the leaf it was built with."""
    return _stage(PagedStep)


@pytest.fixture(scope="module")
def staged_sharded() -> tuple[str, KVCacheParams]:
    """The same graph behind the target's own, vocabulary-sharded tables."""
    return _stage(ShardedPagedStep)


def test_the_paged_block_stages_without_a_cross_device_transfer(
    staged: tuple[str, KVCacheParams],
) -> None:
    """Every rank reads its own cache and its own rows; nothing is shipped."""
    text, _ = staged
    offenders = CROSS_DEVICE.findall(text)
    assert not offenders, (
        f"{len(offenders)} cross-device transfers in the paged block graph:"
        " each one lowers to mgp.buffer.device_to_device between two"
        " mgp.device_wait calls, which CUDA refuses inside a capture"
        " (CUDA_ERROR_STREAM_CAPTURE_MERGE).\n"
        + "\n".join(
            line.strip()
            for line in text.splitlines()
            if CROSS_DEVICE.search(line)
        )
    )


def test_the_paged_block_reads_nothing_back_to_the_host(
    staged: tuple[str, KVCacheParams],
) -> None:
    """A device-to-host readback is the other capture breaker."""
    text, _ = staged
    to_host = [
        line.strip()
        for line in text.splitlines()
        if "mo.transfer" in line
        and line.rsplit(" to ", 1)[-1].strip()[:6] == '<"cpu"'
    ]
    assert not to_host, (
        "device-to-host readback in the paged block graph:\n"
        + "\n".join(to_host)
    )


def test_the_paged_block_gathers_and_stores_once_per_layer(
    staged: tuple[str, KVCacheParams],
) -> None:
    """The two cache ops are the whole point: one of each, per layer per rank.

    The body is emitted once per rank, so a one-layer drafter at eight ranks
    traces eight of each. Pin the counts so a refactor that dropped the paged
    read -- falling back to a dense pool the serving graph cannot hold -- or
    duplicated it has to update this test consciously.
    """
    text, _ = staged
    assert text.count(GATHER) == DEVICES, text.count(GATHER)
    assert text.count(STORE) == DEVICES, text.count(STORE)


def test_the_paged_block_keeps_the_batch_symbolic(
    staged: tuple[str, KVCacheParams],
) -> None:
    """One staged graph must serve every batch size.

    The block's row count and its request count are runtime shapes under the
    block driver, so a graph that baked either would force a recompile per
    batch. Only the depth and the window are static.
    """
    text, kv_params = staged
    assert f"[batch, {DEPTH}, {VOCAB}]" in text, (
        "the logits' leading axis is not the symbolic batch: the block baked"
        " its row count"
    )
    assert f"[batch, {DEPTH}]" in text, (
        "the draft tokens dropped the batch axis"
    )
    # The leaf inside this graph is the speculative one, so the cache operand is
    # three slots wider per device than a plain leaf's nine.
    assert (
        len(list(kv_params.flattened_kv_inputs()))
        == LEAF_RECORD_FIELDS * DEVICES
    )


def test_the_sharded_block_stages_behind_the_targets_own_tables(
    staged_sharded: tuple[str, KVCacheParams],
) -> None:
    """The served shape: the vocabulary split over the ranks, on both ends.

    The fused graph borrows the target's embedding and LM head, and the target
    shards their vocabulary over its tensor-parallel ranks. Staging it used to
    fail at the block's first line -- ``gather(input, indices) needs a
    resharding``, because the drafter looked the anchors up in one rank's
    slice of the table -- and the head failed one step later for the same
    reason. Both are now the target's own modules: the trace this fixture built
    is the assertion.
    """
    text, _ = staged_sharded
    assert text


def test_the_sharded_block_stages_without_a_cross_device_transfer(
    staged_sharded: tuple[str, KVCacheParams],
) -> None:
    """The sharded path's two collectives are kernels, not device copies."""
    text, _ = staged_sharded
    offenders = CROSS_DEVICE.findall(text)
    assert not offenders, (
        f"{len(offenders)} cross-device transfers in the sharded paged block:"
        " each one lowers to mgp.buffer.device_to_device between two"
        " mgp.device_wait calls, which CUDA refuses inside a capture"
        " (CUDA_ERROR_STREAM_CAPTURE_MERGE).\n"
        + "\n".join(
            line.strip()
            for line in text.splitlines()
            if CROSS_DEVICE.search(line)
        )
    )
    to_host = [
        line.strip()
        for line in text.splitlines()
        if "mo.transfer" in line
        and line.rsplit(" to ", 1)[-1].strip()[:6] == '<"cpu"'
    ]
    assert not to_host, (
        "device-to-host readback in the sharded paged block graph:\n"
        + "\n".join(to_host)
    )


def test_the_sharded_block_reaches_the_drafter_through_its_residual_collectives(
    staged_sharded: tuple[str, KVCacheParams],
) -> None:
    """The only cross-rank traffic is the embedding's, the head's and the FFN's.

    One all-gather over the whole graph is the head's vocabulary: without it a
    rank would score a bare slice of the vocabulary and the argmax would be
    taken over that slice. The all-reduces are the embedding's -- one per block,
    not one per rank -- plus the tensor-parallel projections' two per layer:
    the attention's out-projection and the FFN's row-parallel down projection
    each leave every rank a partial sum to complete (`layers/dspark.py`,
    `DSparkAttention._project_values` and `DSparkMLP.forward`). The graph's
    cost is the target's own cost for the same two tables, plus the collectives
    the sharded projections trade their bytes for.
    """
    text, _ = staged_sharded
    assert text.count(ALL_GATHER) == 1, text.count(ALL_GATHER)
    assert text.count(ALL_REDUCE) == 2 * _config().num_layers + 1, text.count(
        ALL_REDUCE
    )
    # The window gather and the context store are per rank, as on the
    # whole-weight path: the cache is replicated, so each rank reads its own.
    assert text.count(GATHER) == DEVICES, text.count(GATHER)
    assert text.count(STORE) == DEVICES, text.count(STORE)


def test_the_sharded_block_keeps_the_batch_symbolic(
    staged_sharded: tuple[str, KVCacheParams],
) -> None:
    """The sharded path's shapes come from the batch, not from the trace."""
    text, kv_params = staged_sharded
    assert f"[batch, {DEPTH}, {VOCAB}]" in text, (
        "the gathered logits' leading axis is not the symbolic batch"
    )
    assert f"[batch, {DEPTH}]" in text, (
        "the draft tokens dropped the batch axis"
    )
    assert (
        len(list(kv_params.flattened_kv_inputs()))
        == LEAF_RECORD_FIELDS * DEVICES
    )


WINDOW_KEY_SHAPE = f"[batch, {WINDOW + DEPTH}, {LATENT}]"
"""One tensor holding the window's rows and the block's own, if they met."""

SCORE_SHAPE = f"[batch, {DEPTH}, {HEADS // DEVICES}, {WINDOW + DEPTH}]"
"""One rank's score rectangle: its own heads, every block row, every key.

The head axis is a tensor-parallel shard, so the trace's shape is the per-rank
one: ``HEADS / DEVICES`` heads each."""


def test_the_window_and_the_block_never_meet_in_one_key_tensor(
    staged: tuple[str, KVCacheParams],
) -> None:
    """The two key sets are concatenated as *scores*, not as keys.

    A key tensor holding both is ``[batch, window + depth, latent_dim]`` in the
    cache's dtype, and the float32 copy the multiplication used to need is
    twice that again. At the deployment's shapes -- a 2048-row window of 576
    latent columns, three block rows -- that is ~150 MB and ~300 MB per layer
    of every step, and a step runs three of them; the score rectangle they are
    concatenated into instead is ``[batch, depth, heads, window + depth]`` of
    float32, ~100 MB while every rank held all the heads and ~13 MB per rank
    once the head axis sharded. This pins the trace, so a refactor that
    reintroduces the concatenated keys fails here rather than in a memory
    profile.

    The rectangle keeps the block-row and head axes apart rather than folding
    them into one query axis: under tensor parallelism the head axis is a
    shard, and a fold behind it cannot be reshaped without gathering the
    queries back to every rank.
    """
    text, _ = staged
    assert SCORE_SHAPE in text, (
        "the block's scores are not one [batch, depth, heads, keys]"
        " rectangle:\n" + "\n".join(text.splitlines()[:40])
    )
    offenders = [
        line.strip() for line in text.splitlines() if WINDOW_KEY_SHAPE in line
    ]
    assert not offenders, (
        "the window's rows and the block's own are one key tensor again,"
        " which is the memory this read is shaped to avoid:\n"
        + "\n".join(offenders)
    )
