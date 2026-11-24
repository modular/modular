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
"""The DSpark Markov chain's sampled-draft path (``draft_proposal="sampled"``).

The chain is the one place K3's drafter must sample inside itself: slot
``k``'s logits are biased by the token slot ``k - 1`` **drew**, so the
per-slot distribution ``q`` the verifier's rejection test reads has to be
the one slot ``k``'s own draw came from -- a distribution computed after
the previous slot's pick, never from the previous slot's argmax.

The driver's :class:`~max.pipelines.speculative.block_driver.
SampledDraftSampler` is the production draw (a GPU kernel), so these tests
pin the chain's contract against a numpy model of that sampler's math --
the tempered full softmax plus a per-(row, position) Gumbel draw, the same
tempering and keying the kernel applies -- while the chain runs for real
on the CPU. The proposer's ``head`` dispatch is pinned end to end through
a traced graph: an argmax sampler runs the greedy chain, anything else is
bridged per position.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from _dspark_test_data import TINY_DRAFT_CONFIG
from _dspark_test_utils import set_dspark_weight
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import MLAKVCacheParams
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    K3DSparkDraftConfig,
    parse_k3_dspark_config,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.dspark import K3DSpark
from max.pipelines.architectures.kimi_k3_modulev3.dspark.proposer import (
    K3DSparkProposer,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
    DraftSampler,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMesh

# Big enough that the bigram bias moves picks, so a chain that biased slot
# ``k`` by the argmax instead of the draw diverges within the block.
_MARKOV_W2_SCALE = 12.0
_ROWS = 3
_SEED = 0x5EED
_DRAFTER_SEED = 20261113


def _mesh() -> DeviceMesh:
    return DeviceMesh((CPU(),), (1,), ("tp",))


def _drafter(
    mesh: DeviceMesh, seed: int
) -> tuple[K3DSpark, K3DSparkDraftConfig]:
    """A tiny drafter with drawn Markov-head weights, on one CPU device.

    Built at float32 so the numpy oracle below reads the very weights the
    chain's own float32 bias math consumes.
    """
    config = parse_k3_dspark_config(TINY_DRAFT_CONFIG)
    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        num_layers=config.num_hidden_layers,
        devices=[DeviceRef.CPU()],
        page_size=128,
        num_q_heads=config.num_attention_heads,
    )
    rng = np.random.default_rng(seed)
    markov_shape = [config.vocab_size, config.markov_rank]
    with default_device(mesh), default_dtype(DType.float32):
        draft = K3DSpark(config, kv_params, max_seq_len=64, mla_buffer_size=128)
    set_dspark_weight(
        draft,
        "markov_head.markov_w1.weight",
        Tensor(
            rng.standard_normal(markov_shape).astype(np.float32),
            dtype=DType.float32,
        ),
    )
    set_dspark_weight(
        draft,
        "markov_head.markov_w2.weight",
        Tensor(
            rng.standard_normal(markov_shape).astype(np.float32)
            * _MARKOV_W2_SCALE,
            dtype=DType.float32,
        ),
    )
    return draft, config


def _block_logits(
    config: K3DSparkDraftConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random per-slot logits and anchors, distinct per row and slot."""
    logits = rng.standard_normal(
        (_ROWS * config.block_size, config.vocab_size)
    ).astype(np.float32)
    anchors = rng.integers(0, config.vocab_size, size=_ROWS).astype(np.int64)
    return logits, anchors


def _markov_bias(draft: K3DSpark, token: np.ndarray) -> np.ndarray:
    """The bigram bias of ``token``, the head's own float32 math."""
    embed = draft.markov_head.markov_w1.weight.to_numpy()
    w2 = draft.markov_head.markov_w2.weight.to_numpy()
    return embed[token].astype(np.float32) @ w2.T.astype(np.float32)


def _tempered_full_softmax(
    step: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """The production draw's distribution: no top-k or top-p truncation."""
    tempered = step / np.maximum(temperature[:, None], np.float32(1e-6))
    shifted = np.exp(tempered - tempered.max(axis=-1, keepdims=True)).astype(
        np.float32
    )
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _gumbel(seed: int, position: int, rows: int, width: int) -> np.ndarray:
    """One Gumbel row per (row, position), so no two slots share a draw."""
    return np.stack(
        [
            -np.log(
                -np.log(
                    np.random.default_rng([seed, position, row]).uniform(
                        1e-9, 1.0, size=width
                    )
                )
            )
            for row in range(rows)
        ]
    ).astype(np.float32)


def _gumbel_pick(dist: np.ndarray, gumbel: np.ndarray) -> np.ndarray:
    """The Gumbel-max draw out of ``dist``: argmax(log p + g)."""
    with np.errstate(divide="ignore"):
        return np.argmax(np.log(dist.astype(np.float64)) + gumbel, axis=-1)


class _NumpyDraftSampler:
    """The production sampler's math, in numpy, recording what it drew.

    Models :meth:`SampledDraftSampler._draw`: the pick is a Gumbel-max
    draw over the tempered full softmax of the logits it is handed -- no
    top-k or top-p truncation, the same choice the GPU kernel makes --
    keyed per (row, position) so no two slots of a request share a draw.
    Records the logits each slot was handed, so a test can prove the
    chain biased slot ``k`` by slot ``k - 1``'s drawn token.
    """

    def __init__(self, temperature: np.ndarray, seed: int) -> None:
        self._temperature = temperature
        self._seed = seed
        self.steps: list[np.ndarray] = []
        self.positions: list[int] = []

    def __call__(self, step: Tensor, position: int) -> Tensor:
        self.positions.append(position)
        handed = step.to_numpy()
        self.steps.append(handed)
        dist = _tempered_full_softmax(handed, self._temperature)
        pick = _gumbel_pick(
            dist,
            _gumbel(self._seed, position, dist.shape[0], dist.shape[-1]),
        )
        return Tensor(pick.astype(np.int64), dtype=DType.int64).to(
            step.mesh if step.is_distributed else _mesh()
        )


def test_the_chain_biases_each_slot_by_the_drawn_token() -> None:
    """Slot ``k``'s logits carry the bias of slot ``k - 1``'s draw.

    The chain is Markov: a chain that biased by the argmax instead of the
    drawn token would hand the sampler a distribution the verifier's ``q``
    never described, silently skewing the ratio test. The picks here are
    drawn, not argmax, so the assert is load-bearing.
    """
    mesh = _mesh()
    draft, config = _drafter(mesh, seed=_DRAFTER_SEED)
    rng = np.random.default_rng(7)
    logits, anchors = _block_logits(config, rng)
    sampler = _NumpyDraftSampler(
        temperature=np.full(_ROWS, 1.0, dtype=np.float32), seed=_SEED
    )
    with default_device(mesh):
        drafts, _ = draft.draft_chain(
            Tensor(logits, dtype=DType.float32).to(mesh),
            Tensor(anchors, dtype=DType.int64).to(mesh),
            sampler,
        )
    picks = drafts.to_numpy().astype(np.int64)
    assert sampler.positions == list(range(config.block_size))
    previous = anchors
    for slot in range(config.block_size):
        expected = logits.reshape(_ROWS, config.block_size, config.vocab_size)[
            :, slot
        ] + _markov_bias(draft, previous)
        np.testing.assert_allclose(
            sampler.steps[slot],
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"slot {slot} was not biased by the drawn token",
        )
        previous = picks[:, slot]


def test_draws_are_deterministic_per_seed_and_match_their_q() -> None:
    """The same per-row seed draws the same block twice, and every pick is
    the Gumbel-max draw out of the distribution the slot was handed."""
    mesh = _mesh()
    draft, config = _drafter(mesh, seed=_DRAFTER_SEED)
    rng = np.random.default_rng(7)
    logits, anchors = _block_logits(config, rng)
    temperature = np.array([1.0, 0.7, 1.3], dtype=np.float32)

    def run(seed: int) -> tuple[np.ndarray, _NumpyDraftSampler]:
        sampler = _NumpyDraftSampler(temperature=temperature, seed=seed)
        with default_device(mesh):
            drafts, _ = draft.draft_chain(
                Tensor(logits, dtype=DType.float32).to(mesh),
                Tensor(anchors, dtype=DType.int64).to(mesh),
                sampler,
            )
        return drafts.to_numpy().astype(np.int64), sampler

    first, sampler = run(_SEED)
    again, _ = run(_SEED)
    np.testing.assert_array_equal(first, again)
    # Every slot's pick is the exact Gumbel draw out of its own recorded
    # distribution: q is the tempered softmax of the logits the chain
    # handed the sampler, and the pick is argmax(log q + gumbel(seed)).
    for slot, step in enumerate(sampler.steps):
        dist = _tempered_full_softmax(step, temperature)
        np.testing.assert_array_equal(
            first[:, slot],
            _gumbel_pick(
                dist, _gumbel(_SEED, slot, dist.shape[0], dist.shape[-1])
            ),
            err_msg=f"slot {slot}'s pick is not the draw from its q",
        )
    # A different seed draws a different block somewhere: the fixture's
    # logits are random over 64 tokens, so identical blocks would mean the
    # key never reached the draw.
    other, _ = run(_SEED + 1)
    assert not np.array_equal(first, other)


def test_greedy_rows_stay_exactly_greedy() -> None:
    """A temperature-0 row draws its argmax at every slot, chain included.

    The production sampler tempers by ``max(temperature, 1e-6)``, so a
    greedy row's softmax is one-hot at its argmax and the Gumbel draw can
    only pick that token -- the sampled chain reproduces the greedy chain
    exactly, slot for slot, down the whole Markov cascade.
    """
    mesh = _mesh()
    draft, config = _drafter(mesh, seed=_DRAFTER_SEED)
    rng = np.random.default_rng(11)
    logits, anchors = _block_logits(config, rng)
    greedy_logits = Tensor(logits, dtype=DType.float32).to(mesh)
    greedy_anchors = Tensor(anchors, dtype=DType.int64).to(mesh)
    with default_device(mesh):
        greedy, _ = draft.draft_chain(greedy_logits, greedy_anchors, None)
        sampled, _ = draft.draft_chain(
            greedy_logits,
            greedy_anchors,
            _NumpyDraftSampler(
                temperature=np.zeros(_ROWS, dtype=np.float32), seed=_SEED
            ),
        )
    np.testing.assert_array_equal(
        sampled.to_numpy(),
        greedy.to_numpy(),
        "a temperature-0 row must draft exactly its argmax chain",
    )


def _head_graph(
    proposer: K3DSparkProposer,
    config: K3DSparkDraftConfig,
    sampler: DraftSampler,
) -> Graph:
    """A traced graph running the proposer's ``head`` over one block.

    The LM head is the identity -- the tiny config's vocabulary and hidden
    size are both 64 -- so the graph's own logits are exactly the ``hidden``
    input, and the expected chains below are derived from it directly.
    """
    rows, block, hidden = _ROWS, config.block_size, config.hidden_size
    cpu = DeviceRef.CPU()
    with (
        Graph(
            "sampled_head",
            input_types=(
                TensorType(DType.float32, [rows * block, hidden], device=cpu),
                TensorType(DType.int64, [rows], device=cpu),
            ),
        ) as graph,
        ensure_context(),
    ):
        hidden_value, anchors = (value.tensor for value in graph.inputs)
        # The head reads only its logits, its anchors and its sampler; the
        # batch is the driver's context it never touches, so the test hands
        # it none rather than staging the whole driver batch. `Any` here is
        # the slot type the head's own signature leaves open.
        batch = cast(BlockBatch[TensorValue, BufferValue, Value[Any]], None)
        accepted = Accepted(
            num_accepted=ops.constant(0, DType.uint32, device=cpu).broadcast_to(
                [rows]
            ),
            next_tokens=anchors,
            commit_lengths=ops.constant(
                1, DType.uint32, device=cpu
            ).broadcast_to([rows]),
            is_prefill=ops.constant(False, DType.bool, device=cpu).broadcast_to(
                [rows]
            ),
        )
        graph.output(proposer.head(batch, [hidden_value], accepted, sampler))
    return graph


class _ConstantSampler:
    """A non-argmax sampler the head must bridge per position.

    Picks are fixed per position, so a test can name the token each slot
    must draft. The chain is Markov and the head bridges ``sample_next``
    only, so ``sample_all`` stays unreached.
    """

    def __init__(self, picks: list[list[int]]) -> None:
        self._picks = picks
        self.positions: list[int] = []

    def sample_all(self, logits: TensorValue) -> TensorValue:
        raise NotImplementedError("the chain samples one position at a time")

    def sample_next(
        self,
        logits: TensorValue,
        position: int,
        token_ids: TensorValue | None = None,
    ) -> TensorValue:
        del logits, token_ids
        self.positions.append(position)
        return ops.constant(
            self._picks[position], DType.int64, device=DeviceRef.CPU()
        )


def test_the_heads_dispatch_bridges_only_a_sampled_sampler() -> None:
    """``ArgmaxDraftSampler`` runs the chain's own greedy walk; any other
    sampler is bridged per position and its picks become the drafts."""
    mesh = _mesh()
    draft, config = _drafter(mesh, seed=_DRAFTER_SEED)
    rng = np.random.default_rng(3)
    logits, anchors = _block_logits(config, rng)
    head_weight = Tensor(
        np.eye(config.vocab_size, config.hidden_size, dtype=np.float32),
        dtype=DType.float32,
    ).to(mesh)
    with default_device(mesh):
        proposer = K3DSparkProposer(draft, head_weight, config)

    def run(sampler: DraftSampler) -> np.ndarray:
        compiled = InferenceSession(devices=[CPU()]).load(
            _head_graph(proposer, config, sampler)
        )
        (out,) = compiled.execute(
            Buffer.from_numpy(logits), Buffer.from_numpy(anchors)
        )
        return np.asarray(out.to_numpy())

    # The greedy head: every slot is the argmax of its own biased logits.
    greedy = run(ArgmaxDraftSampler())
    previous = anchors
    for slot in range(config.block_size):
        step = logits.reshape(_ROWS, config.block_size, config.vocab_size)[
            :, slot
        ] + _markov_bias(draft, previous)
        previous = np.argmax(step, axis=-1)
        np.testing.assert_array_equal(greedy[:, slot], previous)

    # The sampled head: the bridged sampler's picks are the drafts, drawn
    # in position order.
    picks = [
        [int((anchor + slot + 1) % config.vocab_size) for anchor in anchors]
        for slot in range(config.block_size)
    ]
    sampler = _ConstantSampler(picks)
    np.testing.assert_array_equal(
        run(sampler),
        np.asarray(picks).T,
        "the picks the chain consumed",
    )
    assert sampler.positions == list(range(config.block_size))
