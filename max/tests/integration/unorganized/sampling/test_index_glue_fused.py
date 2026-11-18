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
"""The accept path's folded index glue must be the chain it replaced.

``sampler.first_rejected`` and ``sampler.accept_pair`` each answer, in one
launch, what a five- and a nine-launch chain of whole-tensor graph ops answered
before: the first rejected draft position, and the sampled verdict's per
position rejection bit and residual fallback. The fold is an opt-in only the
K3 accept path sets, so the default must keep the chain every other model
built. All of the work is integer or elementwise float, so the contract is
bit-identity -- not a tolerance -- and these tests pin it by building the same
graph twice, once with the ``fold_index_glue`` opt-in and once without it, and
comparing every output element. They also assert the folded op is really in
the folded graph and absent from the default one, so a fold that declined to
fire cannot pass by being identical to the chain by default.

Bit-identity is what makes the fold safe for the sampling verdict as well: the
verdict's draws are keyed off these very tensors, so a key that moved by one
would change committed tokens while every mask-shape assertion still passed.
The verdict tests therefore drive the whole ``stochastic_acceptance_sampler``,
under both draft proposal modes, and compare the accepted counts, the recovered
tokens and the bonus token.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn._graph_ops.sampling import (
    AcceptanceSampler,
    _find_first_rejected,
    stochastic_acceptance_sampler,
)

# K3's decode shape: batch 4 requests, 7 draft positions, plus the bonus slot.
K3_BATCH = 4
K3_STEPS = 7
VOCAB = 256

Build = Callable[
    [Sequence[TensorValue], DeviceRef, bool], tuple[TensorValue, ...]
]
"""Builds a graph's outputs from its inputs, the device, and whether to fold."""


def _device(session: InferenceSession) -> DeviceRef:
    return DeviceRef.from_device(session.devices[0])


_LOADED: dict[str, tuple[Model, Model]] = {}
"""The loaded folded and chain models, one pair per distinct graph.

A load costs seconds and an execute costs microseconds, so a case that
rebuilds the graph every shape it tries pays the load again each time. Every
graph here declares its shapes symbolically, so one loaded pair serves every
shape in the case: the key names the graph, and the shapes are inputs. Eight
keys cover the suite, sixteen loads in place of forty-four.
"""


def _models_for(
    session: InferenceSession,
    key: str,
    input_types: list[TensorType],
    build: Build,
) -> tuple[Model, Model]:
    """Returns the folded and the chain model for ``key``, loading them once."""
    models = _LOADED.get(key)
    if models is None:
        device = _device(session)
        loaded: list[Model] = []
        for folded in (True, False):
            with Graph(f"{key}_{folded}", input_types=input_types) as graph:
                graph.output(
                    *build([v.tensor for v in graph.inputs], device, folded)
                )
            loaded.append(session.load(graph))
        models = (loaded[0], loaded[1])
        _LOADED[key] = models
    return models


def _run_both(
    session: InferenceSession,
    key: str,
    input_types: list[TensorType],
    build: Build,
    inputs: list[npt.NDArray[np.generic]],
) -> tuple[list[npt.NDArray[np.generic]], list[npt.NDArray[np.generic]]]:
    """Runs the graph ``key`` names with the opt-in and without it."""
    folded, chain = _models_for(session, key, input_types, build)
    device_buffer = session.devices[0]
    buffers = [Buffer.from_numpy(i).to(device_buffer) for i in inputs]
    results: list[list[npt.NDArray[np.generic]]] = []
    for model in (folded, chain):
        outputs = model.execute(*buffers)
        results.append([out.to_numpy() for out in outputs])
    return results[0], results[1]


def _assert_identical(
    folded: list[npt.NDArray[np.generic]],
    chain: list[npt.NDArray[np.generic]],
    label: str,
) -> None:
    assert len(folded) == len(chain)
    for index, (got, want) in enumerate(zip(folded, chain, strict=True)):
        assert got.dtype == want.dtype, f"{label}: output {index} dtype moved"
        assert got.shape == want.shape, f"{label}: output {index} shape moved"
        if np.array_equal(got, want):
            continue
        # Name the cells that moved. Two parametrizations failed together in
        # one CI attempt and passed on the retry, and the bare array
        # comparison left nothing to go on: the indices are what say whether
        # the fold or the draw moved.
        moved = np.flatnonzero(np.asarray(got) != np.asarray(want))
        got_values = np.asarray(got).ravel()[moved[:8]].tolist()
        want_values = np.asarray(want).ravel()[moved[:8]].tolist()
        raise AssertionError(
            f"{label}: output {index} moved in {moved.size} of"
            f" {np.asarray(got).size} elements, at flat indices"
            f" {moved[:16].tolist()}; folded {got_values}"
            f" vs chain {want_values}"
        )


def _first_rejected_build(rows: int, steps: int) -> Build:
    """Emits the first rejected position of each row."""

    def build(
        inputs: Sequence[TensorValue], device: DeviceRef, fold_index_glue: bool
    ) -> tuple[TensorValue, ...]:
        (mask,) = inputs
        return (
            ops.squeeze(
                _find_first_rejected(
                    mask, device, fold_index_glue=fold_index_glue
                ),
                axis=-1,
            ),
        )

    del rows, steps
    return build


def _mask(
    rows: int, steps: int, rejected_at: list[int]
) -> npt.NDArray[np.bool_]:
    """Builds a mask whose row ``r`` first rejects at ``rejected_at[r]``.

    A position at or past ``steps`` means the row rejects nothing, which the
    verdict reads as "every draft accepted".
    """
    mask = np.zeros((rows, steps), dtype=np.bool_)
    for row, position in enumerate(rejected_at):
        if position < steps:
            mask[row, position] = True
    return mask


@pytest.mark.parametrize(
    "rows,steps,rejected_at",
    [
        # K3's decode shape: every row accepts every draft.
        (K3_BATCH, K3_STEPS, [K3_STEPS] * K3_BATCH),
        # The first, a middle and the last position, plus an all-accept row.
        (K3_BATCH, K3_STEPS, [0, 3, K3_STEPS - 1, K3_STEPS]),
        # One position only: the shortest row the scan can take.
        (K3_BATCH, 1, [0, 1, 0, 1]),
        # 513 rows: more than one thread block on every target (the block
        # size is 128 on B200 and 256 on CDNA4), so a second block has to run.
        (513, 1, [row % 3 for row in range(513)]),
    ],
)
def test_first_rejected_matches_the_chain(
    session: InferenceSession,
    rows: int,
    steps: int,
    rejected_at: list[int],
) -> None:
    """The folded first-rejected index equals the sentinel/argmax chain."""
    device = _device(session)
    folded, chain = _run_both(
        session,
        f"first_rejected_{steps}",
        # The chain rebinds its leading dim to the graph's ``batch_size``, so
        # the mask has to carry that name; production declares it the same way.
        [TensorType(DType.bool, ["batch_size", steps], device=device)],
        _first_rejected_build(rows, steps),
        [_mask(rows, steps, rejected_at)],
    )
    _assert_identical(folded, chain, "first rejected")
    # The values matter too: an all-zero output would pass the comparison.
    assert [int(v) for v in folded[0]] == [
        min(position, steps) for position in rejected_at
    ]


def test_first_rejected_zero_draft_positions(session: InferenceSession) -> None:
    """A prefill iteration's empty mask reads 0 from both paths.

    ``num_steps == 0`` is a prefill iteration. The fold covers the empty shape
    too, so this pins that the two agree there and that the chain is still
    reachable.
    """
    device = _device(session)
    empty = np.zeros((K3_BATCH, 0), dtype=np.bool_)
    # The chain asserts its leading dim is the graph's ``batch_size``, so the
    # prefill shape has to be declared with that name.
    folded, chain = _run_both(
        session,
        "first_rejected_prefill",
        [TensorType(DType.bool, ["batch_size", 0], device=device)],
        _first_rejected_build(K3_BATCH, 0),
        [empty],
    )
    _assert_identical(folded, chain, "first rejected, no drafts")
    assert (folded[0] == 0).all()


def _verdict_build(
    batch: int, steps: int, vocab: int, *, sampled: bool
) -> Build:
    """Emits the whole acceptance verdict for one draft proposal mode."""

    def build(
        inputs: Sequence[TensorValue], device: DeviceRef, fold_index_glue: bool
    ) -> tuple[TensorValue, ...]:
        draft_tokens, logits, temperature, top_k, top_p, seed = inputs[:6]
        return stochastic_acceptance_sampler(
            draft_tokens,
            logits,
            temperature=temperature,
            top_k=top_k,
            max_k=ops.constant(-1, DType.int64, DeviceRef.CPU()),
            top_p=top_p,
            min_top_p=ops.constant(0.5, DType.float32, DeviceRef.CPU()),
            seed=seed,
            draft_proposal="sampled" if sampled else "argmax",
            draft_probs_full=inputs[6] if sampled else None,
            vocab_size=vocab if sampled else None,
            fold_index_glue=fold_index_glue,
        )

    return build


def _verdict_input_types(
    batch: int, steps: int, vocab: int, *, seed_rows: int, sampled: bool
) -> Callable[[DeviceRef], list[TensorType]]:
    """Declares the verdict's inputs, as symbolic dims.

    The numbers only bound the tensors the test feeds; the declarations stay
    symbolic because that is what the sampler's own rebinds expect.
    """
    del batch, steps, vocab

    def types(device: DeviceRef) -> list[TensorType]:
        # Symbolic dims, as the serving graphs carry them: the sampler rebinds
        # to ``batch_size``/``num_steps``/``vocab_size`` internally, and a
        # literal dim here would not bind to them.
        declared = [
            TensorType(DType.int64, ["batch_size", "num_steps"], device=device),
            TensorType(
                DType.float32,
                ["total_output_len", "vocab_size"],
                device=device,
            ),
            TensorType(DType.float32, ["batch_size"], device=device),
            TensorType(DType.int64, ["batch_size"], device=device),
            TensorType(DType.float32, ["batch_size"], device=device),
            # A per-row seed is the graph's own batch dim; the sampler rebinds
            # it to ``batch_size`` and a literal dim would fail that check.
            TensorType(
                DType.uint64,
                ["batch_size"] if seed_rows > 1 else [1],
                device=device,
            ),
        ]
        if sampled:
            declared.append(
                TensorType(
                    DType.float32,
                    ["batch_size", "num_steps", "vocab_size"],
                    device=device,
                )
            )
        return declared

    return types


def _verdict_inputs(
    batch: int,
    steps: int,
    vocab: int,
    *,
    seed_rows: int,
    sampled: bool,
    accept_some: bool = False,
    zero_q_rows: bool = False,
) -> list[npt.NDArray[np.generic]]:
    rng = np.random.default_rng(7)
    draft_tokens = rng.integers(0, vocab, size=(batch, steps)).astype(np.int64)
    logits = rng.standard_normal((batch * (steps + 1), vocab)).astype(
        np.float32
    )
    if accept_some and steps > 0:
        # Walk the target's opinion of the drafted token across the row: a hard
        # lift every other position makes it the row's mass, and a soft one in
        # between leaves it a middling probability. Without the soft arm the
        # drafted token's masked probability is either ~1 or exactly 0 -- a
        # random token does not make a row's top-k -- and then the accept test
        # reaches the same verdict whether or not a row without a proposal
        # takes q = 1, which is the arm that has to be pinned.
        per_position = logits.reshape(batch, steps + 1, vocab)
        for row in range(batch):
            for step in range(steps):
                per_position[row, step, draft_tokens[row, step]] += (
                    12.0 if step % 2 == 0 else 3.0
                )
    inputs: list[npt.NDArray[np.generic]] = [
        draft_tokens,
        logits,
        np.full(batch, 0.9, dtype=np.float32),
        np.full(batch, 8, dtype=np.int64),
        np.full(batch, 0.95, dtype=np.float32),
        (
            np.array([0x1234567890ABCDEF], dtype=np.uint64)
            if seed_rows == 1
            # A per-row seed is declared over the graph's own ``batch_size``,
            # so it carries one key per request whatever ``seed_rows`` asked
            # for: the mode, not the count, is what the shape has to match.
            else np.arange(batch, dtype=np.uint64) * np.uint64(977)
            + np.uint64(3)
        ),
    ]
    if sampled:
        # A normalized distribution per row, so the ratio test has real work.
        raw = rng.random((batch, steps, vocab)).astype(np.float32) + 1e-3
        probs = raw / raw.sum(axis=-1, keepdims=True)
        if zero_q_rows:
            # Every other request carries no proposal at all: the arm where the
            # accept test takes q = 1 and the residual subtracts a one-hot at
            # the drafted token instead of a proposal it never had.
            for row in range(1, batch, 2):
                probs[row] = 0.0
        inputs.append(probs)
    return inputs


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("seed_rows", [K3_BATCH, 1])
def test_accept_verdict_is_bit_identical(
    session: InferenceSession, sampled: bool, seed_rows: int
) -> None:
    """The whole verdict, both proposal modes, folded and unfolded."""

    def run(batch: int, steps: int) -> None:
        folded, chain = _run_both(
            session,
            f"verdict_{sampled}_{seed_rows}_{VOCAB}",
            _verdict_input_types(
                batch, steps, VOCAB, seed_rows=seed_rows, sampled=sampled
            )(_device(session)),
            _verdict_build(batch, steps, VOCAB, sampled=sampled),
            _verdict_inputs(
                batch, steps, VOCAB, seed_rows=seed_rows, sampled=sampled
            ),
        )
        _assert_identical(
            folded, chain, f"verdict (batch={batch}, steps={steps})"
        )

    # K3's decode shape, then a one-request and a one-position batch.
    run(K3_BATCH, K3_STEPS)
    run(1, K3_STEPS)
    run(K3_BATCH, 1)


def test_accept_verdict_prefill_shape_is_bit_identical(
    session: InferenceSession,
) -> None:
    """A prefill iteration (no draft positions) agrees on both paths."""
    steps = 0
    folded, chain = _run_both(
        session,
        f"verdict_{False}_{K3_BATCH}_{VOCAB}",
        _verdict_input_types(
            K3_BATCH, steps, VOCAB, seed_rows=K3_BATCH, sampled=False
        )(_device(session)),
        _verdict_build(K3_BATCH, steps, VOCAB, sampled=False),
        _verdict_inputs(
            K3_BATCH, steps, VOCAB, seed_rows=K3_BATCH, sampled=False
        ),
    )
    _assert_identical(folded, chain, "verdict, prefill")


def _load_both(
    session: InferenceSession,
    name: str,
    input_types: list[TensorType],
    build: Build,
    inputs: Sequence[npt.NDArray[np.generic]],
) -> tuple[list[object], list[Buffer]]:
    """Loads the folded and the chain graph once each, beside their inputs."""
    device = _device(session)
    models: list[object] = []
    for folded in (True, False):
        with Graph(f"{name}_{folded}", input_types=input_types) as graph:
            graph.output(
                *build([v.tensor for v in graph.inputs], device, folded)
            )
        models.append(session.load(graph))
    buffers = [Buffer.from_numpy(i).to(session.devices[0]) for i in inputs]
    return models, buffers


def _time_rounds(
    models: Sequence[object], buffers: list[Buffer], runs: int
) -> list[float]:
    """Per-execute microseconds per arm, interleaved round by round.

    The arms alternate so a clock or neighbour drift lands on both. The timed
    loop adds no synchronization of its own: a round times back-to-back
    executes, as consecutive decode steps issue them.
    """
    for model in models:
        for _ in range(20):
            model.execute(*buffers)  # type: ignore[attr-defined]
    per_arm: list[list[float]] = [[] for _ in models]
    for _ in range(5):
        for index, model in enumerate(models):
            start = time.perf_counter()
            for _ in range(runs):
                model.execute(*buffers)  # type: ignore[attr-defined]
            elapsed = time.perf_counter() - start
            per_arm[index].append(elapsed * 1e6 / runs)
    return [float(np.median(arm)) for arm in per_arm]


@pytest.mark.skipif(
    not os.environ.get("INDEX_GLUE_BENCH"),
    reason="timing mode: set INDEX_GLUE_BENCH=1 to run",
)
def test_index_glue_timing(session: InferenceSession) -> None:
    """Per-execute cost of each folded family against the chain it replaced.

    Screening only: the arms run back to back on one box in one process, so
    what it ranks is the launch count, which is all the fold changes.
    """
    # Symbolic dims, as the accept graph carries them: the chain arm's
    # sentinel concat rebinds the mask to the graph's own `batch_size`.
    first_rejected_types = [
        TensorType(
            DType.bool,
            ["batch_size", "num_steps"],
            device=_device(session),
        )
    ]
    first_rejected_inputs = [_mask(K3_BATCH, K3_STEPS, [0, 3, 6, K3_STEPS])]

    def first_rejected_build(
        inputs: Sequence[TensorValue], device: DeviceRef, fold_index_glue: bool
    ) -> tuple[TensorValue, ...]:
        (mask,) = inputs
        return (
            ops.squeeze(
                _find_first_rejected(
                    mask, device, fold_index_glue=fold_index_glue
                ),
                axis=-1,
            ),
        )

    models, buffers = _load_both(
        session,
        "timing_first_rejected",
        first_rejected_types,
        first_rejected_build,
        first_rejected_inputs,
    )
    # A third arm: the chain again, so the spread between two runs of the same
    # code is visible next to the difference being reported.
    models.append(models[1])
    fused_us, chain_us, chain_again_us = _time_rounds(models, buffers, 100)
    print(
        f"first_rejected (batch={K3_BATCH}, steps={K3_STEPS}):"
        f" chain {chain_us:.2f} us, chain again {chain_again_us:.2f} us,"
        f" fused {fused_us:.2f} us, ratio {chain_us / fused_us:.2f}x"
    )


@pytest.mark.skipif(
    not os.environ.get("INDEX_GLUE_BENCH"),
    reason="timing mode: set INDEX_GLUE_BENCH=1 to run",
)
def test_sampled_verdict_timing(session: InferenceSession) -> None:
    """The whole sampled-draft verdict: folded glue against the chain."""
    models, buffers = _load_both(
        session,
        "timing_sampled_verdict",
        _verdict_input_types(
            K3_BATCH, K3_STEPS, VOCAB, seed_rows=1, sampled=True
        )(_device(session)),
        _verdict_build(K3_BATCH, K3_STEPS, VOCAB, sampled=True),
        _verdict_inputs(K3_BATCH, K3_STEPS, VOCAB, seed_rows=1, sampled=True),
    )
    models.append(models[1])
    fused_us, chain_us, chain_again_us = _time_rounds(models, buffers, 50)
    print(
        f"sampled_verdict (batch={K3_BATCH}, steps={K3_STEPS}, vocab={VOCAB}):"
        f" chain {chain_us:.2f} us, chain again {chain_again_us:.2f} us,"
        f" fused {fused_us:.2f} us, ratio {chain_us / fused_us:.2f}x"
    )


def _custom_ops(
    name: str,
    input_types: list[TensorType],
    build: Build,
    device: DeviceRef,
    fold_index_glue: bool,
) -> list[str]:
    """Builds ``build`` and returns the names of the custom ops it emitted."""
    seen: list[str] = []
    original = ops.custom

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(str(args[0]) if args else str(kwargs.get("op")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    ops.custom = spy  # type: ignore[assignment]
    try:
        with Graph(name, input_types=input_types) as graph:
            graph.output(
                *build(
                    [v.tensor for v in graph.inputs], device, fold_index_glue
                )
            )
    finally:
        ops.custom = original
    return seen


def test_folded_ops_are_in_the_folded_graph(session: InferenceSession) -> None:
    """The fold really fires, so a declined one cannot pass vacuously.

    Every other test here compares the folded graph with the chain, and a fold
    that declines to fire makes those two the same code: they would agree while
    measuring nothing. This pins the op each graph is supposed to emit, and
    that the default -- the path every non-K3 caller takes -- emits none of
    them, which is main's chain.
    """
    device = _device(session)
    mask_types = [
        TensorType(DType.bool, ["batch_size", "num_steps"], device=device)
    ]

    def mask_build(
        inputs: Sequence[TensorValue], device: DeviceRef, fold_index_glue: bool
    ) -> tuple[TensorValue, ...]:
        (mask,) = inputs
        return (
            ops.squeeze(
                _find_first_rejected(
                    mask, device, fold_index_glue=fold_index_glue
                ),
                axis=-1,
            ),
        )

    verdict_types = _verdict_input_types(
        K3_BATCH, K3_STEPS, VOCAB, seed_rows=1, sampled=True
    )(device)
    verdict_build = _verdict_build(K3_BATCH, K3_STEPS, VOCAB, sampled=True)

    folded_mask = _custom_ops(
        "ops_first_rejected", mask_types, mask_build, device, True
    )
    folded_verdict = _custom_ops(
        "ops_verdict", verdict_types, verdict_build, device, True
    )
    chain_mask = _custom_ops(
        "ops_first_rejected_chain", mask_types, mask_build, device, False
    )
    chain_verdict = _custom_ops(
        "ops_verdict_chain", verdict_types, verdict_build, device, False
    )

    assert "sampler.first_rejected" in folded_mask
    assert "sampler.first_rejected" not in chain_mask
    assert "sampler.accept_pair" in folded_verdict
    assert "sampler.accept_pair" not in chain_verdict


def test_default_sampler_keeps_the_chain(session: InferenceSession) -> None:
    """The AcceptanceSampler a non-K3 driver builds keeps main's chain.

    The fold reaches the graph ops through the sampler the drivers construct,
    so this pins the dispatch level too: a sampler built without the opt-in --
    what every target but K3's adapter passes -- emits no folded op, and one
    built with it emits both.
    """
    device = _device(session)
    verdict_types = _verdict_input_types(
        K3_BATCH, K3_STEPS, VOCAB, seed_rows=1, sampled=True
    )(device)

    def sampler_build(
        inputs: Sequence[TensorValue], device: DeviceRef, fold_index_glue: bool
    ) -> tuple[TensorValue, ...]:
        draft_tokens, logits, temperature, top_k, top_p, seed = inputs[:6]
        sampler = AcceptanceSampler(
            use_stochastic=True,
            draft_proposal="sampled",
            vocab_size=VOCAB,
            fold_index_glue=fold_index_glue,
        )
        return sampler(
            draft_tokens,
            logits,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            max_k=ops.constant(-1, DType.int64, DeviceRef.CPU()),
            top_p=top_p,
            min_top_p=ops.constant(0.5, DType.float32, DeviceRef.CPU()),
            draft_probs_full=inputs[6],
        )

    default_ops = _custom_ops(
        "ops_sampler_default", verdict_types, sampler_build, device, False
    )
    folded_ops = _custom_ops(
        "ops_sampler_folded", verdict_types, sampler_build, device, True
    )
    assert "sampler.first_rejected" not in default_ops
    assert "sampler.accept_pair" not in default_ops
    assert "sampler.first_rejected" in folded_ops
    assert "sampler.accept_pair" in folded_ops


def test_sampled_verdict_arms_are_bit_identical(
    session: InferenceSession,
) -> None:
    """Both verdict arms: accepted positions, and rows with no proposal.

    A random target rejects every row at position 0, which leaves the accept
    branch and the ``q == 0`` substitution untested. This drives both: half the
    positions have their drafted token lifted in the target, and every other
    request carries an all-zero proposal distribution.
    """
    folded, chain = _run_both(
        session,
        f"verdict_{True}_{1}_{VOCAB}",
        _verdict_input_types(
            K3_BATCH, K3_STEPS, VOCAB, seed_rows=1, sampled=True
        )(_device(session)),
        _verdict_build(K3_BATCH, K3_STEPS, VOCAB, sampled=True),
        _verdict_inputs(
            K3_BATCH,
            K3_STEPS,
            VOCAB,
            seed_rows=1,
            sampled=True,
            accept_some=True,
            zero_q_rows=True,
        ),
    )
    _assert_identical(folded, chain, "sampled verdict, accept and q == 0 arms")
    rejected_at = np.asarray(folded[0])
    # Both arms really ran: some row accepted a draft, and some rejected one.
    assert (rejected_at > 0).any(), "no row accepted a draft"
    assert (rejected_at < K3_STEPS).any(), "no row rejected a draft"
    # The rows carrying no proposal reject at position 1, which is the odd
    # position whose target mass was left alone. They only do that because the
    # test takes q = 1 there: with q left at zero the coin can never reject.
    assert (rejected_at[1::2] < K3_STEPS).all(), (
        "a row without a proposal accepted every draft: the q == 0"
        " substitution is not in effect"
    )


@pytest.mark.parametrize("seed_rows", [K3_BATCH, 1])
def test_sampled_verdict_prefill_is_bit_identical(
    session: InferenceSession, seed_rows: int
) -> None:
    """The sampled proposal at zero draft positions, folded and unfolded.

    Both seed shapes matter here: a prefill iteration carries `[batch, 0]`
    draft tokens, so the per-row seed never has a position count to divide by.
    The fold still serves that shape (its kernel launches no threads at zero
    steps), and its result must match the chain's.
    """
    folded, chain = _run_both(
        session,
        f"verdict_{True}_{seed_rows}_{VOCAB}",
        _verdict_input_types(
            K3_BATCH, 0, VOCAB, seed_rows=seed_rows, sampled=True
        )(_device(session)),
        _verdict_build(K3_BATCH, 0, VOCAB, sampled=True),
        _verdict_inputs(K3_BATCH, 0, VOCAB, seed_rows=seed_rows, sampled=True),
    )
    _assert_identical(folded, chain, "sampled verdict, prefill")


def test_sampled_verdict_production_vocab_is_bit_identical(
    session: InferenceSession,
) -> None:
    """The same verdict at K3's real vocabulary width, where the split-K runs.

    At 163840 columns the masked-softmax and the residual draw take their
    split-K paths, so this is the shape the fold meets in serving rather than
    the small-vocab shape the rest of the suite uses.
    """
    vocab = 163840
    folded, chain = _run_both(
        session,
        f"verdict_{True}_{1}_{vocab}",
        _verdict_input_types(
            K3_BATCH, K3_STEPS, vocab, seed_rows=1, sampled=True
        )(_device(session)),
        _verdict_build(K3_BATCH, K3_STEPS, vocab, sampled=True),
        _verdict_inputs(
            K3_BATCH,
            K3_STEPS,
            vocab,
            seed_rows=1,
            sampled=True,
            accept_some=True,
            zero_q_rows=True,
        ),
    )
    _assert_identical(folded, chain, "sampled verdict, production vocab")
    # A swap of the two distributions, or a lost q == 0 substitution, changes
    # these bits, so the case has to reach both outcomes at this width too.
    rejected_at = np.asarray(folded[0])
    assert (rejected_at > 0).any(), "no row accepted a draft"
    assert (rejected_at < K3_STEPS).any(), "no row rejected a draft"
