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
"""Tests the fused MiMo-V2 DFlash graph's sampled acceptance and its
``lm_head`` padding rows, on a random-weight model:

* rows with their own temperature, top-k, top-p and seed commit only
  tokenizer ids, and replaying their seeds replays their output;
* no committed id is a padding row, even with the padding rows' logits
  forced above every real token's and the grammar bitmask allowing them.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from max.driver import Accelerator, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import WeightData
from mimo_dflash_harness import (
    SAMPLEABLE,
    FusedRunner,
    Generation,
    Model,
    PagedTree,
    Pages,
    Sampling,
    bf16_bits,
    fused_runner,
    generate,
    tiny_model,
    to_f32,
    weight_data,
)

ROWS = [
    Sampling(seed=11, temperature=0.7, top_k=8, top_p=0.9),
    Sampling(seed=(1 << 40) + 3, temperature=1.0, top_k=50, top_p=0.95),
    Sampling(seed=97, temperature=1.3, top_k=200, top_p=1.0),
    Sampling(seed=5, temperature=0.9, top_k=20, top_p=0.8),
]
TOKENS = 48


@pytest.fixture(scope="module")
def devices() -> list[Device]:
    return [Accelerator(i) for i in range(min(accelerator_count(), 2))]


@pytest.fixture(scope="module")
def session(devices: list[Device]) -> InferenceSession:
    return InferenceSession(devices=devices)


@pytest.fixture(scope="module")
def model() -> Model:
    return tiny_model()


@pytest.fixture(scope="module")
def sampled(
    model: Model, devices: list[Device], session: InferenceSession
) -> FusedRunner:
    return fused_runner(
        "fused_k7_sampled",
        model,
        devices,
        session,
        bitmask_allows_padding=True,
    )


def _prompts(seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, SAMPLEABLE, int(n)).tolist()
        for n in rng.integers(20, 300, len(ROWS))
    ]


def _run(
    runner: FusedRunner,
    devices: list[Device],
    prompts: list[list[int]],
    rows: list[Sampling],
    guide: list[list[int]] | None = None,
) -> Generation:
    caches = PagedTree(runner.groups, devices, num_pages=64)
    return generate(
        runner,
        caches,
        Pages(),
        prompts,
        max_new=TOKENS,
        guide=guide,
        rng=np.random.default_rng(0) if guide is not None else None,
        sampling=rows,
    )


def _committed_ids(run: Generation) -> set[int]:
    return {t for seq in run.tokens for t in seq}


def test_sampled_rows_replay_from_their_seeds(
    sampled: FusedRunner, devices: list[Device]
) -> None:
    prompts = _prompts(1)
    first = _run(sampled, devices, prompts, ROWS)
    assert first.tokens == _run(sampled, devices, prompts, ROWS).tokens
    assert max(_committed_ids(first)) < SAMPLEABLE

    # Drafting the committed tokens back gets drafts accepted, so recovered
    # and bonus tokens come from every verify position.
    guided = _run(sampled, devices, prompts, ROWS, guide=first.tokens)
    replay = _run(sampled, devices, prompts, ROWS, guide=first.tokens)
    assert guided.tokens == replay.tokens
    assert guided.accepted == replay.accepted
    assert max(a for row in guided.accepted for a in row) > 0
    assert max(_committed_ids(guided)) < SAMPLEABLE

    # Far from the seeds above plus any token count, so no draw is shared.
    reseeded = [dataclasses.replace(r, seed=r.seed + 10**6) for r in ROWS]
    other = _run(sampled, devices, prompts, reseeded)
    differ = [a != b for a, b in zip(first.tokens, other.tokens, strict=True)]
    assert all(differ), differ


def _padding_forced_high(model: Model) -> dict[str, WeightData]:
    """The target's weights with every padding row of ``lm_head`` scoring far
    above every real row, for any final hidden state.

    Each padding row is ``+-c`` times one hidden unit vector: whatever the
    hidden state's sign in that dimension, one of the pair scores ``c`` times
    its magnitude there, and the RMS-normalized state is O(1) in some of the
    dimensions they cover.
    """
    name = "lm_head.weight"
    weight = model.target_state[name]
    head = to_f32(weight.to_buffer())
    for i, row in enumerate(range(SAMPLEABLE, len(head))):
        head[row] = 0
        head[row, i // 2] = 1000.0 if i % 2 == 0 else -1000.0
    state = dict(model.target_state)
    state[name] = weight_data(name, DType.bfloat16, bf16_bits(head))
    return state


@pytest.mark.parametrize("greedy", [True, False])
def test_padding_rows_are_never_committed(
    model: Model,
    devices: list[Device],
    session: InferenceSession,
    greedy: bool,
) -> None:
    runner = fused_runner(
        "fused_k7" if greedy else "fused_k7_sampled",
        model,
        devices,
        session,
        target_state=_padding_forced_high(model),
        bitmask_allows_padding=True,
    )
    prompts = _prompts(2)
    rows = [Sampling()] * len(ROWS) if greedy else ROWS
    own = _run(runner, devices, prompts, rows)
    assert max(_committed_ids(own)) < SAMPLEABLE
    guided = _run(runner, devices, prompts, rows, guide=own.tokens)
    assert max(_committed_ids(guided)) < SAMPLEABLE
