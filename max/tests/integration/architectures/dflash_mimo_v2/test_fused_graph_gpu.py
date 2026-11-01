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
"""Tests the fused MiMo-V2 DFlash graph against the target's own decoding.

A random-weight model runs the fused graph and the plain MiMo-V2 graph over
shared paged caches, on one GPU:

* every token the fused graph commits is the target's argmax, bitmask
  applied, in the forward that committed it, at K 3 and 7, and a graph that
  accepts one extra draft fails that;
* speculation commits the tokens plain greedy decoding would.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.engine import InferenceSession
from mimo_dflash_harness import (
    SAMPLEABLE,
    BaseRunner,
    FusedRunner,
    Model,
    PagedTree,
    Pages,
    generate,
    greedy,
    tiny_model,
)

MAX_GAP = 1.0
"""How far, in logits, a committed token may sit below the one-token decode
path's argmax."""


@pytest.fixture(scope="module")
def device() -> Accelerator:
    return Accelerator(0)


@pytest.fixture(scope="module")
def session(device: Accelerator) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def model() -> Model:
    return tiny_model()


@pytest.fixture(scope="module")
def fused(
    model: Model, device: Accelerator, session: InferenceSession
) -> dict[int, FusedRunner]:
    return {k: FusedRunner(model, k, device, session) for k in (3, 7)}


@pytest.fixture(scope="module")
def base(
    model: Model, device: Accelerator, session: InferenceSession
) -> BaseRunner:
    return BaseRunner(model, device, session)


def _caches(runner: FusedRunner) -> PagedTree:
    return PagedTree(runner.groups, runner.device, num_pages=256)


def _prompts(count: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, SAMPLEABLE, int(n)).tolist()
        for n in rng.integers(20, 300, count)
    ]


@pytest.mark.parametrize("k", [3, 7])
def test_every_committed_token_is_the_target_argmax(
    fused: dict[int, FusedRunner], k: int
) -> None:
    runner = fused[k]
    prompts = _prompts(4, seed=k)
    caches, pages = _caches(runner), Pages()
    own = generate(runner, caches, pages, prompts, max_new=64)
    assert not own.violations, own.violations[:5]
    assert all(t < SAMPLEABLE for seq in own.tokens for t in seq)
    # A random drafter rarely proposes the target's token, so drive the same
    # graph with the committed tokens themselves as drafts.
    guided = generate(
        runner,
        caches,
        pages,
        prompts,
        max_new=64,
        guide=own.tokens,
        rng=np.random.default_rng(0),
    )
    assert not guided.violations, guided.violations[:5]
    accepted = [a for row in guided.accepted for a in row]
    assert max(accepted) == k and min(accepted) < k, accepted


def test_accepting_one_extra_draft_breaks_self_consistency(
    model: Model,
    device: Accelerator,
    session: InferenceSession,
    fused: dict[int, FusedRunner],
) -> None:
    mutant = FusedRunner(model, 7, device, session, accept_one_extra=True)
    prompts = _prompts(4, seed=9)
    caches, pages = _caches(mutant), Pages()
    own = generate(fused[7], caches, pages, prompts, max_new=32)
    broken = generate(
        mutant,
        caches,
        pages,
        prompts,
        max_new=32,
        guide=own.tokens,
        rng=np.random.default_rng(0),
    )
    assert broken.violations


@pytest.mark.parametrize("k", [3, 7])
def test_speculation_commits_what_greedy_decoding_would(
    fused: dict[int, FusedRunner], base: BaseRunner, k: int
) -> None:
    prompts = _prompts(4, seed=20 + k)
    tokens = 48
    caches, pages = _caches(fused[k]), Pages()
    plain, _ = greedy(base, caches, pages, prompts, tokens)
    spec = generate(
        fused[k],
        caches,
        pages,
        prompts,
        max_new=tokens,
        guide=plain,
        rng=np.random.default_rng(1),
    )
    assert not spec.violations, spec.violations[:5]
    # Teacher-forced on what speculation committed, the one-token decode
    # path must rank every committed token within the gap of its argmax.
    _, logits = greedy(base, caches, pages, prompts, tokens, forced=spec.tokens)
    gaps = [
        float(z[:SAMPLEABLE].max() - z[t])
        for seq, zs in zip(spec.tokens, logits, strict=True)
        for t, z in zip(seq, zs, strict=True)
    ]
    same = np.mean(
        [
            a == b
            for p, q in zip(plain, spec.tokens, strict=True)
            for a, b in zip(p, q, strict=True)
        ]
    )
    print(f"K={k}: {same:.3f} of tokens equal, largest gap {max(gaps):.3g}")
    assert max(gaps) <= MAX_GAP, max(gaps)
