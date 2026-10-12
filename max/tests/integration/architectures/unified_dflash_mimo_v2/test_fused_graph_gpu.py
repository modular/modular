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
"""Tests the fused MiMo-V2 DFlash graph and the base-ctx graph together.

A random-weight model runs every graph a spec-on deployment serves, sharing
one set of paged caches, at tensor parallelism 2 when two GPUs are present:

* the drafter's context that base-ctx writes, over chunks and after a prefix
  hit restored from another request's pages, is the context the fused graph
  writes for the same tokens;
* every token the fused graph commits is the target's argmax, bitmask
  applied, in the forward that committed it, at K 3 and 7, and a graph that
  accepts one extra draft fails that;
* speculation commits the tokens plain greedy decoding would.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Accelerator, Device, accelerator_count
from max.engine import InferenceSession
from max.pipelines.architectures.unified_dflash_mimo_v2.model_config import (
    DRAFT,
)
from mimo_dflash_harness import (
    PAGE_SIZE,
    SAMPLEABLE,
    BaseRunner,
    FusedRunner,
    Model,
    PagedTree,
    Pages,
    Row,
    base_runner,
    fused_runner,
    generate,
    greedy,
    tiny_model,
)

BLOCK = 8
CONTEXT = 3000
CHUNK = 1000
HIT_PAGES = 15
"""A prefix hit covers whole pages: 1,920 positions of the 3,000."""
TOLERANCE = 0.05
"""Per-position relative error of the drafter's K or V between two graphs
that wrote it from the same tokens: they differ by BF16 GEMM noise. Without
the writer the error is 1, and without the context V scale 0.63."""
MAX_GAP = 1.0
"""How far, in logits, a committed token may sit below the one-token decode
path's argmax."""


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
def fused(
    model: Model, devices: list[Device], session: InferenceSession
) -> dict[int, FusedRunner]:
    return {
        k: fused_runner(f"fused_k{k}", model, devices, session) for k in (3, 7)
    }


@pytest.fixture(scope="module")
def base_ctx(
    model: Model, devices: list[Device], session: InferenceSession
) -> BaseRunner:
    return base_runner("base_ctx", model, devices, session)


@pytest.fixture(scope="module")
def base(
    model: Model, devices: list[Device], session: InferenceSession
) -> BaseRunner:
    return base_runner("base", model, devices, session)


def _caches(fused: FusedRunner, devices: Sequence[Device]) -> PagedTree:
    return PagedTree(fused.groups, devices, num_pages=256)


def _per_position_error(
    got: npt.NDArray[np.float32], want: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """``[layers, positions, heads, dim]`` -> relative L2 error per position."""
    got, want = np.moveaxis(got, 1, 0), np.moveaxis(want, 1, 0)
    diff = np.linalg.norm((got - want).reshape(len(got), -1), axis=1)
    return diff / np.linalg.norm(want.reshape(len(want), -1), axis=1)


def test_base_ctx_writes_the_context_the_fused_graph_writes(
    fused: dict[int, FusedRunner],
    base_ctx: BaseRunner,
    devices: list[Device],
) -> None:
    caches = _caches(fused[7], devices)
    pages = Pages()
    tokens = np.random.default_rng(1).integers(0, SAMPLEABLE, CONTEXT).tolist()
    whole = pages.take(CONTEXT + 2 * BLOCK)
    fused[7].step(caches, [Row(tokens, 0, whole)])

    chunked = pages.take(CONTEXT + 2 * BLOCK)
    for start in range(0, CONTEXT, CHUNK):
        base_ctx.step(
            caches, [Row(tokens[start : start + CHUNK], start, chunked)]
        )

    # The hit's pages are the fused request's; base-ctx computes the rest on
    # top of that request's target KV.
    hit = HIT_PAGES * PAGE_SIZE
    restored = whole[:HIT_PAGES] + pages.take(CONTEXT - hit + 2 * BLOCK)
    base_ctx.step(caches, [Row(tokens[hit:], hit, restored)])

    print(f"devices: {len(devices)}")
    want = caches.read(DRAFT, whole, range(CONTEXT))
    for name, got_pages, first in (
        ("chunked", chunked, 0),
        ("prefix hit", restored, hit),
    ):
        got = caches.read(DRAFT, got_pages, range(CONTEXT))
        for index, what in enumerate("KV"):
            error = _per_position_error(got[index], want[index])[first:]
            print(
                f"{name} {what}: max {error.max():.3g}, median "
                f"{np.median(error):.3g}"
            )
            assert error.max() < TOLERANCE, (
                f"{name}: drafter {what} differs by {error.max():.3g} at"
                f" position {first + int(error.argmax())}"
            )


def _prompts(count: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, SAMPLEABLE, int(n)).tolist()
        for n in rng.integers(20, 300, count)
    ]


@pytest.mark.parametrize("k", [3, 7])
def test_every_committed_token_is_the_target_argmax(
    fused: dict[int, FusedRunner], devices: list[Device], k: int
) -> None:
    runner = fused[k]
    prompts = _prompts(4, seed=k)
    caches, pages = _caches(runner, devices), Pages()
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
    devices: list[Device],
    session: InferenceSession,
    fused: dict[int, FusedRunner],
) -> None:
    mutant = fused_runner("fused_k7_accept_one_extra", model, devices, session)
    prompts = _prompts(4, seed=9)
    caches, pages = _caches(mutant, devices), Pages()
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
    fused: dict[int, FusedRunner],
    base: BaseRunner,
    devices: list[Device],
    k: int,
) -> None:
    prompts = _prompts(4, seed=20 + k)
    tokens = 48
    caches, pages = _caches(fused[k], devices), Pages()
    plain, _ = greedy(base, caches, pages, prompts, tokens, SAMPLEABLE)
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
    _, logits = greedy(
        base, caches, pages, prompts, tokens, SAMPLEABLE, forced=spec.tokens
    )
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
