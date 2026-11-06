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
"""One end-to-end run of the spec_parity harness.

The end-to-end test serves a tiny random-weight K3 (ModuleV3) checkpoint and
a tiny random DSpark drafter through the harness and requires the
speculative run to decode the plain run's greedy tokens. Random weights prove
little on their own -- greedy parity holds for any drafter, and a random
drafter over this tiny vocabulary almost never has a draft accepted -- so
the test also asserts what proves the pairing engaged: the pipeline's own
draft counts (proposed, and accepted under a synthetic rate) and the
teacher-forced oracle, that every committed token is the plain pipeline's
argmax at its own position.

The tiny pair mirrors the real checkpoints' shapes exactly, scaled down:
the K3 fixture copies the real checkpoint's four-layer layout
(KDA 1-3, MLA 4), checkpoint names and remote-code ``configuration.py``;
the drafter copies the published ``Kimi-K3-DSpark`` snapshot's own
config.json keys -- which ship no remote code, so the drafter loads the
way the real snapshot does -- with its five taps cut to two remapped into
the four-layer target's range.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import spec_parity
from _tiny_pair import DRAFT_WIDTH, MAX_LENGTH, NARROW_LATENT, build_tiny_pair
from max.pipelines.lib.generate import BatchPipeline
from spec_parity import ParityReport

# Building the pipelines warms up a KV manager, which virtual devices mock.
pytestmark = pytest.mark.compile_on_gpu(
    "the pipelines need a real KV manager; virtual devices get a mock"
)

# ===--------------------------------------------------------------------=== #
# The tiny end-to-end pair
# ===--------------------------------------------------------------------=== #


@pytest.fixture(scope="module")
def tiny_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    """A tiny random K3 target whose narrow latent keeps its decompressed
    cache path under test, a tiny random DSpark drafter, and a prompts
    file of token-id lists."""
    pair = build_tiny_pair(
        tmp_path_factory.mktemp("k3_tiny_pair"),
        ([7, 8, 9], [12, 0, 3, 5], [42]),
        geometry=NARROW_LATENT,
    )
    return pair.target, pair.drafter, pair.prompts


_PLAIN_PIPELINE: BatchPipeline | None = None


def _shared_plain_pipeline(
    tiny_pair: tuple[Path, Path, Path],
) -> BatchPipeline:
    """One plain greedy pipeline, shared by every graded run below.

    Built before any speculative pipeline exists, so the cached HF config
    it reads is the plain one a speculative config would rewrite. Each
    run() call would otherwise rebuild the identical pipeline. A plain
    build needs no real hardware, so it must not be a recorded fixture of
    its own: a fixture the record runs to its end replays exactly, and it
    holds no artifact for the interpreter's lazily-compiled CPU casts the
    real weight load compiles (the record's virtual CPU serves the
    adapter's bf16-to-fp32 astype without a graph). Built at the top of
    the clean synthetic's setup instead, its graphs ride that fixture's
    record entry, whose hardware stop its replay already tolerates.
    """
    global _PLAIN_PIPELINE
    if _PLAIN_PIPELINE is None:
        target, _, prompts_path = tiny_pair
        _PLAIN_PIPELINE = spec_parity._pipeline(
            target,
            None,
            devices=1,
            max_length=MAX_LENGTH,
            num_speculative_tokens=DRAFT_WIDTH,
            synthetic_acceptance=None,
            device_graph_capture=False,
            prefer_module_v3=True,
            trust_remote_code=True,
            prompts=spec_parity.read_prompts(prompts_path),
            speculative=False,
        )
    return _PLAIN_PIPELINE


def test_the_tiny_pair_decodes_the_plain_greedy_tokens(
    tiny_pair: tuple[Path, Path, Path],
    clean_synthetic: ParityReport,
) -> None:
    """Parity, engagement and the teacher-forced oracle on the real-acceptance
    run; acceptance traffic through a synthetic run whose parity is not
    asserted (the synthetic sampler commits drafts greedy did not pick) but
    whose target-chosen bonus tokens are graded against the plain pipeline,
    which is what catches a fold short of the accepted block."""
    target, drafter, prompts = tiny_pair
    plain_pipeline = _shared_plain_pipeline(tiny_pair)
    # A 128-token page rounds each prompt up to a page, so the context
    # budget the scheduler defaults out of ``max_length`` must clear a
    # whole page per prompt or the first prefill is refused.
    report = spec_parity.run(
        target,
        drafter,
        prompts,
        devices=1,
        max_new_tokens=32,
        repeat=2,
        max_length=MAX_LENGTH,
        trust_remote_code=True,
        teacher_forced=True,
        plain_pipeline=plain_pipeline,
    )
    assert report.deterministic, "two speculative runs disagreed"
    assert report.identical, [
        (prompt.index, prompt.divergence) for prompt in report.prompts
    ]
    # Parity alone holds for a pipeline whose draft path never ran, so the
    # pipeline's own draft counts must show the pairing engaged.
    assert report.drafts_proposed > 0, "no draft was ever proposed"
    # A random drafter almost never agrees with the target, so the committed
    # tokens are all target-chosen: each must be the plain argmax at its
    # own position, a rounding near-tie at worst.
    assert report.near_ties_only, report.teacher_forced_misses

    synthetic = clean_synthetic
    assert synthetic.deterministic
    assert synthetic.drafts_proposed > 0
    # Accept-everything traffic: every verify folds the full draft block,
    # and the graded oracle checks the target-chosen bonus positions -- the
    # ones whose KDA state a fold short of the accepted block corrupts.
    assert synthetic.drafts_accepted > 0, (
        f"{synthetic.drafts_proposed} drafts proposed, none accepted"
    )
    # The graded bonuses must include at least one that a fold corrupts,
    # or the near-tie check below would pass vacuously.
    assert synthetic.fold_sensitive_graded >= 1, (
        "the synthetic run graded no position past the first verify"
    )
    # The graded bonus positions must be the plain argmax up to a rounding
    # near-tie: the verify computes them through the spec kernel path, so
    # exact equality is a stronger demand than bf16 can make.
    assert synthetic.near_ties_only, synthetic.teacher_forced_misses


@pytest.fixture(scope="module")
def clean_synthetic(
    tiny_pair: tuple[Path, Path, Path],
) -> ParityReport:
    """One clean rate-1.0 synthetic run, shared by every graded assertion.

    Built once so the graded synthetic checks compare against one stream
    instead of rebuilding the pair. 33 tokens span a prefill and four
    decode steps: the graded bonuses sit at the ends of the first four
    chunks (the fifth is final and skipped), and the last two of those
    read a state a fold already landed. The shared plain pipeline is built
    here first, before any speculative pipeline exists.
    """
    plain_pipeline = _shared_plain_pipeline(tiny_pair)
    target, drafter, prompts_path = tiny_pair
    return spec_parity.run(
        target,
        drafter,
        prompts_path,
        devices=1,
        max_new_tokens=33,
        max_length=MAX_LENGTH,
        trust_remote_code=True,
        synthetic_acceptance=1.0,
        teacher_forced=True,
        plain_pipeline=plain_pipeline,
    )
