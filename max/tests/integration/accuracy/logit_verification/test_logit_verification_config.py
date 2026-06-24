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

import csv
from pathlib import Path

from max.tests.integration.accuracy.logit_verification.logit_verification_config import (
    LOGIT_VERIFICATION_CONFIG,
    LogitVerificationPipelineConfig,
)

# hf-repo-lock.tsv (the source of truth for what we cache) lives in
# max/tests/integration/, three directories up; it is pulled in as a data
# dependency of this test.
_HF_REPO_LOCK = Path(__file__).parents[2] / "hf-repo-lock.tsv"

# Sentinel row in hf-repo-lock.tsv that is not a real Hugging Face repo.
_EXAMPLE_KEY = "000EXAMPLE-for-unit-test/repo"


def test_pre_submit_matrix() -> None:
    matrix = LOGIT_VERIFICATION_CONFIG.pre_submit_matrix
    assert len(matrix) > 0


def _locked_repos() -> set[str]:
    """Returns the set of Hugging Face repo ids pinned in ``hf-repo-lock.tsv``."""
    repos: set[str] = set()
    with _HF_REPO_LOCK.open() as lock_file:
        for row in csv.DictReader(lock_file, dialect="excel-tab"):
            repo = row["hf_repo"]
            if repo != _EXAMPLE_KEY:
                repos.add(repo)
    return repos


def _canonical_casing_by_casefold() -> dict[str, str]:
    """Maps each locked repo id to its canonical casing, keyed case-insensitively.

    ``hf-repo-lock.tsv`` is the source of truth for what we cache, and every row
    is independently verified against Hugging Face's canonical casing by
    ``test_hf_repo_lock_casing``.
    """
    return {repo.casefold(): repo for repo in _locked_repos()}


def test_pipeline_models_use_canonical_casing() -> None:
    """Ensures every locked model in the logit verification config uses the exact
    casing from ``hf-repo-lock.tsv``.

    Hugging Face resolves repository IDs case-insensitively, but the local cache
    is keyed on the literal string passed to ``snapshot_download``. A reference
    that differs only in casing therefore downloads a second copy of the weights
    (doubling cache usage) and misses the locked revision under
    ``HF_HUB_OFFLINE``. This guards against reintroducing that drift here.
    """
    canonical = _canonical_casing_by_casefold()
    violations: list[str] = []
    for name, pipeline_config in LOGIT_VERIFICATION_CONFIG.pipelines.items():
        model = pipeline_config.pipeline
        # Only validate Hugging Face repo IDs ("org/model"); skip local paths
        # and bare names that have no canonical Hub casing to compare against.
        if model.startswith("/") or model.count("/") != 1:
            continue
        expected = canonical.get(model.casefold())
        if expected is not None and model != expected:
            violations.append(
                f"  pipeline {name!r}: {model!r} should be {expected!r}"
            )
    assert not violations, (
        "Non-canonical Hugging Face model casing in "
        "logit_verification_config.yaml (must match hf-repo-lock.tsv exactly, "
        "or the local cache double-downloads the weights):\n"
        + "\n".join(violations)
    )


def _weight_repo_ids(
    pipeline_config: LogitVerificationPipelineConfig,
) -> list[str]:
    """Extracts the Hugging Face weight repo ids a pipeline pulls in through
    ``transformer_weight_path`` overrides.

    Pixel-generation pipelines may use a two-repo layout where the
    transformer's quantized weights live in a separate single-file checkpoint
    (e.g. ``FLUX.2-dev`` for the pipeline plus ``FLUX.2-dev-NVFP4`` for the
    transformer). A weight path looks like ``org/repo/.../file.safetensors``;
    the repo id is the leading ``org/repo``.
    """
    overrides = pipeline_config.config_params_override or {}
    weight_paths = overrides.get("transformer_weight_path") or []
    repos: list[str] = []
    for weight_path in weight_paths:
        weight_path = str(weight_path)
        # Only Hub references have a lockable repo id; skip local paths.
        if weight_path.startswith("/"):
            continue
        segments = weight_path.split("/")
        if len(segments) >= 3:
            repos.append("/".join(segments[:2]))
    return repos


def test_pipeline_weight_repos_are_locked() -> None:
    """Ensures every separate weight repo a logit verification pipeline pulls in
    is pinned in ``hf-repo-lock.tsv``.

    Pipelines that use the two-repo layout name their quantized weight repo only
    through a ``transformer_weight_path`` override, not as the pipeline's model
    id. ``hf_repo_lock.apply_to_config`` looks that weight repo up in the lock
    and raises ``No locked revision found for weight repository`` when it is
    missing, crashing the pipeline at run time (and the offline cache cannot
    resolve it under ``HF_HUB_OFFLINE``). Requiring an exact match against the
    lock key additionally enforces the weight repo's canonical casing.
    """
    locked = _locked_repos()
    violations: list[str] = []
    for name, pipeline_config in LOGIT_VERIFICATION_CONFIG.pipelines.items():
        for repo in _weight_repo_ids(pipeline_config):
            if repo not in locked:
                violations.append(f"  pipeline {name!r}: weight repo {repo!r}")
    assert not violations, (
        "Logit verification pipelines reference weight repositories missing "
        "from hf-repo-lock.tsv. apply_to_config raises 'No locked revision "
        "found for weight repository' for these at run time; add a row pinning "
        "each to its Hugging Face commit:\n" + "\n".join(violations)
    )
