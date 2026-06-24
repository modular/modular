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


def _canonical_casing_by_casefold() -> dict[str, str]:
    """Maps each locked repo id to its canonical casing, keyed case-insensitively.

    ``hf-repo-lock.tsv`` is the source of truth for what we cache, and every row
    is independently verified against Hugging Face's canonical casing by
    ``test_hf_repo_lock_casing``.
    """
    canonical: dict[str, str] = {}
    with _HF_REPO_LOCK.open() as lock_file:
        for row in csv.DictReader(lock_file, dialect="excel-tab"):
            repo = row["hf_repo"]
            if repo != _EXAMPLE_KEY:
                canonical[repo.casefold()] = repo
    return canonical


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
