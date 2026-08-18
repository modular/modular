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

"""Verifies that every entry in hf-repo-lock.tsv uses the canonical casing
reported by Hugging Face.

Hugging Face resolves repository IDs case-insensitively, but local caches
(e.g. the one populated by ``snapshot_download``) are keyed on the
canonical casing returned by the API.  When a row in hf-repo-lock.tsv uses
a different casing than HF's canonical form, offline lookups under
``HF_HUB_OFFLINE=1`` can miss the cache and fail at runtime.  This test
issues one HF API call per locked repository and therefore runs only in
the dedicated HF workflow.
"""

from __future__ import annotations

import csv
from pathlib import Path

import huggingface_hub
import pytest

# Sentinel row that is not a real Hugging Face repo.
EXAMPLE_KEY = "000EXAMPLE-for-unit-test/repo"

_HF_REPO_LOCK = Path(__file__).parent / "hf-repo-lock.tsv"


def _locked_repo_ids() -> list[str]:
    with _HF_REPO_LOCK.open() as f:
        return sorted(
            row["hf_repo"]
            for row in csv.DictReader(f, dialect="excel-tab")
            if row["hf_repo"] != EXAMPLE_KEY
        )


@pytest.mark.parametrize("repo_id", _locked_repo_ids())
def test_canonical_casing(repo_id: str) -> None:
    info = huggingface_hub.repo_info(repo_id)
    assert info.id == repo_id, (
        f"{repo_id!r} in hf-repo-lock.tsv does not match the canonical "
        f"casing on Hugging Face (expected {info.id!r}). Update the row "
        "so offline cache lookups resolve correctly."
    )
