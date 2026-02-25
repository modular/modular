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

from __future__ import annotations

import csv
import functools
import logging
from collections.abc import Mapping
from pathlib import Path

import huggingface_hub
from max import pipelines

logger = logging.getLogger(__name__)


@functools.cache
def load_db() -> Mapping[str, str]:
    db = {}
    last_key = None
    # TODO: Switch back to resources.files(__name__).joinpath once support for Python 3.10 is dropped.
    with (Path(__file__).parent / "hf-repo-lock.tsv").open() as f:
        for row in csv.DictReader(f, dialect="excel-tab"):
            key = row["hf_repo"]
            value = row["revision"]
            if last_key is not None and key < last_key:
                raise ValueError(
                    "hf-repo-lock.tsv must be sorted, but I found key "
                    f"{key!r} after {last_key!r}.  Please sort and try again."
                )
            if not value:
                guess = ""
                if any(c.isspace() or c == "," for c in key):
                    guess += (
                        "  Did you perhaps use something "
                        "other than tab to delimit fields?"
                    )
                raise ValueError(
                    f"Missing value for key {key!r} in hf-repo-lock.tsv.{guess}"
                )
            db[key] = value
            last_key = key
    return db


def revision_for_hf_repo(hf_repo_id: str) -> str | None:
    """Get the locked revision for a Hugging Face repository.

    This function looks up the revision hash for a given Hugging Face repository ID
    in the hf-repo-lock.tsv file. If the repository is not found in the lock file,
    it attempts to suggest the main branch's commit hash as a potential revision.

    Args:
        hf_repo_id: The Hugging Face repository ID to look up (e.g. "modularai/Llama-3.2-1B-Instruct-Extended-Vocab")

    Returns:
        The locked revision hash if found in hf-repo-lock.tsv, or None if not found.
        If None is returned, a warning will be logged with a suggested revision if available.
    """
    if hf_repo_id.startswith("/"):
        # This is a local path; there won't be a revision.
        return None
    if hf_repo_id.count("/") != 1:
        raise ValueError(
            f"Invalid Hugging Face repository ID: {hf_repo_id!r}.  "
            "It must be in the format 'org/model'."
        )

    db = load_db()
    if hf_repo_id in db:
        return db[hf_repo_id]
    # Past this point, we're generating an error.  It's just a matter of making
    # the error as helpful as we can.
    suggested_revision = None
    try:
        refs = huggingface_hub.list_repo_refs(hf_repo_id)
        for ref in refs.branches:
            if ref.ref == "refs/heads/main":
                suggested_revision = ref.target_commit
                break
    except Exception:
        # Ignore errors -- we were just trying to be helpful.
        pass

    logger.warning(
        f"No lock revision available for Hugging Face repo {hf_repo_id!r}.  "
        "Add a row to hf-repo-lock.tsv to resolve this error.  "
        f"(Suggested revision: {suggested_revision or 'not available'})"
    )
    return None


def apply_to_config(config: pipelines.PipelineConfig) -> None:
    model_revision = revision_for_hf_repo(config.model.model_path)
    if model_revision is None:
        raise ValueError(
            f"No locked revision found for model repository: {config.model.model_path!r}. "
        )
    config.model.huggingface_model_revision = model_revision

    weight_revision = revision_for_hf_repo(
        config.model.huggingface_weight_repo_id
    )
    if weight_revision is None:
        raise ValueError(
            f"No locked revision found for weight repository: {config.model.huggingface_weight_repo_id!r}. "
        )
    config.model.huggingface_weight_revision = weight_revision
