# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import csv
import functools
from collections.abc import Mapping
from importlib import resources

import huggingface_hub
from max import pipelines


@functools.cache
def load_db() -> Mapping[str, str]:
    db = {}
    last_key = None
    with resources.files(__name__).joinpath("hf-repo-lock.tsv").open() as f:
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


def revision_for_hf_repo(hf_repo_id: str) -> str:
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
    raise KeyError(
        f"No lock revision available for Hugging Face repo {hf_repo_id!r}.  "
        "Add a row to hf-repo-lock.tsv to resolve this error.  "
        f"(Suggested revision: {suggested_revision or 'not available'})"
    )


def apply_to_config(config: pipelines.PipelineConfig) -> None:
    config.model_config.huggingface_model_revision = revision_for_hf_repo(
        config.model_config.model_path
    )
    config.model_config.huggingface_weight_revision = revision_for_hf_repo(
        config.model_config.huggingface_weight_repo_id
    )
