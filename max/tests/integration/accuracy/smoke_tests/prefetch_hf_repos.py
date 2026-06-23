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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8,<9",
#   "huggingface_hub==1.8.0",
#   "pydantic>=2.0,<3",
#   "pyyaml",
#   "requests",
# ]
# ///

"""Make sure every HuggingFace repo a smoke test needs is already cached.

Two modes, switched by the HF_HUB_OFFLINE env var:

* Offline: probe the cache; exit non-zero with a one-line cache-miss message
  if anything is missing.
* Online: download anything missing, canonicalize the base repo's casing if
  needed.

Stdout carries one thing only: the resolved base repo name, emitted last and
only on full success. Progress lines go to stderr.
"""

from __future__ import annotations

import logging
import sys

import click
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_OFFLINE
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from smoke_test import hf_repos_for_model

for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _log(msg: str) -> None:
    click.echo(msg, err=True)


def _cache_path(repo: str, revision: str | None) -> str | None:
    """Return the snapshot path if cached locally, else None.

    Uses local_files_only=True so this never opens a socket.
    """
    try:
        return snapshot_download(repo, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def _ensure(
    repo: str, revision: str | None, *, allow_canonicalize: bool
) -> str:
    """Cache the repo, returning the resolved name.

    In offline mode a probe miss exits non-zero. Online, the base repo gets
    a canonical-name fallback since users type any casing.
    """
    _log(f"Checking the cache for '{repo}' (rev={revision})...")
    path = _cache_path(repo, revision)
    if path is not None:
        _log(f"  Already cached at {path}")
        return repo

    _log("  Not in the local cache.")

    if HF_HUB_OFFLINE:
        _log(
            "  Offline mode is set, so we can't fetch from Hugging Face. "
            "Re-run with network access to download."
        )
        sys.exit(1)

    resolved = repo
    if allow_canonicalize:
        _log(
            "  Looking up the canonical name on Hugging Face (the cache "
            "is case-sensitive, so the casing must match exactly)..."
        )
        try:
            resolved = HfApi().model_info(repo).id
        except HfHubHTTPError as e:
            _log(
                f"  Failed to resolve canonical Hugging Face repo ID for "
                f"'{repo}': {e}"
            )
            sys.exit(1)
        if resolved != repo:
            _log(f"  Resolved '{repo}' to '{resolved}'.")
            path = _cache_path(resolved, revision)
            if path is not None:
                _log(f"  Already cached under the canonical name at {path}")
                return resolved
        else:
            _log("  Canonical name matches the input.")

    _log(f"  Downloading '{resolved}' (rev={revision}) from Hugging Face...")
    path = snapshot_download(resolved, revision=revision)
    _log(f"  Cached '{resolved}' to {path}")
    return resolved


@click.command()
@click.argument("model", type=str)
def main(model: str) -> None:
    repos = hf_repos_for_model(model)
    if not repos:
        _log(f"Nothing to pre-fetch for '{model}'.")
        return
    (base_repo, base_revision), *extras = repos

    _log(f"Base model: '{base_repo}'")
    resolved_base = _ensure(base_repo, base_revision, allow_canonicalize=True)

    for repo, revision in extras:
        _log(f"Draft model: '{repo}'")
        _ensure(repo, revision, allow_canonicalize=False)

    # Stdout is the resolved base name; emit last so a partial run leaves it
    # empty for the caller.
    click.echo(resolved_base)


if __name__ == "__main__":
    main()
