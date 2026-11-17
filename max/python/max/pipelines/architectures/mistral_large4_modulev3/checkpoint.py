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
"""Locating a Mistral Large 4 checkpoint's ``params.json``.

Every per-layer decision this architecture makes arrives in the generated
``config.json``. Two do not: the tap list the drafter reads, and the drafter's
own ``mtp`` block, which only the checkpoint's ``params.json`` states. That file
is also what a release that ships no ``config.json`` is normalized *from*, so it
has to be on disk beside the weights whether the checkpoint was staged by hand
or resolved from a Hub repo id -- and a weights download never brings it. A
serve from a repo id alone therefore captures no taps unless this resolves it,
which is what :func:`params_json_candidates` does.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from max.pipelines.lib import generate_local_model_path
from max.pipelines.lib._hf_config import PARAMS_JSON
from max.pipelines.weights.hf_utils import RepoFileSource, fetch_repo_file

logger = logging.getLogger("max.pipelines")


class LocalCheckpoint(Protocol):
    """The served model config the checkpoint readers need.

    ``MAXModelConfig`` satisfies this as written; the field types are the real
    ones, because ``weight_path`` is a list of *repo-relative* weight filenames
    for a Hub serve and a narrower stand-in hid a crash the standard serve path
    hit.
    """

    @property
    def model_path(self) -> str:
        """The Hugging Face repo id or local model directory."""

    @property
    def weight_path(self) -> list[Path]:
        """The weight files to serve, relative to the repo root."""

    @property
    def huggingface_weight_repo(self) -> RepoFileSource:
        """The handle that says where those files live on disk."""


def _repo_dir(repo: RepoFileSource) -> Path | None:
    """The directory holding a repo's files, or ``None`` when it has none.

    An on-disk repo (a staged checkpoint, or a Hub repo resolved under
    ``HF_HUB_OFFLINE``) names it. Otherwise the Hub cache snapshot is asked
    for, which is a cache lookup only -- no download -- and contributes nothing
    when the repo has not been snapshotted.

    Args:
        repo: The weight repo handle.

    Returns:
        The directory, with the repo's subfolder applied when it has one, or
        ``None`` when the handle names no directory at all.
    """
    root: str
    if repo.repo_type == "local":
        root = repo.local_path
    else:
        try:
            root = generate_local_model_path(repo.repo_id, repo.revision)
        except FileNotFoundError:
            return None
    if not root:
        return None
    directory = Path(root)
    return directory / repo.subfolder if repo.subfolder else directory


def _weight_file_dir(weight: Path) -> Path | None:
    """The directory an explicitly given weight file sits in, if any.

    A relative weight path is a repo-relative *filename* (``model.safetensors``
    in a Hub serve), not a path into the working directory: its parent would be
    ``.``, and a stray ``params.json`` in the CWD would then describe a
    checkpoint this process is not serving. Only an absolute path names a
    directory that holds the shards.

    Args:
        weight: One entry of the config's ``weight_path``.

    Returns:
        Its parent when it is absolute, else ``None``.
    """
    path = Path(weight)
    return path.parent if path.is_absolute() else None


def checkpoint_dirs(model_config: LocalCheckpoint) -> list[Path]:
    """The directories a checkpoint's ``params.json`` may sit in.

    Priority follows "the shards are what is served": the weight repo's own
    directory first, then where any explicitly given absolute weight files sit,
    and the model directory last, for a checkpoint staged somewhere its weights
    are not.

    A repo id that is neither on disk nor in the Hub cache contributes no
    directory, so the process working directory is never a candidate.

    Args:
        model_config: The pipeline's model config.

    Returns:
        The candidate directories, most authoritative first.
    """
    dirs: list[Path] = []
    candidates = [
        _repo_dir(model_config.huggingface_weight_repo),
        *(_weight_file_dir(weight) for weight in model_config.weight_path),
    ]
    if model_config.model_path:
        model_dir = Path(model_config.model_path)
        if model_dir.is_dir():
            candidates.append(model_dir)
    for candidate in candidates:
        if candidate is not None and candidate not in dirs:
            dirs.append(candidate)
    return dirs


def params_json_candidates(model_config: LocalCheckpoint) -> Iterator[Path]:
    """Every place this checkpoint's ``params.json`` could be, best first.

    The checkpoint directories are tried first, in :func:`checkpoint_dirs`
    order. Nothing local means the file is fetched from the weight repo into
    the Hub cache snapshot the weights are then downloaded into -- the one
    case a repo-id serve would otherwise read no ``params.json`` at all --
    which the last candidate reports. The fetch is lazy: a caller that finds
    the file locally never reaches it.

    Args:
        model_config: The pipeline's model config.

    Yields:
        Each candidate path, most authoritative first.
    """
    yield from (
        directory / PARAMS_JSON for directory in checkpoint_dirs(model_config)
    )
    fetched = fetch_repo_file(model_config.huggingface_weight_repo, PARAMS_JSON)
    if fetched is None:
        logger.info(
            "no %s in %s or its weights, and none to fetch from %s.",
            PARAMS_JSON,
            model_config.model_path,
            model_config.huggingface_weight_repo,
        )
        return
    yield fetched
