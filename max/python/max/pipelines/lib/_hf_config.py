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

"""HuggingFace configuration loading with a process-wide cache."""

import importlib
import json
import logging
import os
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from max.pipelines.weights.hf_utils import (
    HuggingFaceRepo,
    fetch_repo_file,
    is_commit_sha,
)
from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger("max.pipelines")

PARAMS_JSON = "params.json"
"""The checkpoint metadata a repo that ships no ``config.json`` may carry.

A Mistral-release checkpoint states its whole geometry in ``params.json`` and
lets ``transformers``' converter write the ``config.json``; such a repo loads
by deriving that config here, through the generator the architecture family
registers (:func:`register_config_generator`). The derived config is built in
memory and nothing is staged on disk: the checkpoint it belongs to may be a
read-only Hugging Face cache snapshot, which no load may write into, and a copy
written from an older generator would be a second, staler statement of metadata
the checkpoint already carries.
"""

ConfigGenerator = Callable[[Path], Mapping[str, Any] | None]
"""Derives a checkpoint's ``config.json`` from its ``params.json``.

Given the path to a ``params.json``, one of these returns the ``config.json``
that belongs beside it -- or ``None`` when that checkpoint is not its family's.
It is the same generator the family ships to prepare a checkpoint by hand, so a
config derived on load and a staged one carry the same fields.
"""


@dataclass(frozen=True)
class _ConfigGenerator:
    """A registered generator, and the checkpoints it claims."""

    spec: str
    """``module:attribute``. Imported only for a checkpoint that claims it."""

    params_keys: frozenset[str]
    """``params.json`` keys every checkpoint of the family carries.

    The claim is data rather than a call so that a repo of another family is
    dropped without importing the generator's module -- an architecture
    package, and the heaviest import in this path.
    """


_CONFIG_GENERATORS: list[_ConfigGenerator] = []

_no_generator_warned = False

# Process-wide cache keyed by HuggingFaceRepo. Avoids redundant network or
# disk access when multiple callers (e.g. MAXModelConfig and PipelineRegistry)
# load the same repo's config. Each worker process has its own independent
# copy: under spawn it starts empty; under fork it is copied from the parent
# at fork time and then diverges independently.
_config_cache: dict[HuggingFaceRepo, PretrainedConfig] = {}


def load_raw_config_json(repo: HuggingFaceRepo) -> dict[str, Any]:
    """Load and parse a raw ``config.json`` from a HuggingFace repository.

    Handles both local directories and remote HuggingFace Hub repos,
    respecting the ``subfolder`` field on *repo*. Also tries
    ``scheduler_config.json`` as a fallback (used by diffusers schedulers).

    Args:
        repo: The repository handle to load from.

    Returns:
        The parsed JSON dictionary.

    Raises:
        FileNotFoundError: If no ``config.json`` can be found.
    """
    filenames = ["config.json", "scheduler_config.json"]
    config_path: str | None = None

    if repo.repo_type == "local":
        for filename in filenames:
            parts = [repo.local_path]
            if repo.subfolder is not None:
                parts.append(repo.subfolder)
            parts.append(filename)
            candidate = os.path.join(*parts)
            if os.path.isfile(candidate):
                config_path = candidate
                break
    else:
        from huggingface_hub import hf_hub_download

        for filename in filenames:
            hf_filename = filename
            if repo.subfolder is not None:
                hf_filename = f"{repo.subfolder}/{filename}"
            try:
                config_path = hf_hub_download(
                    repo_id=repo.repo_id,
                    filename=hf_filename,
                    revision=repo.revision,
                )
                break
            except Exception:
                continue

    if config_path is None:
        location = (
            repo.local_path if repo.repo_type == "local" else repo.repo_id
        )
        raise FileNotFoundError(
            f"No config.json or scheduler_config.json found in"
            f" {location}/{repo.subfolder or ''}"
        )

    with open(config_path) as f:
        return json.load(f)


def register_config_generator(
    module: str, attribute: str, *, params_keys: Collection[str]
) -> None:
    """Registers a checkpoint config generator, imported on first use.

    For a checkpoint family whose published repo ships no ``config.json``: a
    repo whose ``params.json`` a registered generator understands loads from
    that file, at the revision the repo was asked for, with no prepared
    directory.

    Args:
        module: The module the generator is defined in.
        attribute: The generator's name in that module. It takes the path to a
            checkpoint's ``params.json`` and returns the config that belongs
            beside it, or ``None`` for a checkpoint of another family.
        params_keys: The ``params.json`` keys every checkpoint of the family
            carries. A repo whose ``params.json`` lacks any of them is not
            looked at further, so no other family's checkpoint imports this
            generator to be told so.
    """
    _CONFIG_GENERATORS.append(
        _ConfigGenerator(f"{module}:{attribute}", frozenset(params_keys))
    )


def _warn_no_generator() -> None:
    """Says once that a config could not be derived, and why not.

    The registry is filled when MAX's built-in architectures are registered,
    which importing :mod:`max.pipelines` does. A process that reached a config
    load through a narrower import gets an empty registry, so say so once: the
    failure it leads to otherwise reads as a checkpoint that has no config at
    all.
    """
    global _no_generator_warned
    if _no_generator_warned:
        return
    _no_generator_warned = True
    logger.warning(
        "No checkpoint config generator is registered; a checkpoint that"
        " ships only %s cannot be loaded. Import max.pipelines first.",
        PARAMS_JSON,
    )


def _read_params_json(path: Path) -> dict[str, Any] | None:
    """Reads a checkpoint's ``params.json``, or ``None`` when it is unreadable.

    A repo of some other shape that happens to carry the file is the ordinary
    case here, so a file this cannot parse reads as absent rather than failing
    the load: the caller then reports the checkpoint as having no config, which
    is what it has.

    Args:
        path: The file to read.

    Returns:
        The parsed object, or ``None`` when the file is not a JSON object.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            params = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.debug("Cannot read %s: %s", path, error)
        return None
    if not isinstance(params, dict):
        logger.debug("%s does not hold a JSON object.", path)
        return None
    return params


def _claiming_generators(
    params: Mapping[str, Any],
) -> Iterator[ConfigGenerator]:
    """Yields each generator that claims a checkpoint with these parameters.

    A generator is imported only once its family's keys have matched, and one
    whose module is not importable is skipped rather than failing the load:
    only a checkpoint that needs one can reach this, and a missing module then
    surfaces as the repo's "no config.json" error, which names the thing the
    user actually lacks.

    Args:
        params: The checkpoint's parsed ``params.json``.

    Yields:
        Each generator whose family's key set the checkpoint carries.
    """
    if not _CONFIG_GENERATORS:
        _warn_no_generator()
    for registered in _CONFIG_GENERATORS:
        if not registered.params_keys <= params.keys():
            continue
        module_name, _, attribute = registered.spec.partition(":")
        try:
            yield getattr(importlib.import_module(module_name), attribute)
        except ImportError:
            # A missing generator is recoverable: the caller reports the
            # absence of a config. Anything else from the import (a typo in
            # the spec, an attribute that moved) is the registry's bug and
            # should not read as "this checkpoint has no config".
            logger.warning(
                "Cannot import config generator %s.",
                registered.spec,
                exc_info=True,
            )


def _derive_config(repo: HuggingFaceRepo) -> PretrainedConfig | None:
    """Derives a repo's config from its ``params.json``, when it ships none.

    Runs only for a repo that has no config file of any kind, which the raw
    fallback in :func:`load_huggingface_config` has already established, so a
    checkpoint with its own config is never derived over and no model of
    another family pays for the lookup. The generator reads the checkpoint's
    ``params.json`` -- fetched when the weights never brought it -- and returns
    the config a prepared directory would hold, which is then built the way
    ``AutoConfig.from_pretrained`` builds a config it has just parsed.

    Args:
        repo: The repository handle, resolved for metadata.

    Returns:
        The derived config, or ``None`` when no generator claims this
        checkpoint (or its ``params.json`` cannot be resolved).
    """
    params_path = fetch_repo_file(repo, PARAMS_JSON)
    if params_path is None:
        return None
    params = _read_params_json(params_path)
    if params is None:
        return None
    for generate in _claiming_generators(params):
        config = generate(params_path)
        if not config:
            continue
        logger.info(
            "Derived %s's config from %s, which ships none.",
            repo.repo_id,
            params_path,
        )
        # `AutoConfig.for_model` is the constructor `from_pretrained` finishes
        # with (`config_class(**config_dict)`, not `config_class.from_dict`, of
        # which it is its last line); the difference is that this one needs no
        # file. `from_pretrained` then stamps the checkpoint's path and commit
        # on the instance, which is provenance a reader can use -- a standalone
        # quant config is skipped for an empty `name_or_path` (`weights/quant.py`)
        # -- so record them here too: the repo id is what the caller named, and
        # a revision that already is a commit needs no lookup to know.
        derived = AutoConfig.for_model(**dict(config))
        derived.name_or_path = repo.repo_id
        if is_commit_sha(repo.revision):
            derived._commit_hash = repo.revision
        return derived
    return None


def load_huggingface_config(repo: HuggingFaceRepo) -> PretrainedConfig:
    """Load and cache the HuggingFace config for *repo*.

    Tries :func:`AutoConfig.from_pretrained` first (for transformers models),
    then falls back to loading the raw ``config.json`` and wrapping it in a
    :class:`PretrainedConfig` for non-transformers models (such as diffusers
    components). A repo that has neither raises that fallback's
    ``FileNotFoundError``, and only then is a config derived from the
    checkpoint's ``params.json`` (:func:`register_config_generator`): having no
    config file at all is what makes a checkpoint a candidate, so derivation
    never overrides a config and never runs for another model. Results are
    cached for the lifetime of the process.

    Args:
        repo: The repository handle to load from.

    Returns:
        The HuggingFace configuration object for the model.

    Raises:
        FileNotFoundError: If the repo has no ``config.json`` and no
            ``params.json`` a registered generator derives one from.
        Exception: Re-raises any ``AutoConfig`` error when the raw config
            declares a ``model_type`` (indicates an unrecognized transformers
            architecture rather than a non-transformers model).
    """
    if repo in _config_cache:
        return _config_cache[repo]

    kwargs: dict[str, Any] = {
        "trust_remote_code": repo.trust_remote_code,
        "revision": repo.revision,
    }
    if repo.subfolder is not None:
        kwargs["subfolder"] = repo.subfolder

    try:
        result = AutoConfig.from_pretrained(repo.repo_id, **kwargs)
    except Exception:
        try:
            config_dict = load_raw_config_json(repo)
        except FileNotFoundError:
            # No config file of any kind: a checkpoint family that states its
            # geometry in params.json rather than in a config of its own.
            derived = _derive_config(repo)
            if derived is None:
                raise
            result = derived
        else:
            # Fallback for non-transformers models (e.g. diffusers
            # components): load the raw config.json and wrap it in a
            # PretrainedConfig so callers get uniform attribute access.
            # Re-raise if the config declares a model_type — that means it's a
            # real transformers architecture that just failed to load (wrong
            # trust_remote_code, network error, etc.) rather than a diffusers
            # component.
            if "model_type" in config_dict:
                raise
            result = PretrainedConfig.from_dict(config_dict)

    _config_cache[repo] = result
    return result
