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
"""Writes and reads ``.m`` model artifact directories."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar, overload

import huggingface_hub
from max import _core

from .codec import collect_hf_refs, decode, encode, merge_files
from .metadata import ARTIFACT_VERSION, ArtifactMetadata
from .model_artifact import ModelArtifact
from .value_types import (
    ARTIFACT_FILE_NAME,
    ARTIFACT_SUFFIX,
    METADATA_FILE_NAME,
    ArtifactComponent,
    ArtifactError,
    ArtifactFile,
    ArtifactType,
    HuggingFaceRef,
    JsonValue,
    sha256_file,
)

logger = logging.getLogger("max.pipelines")

A = TypeVar("A", bound=ModelArtifact)


def serialize(
    model_artifact: ModelArtifact, path: str | os.PathLike[str]
) -> Path:
    """Serializes a model artifact to a ``.m`` directory.

    The directory is written to a temporary sibling and renamed into place,
    so a failed export never leaves a partial artifact behind.

    Args:
        model_artifact: The artifact to serialize.
        path: The destination. ``.m`` is appended unless it already ends in
            ``.m``.

    Returns:
        The path of the written artifact directory.

    Raises:
        ArtifactError: If the destination exists or a component's serialized
            form is invalid.
    """
    dest = Path(path)
    if dest.suffix != ARTIFACT_SUFFIX:
        dest = dest.with_name(dest.name + ARTIFACT_SUFFIX)
    if dest.exists():
        raise ArtifactError(f"artifact destination {dest} already exists")

    files: dict[str, ArtifactFile] = {}
    sections: dict[str, JsonValue] = {}
    serialized = {
        slot: component.__serialize__()
        for slot, component in model_artifact.components().items()
    }
    for slot, value in serialized.items():
        sections[slot], section_files = encode(value, slot)
        files = merge_files(files, section_files)

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Not tempfile.mkdtemp: it creates the directory 0700 whatever the umask,
    # and the rename below keeps that mode, so no one else could read the
    # artifact.
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir()
    try:
        for file in files.values():
            _write_file(tmp, file)
        (tmp / ARTIFACT_FILE_NAME).write_text(
            json.dumps(sections, indent=2, sort_keys=True)
        )

        metadata = ArtifactMetadata(
            artifact_version=ARTIFACT_VERSION,
            model_sources=collect_hf_refs(serialized.values()),
            max_version=_core.__version__,
            export_time=datetime.now(timezone.utc).isoformat(),
            host_system=platform.system(),
            host_machine=platform.machine(),
            files={
                name: sha256_file(tmp / name)
                for name in sorted([*files, ARTIFACT_FILE_NAME])
            },
        )
        (tmp / METADATA_FILE_NAME).write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True)
        )
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


@overload
def deserialize(path: str | os.PathLike[str]) -> ModelArtifact: ...


@overload
def deserialize(path: str | os.PathLike[str], artifact_type: type[A]) -> A: ...


def deserialize(
    path: str | os.PathLike[str],
    artifact_type: type[ModelArtifact] = ModelArtifact,
) -> ModelArtifact:
    """Deserializes a model artifact from a ``.m`` directory.

    Before any component is loaded, every file listed in the metadata is
    checksummed, and Hugging Face references are snapshotted.

    Args:
        path: The artifact directory.
        artifact_type: The artifact class. Each component is deserialized with
            the class its slot is annotated with, from
            :meth:`ModelArtifact.component_types`.

    Returns:
        The deserialized artifact. File references point into ``path``.

    Raises:
        ArtifactError: If the artifact is missing or modified, a cached
            Hugging Face snapshot lacks a referenced path, or a component
            can't be deserialized.
    """
    root = Path(path)
    if root.suffix != ARTIFACT_SUFFIX:
        raise ArtifactError(
            f"{root} is not a model artifact: expected a {ARTIFACT_SUFFIX}"
            " directory"
        )
    if not root.is_dir():
        raise ArtifactError(f"model artifact {root} does not exist")

    metadata = ArtifactMetadata.from_dict(_read_json(root / METADATA_FILE_NAME))
    stale_hint = (
        f"The artifact was exported by MAX {metadata.max_version} and this is"
        f" MAX {_core.__version__}; re-export it."
    )
    if ARTIFACT_FILE_NAME not in metadata.files:
        raise ArtifactError(
            f"{root}: metadata does not list {ARTIFACT_FILE_NAME}"
        )
    for name, expected in metadata.files.items():
        file_path = root / name
        if not file_path.is_file():
            raise ArtifactError(f"{root}: missing file {name}. {stale_hint}")
        if sha256_file(file_path) != expected:
            raise ArtifactError(
                f"{root}: checksum mismatch for {name}; the file was modified"
                f" after export. {stale_hint}"
            )

    sections = _read_json(root / ARTIFACT_FILE_NAME)
    if not isinstance(sections, dict):
        raise ArtifactError(f"{root}: {ARTIFACT_FILE_NAME} must be an object")

    component_types = artifact_type.component_types()
    values: dict[str, Mapping[str, ArtifactType]] = {}
    for slot, section in sections.items():
        if slot not in component_types:
            raise ArtifactError(f"{root}: unknown component {slot!r}")
        if component_types[slot] is None:
            raise ArtifactError(
                f"{root}: artifact contains component {slot!r}, but"
                f" {artifact_type.__name__}.{slot} has no concrete component"
                " type"
            )
        value = decode(section, root, metadata.files.keys())
        if not isinstance(value, Mapping):
            raise ArtifactError(f"{root}: component {slot!r} must be an object")
        values[slot] = value

    _check_hf_cached(root, collect_hf_refs(values.values()))

    components: dict[str, ArtifactComponent] = {}
    for slot, value in values.items():
        component_type = component_types[slot]
        assert component_type is not None
        try:
            components[slot] = component_type.__deserialize__(value)
        except Exception as e:
            raise ArtifactError(
                f"{root}: failed to deserialize component {slot!r}. {stale_hint}"
            ) from e

    return artifact_type.from_components(components)


def _check_hf_cached(root: Path, refs: Iterable[HuggingFaceRef]) -> None:
    for ref in refs:
        local_repo = Path(ref.repo_id)
        if local_repo.is_dir():
            snapshot = local_repo
        else:
            try:
                snapshot = Path(
                    huggingface_hub.snapshot_download(
                        repo_id=ref.repo_id,
                        revision=ref.revision,
                        local_files_only=True,
                    )
                )
            except huggingface_hub.errors.LocalEntryNotFoundError:
                # Nothing cached yet is fine: the files are fetched at load.
                logger.warning(
                    "%s: %s@%s is not in the local Hugging Face cache",
                    root,
                    ref.repo_id,
                    ref.revision,
                )
                continue
        if ref.path is not None and not (snapshot / ref.path).exists():
            raise ArtifactError(
                f"{root}: {ref.path} from {ref.repo_id}@{ref.revision} is not"
                " in the local Hugging Face cache; download it first"
            )


def _write_file(root: Path, file: ArtifactFile) -> None:
    dest = root / file.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    file.write_to(dest)


def _read_json(path: Path) -> JsonValue:
    try:
        with path.open() as f:
            value: JsonValue = json.load(f)
    except FileNotFoundError as e:
        raise ArtifactError(f"missing {path}") from e
    except json.JSONDecodeError as e:
        raise ArtifactError(f"{path} is not valid JSON: {e}") from e
    return value
