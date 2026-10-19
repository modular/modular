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
"""Converts artifact values to and from JSON.

File and Hugging Face references are encoded as single-key objects tagged
``$file`` and ``$hf``. The Rust reader decodes the same tags.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath

from .value_types import (
    ArtifactError,
    ArtifactFile,
    ArtifactType,
    HuggingFaceRef,
    JsonValue,
)

FILE_TAG = "$file"  # Key used to reference ArtifactFile object
HF_TAG = "$hf"  # Key used to reference HuggingFaceRef object
_RESERVED_TAGS = frozenset({FILE_TAG, HF_TAG})


def encode(
    value: ArtifactType, where: str = ""
) -> tuple[JsonValue, dict[str, ArtifactFile]]:
    """Encodes ``value`` as JSON.

    Args:
        value: The value to encode.
        where: The key path of ``value``, used in error messages.

    Returns:
        The JSON-compatible value, and the files it references keyed by name.

    Raises:
        ArtifactError: If the value has an unsupported type, a mapping uses a
            reserved tag as a key, or two files share a name but not content.
    """
    label = where or "value"
    if value is None or isinstance(value, (bool, int, str)):
        return value, {}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError(f"{label}: {value} is not finite")
        return value, {}
    if isinstance(value, ArtifactFile):
        return {FILE_TAG: value.name}, {value.name: value}
    if isinstance(value, HuggingFaceRef):
        return {HF_TAG: ref_to_dict(value)}, {}

    files: dict[str, ArtifactFile] = {}
    if isinstance(value, Mapping):
        encoded: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactError(
                    f"{label}: mapping key {key!r} is not a string"
                )
            if key in _RESERVED_TAGS:
                raise ArtifactError(f"{label}: mapping key {key!r} is reserved")
            encoded[key], item_files = encode(
                item, f"{where}.{key}" if where else key
            )
            files = merge_files(files, item_files)
        return encoded, files
    # Only lists and tuples: str and bytes are also Sequences.
    if isinstance(value, (list, tuple)):
        items: list[JsonValue] = []
        for i, item in enumerate(value):
            encoded_item, item_files = encode(item, f"{where}[{i}]")
            items.append(encoded_item)
            files = merge_files(files, item_files)
        return items, files
    raise ArtifactError(f"{label}: unsupported type {type(value).__name__}")


def prefix_files(value: ArtifactType, prefix: str) -> ArtifactType:
    """Returns ``value`` with every file it references moved under
    ``prefix``.

    Use this to store several components' values side by side, each under
    its own directory, when their file names may coincide.

    Args:
        value: A serialized value.
        prefix: The directory, a relative POSIX path, to move the files into.
    """
    if isinstance(value, ArtifactFile):
        return ArtifactFile(f"{prefix}/{value.name}", value.content)
    if isinstance(value, Mapping):
        return {key: prefix_files(item, prefix) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [prefix_files(item, prefix) for item in value]
    return value


def merge_files(
    a: Mapping[str, ArtifactFile], b: Mapping[str, ArtifactFile]
) -> dict[str, ArtifactFile]:
    """Returns the union of two sets of files keyed by name.

    Two files with the same name are the same file if they write the same
    bytes, whatever form their content takes.

    Raises:
        ArtifactError: If a name maps to different bytes in ``a`` and ``b``,
            or two names would collide on disk.
    """
    merged = dict(a)
    for name, file in b.items():
        existing = merged.get(name)
        if existing is None:
            _check_name_collisions(name, merged)
            merged[name] = file
        elif existing != file and existing.sha256() != file.sha256():
            raise ArtifactError(
                f"file {name!r} is written twice with different content"
            )
    return merged


def _check_name_collisions(name: str, names: Iterable[str]) -> None:
    # Case-folded, since macOS filesystems are case-insensitive by default.
    path = PurePosixPath(name.casefold())
    for other in names:
        other_path = PurePosixPath(other.casefold())
        if path == other_path:
            raise ArtifactError(
                f"files {other!r} and {name!r} differ only in case"
            )
        if path in other_path.parents or other_path in path.parents:
            raise ArtifactError(
                f"files {other!r} and {name!r} collide: one is a directory"
                " of the other"
            )


def decode(
    value: JsonValue, root: Path, known_files: Collection[str]
) -> ArtifactType:
    """Decodes a JSON value written by :func:`encode`.

    Args:
        value: The JSON value.
        root: The artifact directory that file references resolve against.
        known_files: Names of the files listed in the artifact metadata.

    Returns:
        The decoded value, with file references pointing into ``root``.

    Raises:
        ArtifactError: If a tag is malformed or references an unlisted file.
    """
    if isinstance(value, list):
        return [decode(item, root, known_files) for item in value]
    if not isinstance(value, dict):
        return value
    if not _RESERVED_TAGS & value.keys():
        return {k: decode(v, root, known_files) for k, v in value.items()}
    if len(value) != 1:
        raise ArtifactError(f"tagged object has extra keys: {sorted(value)}")
    if FILE_TAG in value:
        name = value[FILE_TAG]
        if not isinstance(name, str):
            raise ArtifactError(f"{FILE_TAG} must be a string, got {name!r}")
        if name not in known_files:
            raise ArtifactError(
                f"file {name!r} is referenced but not listed in the metadata"
            )
        return ArtifactFile(name, root / name)
    return ref_from_dict(value[HF_TAG])


def iter_hf_refs(value: ArtifactType) -> Iterator[HuggingFaceRef]:
    """Yields every :class:`HuggingFaceRef` leaf in ``value``."""
    if isinstance(value, HuggingFaceRef):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from iter_hf_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_hf_refs(item)


def collect_hf_refs(
    values: Iterable[ArtifactType],
) -> tuple[HuggingFaceRef, ...]:
    """Returns the unique :class:`HuggingFaceRef` leaves of ``values``, sorted.

    Sorting keeps the result stable across a round trip, since
    ``artifact.json`` is written with sorted keys.
    """
    refs = {ref for value in values for ref in iter_hf_refs(value)}
    return tuple(
        sorted(refs, key=lambda r: (r.repo_id, r.revision, r.path or ""))
    )


def ref_to_dict(ref: HuggingFaceRef) -> dict[str, JsonValue]:
    """Returns the JSON form of a Hugging Face reference."""
    return {"repo_id": ref.repo_id, "revision": ref.revision, "path": ref.path}


def ref_from_dict(value: JsonValue) -> HuggingFaceRef:
    """Parses the JSON form written by :func:`ref_to_dict`.

    Raises:
        ArtifactError: If a field is missing or has the wrong type.
    """
    if not isinstance(value, dict):
        raise ArtifactError(
            f"Hugging Face reference must be an object: {value!r}"
        )
    repo_id, revision, path = (
        value.get("repo_id"),
        value.get("revision"),
        value.get("path"),
    )
    if (
        not isinstance(repo_id, str)
        or not isinstance(revision, str)
        or not (path is None or isinstance(path, str))
    ):
        raise ArtifactError(f"malformed Hugging Face reference: {value!r}")
    return HuggingFaceRef(repo_id, revision, path)
