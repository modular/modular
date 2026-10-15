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
"""Value types and the component protocol for pipeline artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeAlias, runtime_checkable

from typing_extensions import Self

METADATA_FILE_NAME = "metadata.json"
ARTIFACT_FILE_NAME = "artifact.json"
ARTIFACT_SUFFIX = ".m"

_RESERVED_FILE_NAMES = frozenset({METADATA_FILE_NAME, ARTIFACT_FILE_NAME})
_CHUNK_SIZE = 1 << 20


class ArtifactError(ValueError):
    """Raised when an artifact cannot be written, read, or validated."""


def sha256_file(path: Path) -> str:
    """Returns the hex sha256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactFile:
    """A file stored inside the ``.m`` artifact directory.

    When serializing, ``str`` content is written as UTF-8 text, ``bytes`` are
    written as-is, and a :class:`~pathlib.Path` is copied. After
    deserialization, ``content`` is always the path of the file inside the
    artifact directory.

    Args:
        name: Relative POSIX path of the file inside the artifact directory.
        content: The file contents, or the path of a file to copy.
    """

    name: str
    content: str | bytes | Path

    def __post_init__(self) -> None:
        path = PurePosixPath(self.name)
        if (
            not self.name
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != self.name
        ):
            raise ArtifactError(
                f"artifact file name {self.name!r} must be a normalized"
                " relative path inside the artifact directory"
            )
        # Case-folded, since macOS filesystems are case-insensitive by default.
        if path.parts[0].casefold() in _RESERVED_FILE_NAMES:
            raise ArtifactError(f"artifact file name {self.name!r} is reserved")

    def sha256(self) -> str:
        """Returns the hex sha256 digest of the bytes this file writes."""
        if isinstance(self.content, Path):
            try:
                return sha256_file(self.content)
            except FileNotFoundError as e:
                raise ArtifactError(
                    f"artifact file {self.name!r}: source {self.content} does"
                    " not exist"
                ) from e
        data = (
            self.content.encode("utf-8")
            if isinstance(self.content, str)
            else self.content
        )
        return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class HuggingFaceRef:
    """A reference to a Hugging Face repo, or a file in one, at a revision.

    Args:
        repo_id: The Hugging Face repo ID.
        revision: The commit the artifact was exported against.
        path: The path of a file within the repo, or ``None`` for the
            whole repo.
    """

    repo_id: str
    revision: str
    path: str | None = None


ArtifactType: TypeAlias = (
    bool
    | int
    | float
    | str
    | ArtifactFile
    | HuggingFaceRef
    | Sequence["ArtifactType"]
    | Mapping[str, "ArtifactType"]
    | None
)
"""A value a component may return from ``__serialize__``."""


@runtime_checkable
class ArtifactComponent(Protocol):
    """A pipeline component that can be stored in a model artifact.

    A component serializes to a JSON-compatible mapping that may reference
    files, but never another component, so each component loads
    independently.
    """

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the serialized form of this component."""
        ...

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the component from the output of ``__serialize__``.

        Args:
            value: The serialized component. Any :class:`ArtifactFile` in it
                points at the file inside the artifact directory.

        Returns:
            The reconstructed component.
        """
        ...
