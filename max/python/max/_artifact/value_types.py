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
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

from max.tree import TreeDef, kind_of, node_of_kind
from typing_extensions import Self

_T = TypeVar("_T")

METADATA_FILE_NAME = "metadata.json"
ARTIFACT_FILE_NAME = "artifact.json"
ARTIFACT_SUFFIX = ".m"

_RESERVED_FILE_NAMES = frozenset({METADATA_FILE_NAME, ARTIFACT_FILE_NAME})
_CHUNK_SIZE = 1 << 20

FileWriter: TypeAlias = Callable[[Path], None]
"""Writes a file's contents to the given path, such as
:meth:`~max.engine.CompiledModel.export_mef`."""


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
    written as-is, a :class:`~pathlib.Path` is copied, and a
    :data:`FileWriter` is called with the destination path. After
    deserialization, ``content`` is always the path of the file inside the
    artifact directory.

    Args:
        name: Relative POSIX path of the file inside the artifact directory.
        content: The file contents, the path of a file to copy, or a function
            that writes the file.
    """

    name: str
    content: str | bytes | Path | FileWriter

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

    def write_to(self, dest: Path) -> None:
        """Writes the file's contents to ``dest``.

        Raises:
            ArtifactError: If ``content`` is a path that does not exist.
        """
        if isinstance(self.content, Path):
            try:
                shutil.copyfile(self.content, dest)
            except FileNotFoundError as e:
                raise self._missing_source(self.content) from e
        elif isinstance(self.content, bytes):
            dest.write_bytes(self.content)
        elif isinstance(self.content, str):
            dest.write_text(self.content, encoding="utf-8")
        else:
            self.content(dest)

    def sha256(self) -> str:
        """Returns the hex sha256 digest of the bytes this file writes.

        A :data:`FileWriter` is run into a temporary file to hash it.
        """
        if isinstance(self.content, Path):
            try:
                return sha256_file(self.content)
            except FileNotFoundError as e:
                raise self._missing_source(self.content) from e
        if isinstance(self.content, bytes):
            return hashlib.sha256(self.content).hexdigest()
        if isinstance(self.content, str):
            return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "file"
            self.content(dest)
            return sha256_file(dest)

    def _missing_source(self, source: Path) -> ArtifactError:
        return ArtifactError(
            f"artifact file {self.name!r}: source {source} does not exist"
        )


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

JsonValue: TypeAlias = (
    bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
)
"""A value as :func:`json.load` returns it."""


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


def treedef_to_json(treedef: TreeDef) -> dict[str, ArtifactType]:
    """Returns a tree structure as a JSON-compatible value.

    Only structures that JSON can describe are supported: leaves, references,
    lists, tuples, ``dict`` and ``OrderedDict`` nodes with string keys, nodes
    of a class :func:`max.tree.kind` names, and static values that are
    ``None``, a ``bool``, an ``int``, a ``float`` or a ``str``. A named node
    records its kind under ``type``, so a reader such as the serving engine
    learns what the node holds without the class.
    :func:`treedef_from_json` rebuilds the structure.

    Args:
        treedef: The structure, as :func:`max.tree.flatten` returns it.

    Raises:
        ArtifactError: If the structure holds a named tuple, a node of an
            unnamed class or with metadata, another ``dict`` subclass or a
            non-string key, or a static value of another type.
    """
    if treedef.kind == "leaf":
        return {"kind": "leaf"}
    if treedef.kind == "ref":
        return {"kind": "ref", "meta": treedef.meta}
    if treedef.kind == "static":
        if treedef.meta is not None and not isinstance(
            treedef.meta, (bool, int, float, str)
        ):
            raise ArtifactError(
                f"static value {treedef.meta!r} of type"
                f" {type(treedef.meta).__name__} is not JSON-encodable"
            )
        return {"kind": "static", "meta": treedef.meta}
    children: list[ArtifactType] = [
        treedef_to_json(child) for child in treedef.children
    ]
    if treedef.kind in ("list", "tuple"):
        return {"kind": treedef.kind, "children": children}
    if treedef.kind == "dict" and treedef.meta is None:
        return {
            "kind": "dict",
            "keys": _str_keys(treedef),
            "children": children,
        }
    if treedef.kind == "dict" and treedef.meta == (OrderedDict, ()):
        return {
            "kind": "ordered_dict",
            "keys": _str_keys(treedef),
            "children": children,
        }
    if treedef.kind == "node":
        cls, meta = treedef.meta
        name = kind_of(cls)
        if name is None or meta is not None:
            raise ArtifactError(
                f"a {cls.__qualname__} node is not JSON-encodable: its class"
                " needs a tree kind and no metadata"
            )
        return {
            "kind": "node",
            "type": name,
            "keys": _str_keys(treedef),
            "children": children,
        }
    raise ArtifactError(f"a {treedef.kind!r} node is not JSON-encodable")


def _str_keys(treedef: TreeDef) -> list[ArtifactType]:
    keys: list[ArtifactType] = []
    for key in treedef.keys:
        if not isinstance(key, str):
            raise ArtifactError(
                f"keys {list(treedef.keys)} are not all strings"
            )
        keys.append(key)
    return keys


def treedef_from_json(value: ArtifactType) -> TreeDef:
    """Rebuilds a tree structure from the output of :func:`treedef_to_json`.

    Args:
        value: The decoded JSON value.

    Raises:
        KeyError: If a node lacks a field.
        TypeError: If a field has the wrong type.
        ValueError: If a node has an unknown kind.
    """
    fields = expect(value, dict)
    kind = expect(fields["kind"], str)
    if kind == "leaf":
        return TreeDef("leaf")
    if kind == "ref":
        return TreeDef("ref", meta=expect(fields["meta"], int))
    if kind == "static":
        meta = fields["meta"]
        if meta is not None and not isinstance(meta, (bool, int, float, str)):
            raise TypeError(f"static value {meta!r} is not a JSON scalar")
        return TreeDef("static", meta=meta)
    if kind not in ("list", "tuple", "dict", "ordered_dict", "node"):
        raise ValueError(f"unknown tree structure kind {kind!r}")
    children = tuple(
        treedef_from_json(child) for child in expect(fields["children"], list)
    )
    if kind in ("list", "tuple"):
        return TreeDef(kind, children)
    keys = tuple(expect(key, str) for key in expect(fields["keys"], list))
    if len(keys) != len(children):
        raise ValueError(f"a {kind} node needs one key per child")
    if kind == "dict":
        return TreeDef("dict", children, keys)
    if kind == "ordered_dict":
        return TreeDef("dict", children, keys, (OrderedDict, ()))
    name = expect(fields["type"], str)
    try:
        cls = node_of_kind(name)
    except KeyError:
        raise ValueError(f"unknown tree node type {name!r}") from None
    return TreeDef("node", children, keys, (cls, None))


def expect(value: object, type_: type[_T]) -> _T:
    """Returns a decoded JSON value, checked to be a ``type_``.

    A ``bool`` is not an ``int`` here, and a ``str`` is not a ``list``.

    Raises:
        TypeError: If ``value`` is not a ``type_``.
    """
    if isinstance(value, type_) and not (
        isinstance(value, bool) and type_ is not bool
    ):
        return value
    raise TypeError(
        f"expected {type_.__name__}, got {type(value).__name__}: {value!r}"
    )


def decode_field(
    value: Mapping[str, ArtifactType],
    field: str,
    decoder: Callable[[ArtifactType], _T],
) -> _T:
    """Decodes one field of a serialized value.

    Args:
        value: The serialized value.
        field: The field to decode.
        decoder: Decodes the field's value, raising ``KeyError``,
            ``TypeError`` or ``ValueError`` if it is malformed.

    Raises:
        ArtifactError: If the field is missing or malformed.
    """
    try:
        return decoder(value[field])
    except (KeyError, TypeError, ValueError) as e:
        raise malformed(field, e) from e


def malformed(where: str, error: Exception) -> ArtifactError:
    """Returns the error for a malformed serialized field, from what a
    decoder raised."""
    reason = (
        f"missing field {error.args[0]!r}"
        if isinstance(error, KeyError)
        else str(error)
    )
    return ArtifactError(f"{where} is malformed: {reason}")
