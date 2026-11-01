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
"""Provenance and integrity metadata stored in ``metadata.json``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .codec import ref_from_dict, ref_to_dict
from .value_types import ArtifactError, HuggingFaceRef, JsonValue

ARTIFACT_VERSION = 0
"""The artifact format version written at export. Not checked at load time yet.
"""


@dataclass(frozen=True)
class ArtifactMetadata:
    """Where an artifact came from and the checksums of its files.

    The MAX version is recorded for provenance and error messages only.
    Artifacts are only guaranteed to load on the build that exported them.
    """

    artifact_version: int
    """The artifact format version. Recorded but not yet checked."""

    model_sources: Sequence[HuggingFaceRef]
    """Every Hugging Face reference used by the artifact's components."""

    max_version: str
    """The MAX version that exported the artifact."""

    export_time: str
    """The ISO-8601 UTC time of the export."""

    host_system: str
    """The OS of the exporting host, from :func:`platform.system`."""

    host_machine: str
    """The CPU architecture of the exporting host, from :func:`platform.machine`.

    This can differ from the target the compiled model runs on.
    """

    files: Mapping[str, str]
    """Maps file names to their hex sha256 checksums."""

    def to_dict(self) -> dict[str, JsonValue]:
        """Returns the JSON form of the metadata."""
        return {
            "artifact_version": self.artifact_version,
            "model_sources": [ref_to_dict(ref) for ref in self.model_sources],
            "max_version": self.max_version,
            "export_time": self.export_time,
            "host": {"system": self.host_system, "machine": self.host_machine},
            "files": dict(self.files),
        }

    @classmethod
    def from_dict(cls, value: JsonValue) -> ArtifactMetadata:
        """Parses the JSON form written by :meth:`to_dict`.

        Args:
            value: The parsed contents of ``metadata.json``.

        Returns:
            The metadata.

        Raises:
            ArtifactError: If a field is missing or has the wrong type.
        """
        data = _expect_dict(value, "metadata")
        artifact_version = data.get("artifact_version")
        if not isinstance(artifact_version, int):
            raise ArtifactError("metadata: artifact_version must be an integer")
        host = _expect_dict(data.get("host"), "host")
        model_sources = data.get("model_sources")
        if not isinstance(model_sources, list):
            raise ArtifactError("metadata: model_sources must be a list")
        files = _expect_dict(data.get("files"), "files")
        if not all(isinstance(v, str) for v in files.values()):
            raise ArtifactError("metadata: file checksums must be strings")
        return cls(
            artifact_version=artifact_version,
            model_sources=tuple(ref_from_dict(ref) for ref in model_sources),
            max_version=_expect_str(data, "max_version"),
            export_time=_expect_str(data, "export_time"),
            host_system=_expect_str(host, "system"),
            host_machine=_expect_str(host, "machine"),
            files={k: str(v) for k, v in files.items()},
        )


def _expect_dict(value: JsonValue, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ArtifactError(f"metadata: {name} must be an object")
    return value


def _expect_str(data: Mapping[str, JsonValue], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ArtifactError(f"metadata: {key} must be a string")
    return value
