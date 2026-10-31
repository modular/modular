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
"""The ``format_spec`` component: a checkpoint's FormatSpec document."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self

from .value_types import ArtifactError, ArtifactFile, ArtifactType

FORMAT_SPEC_FILE_NAME = "format_spec.json"


@dataclass(frozen=True)
class FormatSpec:
    """The FormatSpec document of the exported checkpoint.

    The document is stored unchanged as ``format_spec.json``: the serving
    engine validates it and builds each request's grammar and output parser
    from it.

    Args:
        document: The path of the document.
    """

    document: Path

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the document under ``document``."""
        return {"document": ArtifactFile(FORMAT_SPEC_FILE_NAME, self.document)}

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the component written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If ``document`` is not a file in the artifact.
        """
        document = value.get("document")
        if not isinstance(document, ArtifactFile) or not isinstance(
            document.content, Path
        ):
            raise ArtifactError("format_spec.document must be a file")
        return cls(document.content)
