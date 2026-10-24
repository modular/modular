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
"""The ``model`` component: a pipeline's compiled graphs, one MEF per role."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from max.experimental.compilation import CompiledCallable
from typing_extensions import Self

from .codec import prefix_files
from .value_types import ArtifactError, ArtifactType


@dataclass(frozen=True)
class PipelineModules:
    """A pipeline's compiled graphs, keyed by role.

    Each role's graph is a
    :class:`~max.experimental.compilation.CompiledCallable` whose MEF holds
    only that graph. Each role's files are stored under a directory named
    after the role, such as ``decoder/model.mef``. ``max export`` compiles
    each role on its own so that every MEF holds one graph.

    Args:
        roles: Each role's compiled graph, such as a pipeline's
            ``compiled_modules``.

    Raises:
        ArtifactError: If there are no roles.
    """

    roles: Mapping[str, CompiledCallable[..., Any]]

    def __post_init__(self) -> None:
        if not self.roles:
            raise ArtifactError("no graphs are registered by role")

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns each role's serialized callable.

        Raises:
            ArtifactError: If a role's MEF holds other graphs too.
        """
        return {
            role: prefix_files(compiled.__serialize__(), role)
            for role, compiled in sorted(self.roles.items())
        }

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the graphs written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If a field is malformed.
        """
        roles: dict[str, CompiledCallable[..., Any]] = {}
        for role, raw in value.items():
            if not isinstance(raw, Mapping):
                raise ArtifactError(f"{role} must be an object")
            roles[role] = CompiledCallable.__deserialize__(raw)
        return cls(roles)
