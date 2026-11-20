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
"""The ``batch_processor`` component: the bounds of each per-step input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from typing_extensions import Self

from .value_types import ArtifactError, ArtifactType


@dataclass(frozen=True)
class BatchProcessorDims:
    """The largest size of each symbolic dim of a per-step graph input.

    The serving engine allocates each per-step input once at these sizes and
    slices it per batch. The dims of inputs an owner allocates for the model's
    lifetime, such as KV pages, are not listed: their owners size them.

    Args:
        dims: Each dim's maximum, keyed by name.
    """

    dims: Mapping[str, int]

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the dims under ``dims``."""
        return {"dims": dict(self.dims)}

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the dims written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If ``dims`` is not an object of positive integers.
        """
        dims = value.get("dims")
        if not isinstance(dims, Mapping):
            raise ArtifactError("batch_processor.dims must be an object")
        parsed: dict[str, int] = {}
        for name, size in dims.items():
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ArtifactError(
                    f"batch_processor.dims.{name} must be a positive integer"
                )
            parsed[name] = size
        return cls(parsed)
