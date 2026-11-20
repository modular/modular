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
"""The ``sampling_defaults`` component.

The sampling a request that sets none gets, as MAX Serve resolves it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from typing_extensions import Self

from .value_types import (
    ArtifactError,
    ArtifactType,
    decode_field,
    decode_optional,
    expect,
    unknown_fields,
)


@dataclass(frozen=True)
class SamplingDefaults:
    """Default sampling values for a model.

    Serialized as part of the pipeline artifact exported by ``max export``.
    Prioritizes:

    1) MAX CLI ``--temperature`` and ``--top-k`` flags
    2) The model's ``generation_config.json``
    3) MAX's own built-in defaults.

    The engine will replace any unset value with the defaults defined here.

    Args:
        temperature: The temperature; ``0`` is greedy.
        top_k: The number of most probable tokens to sample from, or ``None``
            for no top-k limit.
        top_p: The nucleus-sampling probability mass.
        repetition_penalty: The penalty on tokens already in the sequence;
            ``1`` is none.
    """

    temperature: float
    top_k: int | None
    top_p: float
    repetition_penalty: float

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the defaults as a JSON object."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the defaults written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If a field is missing, unknown or malformed.
        """
        if unknown := unknown_fields(value, cls):
            raise ArtifactError(f"unknown fields {unknown}")
        return cls(
            temperature=decode_field(value, "temperature", _float),
            top_k=decode_field(value, "top_k", decode_optional(_int)),
            top_p=decode_field(value, "top_p", _float),
            repetition_penalty=decode_field(
                value, "repetition_penalty", _float
            ),
        )


def _float(value: ArtifactType) -> float:
    # JSON writes a whole float such as 1.0 as 1.
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return expect(value, float)


def _int(value: ArtifactType) -> int:
    return expect(value, int)
