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

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import overload


class CommaSeparatedList(Sequence[str]):
    def __init__(self, value: str) -> None:
        self._values = [val.strip() for val in value.strip().split(",")]

    def __iter__(self) -> Iterator[str]:
        return (val for val in self._values)

    def __len__(self) -> int:
        return len(self._values)

    @overload
    def __getitem__(self, idx: int) -> str: ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence[str]: ...

    def __getitem__(self, idx: int | slice) -> str | Sequence[str]:
        return self._values[idx]

    def __repr__(self) -> str:
        return repr(self._values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommaSeparatedList):
            return NotImplemented
        return len(self) == len(other) and all(
            lhs == rhs for lhs, rhs in zip(self, other, strict=True)
        )
