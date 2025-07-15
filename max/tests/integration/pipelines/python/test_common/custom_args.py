# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence


class CommaSeparatedList(Sequence[str]):
    def __init__(self, value: str) -> None:
        self._values = [val.strip() for val in value.strip().split(",")]

    def __iter__(self):
        return (val for val in self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx):  # noqa: ANN001
        return self._values[idx]

    def __repr__(self) -> str:
        return repr(self._values)

    def __eq__(self, other):  # noqa: ANN001
        return len(self) == len(other) and all(
            lhs == rhs for lhs, rhs in zip(self, other)
        )
