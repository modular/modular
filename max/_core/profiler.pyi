# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from types import TracebackType

from typing_extensions import Self

class Trace:
    def __init__(self, message: str, color: str = "") -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...
    def mark(self) -> None: ...

def is_profiling_enabled() -> bool: ...
