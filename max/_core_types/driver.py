# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DLPackArray(Protocol):
    def __dlpack__(self) -> Any: ...

    def __dlpack_device__(self) -> Any: ...
