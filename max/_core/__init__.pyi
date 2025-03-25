# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from max import mlir

from . import (
    dialects as dialects,
)
from . import (
    driver as driver,
)
from . import (
    dtype as dtype,
)
from . import (
    engine as engine,
)
from . import (
    graph as graph,
)
from . import (
    profiler as profiler,
)

class Attribute:
    def __eq__(self, arg: object, /) -> bool: ...

class Block:
    @property
    def end(self) -> InsertPoint: ...

class InsertPoint:
    pass

class OpBuilder:
    def __init__(self, arg: InsertPoint, /) -> None: ...
    def create(
        self, arg0: type[OpState], arg1: mlir.Location, /, *args, **kwargs
    ) -> object: ...

class OpState:
    pass

class Type:
    def __eq__(self, arg: object, /) -> bool: ...
