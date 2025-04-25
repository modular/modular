# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Sequence
from typing import Callable, Generic, TypeVar

from max.mlir import Attribute as MlirAttribute
from max.mlir import Block as MlirBlock
from max.mlir import Location
from max.mlir import Type as MlirType
from max.mlir import Value as MlirValue

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

OpType = TypeVar("OpType", bound=Operation)

class Attribute:
    def __init__(self, arg: MlirAttribute, /) -> None: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def asm(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def _CAPIPtr(self) -> object: ...

class Block:
    def __init__(self, arg: MlirBlock, /) -> None: ...
    @property
    def end(self) -> InsertPoint: ...

class InsertPoint:
    pass

class NamedAttribute:
    def __init__(self, arg0: str, arg1: Attribute, /) -> None: ...
    @property
    def name(self): ...
    @property
    def value(self) -> Attribute: ...
    def __iter__(self) -> tuple: ...

class OpBuilder:
    def __init__(self, arg: InsertPoint, /) -> None: ...
    def create(
        self, type: OpType, location: Location
    ) -> Callable[..., OpType]: ...

class Operation:
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def results(self) -> Sequence[Value]: ...
    @property
    def asm(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def _CAPIPtr(self) -> object: ...

T = TypeVar("T", bound=Type)

class Type:
    def __init__(self, arg: MlirType, /) -> None: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def asm(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def _CAPIPtr(self) -> object: ...

class TypeID:
    pass

class Value(Generic[T]):
    def __init__(self, arg: MlirValue, /) -> None: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def type(self) -> Type: ...
    @property
    def _CAPIPtr(self) -> object: ...

class _BitVector:
    pass

class _LockedSymbolTableCollection:
    pass

class _Operation:
    pass

class _RelocationModel(enum.Enum):
    pass

class _TargetTriple:
    pass

__version__: str
