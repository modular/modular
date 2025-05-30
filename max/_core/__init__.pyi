# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Iterator, Sequence
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
    nixl as nixl,
)
from . import (
    profiler as profiler,
)

class Attribute:
    @staticmethod
    def _from_cmlir(arg: MlirAttribute, /) -> Attribute: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def asm(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def _CAPIPtr(self) -> object: ...

class NamedAttribute(tuple[str, Attribute]):
    def __init__(self, arg0: str, arg1: Attribute, /) -> None: ...
    @property
    def name(self): ...
    @property
    def value(self) -> Attribute: ...
    def __iter__(self) -> Iterator[str | Attribute]: ...

class TypeID:
    pass

class Type:
    @staticmethod
    def _from_cmlir(arg: MlirType, /) -> Type: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def asm(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def _CAPIPtr(self) -> object: ...

T = TypeVar("T", bound=Type)

class Value(Generic[T]):
    @staticmethod
    def _from_cmlir(value: MlirValue, /) -> Value[Type]: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def type(self) -> T: ...
    @property
    def owner(self) -> Block | Operation: ...
    @property
    def _CAPIPtr(self) -> object: ...

class OpOperand:
    @property
    def value(self) -> Value[Type]: ...
    @value.setter
    def value(self, value: Value[Type], /): ...

class InsertPoint:
    pass

class Block:
    @staticmethod
    def _from_cmlir(arg: MlirBlock, /) -> Block: ...
    @property
    def end(self) -> InsertPoint: ...

class Operation:
    @staticmethod
    def _from_cmlir(arg: object, /) -> Operation: ...
    def __eq__(self, arg: object, /) -> bool: ...
    @property
    def results(self) -> Sequence[Value[Type]]: ...
    def verify(self, verify_recursively: bool = True) -> bool: ...
    def move_after(self, arg: Operation, /) -> None: ...
    @property
    def asm(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def _CAPIPtr(self) -> object: ...

class OpBuilder:
    def __init__(self, arg: InsertPoint, /) -> None: ...

    Op = TypeVar("Op", bound=Operation)
    def create(
        self, type: type[Op], location: Location
    ) -> Callable[..., Op]: ...

class _LockedSymbolTableCollection:
    pass

class _BitVector:
    pass

class _TargetTriple:
    pass

class _RelocationModel(enum.Enum):
    pass

class _MemoryEffect:
    pass

__version__: str
