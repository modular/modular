# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Callable
from typing import TypeVar

import _mlir

from . import ir as ir
from . import passmanager as passmanager
from . import rewrite as rewrite

T = TypeVar("T")

U = TypeVar("U")

globals: _Globals = ...

def register_dialect(dialect_class: type) -> type:
    """Class decorator for registering a custom Dialect wrapper"""

def register_operation(
    dialect_class: type, *, replace: bool = False
) -> Callable[[type[T]], type[T]]:
    """
    Produce a class decorator for registering an Operation class as part of a dialect
    """

def register_op_adaptor(
    op_class: type, *, replace: bool = False
) -> Callable[[type[T]], type[T]]:
    """
    Produce a class decorator for registering an OpAdaptor class for an operation.
    """

def register_type_caster(
    typeid: _mlir.ir.TypeID, *, replace: bool = False
) -> Callable[[Callable[[T], U]], Callable[[T], U]]:
    """Register a type caster for casting MLIR types to custom user types."""

def register_value_caster(
    typeid: _mlir.ir.TypeID, *, replace: bool = False
) -> Callable[[Callable[[T], U]], Callable[[T], U]]:
    """Register a value caster for casting MLIR values to custom user values."""
