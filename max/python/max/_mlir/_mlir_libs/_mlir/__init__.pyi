# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import max._mlir.ir

from . import ir as ir
from . import passmanager as passmanager
from . import rewrite as rewrite

globals: _Globals = ...

def register_dialect(dialect_class: type) -> type:
    """Class decorator for registering a custom Dialect wrapper"""

def register_operation(dialect_class: type, *, replace: bool = False) -> object:
    """
    Produce a class decorator for registering an Operation class as part of a dialect
    """

def register_type_caster(
    typeid: max._mlir.ir.TypeID, *, replace: bool = False
) -> object:
    """Register a type caster for casting MLIR types to custom user types."""

def register_value_caster(
    typeid: max._mlir.ir.TypeID, *, replace: bool = False
) -> object:
    """Register a value caster for casting MLIR values to custom user values."""
