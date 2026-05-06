# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .inner import Inner


struct Factory:
    def __init__(out self):
        pass

    # make() references Inner both in its return type and body, causing
    # Inner.__init__'s FuncSymbolAttr to be walked when Factory is
    # body-resolved.  Before the fix, a stale MLIR SymbolTable cache (built
    # before Inner's parent package was materialized) could permanently mark
    # that FuncSymbolAttr as unresolvable, leaving Inner.__init__ absent from
    # declForFuncSymbol and triggering a KGEN verifier error.
    def make(self) -> Inner:
        return Inner(42)
