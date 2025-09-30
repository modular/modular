# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from typing import overload

import max._mlir.ir

class PDLModule:
    @overload
    def __init__(self, module: max._mlir.ir.Module) -> None:
        """Create a PDL module from the given module."""

    @overload
    def __init__(self, module: max._mlir.ir.Module) -> None: ...
    def freeze(self) -> FrozenRewritePatternSet: ...

class FrozenRewritePatternSet:
    pass

@overload
def apply_patterns_and_fold_greedily(
    module: max._mlir.ir.Module, set: FrozenRewritePatternSet
) -> None:
    """
    Applys the given patterns to the given module greedily while folding results.
    """

@overload
def apply_patterns_and_fold_greedily(
    module: max._mlir.ir.Module, set: FrozenRewritePatternSet
) -> None:
    """
    Applys the given patterns to the given module greedily while folding results.
    """

@overload
def apply_patterns_and_fold_greedily(
    op: max._mlir.ir._OperationBase, set: FrozenRewritePatternSet
) -> None:
    """
    Applys the given patterns to the given op greedily while folding results.
    """

@overload
def apply_patterns_and_fold_greedily(
    op: max._mlir.ir._OperationBase, set: FrozenRewritePatternSet
) -> None:
    """
    Applys the given patterns to the given op greedily while folding results.
    """
