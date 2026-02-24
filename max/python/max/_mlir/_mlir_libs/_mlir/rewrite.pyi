# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence
from typing import Any, overload

import max._mlir._mlir_libs._mlir.ir
import max._mlir.ir

class GreedyRewriteStrictness(enum.Enum):
    ANY_OP = 0

    EXISTING_AND_NEW_OPS = 1

    EXISTING_OPS = 2

class GreedySimplifyRegionLevel(enum.Enum):
    DISABLED = 0

    NORMAL = 1

    AGGRESSIVE = 2

class DialectConversionFoldingMode(enum.Enum):
    NEVER = 0

    BEFORE_PATTERNS = 1

    AFTER_PATTERNS = 2

class PatternRewriter:
    @property
    def ip(self) -> max._mlir._mlir_libs._mlir.ir.InsertionPoint:
        """The current insertion point of the PatternRewriter."""

    @overload
    def replace_op(
        self,
        op: max._mlir._mlir_libs._mlir.ir._OperationBase,
        new_op: max._mlir._mlir_libs._mlir.ir._OperationBase,
    ) -> None:
        """Replace an operation with a new operation."""

    @overload
    def replace_op(
        self,
        op: max._mlir._mlir_libs._mlir.ir._OperationBase,
        values: Sequence[max._mlir._mlir_libs._mlir.ir.Value],
    ) -> None:
        """Replace an operation with a list of values."""

    def erase_op(
        self, op: max._mlir._mlir_libs._mlir.ir._OperationBase
    ) -> None:
        """Erase an operation."""

class RewritePatternSet:
    def __init__(self, context: Context | None = None) -> None: ...
    def add(
        self,
        root: type | str,
        fn: Callable[[max._mlir.ir.Operation, PatternRewriter], Any],
        benefit: int = 1,
    ) -> None:
        """
        Add a new rewrite pattern on the specified root operation, using the provided callable
        for matching and rewriting, and assign it the given benefit.

        Args:
          root: The root operation to which this pattern applies.
                This may be either an OpView subclass (e.g., ``arith.AddIOp``) or
                an operation name string (e.g., ``"arith.addi"``).
          fn: The callable to use for matching and rewriting,
              which takes an operation and a pattern rewriter as arguments.
              The match is considered successful iff the callable returns
              a value where ``bool(value)`` is ``False`` (e.g. ``None``).
              If possible, the operation is cast to its corresponding OpView subclass
              before being passed to the callable.
          benefit: The benefit of the pattern, defaulting to 1.
        """

    def add_conversion(
        self,
        root: object,
        fn: Callable,
        type_converter: TypeConverter,
        benefit: int = 1,
    ) -> None:
        """
        Add a new conversion pattern on the specified root operation,
        using the provided callable for matching and rewriting,
        and assign it the given benefit.

        Args:
          root: The root operation to which this pattern applies.
                This may be either an OpView subclass (e.g., ``arith.AddIOp``) or
                an operation name string (e.g., ``"arith.addi"``).
          fn: The callable to use for matching and rewriting,
              which takes an operation, its adaptor,
              the type converter and a pattern rewriter as arguments.
              The match is considered successful iff the callable returns
              a value where ``bool(value)`` is ``False`` (e.g. ``None``).
              If possible, the operation is cast to its corresponding OpView subclass
              before being passed to the callable.
          type_converter: The type converter to convert types in the IR.
          benefit: The benefit of the pattern, defaulting to 1.
        """

    def freeze(self) -> FrozenRewritePatternSet:
        """Freeze the pattern set into a frozen one."""

class ConversionPatternRewriter(PatternRewriter):
    pass

class ConversionTarget:
    def __init__(self, context: Context | None = None) -> None: ...
    def add_legal_op(self, *ops) -> None:
        """Mark the given operations as legal."""

    def add_illegal_op(self, *ops) -> None:
        """Mark the given operations as illegal."""

    def add_legal_dialect(self, *dialects) -> None:
        """Mark the given dialects as legal."""

    def add_illegal_dialect(self, *dialects) -> None:
        """Mark the given dialect as illegal."""

class TypeConverter:
    def __init__(self) -> None:
        """Create a new TypeConverter."""

    def add_conversion(self, convert: Callable) -> None:
        """Register a type conversion function."""

class PDLResultList:
    @overload
    def append(self, arg: max._mlir._mlir_libs._mlir.ir.Value, /) -> None: ...
    @overload
    def append(
        self, arg: max._mlir._mlir_libs._mlir.ir.Operation, /
    ) -> None: ...
    @overload
    def append(self, arg: max._mlir._mlir_libs._mlir.ir.Type, /) -> None: ...
    @overload
    def append(
        self, arg: max._mlir._mlir_libs._mlir.ir.Attribute, /
    ) -> None: ...

class PDLModule:
    @overload
    def __init__(self, module: max._mlir._mlir_libs._mlir.ir.Module) -> None:
        """Create a PDL module from the given module."""

    @overload
    def __init__(
        self, module: max._mlir._mlir_libs._mlir.ir.Module
    ) -> None: ...
    def freeze(self) -> FrozenRewritePatternSet: ...
    def register_rewrite_function(
        self, arg0: str, arg1: Callable, /
    ) -> None: ...
    def register_constraint_function(
        self, arg0: str, arg1: Callable, /
    ) -> None: ...

class GreedyRewriteConfig:
    def __init__(self) -> None:
        """Create a greedy rewrite driver config with defaults"""

    @property
    def max_iterations(self) -> int:
        """Maximum number of iterations"""

    @max_iterations.setter
    def max_iterations(self, arg: int, /) -> None: ...
    @property
    def max_num_rewrites(self) -> int:
        """Maximum number of rewrites per iteration"""

    @max_num_rewrites.setter
    def max_num_rewrites(self, arg: int, /) -> None: ...
    @property
    def use_top_down_traversal(self) -> bool:
        """Whether to use top-down traversal"""

    @use_top_down_traversal.setter
    def use_top_down_traversal(self, arg: bool, /) -> None: ...
    @property
    def enable_folding(self) -> bool:
        """Enable or disable folding"""

    @enable_folding.setter
    def enable_folding(self, arg: bool, /) -> None: ...
    @property
    def strictness(self) -> GreedyRewriteStrictness:
        """Rewrite strictness level"""

    @strictness.setter
    def strictness(self, arg: GreedyRewriteStrictness, /) -> None: ...
    @property
    def region_simplification_level(self) -> GreedySimplifyRegionLevel:
        """Region simplification level"""

    @region_simplification_level.setter
    def region_simplification_level(
        self, arg: GreedySimplifyRegionLevel, /
    ) -> None: ...
    @property
    def enable_constant_cse(self) -> bool:
        """Enable or disable constant CSE"""

    @enable_constant_cse.setter
    def enable_constant_cse(self, arg: bool, /) -> None: ...

class ConversionConfig:
    def __init__(self) -> None:
        """Create a conversion config with defaults"""

    @property
    def folding_mode(self) -> DialectConversionFoldingMode:
        """folding behavior during dialect conversion"""

    @folding_mode.setter
    def folding_mode(self, arg: DialectConversionFoldingMode, /) -> None: ...
    @property
    def build_materializations(self) -> bool:
        """
        Whether the dialect conversion attempts to build source/target materializations
        """

    @build_materializations.setter
    def build_materializations(self, arg: bool, /) -> None: ...

class FrozenRewritePatternSet:
    pass

@overload
def apply_patterns_and_fold_greedily(
    module: max._mlir._mlir_libs._mlir.ir.Module,
    set: FrozenRewritePatternSet,
    config: GreedyRewriteConfig | None = None,
) -> None:
    """
    Applys the given patterns to the given module greedily while folding results.
    """

@overload
def apply_patterns_and_fold_greedily(
    op: max._mlir._mlir_libs._mlir.ir._OperationBase,
    set: FrozenRewritePatternSet,
    config: GreedyRewriteConfig | None = None,
) -> None:
    """
    Applys the given patterns to the given op greedily while folding results.
    """

def walk_and_apply_patterns(
    op: max._mlir._mlir_libs._mlir.ir._OperationBase,
    set: FrozenRewritePatternSet,
) -> None:
    """
    Applies the given patterns to the given op by a fast walk-based driver.
    """

def apply_partial_conversion(
    op: max._mlir._mlir_libs._mlir.ir._OperationBase,
    target: ConversionTarget,
    set: FrozenRewritePatternSet,
    config: ConversionConfig | None = None,
) -> None:
    """Applies a partial conversion on the given operation."""

def apply_full_conversion(
    op: max._mlir._mlir_libs._mlir.ir._OperationBase,
    target: ConversionTarget,
    set: FrozenRewritePatternSet,
    config: ConversionConfig | None = None,
) -> None:
    """Applies a full conversion on the given operation."""
