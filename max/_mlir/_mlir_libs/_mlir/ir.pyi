# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence
from typing import overload

import max._mlir.ir
import mlir
import typing_extensions

class DiagnosticSeverity(enum.Enum):
    ERROR = 0

    WARNING = 1

    NOTE = 2

    REMARK = 3

class WalkOrder(enum.Enum):
    PRE_ORDER = 0

    POST_ORDER = 1

class WalkResult(enum.Enum):
    ADVANCE = 0

    INTERRUPT = 1

    SKIP = 2

class Diagnostic:
    @property
    def severity(self) -> DiagnosticSeverity: ...
    @property
    def location(self) -> Location: ...
    @property
    def message(self) -> str: ...
    @property
    def notes(self) -> tuple: ...
    def __str__(self) -> str: ...

class DiagnosticInfo:
    def __init__(self, arg: Diagnostic, /) -> None: ...
    @property
    def severity(self) -> DiagnosticSeverity: ...
    @property
    def location(self) -> Location: ...
    @property
    def message(self) -> str: ...
    @property
    def notes(self) -> list[DiagnosticInfo]: ...
    def __str__(self) -> str: ...

class DiagnosticHandler:
    def detach(self) -> None: ...
    @property
    def attached(self) -> bool: ...
    @property
    def had_error(self) -> bool: ...
    def __enter__(self) -> object: ...
    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None: ...

class ThreadPool:
    def __init__(self) -> None: ...
    def get_max_concurrency(self) -> int: ...

class Context:
    def __init__(self) -> None: ...
    def __enter__(self) -> object: ...
    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None: ...

    current: Context | None = ...
    """Gets the Context bound to the current thread or raises ValueError"""

    @property
    def dialects(self) -> Dialects:
        """Gets a container for accessing dialects by name"""

    @property
    def d(self) -> Dialects:
        """Alias for 'dialect'"""

    def get_dialect_descriptor(self, dialect_name: str) -> DialectDescriptor:
        """Gets or loads a dialect by name, returning its descriptor object"""

    @property
    def allow_unregistered_dialects(self) -> bool: ...
    @allow_unregistered_dialects.setter
    def allow_unregistered_dialects(self, arg: bool, /) -> None: ...
    def attach_diagnostic_handler(self, callback: object) -> object:
        """Attaches a diagnostic handler that will receive callbacks"""

    def enable_multithreading(self, enable: bool) -> None: ...
    def set_thread_pool(self, arg: ThreadPool, /) -> None: ...
    def get_num_threads(self) -> int: ...
    def is_registered_operation(self, operation_name: str) -> bool: ...
    def append_dialect_registry(self, registry: DialectRegistry) -> None: ...
    @property
    def emit_error_diagnostics(self) -> bool:
        """
        Emit error diagnostics to diagnostic handlers. By default error diagnostics are captured and reported through MLIRError exceptions.
        """

    @emit_error_diagnostics.setter
    def emit_error_diagnostics(self, arg: bool, /) -> None: ...
    def load_all_available_dialects(self) -> None: ...

class DialectDescriptor:
    @property
    def namespace(self) -> str: ...
    def __repr__(self) -> str: ...

class Dialects:
    def __getitem__(self, arg: str, /) -> object: ...
    def __getattr__(self, arg: str, /) -> object: ...

class Dialect:
    def __init__(self, descriptor: object) -> None: ...
    @property
    def descriptor(self) -> object: ...
    def __repr__(self) -> str: ...

class DialectRegistry:
    def __init__(self) -> None: ...

class Location:
    def __enter__(self) -> object: ...
    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None: ...
    @overload
    def __eq__(self, arg: Location, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...

    current: Location | None = ...
    """Gets the Location bound to the current thread or raises ValueError"""

    @staticmethod
    def unknown(context: max._mlir.ir.Context | None = None) -> Location:
        """Gets a Location representing an unknown location"""

    @staticmethod
    def callsite(
        callee: Location,
        frames: Sequence[Location],
        context: max._mlir.ir.Context | None = None,
    ) -> Location:
        """Gets a Location representing a caller and callsite"""

    def is_a_callsite(self) -> bool: ...
    @property
    def callee(self) -> Location: ...
    @property
    def caller(self) -> Location: ...
    @overload
    @staticmethod
    def file(
        filename: str,
        line: int,
        col: int,
        context: max._mlir.ir.Context | None = None,
    ) -> Location:
        """Gets a Location representing a file, line and column"""

    @overload
    @staticmethod
    def file(
        filename: str,
        start_line: int,
        start_col: int,
        end_line: int,
        end_col: int,
        context: max._mlir.ir.Context | None = None,
    ) -> Location:
        """Gets a Location representing a file, line and column range"""

    def is_a_file(self) -> bool: ...
    @property
    def filename(self) -> str: ...
    @property
    def start_line(self) -> int: ...
    @property
    def start_col(self) -> int: ...
    @property
    def end_line(self) -> int: ...
    @property
    def end_col(self) -> int: ...
    @staticmethod
    def fused(
        locations: Sequence[Location],
        metadata: Attribute | None = None,
        context: max._mlir.ir.Context | None = None,
    ) -> Location:
        """Gets a Location representing a fused location with optional metadata"""

    def is_a_fused(self) -> bool: ...
    @property
    def locations(self) -> list[Location]: ...
    @staticmethod
    def name(
        name: str,
        childLoc: Location | None = None,
        context: max._mlir.ir.Context | None = None,
    ) -> Location:
        """
        Gets a Location representing a named location with optional child location
        """

    def is_a_name(self) -> bool: ...
    @property
    def name_str(self) -> str: ...
    @property
    def child_loc(self) -> Location: ...
    @staticmethod
    def from_attr(
        attribute: Attribute, context: max._mlir.ir.Context | None = None
    ) -> Location:
        """Gets a Location from a LocationAttr"""

    @property
    def context(self) -> Context:
        """Context that owns the Location"""

    @property
    def attr(self) -> Attribute:
        """Get the underlying LocationAttr"""

    def emit_error(self, message: str) -> None:
        """Emits an error at this location"""

    def __repr__(self) -> str: ...

class Module:
    @overload
    @staticmethod
    def parse(asm: str, context: max._mlir.ir.Context | None = None) -> Module:
        """
        Parses a module's assembly format from a string.

        Returns a new MlirModule or raises an MLIRError if the parsing fails.

        See also: https://mlir.llvm.org/docs/LangRef/
        """

    @overload
    @staticmethod
    def parse(
        asm: bytes, context: max._mlir.ir.Context | None = None
    ) -> Module: ...
    @staticmethod
    def parseFile(
        path: str, context: max._mlir.ir.Context | None = None
    ) -> Module:
        """
        Parses a module's assembly format from a string.

        Returns a new MlirModule or raises an MLIRError if the parsing fails.

        See also: https://mlir.llvm.org/docs/LangRef/
        """

    @staticmethod
    def create(loc: Location | None = None) -> Module:
        """Creates an empty module"""

    @property
    def context(self) -> Context:
        """Context that created the Module"""

    @property
    def operation(self) -> Operation:
        """Accesses the module as an operation"""

    @property
    def body(self) -> Block:
        """Return the block for this module"""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    def __str__(self) -> str:
        """
        Gets the assembly form of the operation with default options.

        If more advanced control over the assembly formatting or I/O options is needed,
        use the dedicated print or get_asm method, which supports keyword arguments to
        customize behavior.
        """

    def __eq__(self, other: Module) -> bool: ...
    def __hash__(self) -> int: ...

class Operation(_OperationBase):
    @staticmethod
    def create(
        name: str,
        results: Sequence[Type] | None = None,
        operands: Sequence[Value] | None = None,
        attributes: dict | None = None,
        successors: Sequence[Block] | None = None,
        regions: int = 0,
        loc: Location | None = None,
        ip: object | None = None,
        infer_type: bool = False,
    ) -> Operation:
        """
        Creates a new operation.

        Args:
          name: Operation name (e.g. "dialect.operation").
          results: Sequence of Type representing op result types.
          attributes: Dict of str:Attribute.
          successors: List of Block for the operation's successors.
          regions: Number of regions to create.
          location: A Location object (defaults to resolve from context manager).
          ip: An InsertionPoint (defaults to resolve from context manager or set to
            False to disable insertion, even with an insertion point set in the
            context manager).
          infer_type: Whether to infer result types.
        Returns:
          A new "detached" Operation object. Detached operations can be added
          to blocks, which causes them to become "attached."
        """

    @staticmethod
    def parse(
        source: str,
        *,
        source_name: str = "",
        context: max._mlir.ir.Context | None = None,
    ) -> OpView:
        """
        Parses an operation. Supports both text assembly format and binary bytecode format.
        """

    @property
    def operation(self) -> Operation: ...
    @property
    def opview(self) -> OpView: ...
    @property
    def block(self) -> Block: ...
    @property
    def successors(self) -> OpSuccessors:
        """Returns the list of Operation successors."""

class OpView(_OperationBase):
    @overload
    def __init__(self, operation: Operation) -> None: ...
    @overload
    def __init__(
        self,
        name: str,
        opRegionSpec: tuple[int, bool],
        operandSegmentSpecObj: object | None = None,
        resultSegmentSpecObj: object | None = None,
        results: list | None = None,
        operands: list | None = None,
        attributes: dict | None = None,
        successors: Sequence[Block] | None = None,
        regions: int | None = None,
        loc: Location | None = None,
        ip: object | None = None,
    ) -> None: ...
    @property
    def operation(self) -> Operation: ...
    @property
    def opview(self) -> OpView: ...
    def __str__(self) -> str: ...
    @property
    def successors(self) -> OpSuccessors:
        """Returns the list of Operation successors."""

    @classmethod
    def build_generic(*args, **kwargs):
        """
        (cls: object, results: list | None = None, operands: list | None = None, attributes: dict | None = None, successors: collections.abc.Sequence[max._mlir._mlir_libs._mlir.ir.Block] | None = None, regions: int | None = None, loc: max._mlir._mlir_libs._mlir.ir.Location | None = None, ip: object | None = None) -> object

        Builds a specific, generated OpView based on class level attributes.
        """

    @classmethod
    def parse(*args, **kwargs):
        r"""
        (cls: object, source: str, *, source_name: str = \'\', context: max._mlir.ir.Context | None = None) -> max._mlir._mlir_libs._mlir.ir.OpView

        Parses a specific, generated OpView based on class level attributes
        """

class Region:
    @property
    def blocks(self) -> BlockList:
        """Returns a forward-optimized sequence of blocks."""

    @property
    def owner(self) -> OpView:
        """Returns the operation owning this region."""

    def __iter__(self) -> BlockIterator:
        """Iterates over blocks in the region."""

    @overload
    def __eq__(self, arg: Region, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...

class Block:
    @property
    def owner(self) -> OpView:
        """Returns the owning operation of this block."""

    @property
    def region(self) -> Region:
        """Returns the owning region of this block."""

    @property
    def arguments(self) -> BlockArgumentList:
        """Returns a list of block arguments."""

    def add_argument(self, type: Type, loc: Location) -> BlockArgument:
        """
        Append an argument of the specified type to the block and returns the newly added argument.
        """

    def erase_argument(self, arg: int, /) -> None:
        """Erase the argument at 'index' and remove it from the argument list."""

    @property
    def operations(self) -> OperationList:
        """Returns a forward-optimized sequence of operations."""

    @staticmethod
    def create_at_start(
        parent: Region,
        arg_types: Sequence = [],
        arg_locs: Sequence | None = None,
    ) -> Block:
        """
        Creates and returns a new Block at the beginning of the given region (with given argument types and locations).
        """

    def append_to(self, arg: Region, /) -> None:
        """Append this block to a region, transferring ownership if necessary"""

    def create_before(
        self, *arg_types, arg_locs: Sequence | None = None
    ) -> Block:
        """
        Creates and returns a new Block before this block (with given argument types and locations).
        """

    def create_after(
        self, *arg_types, arg_locs: Sequence | None = None
    ) -> Block:
        """
        Creates and returns a new Block after this block (with given argument types and locations).
        """

    def __iter__(self) -> OperationIterator:
        """Iterates over operations in the block."""

    @overload
    def __eq__(self, arg: Block, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str:
        """Returns the assembly form of the block."""

    def append(self, operation: _OperationBase) -> None:
        """
        Appends an operation to this block. If the operation is currently in another block, it will be moved.
        """

    @property
    def successors(self) -> BlockSuccessors:
        """Returns the list of Block successors."""

    @property
    def predecessors(self) -> BlockPredecessors:
        """Returns the list of Block predecessors."""

class InsertionPoint:
    @overload
    def __init__(self, block: Block) -> None:
        """Inserts after the last operation but still inside the block."""

    @overload
    def __init__(self, beforeOperation: _OperationBase) -> None:
        """Inserts before a referenced operation."""

    def __enter__(self) -> object: ...
    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None: ...

    current: InsertionPoint = ...
    """
    Gets the InsertionPoint bound to the current thread or raises ValueError if none has been set
    """

    @staticmethod
    def at_block_begin(block: Block) -> InsertionPoint:
        """Inserts at the beginning of the block."""

    @staticmethod
    def at_block_terminator(block: Block) -> InsertionPoint:
        """Inserts before the block terminator."""

    @staticmethod
    def after(operation: _OperationBase) -> InsertionPoint:
        """Inserts after the operation."""

    def insert(self, operation: _OperationBase) -> None:
        """Inserts an operation."""

    @property
    def block(self) -> Block:
        """Returns the block that this InsertionPoint points to."""

    @property
    def ref_operation(self) -> Operation | None:
        """
        The reference operation before which new operations are inserted, or None if the insertion point is at the end of the block
        """

class Attribute:
    def __init__(self, cast_from_type: Attribute) -> None:
        """Casts the passed attribute to the generic Attribute"""

    @staticmethod
    def parse(
        asm: str, context: max._mlir.ir.Context | None = None
    ) -> Attribute:
        """
        Parses an attribute from an assembly form. Raises an MLIRError on failure.
        """

    @property
    def context(self) -> Context:
        """Context that owns the Attribute"""

    @property
    def type(self) -> Type: ...
    def get_named(self, arg: str, /) -> NamedAttribute:
        """Binds a name to the attribute"""

    @overload
    def __eq__(self, arg: Attribute, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    def __str__(self) -> str:
        """Returns the assembly form of the Attribute."""

    def __repr__(self) -> str: ...
    @property
    def typeid(self) -> TypeID: ...
    def maybe_downcast(self) -> Attribute: ...

class NamedAttribute:
    def __repr__(self) -> str: ...
    @property
    def name(self) -> str:
        """The name of the NamedAttribute binding"""

    @property
    def attr(self) -> Attribute:
        """The underlying generic attribute of the NamedAttribute binding"""

class Type:
    def __init__(self, cast_from_type: Type) -> None:
        """Casts the passed type to the generic Type"""

    @staticmethod
    def parse(asm: str, context: max._mlir.ir.Context | None = None) -> Type:
        """
        Parses the assembly form of a type.

        Returns a Type object or raises an MLIRError if the type cannot be parsed.

        See also: https://mlir.llvm.org/docs/LangRef/#type-system
        """

    @property
    def context(self) -> Context:
        """Context that owns the Type"""

    @overload
    def __eq__(self, arg: Type, /) -> bool: ...
    @overload
    def __eq__(self, other: object | None) -> bool: ...
    def __hash__(self) -> int: ...
    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    def __str__(self) -> str:
        """Returns the assembly form of the type."""

    def __repr__(self) -> str: ...
    def maybe_downcast(self) -> Type: ...
    @property
    def typeid(self) -> TypeID: ...

class TypeID:
    @overload
    def __eq__(self, arg: TypeID, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Value:
    def __init__(self, value: Value) -> None: ...
    @property
    def context(self) -> Context:
        """Context in which the value lives."""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    @property
    def owner(self) -> Operation | Block | None: ...
    @property
    def uses(self) -> OpOperandIterator: ...
    @overload
    def __eq__(self, arg: Value, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str:
        """
        Returns the string form of the value.

        If the value is a block argument, this is the assembly form of its type and the
        position in the argument list. If the value is an operation result, this is
        equivalent to printing the operation that produced it.
        """

    @overload
    def get_name(
        self,
        use_local_scope: bool = False,
        use_name_loc_as_prefix: bool = False,
    ) -> str: ...
    @overload
    def get_name(self, state: AsmState) -> str:
        """Returns the string form of value as an operand (i.e., the ValueID)."""

    @property
    def type(self) -> Type: ...
    def set_type(self, type: Type) -> None: ...
    def replace_all_uses_with(self, arg: Value, /) -> None:
        """
        Replace all uses of value with the new value, updating anything in
        the IR that uses 'self' to use the other value instead.
        """

    @overload
    def replace_all_uses_except(
        self, with_: Value, exceptions: Operation
    ) -> None:
        """
        "Replace all uses of this value with the 'with' value, except for those
        in 'exceptions'. 'exceptions' can be either a single operation or a list of
        operations.
        """

    @overload
    def replace_all_uses_except(
        self, with_: Value, exceptions: Sequence[Operation]
    ) -> None: ...
    @overload
    def replace_all_uses_except(
        self, with_: Value, exceptions: Operation
    ) -> None: ...
    @overload
    def replace_all_uses_except(
        self, with_: Value, exceptions: Sequence[Operation]
    ) -> None: ...
    def maybe_downcast(self) -> Value: ...
    @property
    def location(self) -> Location:
        """Returns the source location the value"""

class BlockArgument(Value):
    def __init__(self, value: Value) -> None: ...
    @staticmethod
    def isinstance(other_value: Value) -> bool: ...
    def maybe_downcast(self) -> BlockArgument: ...
    @property
    def owner(self) -> Block: ...
    @property
    def arg_number(self) -> int: ...
    def set_type(self, type: Type) -> None: ...

class OpResult(Value):
    def __init__(self, value: Value) -> None: ...
    @staticmethod
    def isinstance(other_value: Value) -> bool: ...
    def maybe_downcast(self) -> OpResult: ...
    @property
    def owner(self) -> Operation: ...
    @property
    def result_number(self) -> int: ...

class OpOperand:
    @property
    def owner(self) -> OpView: ...
    @property
    def operand_number(self) -> int: ...

class AsmState:
    @overload
    def __init__(self, value: Value, use_local_scope: bool = False) -> None: ...
    @overload
    def __init__(
        self, op: _OperationBase, use_local_scope: bool = False
    ) -> None: ...

class SymbolTable:
    def __init__(self, arg: _OperationBase, /) -> None: ...
    def __getitem__(self, arg: str, /) -> OpView: ...
    def insert(self, operation: _OperationBase) -> StringAttr: ...
    def erase(self, operation: _OperationBase) -> None: ...
    def __delitem__(self, arg: str, /) -> None: ...
    def __contains__(self, arg: str, /) -> bool: ...
    @staticmethod
    def set_symbol_name(symbol: _OperationBase, name: str) -> None: ...
    @staticmethod
    def get_symbol_name(symbol: _OperationBase) -> StringAttr: ...
    @staticmethod
    def get_visibility(symbol: _OperationBase) -> StringAttr: ...
    @staticmethod
    def set_visibility(symbol: _OperationBase, visibility: str) -> None: ...
    @staticmethod
    def replace_all_symbol_uses(
        old_symbol: str, new_symbol: str, from_op: _OperationBase
    ) -> None: ...
    @staticmethod
    def walk_symbol_tables(
        from_op: _OperationBase, all_sym_uses_visible: bool, callback: object
    ) -> None: ...

class BlockArgumentList:
    def __add__(self, arg: BlockArgumentList, /) -> list[BlockArgument]: ...
    @property
    def types(self) -> list[Type]: ...

class BlockIterator:
    def __iter__(self) -> BlockIterator: ...
    def __next__(self) -> Block: ...

class BlockList:
    def __getitem__(self, arg: int, /) -> Block: ...
    def __iter__(self) -> BlockIterator: ...
    def __len__(self) -> int: ...
    def append(self, *args, arg_locs: Sequence | None = None) -> Block:
        """
        Appends a new block, with argument types as positional args.

        Returns:
          The created block.
        """

class BlockSuccessors:
    def __add__(self, arg: BlockSuccessors, /) -> list[Block]: ...

class BlockPredecessors:
    def __add__(self, arg: BlockPredecessors, /) -> list[Block]: ...

class OperationIterator:
    def __iter__(self) -> OperationIterator: ...
    def __next__(self) -> OpView: ...

class OperationList:
    def __getitem__(self, arg: int, /) -> OpView: ...
    def __iter__(self) -> OperationIterator: ...
    def __len__(self) -> int: ...

class OpAttributeMap:
    def __contains__(self, arg: str, /) -> bool: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, arg: str, /) -> Attribute: ...
    @overload
    def __getitem__(self, arg: int, /) -> NamedAttribute: ...
    def __setitem__(self, arg0: str, arg1: Attribute, /) -> None: ...
    def __delitem__(self, arg: str, /) -> None: ...

class OpOperandIterator:
    def __iter__(self) -> OpOperandIterator: ...
    def __next__(self) -> OpOperand: ...

class OpOperandList:
    def __add__(self, arg: OpOperandList, /) -> list[Value]: ...
    def __setitem__(self, arg0: int, arg1: Value, /) -> None: ...

class OpResultList:
    def __add__(self, arg: OpResultList, /) -> list[OpResult]: ...
    @property
    def types(self) -> list[Type]: ...
    @property
    def owner(self) -> OpView: ...

class OpSuccessors:
    def __add__(self, arg: OpSuccessors, /) -> list[Block]: ...
    def __setitem__(self, arg0: int, arg1: Block, /) -> None: ...

class RegionIterator:
    def __iter__(self) -> RegionIterator: ...
    def __next__(self) -> Region: ...

class RegionSequence:
    def __add__(self, arg: RegionSequence, /) -> list[Region]: ...
    def __iter__(self) -> RegionIterator: ...

class AttrBuilder:
    @staticmethod
    def contains(arg: str, /) -> bool: ...
    @staticmethod
    def get(arg: str, /) -> Callable: ...
    @staticmethod
    def insert(
        attribute_kind: str, attr_builder: Callable, replace: bool = False
    ) -> None:
        """
        Register an attribute builder for building MLIR attributes from python values.
        """

class AffineExpr:
    @overload
    def __add__(self, arg: AffineExpr, /) -> AffineAddExpr: ...
    @overload
    def __add__(self, arg: int, /) -> AffineAddExpr: ...
    def __radd__(self, arg: int, /) -> AffineAddExpr: ...
    @overload
    def __mul__(self, arg: AffineExpr, /) -> AffineMulExpr: ...
    @overload
    def __mul__(self, arg: int, /) -> AffineMulExpr: ...
    def __rmul__(self, arg: int, /) -> AffineMulExpr: ...
    @overload
    def __mod__(self, arg: AffineExpr, /) -> AffineModExpr: ...
    @overload
    def __mod__(self, arg: int, /) -> AffineModExpr: ...
    def __rmod__(self, arg: int, /) -> AffineModExpr: ...
    @overload
    def __sub__(self, arg: AffineExpr, /) -> AffineAddExpr: ...
    @overload
    def __sub__(self, arg: int, /) -> AffineAddExpr: ...
    def __rsub__(self, arg: int, /) -> AffineAddExpr: ...
    @overload
    def __eq__(self, arg: AffineExpr, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    @property
    def context(self) -> Context: ...
    def compose(self, arg: AffineMap, /) -> AffineExpr: ...
    def shift_dims(
        self, num_dims: int, shift: int, offset: int = 0
    ) -> AffineExpr: ...
    def shift_symbols(
        self, num_symbols: int, shift: int, offset: int = 0
    ) -> AffineExpr: ...
    @staticmethod
    def simplify_affine_expr(
        expr: AffineExpr, num_dims: int, num_symbols: int
    ) -> AffineExpr:
        """
        Simplify an affine expression by flattening and some amount of simple analysis.
        """

    @overload
    @staticmethod
    def get_add(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineAddExpr:
        """Gets an affine expression containing a sum of two expressions."""

    @overload
    @staticmethod
    def get_add(arg0: int, arg1: AffineExpr, /) -> AffineAddExpr:
        """
        Gets an affine expression containing a sum of a constant and another expression.
        """

    @overload
    @staticmethod
    def get_add(arg0: AffineExpr, arg1: int, /) -> AffineAddExpr:
        """
        Gets an affine expression containing a sum of an expression and a constant.
        """

    @overload
    @staticmethod
    def get_mul(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineMulExpr:
        """Gets an affine expression containing a product of two expressions."""

    @overload
    @staticmethod
    def get_mul(arg0: int, arg1: AffineExpr, /) -> AffineMulExpr:
        """
        Gets an affine expression containing a product of a constant and another expression.
        """

    @overload
    @staticmethod
    def get_mul(arg0: AffineExpr, arg1: int, /) -> AffineMulExpr:
        """
        Gets an affine expression containing a product of an expression and a constant.
        """

    @overload
    @staticmethod
    def get_mod(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineModExpr:
        """
        Gets an affine expression containing the modulo of dividing one expression by another.
        """

    @overload
    @staticmethod
    def get_mod(arg0: int, arg1: AffineExpr, /) -> AffineModExpr:
        """
        Gets a semi-affine expression containing the modulo of dividing a constant by an expression.
        """

    @overload
    @staticmethod
    def get_mod(arg0: AffineExpr, arg1: int, /) -> AffineModExpr:
        """
        Gets an affine expression containing the module of dividingan expression by a constant.
        """

    @overload
    @staticmethod
    def get_floor_div(
        arg0: AffineExpr, arg1: AffineExpr, /
    ) -> AffineFloorDivExpr:
        """
        Gets an affine expression containing the rounded-down result of dividing one expression by another.
        """

    @overload
    @staticmethod
    def get_floor_div(arg0: int, arg1: AffineExpr, /) -> AffineFloorDivExpr:
        """
        Gets a semi-affine expression containing the rounded-down result of dividing a constant by an expression.
        """

    @overload
    @staticmethod
    def get_floor_div(arg0: AffineExpr, arg1: int, /) -> AffineFloorDivExpr:
        """
        Gets an affine expression containing the rounded-down result of dividing an expression by a constant.
        """

    @overload
    @staticmethod
    def get_ceil_div(
        arg0: AffineExpr, arg1: AffineExpr, /
    ) -> AffineCeilDivExpr:
        """
        Gets an affine expression containing the rounded-up result of dividing one expression by another.
        """

    @overload
    @staticmethod
    def get_ceil_div(arg0: int, arg1: AffineExpr, /) -> AffineCeilDivExpr:
        """
        Gets a semi-affine expression containing the rounded-up result of dividing a constant by an expression.
        """

    @overload
    @staticmethod
    def get_ceil_div(arg0: AffineExpr, arg1: int, /) -> AffineCeilDivExpr:
        """
        Gets an affine expression containing the rounded-up result of dividing an expression by a constant.
        """

    @staticmethod
    def get_constant(
        value: int, context: max._mlir.ir.Context | None = None
    ) -> AffineConstantExpr:
        """Gets a constant affine expression with the given value."""

    @staticmethod
    def get_dim(
        position: int, context: max._mlir.ir.Context | None = None
    ) -> AffineDimExpr:
        """Gets an affine expression of a dimension at the given position."""

    @staticmethod
    def get_symbol(
        position: int, context: max._mlir.ir.Context | None = None
    ) -> AffineSymbolExpr:
        """Gets an affine expression of a symbol at the given position."""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

class AffineConstantExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(
        value: int, context: max._mlir.ir.Context | None = None
    ) -> AffineConstantExpr: ...
    @property
    def value(self) -> int: ...

class AffineDimExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(
        position: int, context: max._mlir.ir.Context | None = None
    ) -> AffineDimExpr: ...
    @property
    def position(self) -> int: ...

class AffineSymbolExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(
        position: int, context: max._mlir.ir.Context | None = None
    ) -> AffineSymbolExpr: ...
    @property
    def position(self) -> int: ...

class AffineBinaryExpr(AffineExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @property
    def lhs(self) -> AffineExpr: ...
    @property
    def rhs(self) -> AffineExpr: ...

class AffineAddExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineAddExpr: ...

class AffineMulExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineMulExpr: ...

class AffineModExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineModExpr: ...

class AffineFloorDivExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineFloorDivExpr: ...

class AffineCeilDivExpr(AffineBinaryExpr):
    def __init__(self, expr: AffineExpr) -> None: ...
    @staticmethod
    def isinstance(other: AffineExpr) -> bool: ...
    @staticmethod
    def get(arg0: AffineExpr, arg1: AffineExpr, /) -> AffineCeilDivExpr: ...

class AffineMap:
    @overload
    def __eq__(self, arg: AffineMap, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    @staticmethod
    def compress_unused_symbols(
        arg0: list, arg1: max._mlir.ir.Context, /
    ) -> list[AffineMap]: ...
    @property
    def context(self) -> Context:
        """Context that owns the Affine Map"""

    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    @staticmethod
    def get(
        dim_count: int,
        symbol_count: int,
        exprs: list,
        context: max._mlir.ir.Context | None = None,
    ) -> AffineMap:
        """Gets a map with the given expressions as results."""

    @staticmethod
    def get_constant(
        value: int, context: max._mlir.ir.Context | None = None
    ) -> AffineMap:
        """Gets an affine map with a single constant result"""

    @staticmethod
    def get_empty(context: max._mlir.ir.Context | None = None) -> AffineMap:
        """Gets an empty affine map."""

    @staticmethod
    def get_identity(
        n_dims: int, context: max._mlir.ir.Context | None = None
    ) -> AffineMap:
        """Gets an identity map with the given number of dimensions."""

    @staticmethod
    def get_minor_identity(
        n_dims: int, n_results: int, context: max._mlir.ir.Context | None = None
    ) -> AffineMap:
        """
        Gets a minor identity map with the given number of dimensions and results.
        """

    @staticmethod
    def get_permutation(
        permutation: Sequence[int], context: max._mlir.ir.Context | None = None
    ) -> AffineMap:
        """Gets an affine map that permutes its inputs."""

    def get_submap(self, result_positions: Sequence[int]) -> AffineMap: ...
    def get_major_submap(self, n_results: int) -> AffineMap: ...
    def get_minor_submap(self, n_results: int) -> AffineMap: ...
    def replace(
        self,
        expr: AffineExpr,
        replacement: AffineExpr,
        n_result_dims: int,
        n_result_syms: int,
    ) -> AffineMap: ...
    @property
    def is_permutation(self) -> bool: ...
    @property
    def is_projected_permutation(self) -> bool: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_symbols(self) -> int: ...
    @property
    def results(self) -> AffineExprList: ...

class AffineExprList:
    def __add__(self, arg: AffineExprList, /) -> list[AffineExpr]: ...

class IntegerSet:
    @overload
    def __eq__(self, arg: IntegerSet, /) -> bool: ...
    @overload
    def __eq__(self, arg: object, /) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    @property
    def context(self) -> Context: ...
    def dump(self) -> None:
        """Dumps a debug representation of the object to stderr."""

    @staticmethod
    def get(
        num_dims: int,
        num_symbols: int,
        exprs: list,
        eq_flags: Sequence[bool],
        context: max._mlir.ir.Context | None = None,
    ) -> IntegerSet: ...
    @staticmethod
    def get_empty(
        num_dims: int,
        num_symbols: int,
        context: max._mlir.ir.Context | None = None,
    ) -> IntegerSet: ...
    def get_replaced(
        self,
        dim_exprs: list,
        symbol_exprs: list,
        num_result_dims: int,
        num_result_symbols: int,
    ) -> IntegerSet: ...
    @property
    def is_canonical_empty(self) -> bool: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def n_symbols(self) -> int: ...
    @property
    def n_inputs(self) -> int: ...
    @property
    def n_equalities(self) -> int: ...
    @property
    def n_inequalities(self) -> int: ...
    @property
    def constraints(self) -> IntegerSetConstraintList: ...

class IntegerSetConstraint:
    @property
    def expr(self) -> AffineExpr: ...
    @property
    def is_eq(self) -> bool: ...

class IntegerSetConstraintList:
    def __add__(
        self, arg: IntegerSetConstraintList, /
    ) -> list[IntegerSetConstraint]: ...

class AffineMapAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(affine_map: AffineMap) -> AffineMapAttr:
        """Gets an attribute wrapping an AffineMap."""

    @property
    def value(self) -> AffineMap:
        """Returns the value of the AffineMap attribute"""

class DenseBoolArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        values: Sequence, context: max._mlir.ir.Context | None = None
    ) -> DenseBoolArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseBoolArrayIterator: ...
    def __add__(self, arg: list, /) -> DenseBoolArrayAttr: ...

class DenseBoolArrayIterator:
    def __iter__(self) -> DenseBoolArrayIterator: ...
    def __next__(self) -> bool: ...

class DenseI8ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        values: Sequence[int], context: max._mlir.ir.Context | None = None
    ) -> DenseI8ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI8ArrayIterator: ...
    def __add__(self, arg: list, /) -> DenseI8ArrayAttr: ...

class DenseI8ArrayIterator:
    def __iter__(self) -> DenseI8ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI16ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        values: Sequence[int], context: max._mlir.ir.Context | None = None
    ) -> DenseI16ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI16ArrayIterator: ...
    def __add__(self, arg: list, /) -> DenseI16ArrayAttr: ...

class DenseI16ArrayIterator:
    def __iter__(self) -> DenseI16ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI32ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        values: Sequence[int], context: max._mlir.ir.Context | None = None
    ) -> DenseI32ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI32ArrayIterator: ...
    def __add__(self, arg: list, /) -> DenseI32ArrayAttr: ...

class DenseI32ArrayIterator:
    def __iter__(self) -> DenseI32ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseI64ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        values: Sequence[int], context: max._mlir.ir.Context | None = None
    ) -> DenseI64ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseI64ArrayIterator: ...
    def __add__(self, arg: list, /) -> DenseI64ArrayAttr: ...

class DenseI64ArrayIterator:
    def __iter__(self) -> DenseI64ArrayIterator: ...
    def __next__(self) -> int: ...

class DenseF32ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        values: Sequence[float], context: max._mlir.ir.Context | None = None
    ) -> DenseF32ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> float: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseF32ArrayIterator: ...
    def __add__(self, arg: list, /) -> DenseF32ArrayAttr: ...

class DenseF32ArrayIterator:
    def __iter__(self) -> DenseF32ArrayIterator: ...
    def __next__(self) -> float: ...

class DenseF64ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        values: Sequence[float], context: max._mlir.ir.Context | None = None
    ) -> DenseF64ArrayAttr:
        """Gets a uniqued dense array attribute"""

    def __getitem__(self, arg: int, /) -> float: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> DenseF64ArrayIterator: ...
    def __add__(self, arg: list, /) -> DenseF64ArrayAttr: ...

class DenseF64ArrayIterator:
    def __iter__(self) -> DenseF64ArrayIterator: ...
    def __next__(self) -> float: ...

class ArrayAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        attributes: list, context: max._mlir.ir.Context | None = None
    ) -> ArrayAttr:
        """Gets a uniqued Array attribute"""

    def __getitem__(self, arg: int, /) -> Attribute: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> ArrayAttributeIterator: ...
    def __add__(self, arg: list, /) -> ArrayAttr: ...

class ArrayAttributeIterator:
    def __iter__(self) -> ArrayAttributeIterator: ...
    def __next__(self) -> Attribute: ...

class BoolAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        value: bool, context: max._mlir.ir.Context | None = None
    ) -> BoolAttr:
        """Gets an uniqued bool attribute"""

    @property
    def value(self) -> bool:
        """Returns the value of the bool attribute"""

    def __bool__(self) -> bool:
        """Converts the value of the bool attribute to a Python bool"""

class DenseElementsAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    def __buffer__(self, flags, /):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """

    def __release_buffer__(self, buffer, /):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """

    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    @overload
    @staticmethod
    def get(
        array: typing_extensions.Buffer,
        signless: bool = True,
        type: Type | None = None,
        shape: Sequence[int] | None = None,
        context: Context | None = None,
    ) -> DenseElementsAttr:
        """
        Gets a DenseElementsAttr from a Python buffer or array.

        When `type` is not provided, then some limited type inferencing is done based
        on the buffer format. Support presently exists for 8/16/32/64 signed and
        unsigned integers and float16/float32/float64. DenseElementsAttrs of these
        types can also be converted back to a corresponding buffer.

        For conversions outside of these types, a `type=` must be explicitly provided
        and the buffer contents must be bit-castable to the MLIR internal
        representation:

          * Integer types (except for i1): the buffer must be byte aligned to the
            next byte boundary.
          * Floating point types: Must be bit-castable to the given floating point
            size.
          * i1 (bool): Bit packed into 8bit words where the bit pattern matches a
            row major ordering. An arbitrary Numpy `bool_` array can be bit packed to
            this specification with: `np.packbits(ary, axis=None, bitorder='little')`.

        If a single element buffer is passed (or for i1, a single byte with value 0
        or 255), then a splat will be created.

        Args:
          array: The array or buffer to convert.
          signless: If inferring an appropriate MLIR type, use signless types for
            integers (defaults True).
          type: Skips inference of the MLIR element type and uses this instead. The
            storage size must be consistent with the actual contents of the buffer.
          shape: Overrides the shape of the buffer when constructing the MLIR
            shaped type. This is needed when the physical and logical shape differ (as
            for i1).
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """

    @overload
    @staticmethod
    def get(
        attrs: list,
        type: Type | None = None,
        context: max._mlir.ir.Context | None = None,
    ) -> DenseElementsAttr:
        """
        Gets a DenseElementsAttr from a Python list of attributes.

        Note that it can be expensive to construct attributes individually.
        For a large number of elements, consider using a Python buffer or array instead.

        Args:
          attrs: A list of attributes.
          type: The desired shape and type of the resulting DenseElementsAttr.
            If not provided, the element type is determined based on the type
            of the 0th attribute and the shape is `[len(attrs)]`.
          context: Explicit context, if not from context manager.

        Returns:
          DenseElementsAttr on success.

        Raises:
          ValueError: If the type of the attributes does not match the type
            specified by `shaped_type`.
        """

    @staticmethod
    def get_splat(
        shaped_type: Type, element_attr: Attribute
    ) -> DenseElementsAttr:
        """Gets a DenseElementsAttr where all values are the same"""

    @property
    def is_splat(self) -> bool: ...
    def get_splat_value(self) -> Attribute: ...

class DenseFPElementsAttr(DenseElementsAttr):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, arg: int, /) -> float: ...

class DenseIntElementsAttr(DenseElementsAttr):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, arg: int, /) -> int: ...

class DenseResourceElementsAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get_from_buffer(
        array: typing_extensions.Buffer,
        name: str,
        type: Type,
        alignment: int | None = None,
        is_mutable: bool = False,
        context: Context | None = None,
    ) -> DenseResourceElementsAttr:
        """
        Gets a DenseResourceElementsAttr from a Python buffer or array.

        This function does minimal validation or massaging of the data, and it is
        up to the caller to ensure that the buffer meets the characteristics
        implied by the shape.

        The backing buffer and any user objects will be retained for the lifetime
        of the resource blob. This is typically bounded to the context but the
        resource can have a shorter lifespan depending on how it is used in
        subsequent processing.

        Args:
          buffer: The array or buffer to convert.
          name: Name to provide to the resource (may be changed upon collision).
          type: The explicit ShapedType to construct the attribute with.
          context: Explicit context, if not from context manager.

        Returns:
          DenseResourceElementsAttr on success.

        Raises:
          ValueError: If the type of the buffer or array cannot be matched to an MLIR
            type or if the buffer does not meet expectations.
        """

class DictAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    def __contains__(self, arg: str, /) -> bool: ...
    def __len__(self) -> int: ...
    @staticmethod
    def get(
        value: dict = {}, context: max._mlir.ir.Context | None = None
    ) -> DictAttr:
        """Gets an uniqued dict attribute"""

    @overload
    def __getitem__(self, arg: str, /) -> Attribute: ...
    @overload
    def __getitem__(self, arg: int, /) -> NamedAttribute: ...

class SymbolRefAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        symbols: Sequence[str], context: max._mlir.ir.Context | None = None
    ) -> SymbolRefAttr:
        """Gets a uniqued SymbolRef attribute from a list of symbol names"""

    @property
    def value(self) -> list[str]:
        """Returns the value of the SymbolRef attribute as a list[str]"""

class FlatSymbolRefAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        value: str, context: max._mlir.ir.Context | None = None
    ) -> FlatSymbolRefAttr:
        """Gets a uniqued FlatSymbolRef attribute"""

    @property
    def value(self) -> str:
        """Returns the value of the FlatSymbolRef attribute as a string"""

class OpaqueAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        dialect_namespace: str,
        buffer: typing_extensions.Buffer,
        type: Type,
        context: Context | None = None,
    ) -> OpaqueAttr:
        """Gets an Opaque attribute."""

    @property
    def dialect_namespace(self) -> str:
        """Returns the dialect namespace for the Opaque attribute as a string"""

    @property
    def data(self) -> bytes:
        """Returns the data for the Opaqued attributes as `bytes`"""

class FloatAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        type: Type, value: float, loc: max._mlir.ir.Location | None = None
    ) -> FloatAttr:
        """Gets an uniqued float point attribute associated to a type"""

    @staticmethod
    def get_f32(
        value: float, context: max._mlir.ir.Context | None = None
    ) -> FloatAttr:
        """Gets an uniqued float point attribute associated to a f32 type"""

    @staticmethod
    def get_f64(
        value: float, context: max._mlir.ir.Context | None = None
    ) -> FloatAttr:
        """Gets an uniqued float point attribute associated to a f64 type"""

    @property
    def value(self) -> float:
        """Returns the value of the float attribute"""

    def __float__(self) -> float:
        """Converts the value of the float attribute to a Python float"""

class IntegerAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(type: Type, value: int) -> IntegerAttr:
        """Gets an uniqued integer attribute associated to a type"""

    @property
    def value(self) -> int:
        """Returns the value of the integer attribute"""

    def __int__(self) -> int:
        """Converts the value of the integer attribute to a Python int"""

class IntegerSetAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(integer_set: IntegerSet) -> IntegerSetAttr:
        """Gets an attribute wrapping an IntegerSet."""

class StringAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @overload
    @staticmethod
    def get(
        value: str, context: max._mlir.ir.Context | None = None
    ) -> StringAttr:
        """Gets a uniqued string attribute"""

    @overload
    @staticmethod
    def get(
        value: bytes, context: max._mlir.ir.Context | None = None
    ) -> StringAttr: ...
    @staticmethod
    def get_typed(type: Type, value: str) -> StringAttr:
        """Gets a uniqued string attribute associated to a type"""

    @property
    def value(self) -> str:
        """Returns the value of the string attribute"""

    @property
    def value_bytes(self) -> bytes:
        """Returns the value of the string attribute as `bytes`"""

class TypeAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        value: Type, context: max._mlir.ir.Context | None = None
    ) -> TypeAttr:
        """Gets a uniqued Type attribute"""

    @property
    def value(self) -> Type: ...

class UnitAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> UnitAttr:
        """Create a Unit attribute."""

class StridedLayoutAttr(Attribute):
    def __init__(self, cast_from_attr: Attribute) -> None: ...
    @staticmethod
    def isinstance(other: Attribute) -> bool: ...
    @property
    def type(self) -> Type: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        offset: int,
        strides: Sequence[int],
        context: max._mlir.ir.Context | None = None,
    ) -> StridedLayoutAttr:
        """Gets a strided layout attribute."""

    @staticmethod
    def get_fully_dynamic(
        rank: int, context: max._mlir.ir.Context | None = None
    ) -> StridedLayoutAttr:
        """
        Gets a strided layout attribute with dynamic offset and strides of a given rank.
        """

    @property
    def offset(self) -> int:
        """Returns the value of the float point attribute"""

    @property
    def strides(self) -> list[int]:
        """Returns the value of the float point attribute"""

class InferTypeOpInterface:
    def __init__(
        self, object: object, context: max._mlir.ir.Context | None = None
    ) -> None:
        """
        Creates an interface from a given operation/opview object or from a
        subclass of OpView. Raises ValueError if the operation does not implement the
        interface.
        """

    @property
    def operation(self) -> Operation:
        """Returns an Operation for which the interface was constructed."""

    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was
        constructed
        """

    def inferReturnTypes(
        self,
        operands: list | None = None,
        attributes: Attribute | None = None,
        properties: typing_extensions.CapsuleType | None = None,
        regions: Sequence[Region] | None = None,
        context: max._mlir.ir.Context | None = None,
        loc: max._mlir.ir.Location | None = None,
    ) -> list[Type]:
        """
        Given the arguments required to build an operation, attempts to infer
        its return types. Raises ValueError on failure.
        """

class ShapedTypeComponents:
    @property
    def element_type(self) -> Type:
        """Returns the element type of the shaped type components."""

    @overload
    @staticmethod
    def get(element_type: Type) -> ShapedTypeComponents:
        """Create an shaped type components object with only the element type."""

    @overload
    @staticmethod
    def get(shape: list, element_type: Type) -> ShapedTypeComponents:
        """Create a ranked shaped type components object."""

    @overload
    @staticmethod
    def get(
        shape: list, element_type: Type, attribute: Attribute
    ) -> ShapedTypeComponents:
        """Create a ranked shaped type components object with attribute."""

    @property
    def has_rank(self) -> bool:
        """Returns whether the given shaped type component is ranked."""

    @property
    def rank(self) -> int | None:
        """
        Returns the rank of the given ranked shaped type components. If the shaped type components does not have a rank, None is returned.
        """

    @property
    def shape(self) -> list | None:
        """
        Returns the shape of the ranked shaped type components as a list of integers. Returns none if the shaped type component does not have a rank.
        """

class InferShapedTypeOpInterface:
    def __init__(
        self, object: object, context: max._mlir.ir.Context | None = None
    ) -> None:
        """
        Creates an interface from a given operation/opview object or from a
        subclass of OpView. Raises ValueError if the operation does not implement the
        interface.
        """

    @property
    def operation(self) -> Operation:
        """Returns an Operation for which the interface was constructed."""

    @property
    def opview(self) -> OpView:
        """
        Returns an OpView subclass _instance_ for which the interface was
        constructed
        """

    def inferReturnTypeComponents(
        self,
        operands: list | None = None,
        attributes: Attribute | None = None,
        regions: typing_extensions.CapsuleType | None = None,
        properties: Sequence[Region] | None = None,
        context: max._mlir.ir.Context | None = None,
        loc: max._mlir.ir.Location | None = None,
    ) -> list[ShapedTypeComponents]:
        """
        Given the arguments required to build an operation, attempts to infer
        its return shaped type components. Raises ValueError on failure.
        """

class IntegerType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get_signless(
        width: int, context: max._mlir.ir.Context | None = None
    ) -> IntegerType:
        """Create a signless integer type"""

    @staticmethod
    def get_signed(
        width: int, context: max._mlir.ir.Context | None = None
    ) -> IntegerType:
        """Create a signed integer type"""

    @staticmethod
    def get_unsigned(
        width: int, context: max._mlir.ir.Context | None = None
    ) -> IntegerType:
        """Create an unsigned integer type"""

    @property
    def width(self) -> int:
        """Returns the width of the integer type"""

    @property
    def is_signless(self) -> bool:
        """Returns whether this is a signless integer"""

    @property
    def is_signed(self) -> bool:
        """Returns whether this is a signed integer"""

    @property
    def is_unsigned(self) -> bool:
        """Returns whether this is an unsigned integer"""

class FloatType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @property
    def width(self) -> int:
        """Returns the width of the floating-point type"""

class IndexType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> IndexType:
        """Create a index type."""

class Float4E2M1FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float4E2M1FNType:
        """Create a float4_e2m1fn type."""

class Float6E2M3FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float6E2M3FNType:
        """Create a float6_e2m3fn type."""

class Float6E3M2FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float6E3M2FNType:
        """Create a float6_e3m2fn type."""

class Float8E4M3FNType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float8E4M3FNType:
        """Create a float8_e4m3fn type."""

class Float8E5M2Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float8E5M2Type:
        """Create a float8_e5m2 type."""

class Float8E4M3Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float8E4M3Type:
        """Create a float8_e4m3 type."""

class Float8E4M3FNUZType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float8E4M3FNUZType:
        """Create a float8_e4m3fnuz type."""

class Float8E4M3B11FNUZType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        context: max._mlir.ir.Context | None = None,
    ) -> Float8E4M3B11FNUZType:
        """Create a float8_e4m3b11fnuz type."""

class Float8E5M2FNUZType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float8E5M2FNUZType:
        """Create a float8_e5m2fnuz type."""

class Float8E3M4Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float8E3M4Type:
        """Create a float8_e3m4 type."""

class Float8E8M0FNUType(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> Float8E8M0FNUType:
        """Create a float8_e8m0fnu type."""

class BF16Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> BF16Type:
        """Create a bf16 type."""

class F16Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> F16Type:
        """Create a f16 type."""

class FloatTF32Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> FloatTF32Type:
        """Create a tf32 type."""

class F32Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> F32Type:
        """Create a f32 type."""

class F64Type(FloatType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> F64Type:
        """Create a f64 type."""

class NoneType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(context: max._mlir.ir.Context | None = None) -> NoneType:
        """Create a none type."""

class ComplexType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(arg: Type, /) -> ComplexType:
        """Create a complex type"""

    @property
    def element_type(self) -> Type:
        """Returns element type."""

class ShapedType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @property
    def element_type(self) -> Type:
        """Returns the element type of the shaped type."""

    @property
    def has_rank(self) -> bool:
        """Returns whether the given shaped type is ranked."""

    @property
    def rank(self) -> int:
        """Returns the rank of the given ranked shaped type."""

    @property
    def has_static_shape(self) -> bool:
        """Returns whether the given shaped type has a static shape."""

    def is_dynamic_dim(self, dim: int) -> bool:
        """
        Returns whether the dim-th dimension of the given shaped type is dynamic.
        """

    def is_static_dim(self, dim: int) -> bool:
        """
        Returns whether the dim-th dimension of the given shaped type is static.
        """

    def get_dim_size(self, dim: int) -> int:
        """Returns the dim-th dimension of the given ranked shaped type."""

    @staticmethod
    def is_dynamic_size(dim_size: int) -> bool:
        """
        Returns whether the given dimension size indicates a dynamic dimension.
        """

    @staticmethod
    def is_static_size(dim_size: int) -> bool:
        """Returns whether the given dimension size indicates a static dimension."""

    def is_dynamic_stride_or_offset(self, dim_size: int) -> bool:
        """
        Returns whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
        """

    def is_static_stride_or_offset(self, dim_size: int) -> bool:
        """
        Returns whether the given shaped type stride or offset value is statically-sized.
        """

    @property
    def shape(self) -> list[int]:
        """Returns the shape of the ranked shaped type as a list of integers."""

    @staticmethod
    def get_dynamic_size() -> int:
        """Returns the value used to indicate dynamic dimensions in shaped types."""

    @staticmethod
    def get_dynamic_stride_or_offset() -> int:
        """
        Returns the value used to indicate dynamic strides or offsets in shaped types.
        """

class VectorType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        shape: Sequence[int],
        element_type: Type,
        *,
        scalable: list | None = None,
        scalable_dims: Sequence[int] | None = None,
        loc: max._mlir.ir.Location | None = None,
    ) -> VectorType:
        """Create a vector type"""

    @property
    def scalable(self) -> bool: ...
    @property
    def scalable_dims(self) -> list[bool]: ...

class RankedTensorType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        shape: Sequence[int],
        element_type: Type,
        encoding: Attribute | None = None,
        loc: max._mlir.ir.Location | None = None,
    ) -> RankedTensorType:
        """Create a ranked tensor type"""

    @property
    def encoding(self) -> Attribute | None: ...

class UnrankedTensorType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        element_type: Type, loc: max._mlir.ir.Location | None = None
    ) -> UnrankedTensorType:
        """Create a unranked tensor type"""

class MemRefType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        shape: Sequence[int],
        element_type: Type,
        layout: Attribute | None = None,
        memory_space: Attribute | None = None,
        loc: max._mlir.ir.Location | None = None,
    ) -> MemRefType:
        """Create a memref type"""

    @property
    def layout(self) -> Attribute:
        """The layout of the MemRef type."""

    def get_strides_and_offset(self) -> tuple[list[int], int]:
        """The strides and offset of the MemRef type."""

    @property
    def affine_map(self) -> AffineMap:
        """The layout of the MemRef type as an affine map."""

    @property
    def memory_space(self) -> Attribute | None:
        """Returns the memory space of the given MemRef type."""

class UnrankedMemRefType(ShapedType):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        element_type: Type,
        memory_space: Attribute | None,
        loc: max._mlir.ir.Location | None = None,
    ) -> UnrankedMemRefType:
        """Create a unranked memref type"""

    @property
    def memory_space(self) -> Attribute | None:
        """Returns the memory space of the given Unranked MemRef type."""

class TupleType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @overload
    @staticmethod
    def get_tuple(
        elements: Sequence[Type], context: max._mlir.ir.Context | None = None
    ) -> TupleType:
        """Create a tuple type"""

    @overload
    @staticmethod
    def get_tuple(
        elements: Sequence[Type], context: mlir.ir.Context | None = None
    ) -> TupleType: ...
    def get_type(self, pos: int) -> Type:
        """Returns the pos-th type in the tuple type."""

    @property
    def num_types(self) -> int:
        """Returns the number of types contained in a tuple."""

class FunctionType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @overload
    @staticmethod
    def get(
        inputs: Sequence[Type],
        results: Sequence[Type],
        context: max._mlir.ir.Context | None = None,
    ) -> FunctionType:
        """Gets a FunctionType from a list of input and result types"""

    @overload
    @staticmethod
    def get(
        inputs: Sequence[Type],
        results: Sequence[Type],
        context: mlir.ir.Context | None = None,
    ) -> FunctionType: ...
    @property
    def inputs(self) -> list:
        """Returns the list of input types in the FunctionType."""

    @property
    def results(self) -> list:
        """Returns the list of result types in the FunctionType."""

class OpaqueType(Type):
    def __init__(self, cast_from_type: Type) -> None: ...
    @staticmethod
    def isinstance(other: Type) -> bool: ...

    static_typeid: TypeID = ...
    """static_typeid(/) -> TypeID"""

    @property
    def typeid(self) -> TypeID: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def get(
        dialect_namespace: str,
        buffer: str,
        context: max._mlir.ir.Context | None = None,
    ) -> OpaqueType:
        """Create an unregistered (opaque) dialect type."""

    @property
    def dialect_namespace(self) -> str:
        """Returns the dialect namespace for the Opaque type as a string."""

    @property
    def data(self) -> str:
        """Returns the data for the Opaque type as a string."""
