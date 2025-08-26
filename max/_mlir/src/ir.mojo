# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


from collections.optional import Optional
from collections.string import StaticString


from utils import Variant

import ._c
import ._c.IR
from ._c.ffi import MLIR_func
from .diagnostics import (
    Diagnostic,
    DiagnosticHandler,
    ErrorCapturingDiagnosticHandler,
)

# Ownership:
#
#   See https://mlir.llvm.org/docs/Bindings/Python/#ownership-in-the-core-ir
#   for full ownership semantics. We'll attempt to follow the same patterns
#   for consistency.
#
# - Context owns most things, we don't need to memory manage them
# - Most things are value-semantic and are actually implicitly mutable references
#   with the same lifetime as the owning Context
# - Context ownership is therefore the main thing people might need to consider
# - Exceptions:
#   - When objects are created without a context, they're often "owned" objects
#     _until_ they've been added to another context-owned object.


trait DialectType:
    fn to_mlir(self) -> Type:
        ...

    @staticmethod
    fn from_mlir(type: Type) raises -> Self:
        ...


trait DialectAttribute:
    fn to_mlir(self) -> Attribute:
        ...

    @staticmethod
    fn from_mlir(attr: Attribute) raises -> Self:
        ...


struct DialectRegistry(Defaultable):
    alias cType = _c.IR.MlirDialectRegistry
    var c: Self.cType

    fn __init__(out self):
        self.c = _c.IR.mlirDialectRegistryCreate()

    fn __del__(deinit self):
        # We only want to do this for objects which are not added to a context
        pass  # _c.IR.mlirDialectRegistryDestroy(self.c)

    fn insert(mut self, handle: DialectHandle):
        _c.IR.mlirDialectHandleInsertDialect(handle.c, self.c)

    fn load_modular_dialects(self):
        MLIR_func["MAXG_loadModularDialects", NoneType._mlir_type](self.c)


@register_passable("trivial")
struct Dialect(Copyable, Movable):
    alias cType = _c.IR.MlirDialect
    var c: Self.cType

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn context(self) -> Context:
        return _c.IR.mlirDialectGetContext(self.c)

    fn __eq__(self, other: Self) -> Bool:
        return _c.IR.mlirDialectEqual(self.c, other.c)

    fn namespace(self) -> StaticString:
        return _c.IR.mlirDialectGetNamespace(self.c)


@fieldwise_init
@register_passable("trivial")
struct DialectHandle(Copyable, Movable):
    alias cType = _c.IR.MlirDialectHandle
    var c: Self.cType

    fn namespace(self) -> StaticString:
        return _c.IR.mlirDialectHandleGetNamespace(self.c)


@register_passable("trivial")
struct Context(Copyable, Defaultable, Movable):
    alias cType = _c.IR.MlirContext
    var c: Self.cType

    fn __init__(out self):
        self.c = _c.IR.mlirContextCreateWithThreading(False)

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    @implicit
    fn __init__(out self, threading_enabled: Bool):
        self.c = _c.IR.mlirContextCreateWithThreading(threading_enabled)

    fn __init__(
        out self, var registry: DialectRegistry, threading_enabled: Bool
    ):
        self.c = _c.IR.mlirContextCreateWithRegistry(
            registry.c, threading_enabled
        )

    fn __enter__(mut self) -> Self:
        return self

    fn __exit__(mut self):
        _c.IR.mlirContextDestroy(self.c)

    fn __exit__(self, err: Error) -> Bool:
        _c.IR.mlirContextDestroy(self.c)
        return False

    fn __eq__(self, other: Self) -> Bool:
        return _c.IR.mlirContextEqual(self.c, other.c)

    fn append(mut self, var registry: DialectRegistry):
        return _c.IR.mlirContextAppendDialectRegistry(self.c, registry.c)

    fn register(mut self, handle: DialectHandle):
        _c.IR.mlirDialectHandleRegisterDialect(handle.c, self.c)

    fn load(self, handle: DialectHandle) -> Dialect:
        return _c.IR.mlirDialectHandleLoadDialect(handle.c, self.c)

    fn load_modular_dialects(mut self):
        var registry = DialectRegistry()
        registry.load_modular_dialects()
        self.append(registry^)

    fn allow_unregistered_dialects(mut self, allow: Bool = True):
        _c.IR.mlirContextSetAllowUnregisteredDialects(self.c, allow)

    fn allows_unregistered_dialects(self) -> Bool:
        return _c.IR.mlirContextGetAllowUnregisteredDialects(self.c)

    fn num_registered_dialects(self) -> Int:
        return _c.IR.mlirContextGetNumRegisteredDialects(self.c)

    fn num_loaded_dialects(self) -> Int:
        return _c.IR.mlirContextGetNumLoadedDialects(self.c)

    fn get_or_load_dialect(self, dialect_name: String) -> Optional[Dialect]:
        var result = _c.IR.mlirContextGetOrLoadDialect(
            self.c,
            StaticString(
                ptr=dialect_name.unsafe_ptr(), length=len(dialect_name)
            ),
        )
        return Optional(Dialect(result)) if result.ptr else None

    fn enable_multithreading(mut self, enable: Bool = True):
        _c.IR.mlirContextEnableMultithreading(self.c, enable)

    fn load_all_available_dialects(mut self):
        _c.IR.mlirContextLoadAllAvailableDialects(self.c)

    fn is_registered_operation(self, opname: String) -> Bool:
        var result = _c.IR.mlirContextIsRegisteredOperation(
            self.c, StaticString(ptr=opname.unsafe_ptr(), length=len(opname))
        )
        return result

    fn print_on_diagnostic(self):
        fn print_diagnostic(diagnostic: Diagnostic) -> Bool:
            print(diagnostic)
            return False

        # For now no way to detach
        _ = DiagnosticHandler[print_diagnostic].attach(self)

    fn diagnostic_error(self) -> ErrorCapturingDiagnosticHandler:
        """Uses a DiagnosticHandler to capture errors from MLIR.
        If the handler catches an error, it will re-raise with the error
        message provided by MLIR.
        This will drop any information in the original error message.
        """
        return ErrorCapturingDiagnosticHandler(self)

    # TODO: mlirContextSetThreadPool


@register_passable("trivial")
struct Location(Copyable, Movable, Stringable, Writable):
    alias cType = _c.IR.MlirLocation
    var c: Self.cType

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn __init__(out self, ctx: Context, filename: String, line: Int, col: Int):
        self.c = _c.IR.mlirLocationFileLineColGet(
            ctx.c,
            StaticString(ptr=filename.unsafe_ptr(), length=len(filename)),
            line,
            col,
        )

    @staticmethod
    fn from_attribute(attr: Attribute) -> Self:
        return Self(_c.IR.mlirLocationFromAttribute(attr.c))

    @staticmethod
    fn call_site(callee: Self, caller: Self) -> Self:
        return Self(_c.IR.mlirLocationCallSiteGet(callee.c, caller.c))

    # TODO: locationFusedGet, locationNameGet

    @staticmethod
    fn unknown(ctx: Context) -> Self:
        return Self(_c.IR.mlirLocationUnknownGet(ctx.c))

    fn attribute(self) -> Attribute:
        return Attribute(_c.IR.mlirLocationGetAttribute(self.c))

    fn context(self) -> Context:
        return _c.IR.mlirLocationGetContext(self.c)

    fn __eq__(self, other: Self) -> Bool:
        return _c.IR.mlirLocationEqual(self.c, other.c)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        _c.IR.mlirLocationPrint(writer, self.c)


@register_passable("trivial")
struct Module(Copyable, Movable, Stringable, Writable):
    alias cType = _c.IR.MlirModule
    var c: Self.cType

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    @implicit
    fn __init__(out self, location: Location):
        self.c = _c.IR.mlirModuleCreateEmpty(location.c)

    # TODO: The lifetime of module appears to be iffy in the current codebase.
    # For now, this is manually called when known to be safe to prevent ASAN
    # from complaining for certain tests.
    fn destroy(var self):
        _c.IR.mlirModuleDestroy(self.c)

    @staticmethod
    fn parse(ctx: Context, module: String) -> Self:
        # TODO: how can this fail?
        var c = _c.IR.mlirModuleCreateParse(
            ctx.c, StaticString(ptr=module.unsafe_ptr(), length=len(module))
        )
        return Self(c)

    @staticmethod
    fn from_op(module_op: Operation) raises -> Self:
        var module = _c.IR.mlirModuleFromOperation(module_op.c)
        if not module.ptr:
            raise "Op must be a ModuleOp"
        return module

    fn context(self) -> Context:
        return _c.IR.mlirModuleGetContext(self.c)

    fn body(self) -> Block:
        return Block(_c.IR.mlirModuleGetBody(self.c))

    fn as_op(self) -> Operation:
        return Operation(_c.IR.mlirModuleGetOperation(self.c))

    fn debug_str(self, pretty_print: Bool = False) -> String:
        return self.as_op().debug_str(pretty_print)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_op())


# Helper class with a bunch of implicit conversions for things that go on
# Operations.
struct _OpBuilderList[T: Copyable & Movable](Defaultable):
    var elements: List[T]

    fn __init__(out self):
        self.elements = []

    @implicit
    fn __init__(out self, var elements: List[T]):
        self.elements = elements^

    @implicit
    fn __init__(out self, element: T):
        self.elements = []
        self.elements.append(element)

    fn __bool__(self) -> Bool:
        return len(self.elements).__bool__()


@fieldwise_init
struct NamedAttribute(Copyable, Movable):
    alias cType = _c.IR.MlirNamedAttribute
    var name: Identifier
    var attr: Attribute

    @implicit
    fn __init__(out self, attr: Self.cType):
        self.name = Identifier(attr.name)
        self.attr = Attribute(attr.attribute)

    fn c(self) -> Self.cType:
        return Self.cType(self.name.c, self.attr.c)

    # TODO: tuple init so we can write these a bit less verbosely.


@fieldwise_init
struct _WriteState(Copyable, Movable):
    var handle: UnsafePointer[FileHandle]
    var errors: List[String]


# TODO: how to correctly destroy "owned" Operations?
@register_passable("trivial")
struct Operation(Copyable, Movable, Stringable, Writable):
    alias cType = _c.IR.MlirOperation
    var c: Self.cType

    @implicit
    fn __init__(out self, op: Self.cType):
        self.c = op

    fn __init__(
        out self,
        name: String,
        location: Location,
        *,
        attributes: _OpBuilderList[NamedAttribute] = _OpBuilderList[
            NamedAttribute
        ](),
        operands: _OpBuilderList[Value] = _OpBuilderList[Value](),
        results: _OpBuilderList[Type] = _OpBuilderList[Type](),
        regions: _OpBuilderList[Region] = _OpBuilderList[Region](),
        successors: _OpBuilderList[Block] = _OpBuilderList[Block](),
    ):
        var state = _c.IR.mlirOperationStateGet(
            StaticString(ptr=name.unsafe_ptr(), length=len(name)), location.c
        )
        Self._init_op_state(
            state,
            attributes.elements,
            operands.elements,
            results.elements,
            regions.elements,
            successors.elements,
        )
        self.c = _c.IR.mlirOperationCreate(UnsafePointer(to=state))

    fn __init__(
        out self,
        name: String,
        location: Location,
        *,
        enable_result_type_inference: Bool,
        attributes: _OpBuilderList[NamedAttribute] = _OpBuilderList[
            NamedAttribute
        ](),
        operands: _OpBuilderList[Value] = _OpBuilderList[Value](),
        results: _OpBuilderList[Type] = _OpBuilderList[Type](),
        regions: _OpBuilderList[Region] = _OpBuilderList[Region](),
        successors: _OpBuilderList[Block] = _OpBuilderList[Block](),
    ) raises:
        var state = _c.IR.mlirOperationStateGet(
            StaticString(ptr=name.unsafe_ptr(), length=len(name)), location.c
        )
        Self._init_op_state(
            state,
            attributes.elements,
            operands.elements,
            results.elements,
            regions.elements,
            successors.elements,
        )
        if enable_result_type_inference:
            _c.IR.mlirOperationStateEnableResultTypeInference(
                UnsafePointer(to=state)
            )

        var result: Self.cType
        with location.context().diagnostic_error():
            result = _c.IR.mlirOperationCreate(UnsafePointer(to=state))
            if not result.ptr:
                raise "operation create failed"

        self.c = result

    @staticmethod
    fn _init_op_state(
        ref state: _c.IR.MlirOperationState,
        attributes: List[NamedAttribute],
        operands: List[Value],
        results: List[Type],
        regions: List[Region],
        successors: List[Block],
    ):
        if attributes:
            _c.IR.mlirOperationStateAddAttributes(
                UnsafePointer(to=state),
                len(attributes),
                # This technically works as long as `Attribute` is only `MlirAttribute`.
                attributes.unsafe_ptr().bitcast[NamedAttribute.cType](),
            )
        if operands:
            _c.IR.mlirOperationStateAddOperands(
                UnsafePointer(to=state),
                len(operands),
                operands.unsafe_ptr().bitcast[Value.cType](),
            )
        if results:
            _c.IR.mlirOperationStateAddResults(
                UnsafePointer(to=state),
                len(results),
                results.unsafe_ptr().bitcast[Type.cType](),
            )
        # TODO: how to express to the caller that we're taking ownership
        #       over Regions.
        if regions:
            _c.IR.mlirOperationStateAddOwnedRegions(
                UnsafePointer(to=state),
                len(regions),
                regions.unsafe_ptr().bitcast[Region.cType](),
            )
        if successors:
            _c.IR.mlirOperationStateAddSuccessors(
                UnsafePointer(to=state),
                len(successors),
                successors.unsafe_ptr().bitcast[Block.cType](),
            )

    @staticmethod
    fn parse(ctx: Context, source: String, source_name: String) raises -> Self:
        var result = _c.IR.mlirOperationCreateParse(
            ctx.c,
            StaticString(ptr=source.unsafe_ptr(), length=len(source)),
            StaticString(ptr=source_name.unsafe_ptr(), length=len(source_name)),
        )
        if not result.ptr:
            raise "Operation.parse failed"
        return Self(result)

    fn destroy(var self):
        _c.IR.mlirOperationDestroy(self.c)

    fn context(self) -> Context:
        return _c.IR.mlirOperationGetContext(self.c)

    fn location(self) -> Location:
        return _c.IR.mlirOperationGetLocation(self.c)

    fn verify(self) -> Bool:
        return _c.IR.mlirOperationVerify(self.c)

    fn write[
        origin: MutableOrigin
    ](
        self, ref [origin]file: FileHandle, version: Optional[Int64] = None
    ) raises:
        if not file.handle:
            raise "Writing op bytecode to file failed: invalid file handle"

        var config = _c.IR.mlirBytecodeWriterConfigCreate()
        if version:
            _c.IR.mlirBytecodeWriterConfigDesiredEmitVersion(
                config, version.value()
            )

        var result = _c.IR.mlirOperationWriteBytecodeWithConfig(
            file,
            self.c,
            config,
        )
        _c.IR.mlirBytecodeWriterConfigDestroy(config)

        if result.value == 0:
            raise "Writing op bytecode to file failed"

    fn name(self) -> Identifier:
        return _c.IR.mlirOperationGetName(self.c)

    fn block(self) -> Block:
        return _c.IR.mlirOperationGetBlock(self.c)

    fn parent(self) -> Self:
        return _c.IR.mlirOperationGetParentOperation(self.c)

    fn successor(self, successor_idx: Int) raises -> Block:
        var block = _c.IR.mlirOperationGetSuccessor(self.c, successor_idx)
        if not block.ptr:
            raise Error("IndexError")
        return block

    fn region(self, region_idx: Int) raises -> Region:
        var region = _c.IR.mlirOperationGetRegion(self.c, region_idx)
        if not region.ptr:
            raise Error("IndexError")
        return region

    fn num_results(self) -> Int:
        return _c.IR.mlirOperationGetNumResults(self.c)

    fn result(self, idx: Int) -> Value:
        return _c.IR.mlirOperationGetResult(self.c, idx)

    fn num_operands(self) -> Int:
        return _c.IR.mlirOperationGetNumOperands(self.c)

    fn operand(self, idx: Int) -> Value:
        return _c.IR.mlirOperationGetOperand(self.c, idx)

    fn set_inherent_attr(mut self, name: String, attr: Attribute):
        _c.IR.mlirOperationSetInherentAttributeByName(
            self.c,
            StaticString(ptr=name.unsafe_ptr(), length=len(name)),
            attr.c,
        )

    fn get_inherent_attr(self, name: String) -> Attribute:
        var result = _c.IR.mlirOperationGetInherentAttributeByName(
            self.c, StaticString(ptr=name.unsafe_ptr(), length=len(name))
        )
        return result

    fn set_discardable_attr(mut self, name: String, attr: Attribute):
        _c.IR.mlirOperationSetDiscardableAttributeByName(
            self.c,
            StaticString(ptr=name.unsafe_ptr(), length=len(name)),
            attr.c,
        )

    fn get_discardable_attr(self, name: String) -> Attribute:
        var result = _c.IR.mlirOperationGetDiscardableAttributeByName(
            self.c, StaticString(ptr=name.unsafe_ptr(), length=len(name))
        )
        return result

    fn debug_str(self, pretty_print: Bool = False) -> String:
        var flags = _c.IR.mlirOpPrintingFlagsCreate()
        _c.IR.mlirOpPrintingFlagsEnableDebugInfo(flags, True, pretty_print)
        var result = String()
        _c.IR.mlirOperationPrintWithFlags(
            result,
            self.c,
            flags,
        )
        return result

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        _c.IR.mlirOperationPrint(writer, self.c)


@register_passable("trivial")
struct Identifier(Copyable, Movable, Stringable):
    alias cType = _c.IR.MlirIdentifier
    var c: Self.cType

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn __init__(out self, ctx: Context, identifier: String):
        self.c = _c.IR.mlirIdentifierGet(
            ctx.c,
            StaticString(ptr=identifier.unsafe_ptr(), length=len(identifier)),
        )

    fn __str__(self) -> String:
        return String(_c.IR.mlirIdentifierStr(self.c))


@register_passable("trivial")
struct Type(Copyable, Movable, Stringable, Writable):
    alias cType = _c.IR.MlirType
    var c: Self.cType

    @implicit
    fn __init__[T: DialectType](out self, type: T):
        self = type.to_mlir()

    @implicit
    fn __init__(out self, type: Self.cType):
        self.c = type

    @staticmethod
    fn parse(ctx: Context, s: String) raises -> Self:
        var result = _c.IR.mlirTypeParseGet(
            ctx.c, StaticString(ptr=s.unsafe_ptr(), length=len(s))
        )
        if not result.ptr:
            raise "Failed to parse type: " + s
        return result

    fn context(self) -> Context:
        return _c.IR.mlirTypeGetContext(self.c)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        _c.IR.mlirTypePrint(writer, self.c)


@register_passable("trivial")
struct Value(Copyable, Movable, Stringable, Writable):
    alias cType = _c.IR.MlirValue
    var c: Self.cType

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn type(self) -> Type:
        return _c.IR.mlirValueGetType(self.c)

    fn context(self) -> Context:
        return self.type().context()

    fn parent(self) -> Variant[Block, Operation]:
        if self.is_block_argument():
            return self._block()
        else:
            debug_assert(self.is_op_result(), "Invalid Value state")
            return self._defining_op()

    fn is_block_argument(self) -> Bool:
        return _c.IR.mlirValueIsABlockArgument(self.c)

    fn is_op_result(self) -> Bool:
        return _c.IR.mlirValueIsAOpResult(self.c)

    fn set_type(mut self, type: Type):
        return _c.IR.mlirValueSetType(self.c, type.c)

    fn _block(self) -> Block:
        return _c.IR.mlirBlockArgumentGetOwner(self.c)

    fn _defining_op(self) -> Operation:
        return _c.IR.mlirOpResultGetOwner(self.c)

    fn replace_all_uses_with(self, other: Self):
        _c.IR.mlirValueReplaceAllUsesOfWith(of=self.c, `with`=other.c)

    fn __eq__(self, other: Self) -> Bool:
        return _c.IR.mlirValueEqual(self.c, other.c)

    fn __ne__(self, other: Self) -> Bool:
        return not _c.IR.mlirValueEqual(self.c, other.c)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        _c.IR.mlirValuePrint(writer, self.c)


@register_passable("trivial")
struct Attribute(Copyable, Movable, Stringable, Writable):
    alias cType = _c.IR.MlirAttribute
    var c: Self.cType

    @implicit
    fn __init__[T: DialectAttribute](out self, attr: T):
        self = attr.to_mlir()

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn context(self) -> Context:
        return _c.IR.mlirAttributeGetContext(self.c)

    @staticmethod
    fn parse(ctx: Context, attr: String) raises -> Self:
        var result = _c.IR.mlirAttributeParseGet(
            ctx.c, StaticString(ptr=attr.unsafe_ptr(), length=len(attr))
        )
        if not result.ptr:
            raise "Failed to parse attribute:" + attr
        return result

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        _c.IR.mlirAttributePrint(writer, self.c)


@register_passable("trivial")
struct Block(Copyable, Movable, Stringable, Writable):
    alias cType = _c.IR.MlirBlock
    var c: Self.cType

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    @implicit
    fn __init__(out self, args: List[Type]):
        var locations = List[Location]()
        for i in range(len(args)):
            var ctx = args[i].context()
            locations.append(Location.unknown(ctx))
        self = Self(args, locations)

    fn __init__(
        out self,
        args: List[Type],
        locations: List[Location],
    ):
        debug_assert(
            len(args) == len(locations), "Each arg must have a location"
        )
        self.c = _c.IR.mlirBlockCreate(
            len(args),
            args.unsafe_ptr().bitcast[Type.cType](),
            locations.unsafe_ptr().bitcast[Location.cType](),
        )

    fn region(self) -> Region:
        return _c.IR.mlirBlockGetParentRegion(self.c)

    fn parent(self) -> Operation:
        return _c.IR.mlirBlockGetParentOperation(self.c)

    fn first_operation(self) -> Operation:
        return _c.IR.mlirBlockGetFirstOperation(self.c)

    fn num_arguments(self) -> Int:
        return _c.IR.mlirBlockGetNumArguments(self.c)

    fn argument(self, idx: Int) -> Value:
        return _c.IR.mlirBlockGetArgument(self.c, idx)

    fn append(self, op: Operation):
        return _c.IR.mlirBlockAppendOwnedOperation(self.c, op.c)

    fn insert_before(self, reference: Operation, op: Operation):
        return _c.IR.mlirBlockInsertOwnedOperationBefore(
            self.c, reference.c, op.c
        )

    fn insert_after(self, reference: Operation, op: Operation):
        return _c.IR.mlirBlockInsertOwnedOperationAfter(
            self.c, reference.c, op.c
        )

    fn terminator(self) -> Optional[Operation]:
        var op = _c.IR.mlirBlockGetTerminator(self.c)
        return Optional(Operation(op)) if op.ptr else None

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        _c.IR.mlirBlockPrint(writer, self.c)


@register_passable("trivial")
struct Region(Copyable, Defaultable, Movable):
    alias cType = _c.IR.MlirRegion
    var c: Self.cType

    fn __init__(out self):
        self.c = _c.IR.mlirRegionCreate()

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn append(self, block: Block):
        _c.IR.mlirRegionAppendOwnedBlock(self.c, block.c)

    fn first_block(self) raises -> Block:
        var block = _c.IR.mlirRegionGetFirstBlock(self.c)
        if not block.ptr:
            raise "Region has no block"
        return block
