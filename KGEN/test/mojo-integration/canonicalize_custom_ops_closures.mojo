# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate -S %s | FileCheck %s

import _mlir
from _mlir import (
    Operation,
    Rewriter,
    Value,
    Type,
    Location,
    NamedAttribute,
    Attribute,
    Identifier,
)
from compile.reflection import get_linkage_name
from collections import Optional
from sys import external_call
from memory import UnsafePointer


fn Op[
    T: AnyTrivialRegType, //, func: T
](
    loc: Location,
    operands: List[Value],
    results: List[Type],
    params: Optional[Attribute] = None,
) -> Operation:
    alias name: StringLiteral = "custom." + get_linkage_name[func]()
    var attrs = List[NamedAttribute]()
    if params:
        attrs.append(
            NamedAttribute(
                Identifier(loc.context(), "params"), params.unsafe_value()
            )
        )
    return Operation(
        name, loc, operands=operands, results=results, attributes=attrs
    )


@register_passable("trivial")
struct OpArrayRef:
    var ptr: UnsafePointer[Operation]
    var size: Int

    fn __init__(out self, list: List[Operation]):
        self.ptr = list.unsafe_ptr()
        self.size = len(list)


@register_passable("trivial")
struct ModRefAnalysis:
    var impl: UnsafePointer[NoneType]

    fn rauw(inout self, src: List[Operation], dst: List[Operation]):
        external_call["mlirCMRAnalysisRAUW", NoneType](
            UnsafePointer.address_of(self), OpArrayRef(src), OpArrayRef(dst)
        )

    fn rauw(inout self, src: Operation, dst: Operation):
        srcOps = List[Operation]()
        srcOps.append(src)
        dstOps = List[Operation]()
        dstOps.append(dst)
        self.rauw(srcOps, dstOps)

    fn get_next_modref(inout self, op: Operation) -> List[Operation]:
        size = external_call["mlirCMRAnalysisGetNextModRefCount", Int](
            UnsafePointer.address_of(self), op
        )
        ptr = UnsafePointer[Operation].alloc(size)
        external_call["mlirCMRAnalysisGetNextModRefValues", NoneType](
            UnsafePointer.address_of(self), op, ptr
        )
        result = List[Operation]()
        for i in range(size):
            result.append(ptr[i])
        ptr.free()
        return result^

    fn get_prev_modref(inout self, op: Operation) -> List[Operation]:
        size = external_call["mlirCMRAnalysisGetPrevModRefCount", Int](
            UnsafePointer.address_of(self), op
        )
        ptr = UnsafePointer[Operation].alloc(size)
        external_call["mlirCMRAnalysisGetPrevModRefValues", NoneType](
            UnsafePointer.address_of(self), op, ptr
        )
        result = List[Operation]()
        for i in range(size):
            result.append(ptr[i])
        ptr.free()
        return result^

    fn get_symtab(inout self) -> _mlir._c.IR.MlirSymbolTable:
        return external_call[
            "mlirCMRAnalysisGetSymbolTable", _mlir._c.IR.MlirSymbolTable
        ](UnsafePointer.address_of(self))

    fn lookup_function(inout self, attr: Attribute) -> Operation:
        symbol = external_call[
            "mlirSymbolConstantGetSymbolRef", _mlir._c.IR.MlirAttribute
        ](attr.c)
        strref = _mlir._c.BuiltinAttributes.mlirSymbolRefAttrGetRootReference(
            symbol
        )
        return _mlir._c.IR.mlirSymbolTableLookup(self.get_symtab(), strref)

    fn insert_function(inout self, op: Operation):
        _ = _mlir._c.IR.mlirSymbolTableInsert(self.get_symtab(), op.c)


fn test_pattern(
    inout op: Operation, inout b: Rewriter, inout mr: ModRefAnalysis
) -> Bool:
    params = op.get_discardable_attr("params")
    attr = Attribute(
        _mlir._c.BuiltinAttributes.mlirArrayAttrGetElement(params.c, 0)
    )
    func = mr.lookup_function(attr)
    b.set_insertion_point_after(func)
    copy = b.clone(func)
    mr.insert_function(copy)
    strref = _mlir._c.BuiltinAttributes.mlirStringAttrGetValue(
        copy.get_inherent_attr("sym_name").c
    )
    attr_str = (
        String('#kgen.symbol.constant<@"')
        + strref
        + '"> : !kgen.signature<() capturing -> index>'
    )
    var symbolcst: Attribute
    try:
        symbolcst = Attribute.parse(op.context(), attr_str)
    except e:
        print(e)
        return False

    operands = List[Value]()
    for i in range(op.num_operands()):
        operands.append(op.operand(i))
    result_types = List[Type]()
    for i in range(op.num_results()):
        result_types.append(op.result(i).type())

    paramstr = symbolcst.c
    params_attr = _mlir._c.BuiltinAttributes.mlirArrayAttrGet(
        op.context().c, 1, UnsafePointer.address_of(paramstr)
    )
    new_op = Op[take_closure2](
        op.location(), operands, result_types, Attribute(params_attr)
    )
    b.set_insertion_point_after(op)
    _ = b.insert(new_op)
    mr.rauw(List[Operation](op), List[Operation]())

    try:
        b.set_insertion_point_to_start(copy.region(0).first_block())
        var result_types = List[Type]()
        for i in range(copy.region(0).first_block().num_arguments()):
            result_types.append(copy.region(0).first_block().argument(i).type())
        var attrs = List[NamedAttribute]()
        attrs.append(
            NamedAttribute(
                Identifier(op.context(), "name"),
                Attribute(
                    _mlir._c.BuiltinAttributes.mlirStringAttrGet(
                        op.context().c, "__op_inst_capture"
                    )
                ),
            )
        )
        load = Operation(
            "pop.compiler.global_load",
            op.location(),
            results=result_types,
            attributes=attrs,
        )
        copy.set_inherent_attr(
            "signature",
            Attribute.parse(
                op.context(), "!kgen.signature<() capturing -> index>"
            ),
        )
        copy.region(0).first_block().argument(0).replace_all_uses_with(
            load.result(0)
        )
        _mlir._c.IR.mlirBlockEraseArgument(copy.region(0).first_block().c, 0)
        _ = b.insert(load)
    except e:
        print(e)

    b.replace_op_with(op, new_op)

    return True


@no_inline
@op(test_pattern)
fn take_closure[f: fn () capturing -> Int]() -> Int:
    return f()


@no_inline
@op
fn take_closure2[f: fn () capturing -> Int]() -> Int:
    return f()


# CHECK-LABEL: kgen.func export @test
@export
fn test(x: Int) -> Int:
    @no_inline
    @parameter
    fn foo() -> Int:
        return x

    # CHECK-NEXT: %0 = kgen.call @__op_inst_0(%arg0)
    # CHECK-NEXT: return %0
    return take_closure[foo]()


# CHECK: kgen.func @__op_inst_0(%arg0: index) capturing -> index
# CHECK-NEXT: %0 = kgen.call {{.*}}take_closure2{{.*}}(%arg0)
# CHECK-NEXT: return %0


# CHECK: kgen.func @{{.*}}take_closure2{{.*}}(%arg0: index) capturing -> index
# CHECK-NEXT: %0 = kgen.call @"test_foo()_0"(%arg0)
# CHECK-NEXT: return %0
