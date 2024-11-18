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

    @implicit
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


@op
fn foop():
    pass


fn isa[T: AnyTrivialRegType, //, func: T](op: Operation) -> Bool:
    alias name: StringLiteral = "custom." + get_linkage_name[func]()
    return str(op.name()) == name


fn test_pattern(
    inout op: Operation, inout b: Rewriter, inout mr: ModRefAnalysis
) -> Bool:
    closure = op.operand(0)
    alloc = closure._defining_op()
    nextOps = mr.get_next_modref(alloc)
    if len(nextOps) != 1:
        return False
    store = nextOps[0]
    gep = store.operand(1)._defining_op()
    if str(gep.name()) != "kgen.struct.gep":
        return False
    if str(gep.get_inherent_attr("index")) != "3 : index":
        return False
    if gep.operand(0) != alloc.result(0):
        return False
    stage_closure = store.operand(0)._defining_op()
    try:
        first = stage_closure.region(0).first_block().first_operation()
        if isa[foop](first):
            return False
        b.set_insertion_point_before(first)
        _ = b.insert(Op[foop](op.location(), List[Value](), List[Type]()))
        return True
    except e:
        print(e)
        return False


fn print_me(
    inout op: Operation, inout b: Rewriter, inout mr: ModRefAnalysis
) -> Bool:
    p = op
    while str(p.name()) != "builtin.module":
        p = p.parent()
    # print(p)

    mr.rauw(List[Operation](op), List[Operation]())
    b.erase_op(op)
    return True


@op(print_me)
fn lol():
    pass


@no_inline
@op(test_pattern)
fn take_closure(f: fn () escaping -> Int) -> Int:
    return f()


# CHECK-LABEL: kgen.func @test_closure_0
# CHECK-NEXT: call @__op_inst_0


# CHECK-LABEL: kgen.func export @test
@export
fn test() -> Int:
    lol()

    var x = 1

    fn closure() -> Int:
        return x

    return take_closure(closure)
