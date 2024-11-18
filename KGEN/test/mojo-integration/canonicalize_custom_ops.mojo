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


fn isa[T: AnyTrivialRegType, //, func: T](op: Operation) -> Bool:
    alias name: StringLiteral = "custom." + get_linkage_name[func]()
    return str(op.name()) == name


@op
fn mul_two[T: AnyType](x: Int32) -> Int32:
    return x * 2


@op(add_mul_two[1])
fn add[T: AnyType](x: Int32, y: Int32) -> Int32:
    return x + y


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


fn add_mul_two[
    x: Int
](inout op: Operation, inout b: Rewriter, inout mr: ModRefAnalysis) -> Bool:
    var loc = op.location()
    var lhs = op.operand(0)
    var rhs = op.operand(1)
    if lhs != rhs:
        return True

    var new_op = Op[mul_two](
        loc,
        operands=List[Value](lhs),
        results=List[Type](op.result(0).type()),
        params=op.get_discardable_attr("params"),
    )
    _ = b.insert(new_op)
    mr.rauw(op, new_op)
    b.replace_op_with(op, new_op)
    return True


# CHECK-LABEL: kgen.func export @top
@export
fn top(x: Int32) -> Int32:
    # CHECK-NEXT: call @__op_inst_0(%arg0)
    return add[Int32](x, x)


fn unused_str(
    inout op: Operation, inout b: Rewriter, inout mr: ModRefAnalysis
) -> Bool:
    nextOps = mr.get_next_modref(op)
    prevOps = mr.get_prev_modref(op)
    if len(nextOps) != 0:
        return False
    if len(prevOps) != 1:
        return False
    if not isa[UnusedStr.__init__](prevOps[0]):
        return False
    repls = List[Operation]()
    repls.append(prevOps[0])
    repls.append(op)
    mr.rauw(repls, List[Operation]())
    b.erase_op(op)
    b.erase_op(prevOps[0])
    return True


struct UnusedStr:
    var ptr: UnsafePointer[Int]
    var size: Int

    @no_inline
    @op
    fn __init__(out self):
        self.ptr = UnsafePointer[Int].alloc(1)
        self.size = 0

    @no_inline
    @op(unused_str)
    fn __del__(owned self):
        self.ptr.free()


# CHECK-LABEL: kgen.func export @test_unused
@export
fn test_unused():
    var unused = UnusedStr()
    # CHECK-NEXT: kgen.return


fn str_attr[T: AnyTrivialRegType, //, value: T]() -> StringLiteral:
    alias result = __mlir_attr[
        `#kgen.param.expr<attr_to_str,`, value, `> : !kgen.string`
    ]
    return result


fn opt_push_pop[
    T: Movable
](inout op: Operation, inout b: Rewriter, inout mr: ModRefAnalysis) -> Bool:
    prevOps = mr.get_prev_modref(op)
    if len(prevOps) != 1:
        return False
    prev = prevOps[0]
    if not isa[Stack.push](prev):
        return False
    alias type_str = str_attr[T]()
    params = op.get_discardable_attr("params")
    attr = Attribute(
        _mlir._c.BuiltinAttributes.mlirArrayAttrGetElement(params.c, 0)
    )
    if str(attr) != type_str:
        return False

    repls = List[Operation]()
    repls.append(prev)
    repls.append(op)

    if op.num_results() == 0:
        var new_op = Op[T.__moveinit__](
            op.location(),
            operands=List[Value](op.operand(1), prev.operand(1)),
            results=List[Type](),
        )
        b.set_insertion_point_before(prev)
        _ = b.insert(new_op)
    mr.rauw(repls, List[Operation]())
    if op.num_results() == 1:
        b.replace_op_with(op, prev.operand(1))
    else:
        b.erase_op(op)
    b.erase_op(prev)
    return True


@value
struct Stack[T: Movable]:
    @no_inline
    @op
    fn push(inout self, owned value: T):
        pass

    @no_inline
    @op(opt_push_pop[T])
    fn pop(inout self) -> T:
        while True:
            pass


@no_inline
fn keep[T: AnyTrivialRegType](value: T):
    external_call["keep", NoneType](value)


@no_inline
fn unused():
    pass


@no_inline
fn capture(inout s: UnusedStr):
    keep(UnsafePointer.address_of(s))


@no_inline
fn use(x: Int):
    keep(x)


# CHECK-LABEL: kgen.func export @test_push_pop
@export
fn test_push_pop():
    var str = UnusedStr()

    var stack = Stack[Int]()

    unused()

    # CHECK-NOT: push(
    stack.push(2)

    capture(str)
    unused()

    # CHECK-NOT: pop(
    use(stack.pop())
    # CHECK: call {{.*}}canonicalize_custom_ops::use{{.*}}(%index2)


@value
struct Thing:
    var x: Int

    @no_inline
    fn __moveinit__(out self, owned existing: Self):
        self.x = existing.x


# CHECK-LABEL: kgen.func export @test_push_pop_nontrivial
@export
fn test_push_pop_nontrivial() -> Thing:
    # CHECK-NEXT: %struct = kgen.param.constant: struct<(index) memoryOnly> = <{ 1 }>
    # CHECK-NEXT: %0 = kgen.call {{.*}}(%struct)
    # CHECK-NEXT: store %0, %arg0
    var stack = Stack[Thing]()

    stack.push(Thing(1))
    return stack.pop()


# CHECK-LABEL: kgen.func @__op_inst_0
# CHECK-NEXT: call [[CALLEE:.*mul_two.*,T=.*"]](%arg0)

# CHECK: kgen.func [[CALLEE]]
# CHECK-NEXT: %simd = kgen.param.constant: scalar<si32> = <2>
# CHECK-NEXT: pop.mul %arg0, %simd
