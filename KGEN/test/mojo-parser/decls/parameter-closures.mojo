# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index
alias AnyRegType = __mlir_type.`!kgen.type`

alias `1` = __mlir_attr.`1 : index`
alias `2` = __mlir_attr.`2 : index`
alias `3` = __mlir_attr.`3 : index`
alias `4` = __mlir_attr.`4 : index`
alias `5` = __mlir_attr.`5 : index`


@register_passable
struct Error:
    pass


# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


fn fat_signature_types():
    let x = `4`

    # CHECK: lit.func *"g(__mlir_type.index)"(%y: index borrow) capturing -> index
    @parameter
    fn g(y: __mlir_type.index) -> __mlir_type.index:
        return x

    # CHECK: lit.func *"h[__mlir_type.index](__mlir_type.index,__mlir_type.index)"<[[N:.*]]>(%y: index borrow, %z: index borrow) capturing -> index
    @parameter
    fn h[
        N: __mlir_type.index
    ](y: __mlir_type.index, z: __mlir_type.index) -> __mlir_type.index:
        return x


fn take_closure(
    g: fn (borrowed __mlir_type.index) capturing -> __mlir_type.index,
    x: __mlir_type.index,
):
    # CHECK: %0 = lit.call_signature %g(%x) : !lit.signature<(index borrow, |) capturing -> index>
    let result = g(x)


fn take_closure_no_param_main():
    let x = `4`

    @parameter
    fn g(y: __mlir_type.index) -> __mlir_type.index:
        return x

    # CHECK: %0 = kgen.create_closure[!lit.signature<("y": index borrow) capturing -> index>: *"g(__mlir_type.index)"]()
    # CHECK: %W = lit.letreg.decl "W" = %0
    # CHECK: %1 = kgen.rebind %W : !kgen.signature<!lit.signature<("y": index borrow) capturing -> index>>
    # CHECK-SAME: to !kgen.signature<!lit.signature<(index borrow, |) capturing -> index>>
    let W = g
    # CHECK: lit.call @{{.*}}::@"take_closure{{.*}}"(%1, %index3) : !lit.signature<("g": !kgen.signature<!lit.signature<(index borrow, |) capturing -> index>> borrow, "x": index borrow) -> !kgen.none>
    take_closure(W, `3`)


fn take_closure_with_param_main():
    let x = `4`
    let w = `5`

    @parameter
    fn h[
        N: __mlir_type.index, M: __mlir_type.index
    ](y: __mlir_type.index) -> __mlir_type.index:
        return x

    @parameter
    fn g[N: __mlir_type.index](y: __mlir_type.index) -> __mlir_type.index:
        return x

    # CHECK: lit.alias.decl [[BOUND:.*]]: !lit.signature<("y": index borrow) capturing -> index> =
    # CHECK-SAME: <bind_signature(:!lit.signature<<"N": index>("y": index borrow) capturing -> index> *"g[__mlir_type.index](__mlir_type.index)", 3)>
    # CHECK: %0 = kgen.create_closure[!lit.signature<("y": index borrow) capturing -> index>: [[BOUND]]]()
    alias Bound = g[`3`]

    # CHECK: %value = lit.letreg.decl "value" = %0
    let value = Bound
    # CHECK: %1 = kgen.rebind %value
    # CHECK: lit.call @{{.*}}::@"take_closure{{.*}}"(%1, %x)
    take_closure(value, x)

    # CHECK: %3 = kgen.create_closure[!lit.signature<("y": index borrow) capturing -> index>: bind_signature(:!lit.signature<<"N": index, "M": index>("y": index borrow) capturing -> index> *"h[__mlir_type.index,__mlir_type.index](__mlir_type.index)", 1, 2)]()
    # CHECK: %Q = lit.letreg.decl "Q" = %3
    let Q = h[`1`, `2`]
    take_closure(Q, x)


fn take_closure_raises(
    g: fn (borrowed __mlir_type.index) raises capturing -> __mlir_type.index,
    x: __mlir_type.index,
) raises:
    # CHECK: %0 = lit.call_signature %g(%x) : !lit.signature<(index borrow, |) throws|capturing|ownedresult -> !kgen.variant<!Error, index>>
    let result = g(x)


fn throws_main() raises:
    let x = `4`

    # CHECK: lit.func *"g(__mlir_type.index)"(%y: index borrow) throws|capturing|ownedresult -> !kgen.variant<!Error, index>
    @parameter
    fn g(y: __mlir_type.index) raises -> __mlir_type.index:
        return x

    let W = g
    take_closure_raises(W, `3`)


@register_passable
struct BoxedInt:
    var value: Int

    fn __init__(value: Int) -> Self:
        return Self {value: value}

    fn __copyinit__(existing: Self) -> Self:
        return Self {value: existing.value}

    fn boxedAdd(self, rhs: Int) -> Int:
        return __mlir_op.`index.add`(self.value, rhs)


fn member_method_reference():
    let x = BoxedInt(`3`)
    # CHECK: %[[SELF:.*]] = lit.call {{.*}}__copyinit__
    # CHECK: %[[C:.*]] = kgen.create_closure[{{.*}}boxedAdd{{.*}}](%[[SELF]])
    # CHECK: lit.letreg.decl "closure" = %[[C]]
    let closure = x.boxedAdd
    # CHECK: %[[CST:.*]] = kgen.param.constant
    # CHECK: call_signature %closure(%[[CST]])
    _ = closure(`2`)


# CHECK-LABEL: lit.func @"capture_by_copy()"
fn capture_by_copy():
    var c: BoxedInt = `2`

    # CHECK: [[TMP:%.*]] = pop.stack_allocation
    # CHECK-NEXT: [[TMPREF:%.*]] = lit.ref.from_pointer [[TMP]]
    # CHECK-NEXT: %[[VAL:.*]] = lit.ref.load %c
    # CHECK-NEXT: %[[COPY:.*]] = lit.call {{.*}}__copyinit__{{.*}}(%[[VAL]])
    # CHECK-NEXT: lit.ref.store %[[COPY]], [[TMPREF]]
    # CHECK-NEXT: %[[RAW:.*]] = lit.ref.load [[TMPREF]]
    # CHECK-NEXT: lit.func *"value_closure
    fn value_closure(x: Int):
        let arg = x
        # CHECK-NEXT: [[STATE:%.*]] = pop.stack_allocation
        # CHECK-NEXT: [[STATEREF:%.*]] = lit.ref.from_pointer [[STATE]]
        # CHECK-NEXT: lit.ref.store %[[RAW]], [[STATEREF]]
        let capture = c

    # CHECK: %[[CLS:.*]] = kgen.create_closure{{.*}}*"value_closure

    # CHECK: call_signature %[[CLS]](
    value_closure(`3`)

    # CHECK: lit.func *"doesnt_capture
    fn doesnt_capture(x: Int):
        let arg = x

    # CHECK: lit.alias.decl {{.*}}f: {{.*}} = <*"doesnt_capture
    alias f = doesnt_capture


struct Param[T: AnyRegType]:
    pass


# CHECK-LABEL: lit.func @"capturing_in_struct
# CHECK-SAME: capturing -> !kgen.none attributes {isParametric
fn capturing_in_struct[x: Param[fn () capturing -> Int]]():
    pass


# CHECK-LABEL: lit.struct.decl @CapturingMember
struct CapturingMember[f: fn () capturing -> None]:
    # CHECK-LABEL: lit.func @"member
    # CHECK-SAME: capturing -> !kgen.none attributes
    fn member(self):
        pass

    # CHECK-LABEL: lit.func @"static_method
    # CHECK-SAME: capturing -> !kgen.none attributes
    @staticmethod
    fn static_method():
        pass
