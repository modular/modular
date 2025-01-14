# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


@register_passable
struct BoxedInt:
    var value: Index

    @implicit
    fn __init__(out self, value: Index):
        self.value = value

    fn __copyinit__(out self, existing: Self):
        self.value = existing.value

    fn boxedAdd(self, rhs: Index) -> Index:
        return __mlir_op.`index.add`(self.value, rhs)


struct Param[T: AnyTrivialRegType]:
    pass


# CHECK-LABEL: lit.fn @"capturing_in_struct
# CHECK-SAME: capturing -> !kgen.none
fn capturing_in_struct[x: Param[fn () capturing -> Index]]():
    pass


# CHECK-LABEL: lit.struct.decl @CapturingMember
struct CapturingMember[f: fn () capturing -> None]:
    # CHECK-LABEL: lit.fn @"member
    # CHECK-SAME: capturing -> !kgen.none attributes
    fn member(self):
        pass

    # CHECK-LABEL: lit.fn @"static_method
    # CHECK-SAME: capturing -> !kgen.none attributes
    @staticmethod
    fn static_method():
        pass


fn makeClosure[p: Index](x: Index) -> Index:
    var z = __mlir_op.`index.add`(x, x)

    # CHECK: [[COPY_VAL:%.*]] = lit.ref.load %z : <index, mut *"z`">
    # CHECK: %index = kgen.param.constant = <p>
    @__copy_capture(z, p)
    @parameter
    fn writer() -> Index:
        # CHECK: lit.return [[COPY_VAL]] : index
        return z

    return writer()


fn foo():
    var x = `3`
    var y = `2`
    _ = makeClosure[`3`](x)
