# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


@register_passable
struct BoxedInt:
    var value: int

    fn __init__(inout self, value: int):
        self.value = value

    fn __copyinit__(inout self, existing: Self):
        self.value = existing.value

    fn boxedAdd(self, rhs: int) -> int:
        return __mlir_op.`index.add`(self.value, rhs)


struct Param[T: AnyTrivialRegType]:
    pass


# CHECK-LABEL: lit.func @"capturing_in_struct
# CHECK-SAME: capturing -> !kgen.none
fn capturing_in_struct[x: Param[fn () capturing -> int]]():
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


fn makeClosure[p: int](x: int) -> int:
    var z = __mlir_op.`index.add`(x, x)

    # CHECK: [[COPY_VAL:%.*]] = lit.ref.load %z : <index, mut *"z`">
    # CHECK: %index = kgen.param.constant = <p>
    @__copy_capture(z, p)
    @parameter
    fn formatter() -> int:
        # CHECK: lit.return [[COPY_VAL]] : index
        return z

    return formatter()


fn foo():
    var x = `3`
    var y = `2`
    _ = makeClosure[`3`](x)
