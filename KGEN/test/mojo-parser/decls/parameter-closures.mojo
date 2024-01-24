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


@register_passable
struct BoxedInt:
    var value: Int

    fn __init__(value: Int) -> Self:
        return Self {value: value}

    fn __copyinit__(existing: Self) -> Self:
        return Self {value: existing.value}

    fn boxedAdd(self, rhs: Int) -> Int:
        return __mlir_op.`index.add`(self.value, rhs)


struct Param[T: AnyRegType]:
    pass


# CHECK-LABEL: lit.func @"capturing_in_struct
# CHECK-SAME: capturing -> !kgen.none
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

fn makeClosure[p:Int](x:Int) -> Int:
    var z = __mlir_op.`index.add`(x, x)
    # CHECK: [[COPY_VAL:%.*]] = lit.ref.load %z : <index, mut *"z`0">
    # CHECK: %index = kgen.param.constant = <p>
    @__copy_capture(z, p)
    @parameter
    fn formatter() -> Int:
        # CHECK: lit.return [[COPY_VAL]] : index
        return z
    return formatter()


fn foo():
    let x = `3`
    let y = `2`
    _ = makeClosure[`3`](x)
