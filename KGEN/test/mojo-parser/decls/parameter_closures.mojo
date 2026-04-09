# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct BoxedInt(Copyable, RegisterPassable):
    var value: Int

    @implicit
    def __init__(out self, value: Int):
        self.value = value

    def boxedAdd(self, rhs: Int) -> Int:
        return self.value + rhs


struct Param[T: TrivialRegisterPassable]:
    pass


# CHECK-LABEL: lit.fn @"capturing_in_struct
# CHECK-SAME: capturing -> !kgen.none
def capturing_in_struct[x: Param[def() capturing -> Int]]():
    pass


# CHECK-LABEL: lit.struct.decl @CapturingMember
struct CapturingMember[f: def() capturing -> None]:
    # CHECK-LABEL: lit.fn @"member
    # CHECK-SAME: capturing -> !kgen.none attributes
    def member(self):
        pass

    # CHECK-LABEL: lit.fn @"static_method
    # CHECK-SAME: capturing -> !kgen.none attributes
    @staticmethod
    def static_method():
        pass


def makeClosure[p: Int](x: Int) -> Int:
    var z = x + x

    # CHECK: [[COPY_VAL:%.*]] = lit.ref.load %z : <!Int, mut *"z`">
    # CHECK:  = kgen.param.constant: !Int = <p>
    @__copy_capture(z, p)
    @parameter
    def writer() -> Int:
        # CHECK: lit.return [[COPY_VAL]] : !Int
        return z

    return writer()


def foo():
    var x = 3
    var y = 2
    _ = makeClosure[3](x)
