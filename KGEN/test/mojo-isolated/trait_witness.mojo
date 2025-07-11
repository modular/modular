# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


trait R1:
    alias N: Int

    fn f1(self, x: Bool):
        ...

    fn f1(self, x: Int):
        ...


trait R2:
    alias T: AnyType

    fn f2(self, x: T):
        ...


trait R1Child(R1):
    pass


@register_passable("trivial")
# CHECK-LABEL: lit.struct.decl @S1<X: !Int>
struct S1[X: Int](R1, R1Child, R2, Movable):
    alias N: Int = X
    alias T: AnyType = Int

    # CHECK: lit.fn @"f1[[F1_BOOL_NAME:.+]]"({{.*}}, %x: !Bool)
    fn f1(self, x: Bool):
        pass

    # CHECK: lit.fn @"f1[[F1_INT_NAME:.+]]"({{.*}}, %x: !Int)
    fn f1(self, x: Int):
        pass

    # CHECK: lit.fn @"f2[[F2_NAME:.+]]"({{.*}}, %x: !Int)
    fn f2(self, x: Int):
        pass

    # CHECK: lit.fn @"__del__[[DEL_NAME:.+]]"[
    # Synthesized function:
    # CHECK: lit.fn @"__moveinit__[[MOVEINIT_NAME:.+]]"[

    # CHECK: kgen.conformance @{{.*}}AnyType
    # CHECK-NEXT: kgen.witness "__del__" : {{.*}} = {{.*}}@S1::@"__del__[[DEL_NAME]]"<:!Int X>
    # CHECK-NEXT: traitRef = [[ANYTYPE_NAME:(@[^:]+::)+@AnyType]]

    # CHECK: kgen.conformance @{{.*}}Movable
    # CHECK-NEXT: kgen.witness "__moveinit__" : {{.*}} = {{.*}}@S1::@"__moveinit__[[MOVEINIT_NAME]]"<:!Int X>
    # CHECK-NEXT: immediateParents = #M<symbols[[[ANYTYPE_NAME]]]>

    # CHECK: kgen.conformance @{{.*}}R1
    # CHECK-NEXT: kgen.witness "N" : !Int = X
    # CHECK-NEXT: kgen.witness "f1" : {{.*}} = {{.*}}@S1::@"f1[[F1_BOOL_NAME]]"<:!Int X>
    # CHECK-NEXT: kgen.witness "f1" : {{.*}} = {{.*}}@S1::@"f1[[F1_INT_NAME]]"<:!Int X>
    # CHECK-NEXT: immediateParents = #M<symbols[[[ANYTYPE_NAME]]]>
    # CHECK-SAME: traitRef = [[R1_NAME:(@[^:]+::)+@R1]]

    # CHECK: kgen.conformance @{{.*}}R1Child
    # CHECK-NEXT: immediateParents = #M<symbols[[[R1_NAME]]]>

    # CHECK: kgen.conformance @{{.*}}R2
    # CHECK-NEXT: kgen.witness "T" : !AnyType = !Int
    # CHECK-NEXT: kgen.witness "f2" : {{.*}} = {{.*}}@S1::@"f2[[F2_NAME]]"<:!Int X>
