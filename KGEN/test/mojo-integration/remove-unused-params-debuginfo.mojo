# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -emit-llvm --debug-level line-tables %s | FileCheck %s


# CHECK: define {{.*}}agnostic_user{{.*}} !dbg ![[SP:[0-9]+]]
@no_inline
fn agnostic_user[T: AnyTrivialRegType](b: Pointer[T]):
    print(bool(b.bitcast[UInt32]()))


# There should only be one instantiation of agnostic_user
# CHECK-NOT define {{.*}}agnostic_user


fn main():
    var x: Int = 8
    var y: Float64 = 42.5
    agnostic_user(Pointer[Int].address_of(x))
    agnostic_user(Pointer[Float64].address_of(y))


# The arg type for `agnostic_user` should be an unspecified type.
# CHECK-DAG: ![[SP]] = distinct !DISubprogram({{.*}}name:{{.*}}agnostic_user{{.*}}, type: ![[SP_TYPE:[0-9]+]],
# CHECK-DAG: ![[SP_TYPE]] = !DISubroutineType({{.*}}types: ![[SP_MEMBER_TYPES:[0-9]+]]
# CHECK-DAG: ![[SP_MEMBER_TYPES]] = !{null, ![[ARG_TYPE:[0-9]+]]}
# CHECK-DAG: ![[ARG_TYPE]] = !DIDerivedType({{.*}}baseType: ![[BASE_TYPE:[0-9]+]]
# CHECK-DAG: ![[BASE_TYPE]] = !DIBasicType(tag: DW_TAG_unspecified_type
