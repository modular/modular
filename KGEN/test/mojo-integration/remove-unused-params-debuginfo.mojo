# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -emit-llvm --debug-level line-tables %s | FileCheck %s


# CHECK: define {{.*}}agnostic_user{{.*}} !dbg ![[SP:[0-9]+]]
@no_inline
fn agnostic_user[
    T: AnyType, dt: DType
](b: UnsafePointer[T], dp: UnsafePointer[Scalar[dt]]):
    print(bool(b.bitcast[UInt32]()))
    print(bool(dp.bitcast[UInt32]()))


# There should only be one instantiation of agnostic_user
# CHECK-NOT define {{.*}}agnostic_user


fn main():
    var x: Int = 8
    var y: Float64 = 42.5
    var d: Scalar[DType.uint8] = 9
    agnostic_user[Int, DType.uint8](
        UnsafePointer[Int].address_of(x),
        UnsafePointer[Scalar[DType.uint8]].address_of(d),
    )
    agnostic_user[Float64, DType.uint8](
        UnsafePointer[Float64].address_of(y),
        UnsafePointer[Scalar[DType.uint8]].address_of(d),
    )


# The arg type for `agnostic_user` should be an unspecified type.
# CHECK-DAG: ![[SP]] = distinct !DISubprogram({{.*}}name:{{.*}}agnostic_user{{.*}}, type: ![[SP_TYPE:[0-9]+]],
# CHECK-DAG: ![[SP_TYPE]] = !DISubroutineType({{.*}}types: ![[SP_MEMBER_TYPES:[0-9]+]]
# CHECK-DAG: ![[SP_MEMBER_TYPES]] = !{null, ![[ARG_TYPE0:[0-9]+]], ![[ARG_TYPE1:[0-9]+]]}
# CHECK-DAG: ![[ARG_TYPE0]] = !DIDerivedType({{.*}}baseType: ![[BASE_TYPE:[0-9]+]]
# CHECK-DAG: ![[BASE_TYPE]] = !DIBasicType(tag: DW_TAG_unspecified_type
# CHECK-DAG: ![[ARG_TYPE1]] = !DIDerivedType({{.*}}baseType: ![[BASE_SCALAR_TYPE:[0-9]+]]
# CHECK-DAG: ![[BASE_SCALAR_TYPE]] = !DICompositeType(tag: DW_TAG_array_type{{.*}}baseType: ![[BASE_TYPE]]
