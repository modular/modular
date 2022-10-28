// RUN: kgen-opt %s -lower-kgen-to-llvm | FileCheck %s
kgen.struct.decl @ArrayRef<T: type> {
  data : !pop.pointer<T>
  size : index
}

kgen.struct.decl @OpaqueFunction {
  data : !pop.pointer<simd<1, invalid>>
  fn : (!pop.pointer<simd<1, invalid>>, !pop.pointer<simd<1, invalid>>) -> !pop.pointer<simd<1, invalid>>
}

// CHECK-LABEL: @makeNested
// CHECK-SAME: !llvm.struct<(ptr<struct<(ptr, ptr<func<ptr (ptr, ptr)>>)>>, i64)>, %arg1: !llvm.ptr<func<ptr (ptr, ptr)>>
kgen.func @makeNested(%arg0: !kgen.ref<@ArrayRef<T: type = !kgen.ref<@OpaqueFunction>>>, %arg1: (!pop.pointer<simd<1, invalid>>, !pop.pointer<simd<1, invalid>>) -> !pop.pointer<simd<1, invalid>>) {
  // CHECK: pop.struct.construct
  // CHECK-SAME: !pop.struct<!llvm.struct<(!pop.pointer<!llvm.struct<(!pop.pointer<simd<1, invalid>>, (!pop.pointer<simd<1, invalid>>, !pop.pointer<simd<1, invalid>>) -> !pop.pointer<simd<1, invalid>>)>>, index)>, (!pop.pointer<simd<1, invalid>>, !pop.pointer<simd<1, invalid>>) -> !pop.pointer<simd<1, invalid>>>
  %0 = pop.struct.construct(%arg0, %arg1) : !pop.struct<!kgen.ref<@ArrayRef<T: type = !kgen.ref<@OpaqueFunction>>>, (!pop.pointer<simd<1, invalid>>, !pop.pointer<simd<1, invalid>>) -> !pop.pointer<simd<1, invalid>>>
  kgen.return
}

kgen.struct.decl @Empty {}

kgen.struct.decl @StructInStruct {
  x : !kgen.ref<@Empty>
}

// CHECK-LABEL: @refStructInRefStruct
// CHECK-SAME: !llvm.struct<(struct<()>)>
kgen.func @refStructInRefStruct(%a: !kgen.ref<@StructInStruct>) {
  kgen.return
}
