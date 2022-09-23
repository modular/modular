// RUN: kgen-opt -pass-pipeline='kgen.func(lower-pop-to-llvm)' %s | FileCheck %s

!struct1 = !pop.struct<!pop.struct<!pop.scalar<f32>>, !pop.array<4, !pop.scalar<f32>>>

// CHECK-LABEL: @struct_construct
kgen.func @struct_construct(
    %a: !pop.struct<!pop.scalar<f32>>,
    %b: !pop.array<4, !pop.scalar<f32>>
) -> !struct1 {
  // CHECK: %[[S0:.*]] = llvm.mlir.undef : !llvm.struct<(struct<(f32)>, array<4 x f32>)>
  // CHECK: %[[S1:.*]] = llvm.insertvalue %{{.*}}, %[[S0]][0]
  // CHECK: %[[S2:.*]] = llvm.insertvalue %{{.*}}, %[[S1]][1]
  %0 = pop.struct.construct(%a, %b) : !struct1
  kgen.return %0 : !struct1
}

// CHECK-LABEL: @struct_insert
kgen.func @struct_insert(
    %a: !pop.struct<!pop.scalar<f32>>,
    %b: !pop.scalar<f32>
) -> !pop.struct<!pop.scalar<f32>> {
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(f32)>
  %0 = pop.replace_element %b, %a[0] : !pop.struct<!pop.scalar<f32>>
  kgen.return %0 : !pop.struct<!pop.scalar<f32>>
}

// CHECK-LABEL: @struct_extract
kgen.func @struct_extract(%a: !pop.struct<!pop.scalar<f32>>) -> !pop.scalar<f32> {
  // CHECK: llvm.extractvalue %{{.*}}[0]
  %0 = pop.get_element %a[0] : !pop.struct<!pop.scalar<f32>>
  kgen.return %0 : !pop.scalar<f32>
}
