// Check that SROA gives a struct's union field its own allocation.
// RUN: kgen-opt %s -sroa -kgen-verifier | FileCheck %s

// CHECK-LABEL: kgen.func @field
kgen.func @field(%value: !kgen.struct<(untagged_union<i64>, i8)>) -> i64 {
  // CHECK: pop.stack_allocation 1 x untagged_union<i64>
  // CHECK: pop.stack_allocation 1 x i8
  %storage = pop.stack_allocation 1 x !kgen.struct<(untagged_union<i64>, i8)>
  // CHECK: pop.store
  pop.store %value, %storage : !kgen.pointer<struct<(untagged_union<i64>, i8)>>
  %member = kgen.struct.gep %storage[0] : <struct<(untagged_union<i64>, i8)>>
  %pointer = pop.pointer.bitcast %member : !kgen.pointer<untagged_union<i64>> to !kgen.pointer<i64>
  %loaded = pop.load %pointer : !kgen.pointer<i64>
  hlcf.return %loaded : i64
}
