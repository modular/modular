// RUN: kgen-opt -sroa -split-input-file -allow-unregistered-dialect \
// RUN:   --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false %s \
// RUN:   | FileCheck %s

// SROA must not assume a store operand matches the pointer's element type.
// `replaceAllUsesWith` does not type-check, so a pass can leave the two
// disagreeing for a later pass to observe. IR of that shape does not parse
// normally, hence the unsafe flags.

// CHECK: sym_name = "sroa_mismatched_struct_store"
// CHECK: "pop.stack_allocation"() <{count = 1 : index}> : () -> !kgen.pointer<struct<(struct<()>, struct<()>)>>
// CHECK-NOT: kgen.struct.extract
// CHECK: "pop.store"
kgen.func @sroa_mismatched_struct_store() {
  %v = kgen.param.constant: struct<()> = <{  }>
  %a = pop.stack_allocation 1 x !kgen.struct<(struct<()>, struct<()>)>
  "pop.store"(%v, %a) : (!kgen.struct<()>, !kgen.pointer<struct<(struct<()>, struct<()>)>>) -> ()
  hlcf.return
}

// -----

// CHECK: sym_name = "sroa_mismatched_array_store"
// CHECK: "pop.stack_allocation"() <{count = 1 : index}> : () -> !kgen.pointer<array<2, index>>
// CHECK-NOT: pop.array.get
// CHECK: "pop.store"
kgen.func @sroa_mismatched_array_store() {
  %v = kgen.param.constant: struct<()> = <{  }>
  %a = pop.stack_allocation 1 x !pop.array<2, index>
  "pop.store"(%v, %a) : (!kgen.struct<()>, !kgen.pointer<array<2, index>>) -> ()
  hlcf.return
}
