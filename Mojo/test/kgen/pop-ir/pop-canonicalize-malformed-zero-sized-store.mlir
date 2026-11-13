// RUN: kgen-opt -canonicalize -allow-unregistered-dialect \
// RUN:   --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false %s \
// RUN:   | FileCheck %s

// A zero-sized value stored into a slot of another type only arises in
// not-yet-legalized IR, which cannot pass the verifier, hence the unsafe
// flags, and the generic form has to spell out the store's defaults.
// The store still writes no bytes, so canonicalization drops it.
// CHECK-LABEL: kgen.func @canonicalize_mismatched_zero_sized_store
// CHECK-NOT: pop.store
kgen.func @canonicalize_mismatched_zero_sized_store() {
  %v = kgen.param.constant: struct<()> = <{  }>
  %a = pop.stack_allocation 1 x !pop.array<2, index>
  "pop.store"(%v, %a) <{isNonTemporal = false, isVolatile = false,
      ordering = #pop.atomic_ordering<not_atomic>}>
      : (!kgen.struct<()>, !kgen.pointer<array<2, index>>) -> ()
  hlcf.return
}
