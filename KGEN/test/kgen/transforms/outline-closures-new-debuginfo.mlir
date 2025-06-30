// RUN: kgen-opt %s -split-input-file -outline-closures-new=debug-build=true -mlir-print-debuginfo | FileCheck %s

// COM: Use of 'C' appears only in a location inside the closure.

// CHECK: kgen.generator @foo_fn<C>(%arg0: !kgen.pointer<none> loc("{{.*}}":{{.*}}:{{.*}}) read_mem) {
// CHECK-NEXT:   kgen.return loc(#loc
// CHECK-NEXT: } loc(#loc
kgen.generator @foo<C>() {
  %3 = kgen.closure.init()() -> () {
	  kgen.return loc(fused<#kgen.param.decl.ref<"C"> : index>["C:0:0"])
  } : (), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
  kgen.return
}
