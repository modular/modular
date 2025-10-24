// RUN: kgen-opt %s -split-input-file -outline-closures-new=debug-build=true -mlir-print-debuginfo | FileCheck %s

// COM: Use of 'C' appears only in a location inside the closure.

// Provide a struct generator for the escaping closure
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping> {
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@foo, "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> read_mem) -> () = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}

// CHECK: kgen.generator @foo_fn<CAPTURES>(%arg0: !kgen.pointer<none> loc({{.*}}) read_mem) {
// CHECK-NEXT:  kgen.param.declare C = <CAPTURES>
// CHECK-NEXT:   kgen.return loc(#loc
// CHECK-NEXT: } loc(#loc
kgen.generator @foo<C>() {
  %3 = kgen.closure.init()() -> () {
	  kgen.return loc(fused<#kgen.param.decl.ref<"C"> : index>["C:0:0"])
  } : (), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
  kgen.return
}
