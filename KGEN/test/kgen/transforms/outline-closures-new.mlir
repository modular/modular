// RUN: kgen-opt %s -split-input-file -outline-closures-new=debug-build=true -o %t
// RUN: kgen-opt -split-input-file -verify-parameters %t | FileCheck %s

// COM: Test that the move constructor is synthesized correctly.

// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn" = struct_inst<"foo::fn" memoryOnly> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>>
  }
  kgen.conformance @"Movable" {
    // CHECK: kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = @foo__move__fn
    kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>>
  }
}

// CHECK: kgen.generator @foo__move__fn(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, %arg1: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none
// CHECK-NEXT:  [[V0:%.*]] = kgen.struct.gep %arg1[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  [[V1:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none: @move]([[V1]], [[V0]])
// CHECK-NEXT:  [[V2:%.*]] = kgen.struct.gep %arg1[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  [[V3:%.*]] = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none: @move]([[V3]], [[V2]])
// CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
// CHECK-NEXT: kgen.return %none : !kgen.none
kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1 : !kgen.pointer<struct<(index, pointer<index>)>>) {
  kgen.closure.init(%arg0[@move, @del move], %arg1[@copy, @move, @del])() -> index : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>), !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> {closureType = !kgen.closure<@foo, "fn" escaping>}
  kgen.return
}

kgen.generator @copy(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> read_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
kgen.generator @move(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
kgen.generator @del(%arg0: !kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// -----

// COM: Test that the del method is synthesized correctly.

// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn" = struct_inst<"foo::fn" memoryOnly> {
  kgen.conformance @"AnyType" {
    // CHECK: kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = @foo__del__fn
    kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>>
  }
}

// CHECK: kgen.generator @foo__del__fn(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none
// CHECK-NEXT:  [[V0:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none: @del]([[V0]])
// CHECK-NEXT:  [[V1:%.*]] = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none: @del]([[V1]])
// CHECK-NEXT:  kgen.param.constant: none
// CHECK-NEXT:  kgen.return
kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1 : !kgen.pointer<struct<(index, pointer<index>)>>) {
  kgen.closure.init(%arg0[@move, @del move], %arg1[@copy, @move, @del])() -> index : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>), !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> {closureType = !kgen.closure<@foo, "fn" escaping>}
  kgen.return
}

kgen.generator @copy(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> read_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}
kgen.generator @move(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}
kgen.generator @del(%arg0: !kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}

// -----

// COM: Ensure the referenced kgen.struct.generator's closure types and symbols are lowered
// COM: with both del and move conformances present.

// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn" = struct_inst<"foo::fn" memoryOnly> {
  kgen.conformance @"AnyType" {
    // CHECK: kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = @foo__del__fn
    kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>>
  }
  kgen.conformance @"Movable" {
    // CHECK: kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = @foo__move__fn
    kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>>
  }
}

kgen.generator @foo<C, D>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1 : !kgen.pointer<struct<(index, pointer<index>)>>) {
  kgen.closure.init(%arg0[@move, @del move], %arg1[@copy, @move, @del])() -> index : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>), !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> {closureType = !kgen.closure<@foo, "fn" escaping>}
  kgen.return
}

kgen.generator @copy(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> read_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}
kgen.generator @move(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}
kgen.generator @del(%arg0: !kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}

// -----

// COM: Test that the copy constructor is synthesized correctly.

kgen.struct.generator @"foo::fn" = struct_inst<"foo::fn" memoryOnly> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> deinit_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>>
  }
  kgen.conformance @"Copyable" {
    kgen.witness "__copyinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> read_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<copy>>
  }
}

// CHECK-LABEL: kgen.generator @foo__copy__fn(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> read_mem, %arg1: !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> byref_result) -> !kgen.none
// CHECK:   [[V0:%.*]] = kgen.struct.gep %arg1[0] : <struct<(struct<(index, pointer<index>)>, index) memoryOnly>>
// CHECK:   [[V1:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, index) memoryOnly>>
// CHECK:   kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> read_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none: @copy]([[V1]], [[V0]])
// CHECK:   [[V3:%.*]] = kgen.struct.gep %arg1[1] : <struct<(struct<(index, pointer<index>)>, index) memoryOnly>>
// CHECK:   [[V4:%.*]] = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, index) memoryOnly>>
// CHECK:   [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<index>
// CHECK:   pop.store [[V5]], [[V3]] : !kgen.pointer<index>
// CHECK:   %none = kgen.param.constant: none = <#kgen.none>
// CHECK:   kgen.return %none : !kgen.none
// CHECK: }
kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1: index) {
  kgen.closure.init(%arg0[@copy, @move, @del move], %arg1)() -> index : (!kgen.pointer<struct<(index, pointer<index>)>>, index), !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> {closureType = !kgen.closure<@foo, "fn" escaping>}
  kgen.return
}

kgen.generator @copy(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> read_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
kgen.generator @move(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
kgen.generator @del(%arg0: !kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// -----

// COM: Rely on structured metadata instead of strings for matching symbol to synthesized function.
// COM: Verify del, move, and copy are all routed to the correctly-named generators.

// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn" = struct_inst<"foo::fn" memoryOnly> {
  kgen.conformance @"AnyType" {
    // CHECK: @foo__del__fn
    kgen.witness "weird_mangle__del__[]()" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> deinit_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>>
  }
  kgen.conformance @"Movable" {
    // CHECK: @foo__move__fn
    kgen.witness "weird_mangle__moveinit__[]()" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>>
  }
  kgen.conformance @"Copyable" {
    // CHECK: @foo__copy__fn
    kgen.witness "weird_mangle__copyinit__[]()" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> read_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<copy>>
  }
}

kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1: index) {
  kgen.closure.init(%arg0[@copy, @move, @del move], %arg1)() -> index : (!kgen.pointer<struct<(index, pointer<index>)>>, index), !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> {closureType = !kgen.closure<@foo, "fn" escaping>}
  kgen.return
}

kgen.generator @copy(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> read_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
kgen.generator @move(%arg0:!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, %arg1:!kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
kgen.generator @del(%arg0: !kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}
