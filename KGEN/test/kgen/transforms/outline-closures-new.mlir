// RUN: kgen-opt %s -split-input-file -outline-closures-new=debug-build=true -o %t
// RUN: kgen-opt -split-input-file -verify-parameters %t | FileCheck %s

// COM: Test that closure init is replaced


kgen.struct.generator @"foo::fn" = struct_inst<"foo::fn" memoryOnly> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>>
  }
}

// CHECK-LABEL: kgen.generator @foo<C>(
// CHECK-SAME:    %[[ARG0:.*]]: !kgen.pointer<struct<(index, pointer<index>)>>,
// CHECK-SAME:    %[[ARG1:.*]]: !kgen.pointer<struct<(index, pointer<index>)>>) {
// CHECK:         %[[SIZE:.*]] = kgen.param.constant = <get_sizeof(struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>, current_target())>
// CHECK:         %[[ALIGN:.*]] = kgen.param.constant = <get_alignof(struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>, current_target())>
// CHECK:         %[[ALLOC:.*]] = pop.aligned_alloc %[[ALIGN]], %[[SIZE]] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK:         %[[GEP0:.*]] = kgen.struct.gep %[[ALLOC]][0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK:         kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none: @move](%[[ARG0]], %[[GEP0]])
// CHECK:         %[[GEP1:.*]] = kgen.struct.gep %[[ALLOC]][1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK:         kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> read_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none: @copy](%[[ARG1]], %[[GEP1]])
// CHECK:         kgen.return
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
