// RUN: kgen-opt %s -split-input-file -outline-closures-new=debug-build=true -o %t
// RUN: kgen-opt -split-input-file -verify-parameters %t | FileCheck %s

// COM: Verify that the Parameter region is lifted and the closure capture and closure init is lowered away.

kgen.generator @foo_copy(%self:!kgen.pointer<struct<(index,index)>> read_mem, %src:!kgen.pointer<struct<(index,index)>> byref_result) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}
kgen.generator @foo_move(%self:!kgen.pointer<struct<(index,index)>> owned_in_mem, %src:!kgen.pointer<struct<(index,index)>> byref_result) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}
kgen.generator @foo_del(%self:!kgen.pointer<struct<(index,index)>> owned_in_mem) -> !kgen.none {
   %none = kgen.param.constant: none = <#kgen.none>
   kgen.return %none : !kgen.none
}

// The closure generator for @closure_types
kgen.struct.generator @"closure_types::fn"<CAPTURES: !kgen.param_closure<@"closure_types" "fn">> = !kgen.closure<@"closure_types", "fn" nonescaping> {
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"closure_types", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"closure_types", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"closure_types", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"closure_types" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"closure_types", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"closure_types", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"closure_types" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"closure_types", "fn" nonescaping>> read_mem, index) -> index = #kgen.closure.symbol<@"closure_types", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"closure_types" "fn"> CAPTURES>>
  }
}

// The closure generator for @closure_types_escaping
kgen.struct.generator @"closure_types_escaping::fn"<CAPTURES: !kgen.param_closure<@"closure_types_escaping" "fn">> = !kgen.closure<@"closure_types_escaping", "fn" escaping> {
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"closure_types_escaping", "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"closure_types_escaping", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"closure_types_escaping", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"closure_types_escaping" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"closure_types_escaping", "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"closure_types_escaping", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"closure_types_escaping" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"closure_types_escaping", "fn" escaping>> read_mem, index) -> index = #kgen.closure.symbol<@"closure_types_escaping", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"closure_types_escaping" "fn"> CAPTURES>>
  }
}

// CHECK: kgen.generator @closure_types_fn(%arg0: !kgen.pointer<struct<(struct<(index, index)>) memoryOnly>> read_mem, %arg1: index) -> index attributes {sourceName = "fn"} {
// CHECK-NEXT: [[CAP:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, index)>) memoryOnly>
// CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[CAP]][0] : <struct<(index, index)>>
// CHECK-NEXT: [[FIELD:%.*]] = pop.load [[SLOT]] : !kgen.pointer<index>
// CHECK-NEXT: kgen.return [[FIELD]] : index
// CHECK-NEXT: }

// CHECK: kgen.generator @closure_types(%arg0: index, %arg1: !kgen.pointer<struct<(index, index)>>) {
// CHECK-NEXT: [[CAP:%.*]] = pop.stack_allocation 1 x struct<(struct<(index, index)>) memoryOnly> marked
// CHECK-NEXT: %1 = kgen.struct.gep %0[0] : <struct<(struct<(index, index)>) memoryOnly>>
// CHECK-NEXT: kgen.call_param[(!kgen.pointer<struct<(index, index)>> read_mem, !kgen.pointer<struct<(index, index)>> byref_result) -> !kgen.none: @foo_copy](%arg1, %1)
// CHECK-NEXT: kgen.return
// CHECK-NEXT: }
kgen.generator @closure_types(%arg0 : index, %arg1: !kgen.pointer<struct<(index,index)>>) {
  %3 = kgen.closure.init(%arg1[@foo_copy, @foo_move, @foo_del])(%arg2: index) -> index {
    %0 = kgen.struct.gep %arg1[0] : <struct<(index, index)>>
    %1 = pop.load %0 : !kgen.pointer<index>
    kgen.return %1 : index
  } : (!kgen.pointer<struct<(index,index)>>), !kgen.pointer<!kgen.closure<@closure_types, "fn" nonescaping>>

  kgen.return
}

// CHECK-LABEL: kgen.generator @closure_types_escaping
// CHECK: %index = kgen.param.constant = <get_sizeof(struct<(struct<(index, index)>) memoryOnly>, current_target())>
// CHECK-NEXT: %index_0 = kgen.param.constant = <get_alignof(struct<(struct<(index, index)>) memoryOnly>, current_target())>
// CHECK-NEXT: %0 = pop.aligned_alloc %index_0, %index : <struct<(struct<(index, index)>) memoryOnly>>
kgen.generator @closure_types_escaping(%arg0 : index, %arg1: !kgen.pointer<struct<(index,index)>>) {
  %3 = kgen.closure.init(%arg1[@foo_copy, @foo_move, @foo_del])(%arg2: index) -> index {
    %0 = kgen.struct.gep %arg1[0] : <struct<(index, index)>>
    %1 = pop.load %0 : !kgen.pointer<index>
    kgen.return %1 : index
  } : (!kgen.pointer<struct<(index,index)>>), !kgen.pointer<!kgen.closure<@closure_types_escaping, "fn" escaping>>

  kgen.return
}

// -----

// COM: Verify ClosureSymbols and ClosureTypes are lowered correctly.

// CHECK: #type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn">>, struct<(index) memoryOnly>> : !kgen.type
#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn">>, struct<(index) memoryOnly>> : !kgen.type

kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" nonescaping> {
  // CHECK: kgen.witness "__call__" : (!kgen.pointer<struct<(index) memoryOnly>> read_mem, index) -> index = @foo_fn
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> read_mem, index) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
  kgen.param.declare call: (!kgen.pointer<x> read_mem, index) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %idx7 = kgen.param.constant = <7>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem, index) -> index: call](%arg0, %idx7)
  kgen.return %0 : index
}

// CHECK: kgen.generator @foo_fn(%arg0: !kgen.pointer<struct<(index) memoryOnly>> read_mem, %arg1: index) -> index attributes {sourceName = "fn"} {
// CHECK-NEXT: [[V0:%.*]] = kgen.struct.gep %arg0[0] : <struct<(index) memoryOnly>>
// CHECK-NEXT: [[V1:%.*]] = pop.load [[V0]] : !kgen.pointer<index>
// CHECK-NEXT: kgen.return [[V1]] : index

// CHECK: kgen.generator @foo(%arg0: index) {
// CHECK-NEXT: [[V0:%.*]] = pop.stack_allocation 1 x struct<(index) memoryOnly> marked
// CHECK-NEXT: [[V1:%.*]] = kgen.struct.gep [[V0]][0] : <struct<(index) memoryOnly>>
// CHECK-NEXT: pop.store %arg0, [[V1]] : !kgen.pointer<index>
// CHECK-NEXT: [[V2:%.*]] = kgen.call @consume<:type #type_value>([[V0]]) : (!kgen.pointer<struct<(index) memoryOnly>> read_mem) -> index
// CHECK-NEXT: kgen.return
kgen.generator @foo(%arg0 : index) {
  %3 = kgen.closure.init(%arg0)(%arg1: index) -> index {
    kgen.return %arg0 : index
  } : (index), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>> read_mem) -> index
  kgen.return
}

// -----

// COM: Thin closures (todo: optimize away the none arguments MOCO 1702 and MOCO 1762)

// CHECK: #type_value = #kgen.type<typevalue<#kgen.genref<@"thin::fn">>, none> : !kgen.type
#type_value = #kgen.type<typevalue<#kgen.genref<@"thin::fn">>, !kgen.closure<@"thin", "fn" nonescaping>> : !kgen.type

kgen.struct.generator @"thin::fn"<CAPTURES: !kgen.param_closure<@"thin" "fn">> = !kgen.closure<@"thin", "fn" nonescaping> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"thin", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"thin", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"thin" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"thin", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"thin", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"thin", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"thin" "fn"> CAPTURES>>
  }
  // CHECK: kgen.witness "__call__" : (!kgen.pointer<none> read_mem, index) -> index = @thin_fn
  kgen.conformance @"closure_trait" {

    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"thin", "fn" nonescaping>> read_mem, index) -> index = #kgen.closure.symbol<@"thin", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"thin" "fn"> CAPTURES>>
  }
}

kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
  kgen.param.declare call: (!kgen.pointer<x> read_mem, index) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %index3 = kgen.param.constant = <3>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem, index) -> index: call](%arg0, %index3)
  kgen.return %0 : index
}

// CHECK:  kgen.generator @thin_fn(%arg0: !kgen.pointer<none> read_mem, %arg1: index) -> index
// CHECK-NEXT:    kgen.return %arg1 : index
// CHECK-NEXT:  }

// CHECK-LABEL: kgen.generator @thin()
// CHECK-NEXT: pop.stack_allocation 1 x none marked
// CHECK-NEXT: kgen.call @consume<:type #type_value>(%{{.*}}) : (!kgen.pointer<none> read_mem) -> index
// CHECK-NEXT: kgen.return
kgen.generator @thin() {
  %3 = kgen.closure.init()(%arg2: index) -> index {
    kgen.return %arg2 : index
  } : (), !kgen.pointer<!kgen.closure<@thin, "fn" nonescaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@thin, "fn" nonescaping>> read_mem) -> index
  kgen.return
}

// -----

// COM: Register passable closures (TODO: remove none params MOCO 1762)

// CHECK: #type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn">>, struct<(index)>> : !kgen.type
#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn">>, !kgen.closure<@"foo", "fn" trivial>> : !kgen.type

// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" trivial> {
  // CHECK: kgen.witness "__call__" : (!kgen.struct<(index)>, index) -> index = @foo_fn
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.closure<@"foo", "fn" trivial>, index) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}

kgen.generator @consume<x: type>(%arg0: !kgen.param<x>, %arg1: index) -> index {
  kgen.param.declare call: (!kgen.param<x>, index) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.param<x>, index) -> index: call](%arg0, %arg1)
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @foo_fn(%arg0: !kgen.struct<(index)>, %arg1: index) -> index
// CHECK-NEXT: [[V0:%.*]] = kgen.struct.extract %arg0[0] : <(index)>
// CHECK-NEXT: kgen.return [[V0]] : index
// CHECK-NEXT: }

// CHECK: kgen.generator @foo(%arg0: index) {
// CHECK-NEXT: [[W0:%.*]] = pop.stack_allocation 1 x struct<(index)> marked
// CHECK-NEXT: [[W1:%.*]] = kgen.struct.gep [[W0]][0] : <struct<(index)>>
// CHECK-NEXT: pop.store %arg0, [[W1]] : !kgen.pointer<index>
// CHECK-NEXT: [[W2:%.*]] = pop.load [[W0]] : !kgen.pointer<struct<(index)>>
// CHECK-NEXT: [[W3:%.*]] = kgen.call @consume<:type #type_value>([[W2]], %arg0) : (!kgen.struct<(index)>, index) -> index
// CHECK-NEXT: kgen.return
// CHECK-NEXT: }

kgen.generator @foo(%arg0 : index) {
  %3 = kgen.closure.init(%arg0)(%arg1: index) -> index {
    kgen.return %arg0 : index
  } : (index), !kgen.closure<@foo, "fn" trivial>
  %2 = kgen.call @consume<:type #type_value>(%3, %arg0) : (!kgen.closure<@foo, "fn" trivial>, index) -> index
  kgen.return
}

// -----

// COM: Register Passable Thin closures (todo: MOCO 1702 and MOCO 1762)

// CHECK: #type_value = #kgen.type<typevalue<#kgen.genref<@"thin::fn">>, none> : !kgen.type
#type_value = #kgen.type<typevalue<#kgen.genref<@"thin::fn">>, !kgen.closure<@"thin", "fn" trivial>> : !kgen.type

kgen.struct.generator @"thin::fn"<CAPTURES: !kgen.param_closure<@"thin" "fn">> = !kgen.closure<@"thin", "fn" trivial> {
  // CHECK: kgen.witness "__call__" : (!kgen.none, index) -> index = @thin_fn
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.closure<@"thin", "fn" trivial>, index) -> index = #kgen.closure.symbol<@"thin", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"thin" "fn"> CAPTURES>>
  }
}

kgen.generator @consume<x: type>(%arg0: !kgen.param<x>) -> index {
  kgen.param.declare call: (!kgen.param<x>) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.param<x>) -> index: call](%arg0)
  kgen.return %0 : index
}

// CHECK:  kgen.generator @thin_fn(%arg0: !kgen.none, %arg1: index) -> index
// CHECK-NEXT:    kgen.return %arg1 : index
// CHECK-NEXT:  }

// CHECK: kgen.generator @thin()
// CHECK: kgen.call @consume<:type #type_value>(%{{.*}}) : (!kgen.none) -> index
// CHECK-NEXT: kgen.return
kgen.generator @thin() {
  %3 = kgen.closure.init()(%arg2: index) -> index {
    kgen.return %arg2 : index
  } : (), !kgen.closure<@thin, "fn" trivial>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.closure<@thin, "fn" trivial>) -> index
  kgen.return
}

// -----

// COM: Test that a Parametric Closure that Captures Parameters Is Lifted Correctly

#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" nonescaping>> : !kgen.type

kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" nonescaping> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> read_mem) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x>) -> index {
  kgen.param.declare call: <index>(!kgen.pointer<x> read_mem) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: bind_params(:<index>(!kgen.pointer<x> read_mem) -> index call, 3)](%arg0)
  kgen.return %0 : index
}

// COM: Verify that single param capture does not result in disassembly
// CHECK-LABEL: kgen.generator @foo_fn
// CHECK-SAME: <C, A>(%arg0: !kgen.pointer<struct<(index) memoryOnly>> read_mem) -> index
// CHECK-NEXT: kgen.struct.gep
// CHECK-NEXT: pop.load
// CHECK-NEXT: <mul(A, C)>

kgen.generator @foo<C>(%arg0 : index) {
  %3 = kgen.closure.init(%arg0)<A>() -> index {
	%0 = kgen.param.constant = <mul(C, A)>
	kgen.return %arg0 : index
  } : (index), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index
  kgen.return
}

// -----

// COM: Test that a Parametric Closure that Captures Parameters Is Lifted Correctly

#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" nonescaping>> : !kgen.type

kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" nonescaping> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> read_mem) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x>) -> index {
  kgen.param.declare call: <index>(!kgen.pointer<x> read_mem) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: bind_params(:<index>(!kgen.pointer<x> read_mem) -> index call, 3)](%arg0)
  kgen.return %0 : index
}

// COM: Verify that single param capture does not result in disassembly
// CHECK-LABEL: kgen.generator @foo_fn
// CHECK-SAME: <C, A>(%arg0: !kgen.pointer<struct<(index) memoryOnly>> read_mem) -> index
// CHECK-NEXT: kgen.struct.gep
// CHECK-NEXT: pop.load
// CHECK-NEXT: <mul(A, C)>

kgen.generator @foo<C>(%arg0 : index) {
  %3 = kgen.closure.init(%arg0)<A>() -> index {
	%0 = kgen.param.constant = <mul(C, A)>
	kgen.return %arg0 : index
  } : (index), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index
  kgen.return
}

// -----

// COM: Test that the move constructor is synthesized correctly.


// CHECK: #type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<C>>>, struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> : !kgen.type
#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" escaping>> : !kgen.type
// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    // CHECK: kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = @foo__move__fn<C>
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> read_mem) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x>) -> index {
  kgen.param.declare call: <index>(!kgen.pointer<x> read_mem) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: bind_params(:<index>(!kgen.pointer<x> read_mem) -> index call, 3)](%arg0)
  kgen.return %0 : index
}
// CHECK-LABEL: kgen.generator @consume
// CHECK-LABEL: kgen.generator @foo_fn<C, A>
// CHECK-LABEL: kgen.generator
// CHECK-SAME: @foo__move__fn<C>(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, %arg1: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none
// CHECK-NEXT:  [[V0:%.*]] = kgen.struct.gep %arg1[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  [[V1:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none: @move]([[V1]], [[V0]])
// CHECK-NEXT:  [[V2:%.*]] = kgen.struct.gep %arg1[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  [[V3:%.*]] = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none: @move]([[V3]], [[V2]])
// CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
// CHECK-NEXT: kgen.return %none : !kgen.none
kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1 : !kgen.pointer<struct<(index, pointer<index>)>>) {
  %3 = kgen.closure.init(%arg0[@move, @del move], %arg1[@copy, @move, @del])<A>() -> index {
	%0 = kgen.param.constant = <mul(C, A)>
	kgen.return %0 : index
  } : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>>) -> index
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

// COM: Ensure that captured parameters in arguments are expressed correctly.

#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" nonescaping>> : !kgen.type

kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" nonescaping> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> read_mem) -> () = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}
// CHECK-LABEL: kgen.generator @foo_fn<D: type, E: type>(%arg0: !kgen.pointer<struct<(pointer<struct<(E, D)>>) memoryOnly>> read_mem) attributes {sourceName = "fn"}

kgen.generator @foo<D: type, E: type>(%arg0 : !kgen.pointer<struct<(E, D)>>) {
%3 = kgen.closure.init(%arg0)() {
  %1 = kgen.struct.gep %arg0[1] : <struct<(E, D)>>
  %2 = pop.load %1 : !kgen.pointer<D>
  kgen.return
} : (!kgen.pointer<struct<(E, D)>>), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
kgen.return
}

// -----

// COM: Test that the del method is synthesized correctly.

// CHECK: #type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<C>>>, struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> : !kgen.type
#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" escaping>> : !kgen.type

// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping>{
    kgen.conformance @"AnyType" {
      // CHECK: kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = @foo__del__fn<C>
      kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
    kgen.conformance @"Movable" {
      kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
    kgen.conformance @"closure_trait" {
      kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> read_mem) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
}

kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x>) -> !kgen.none {
  kgen.param.declare del: (!kgen.pointer<x>) -> !kgen.none = <#kgen.get_witness<x, "AnyType", "__del__">>
  %0 = kgen.call_param[(!kgen.pointer<x>) -> !kgen.none: del](%arg0)
  kgen.return %0 : !kgen.none
}
// CHECK: kgen.generator @foo__del__fn<C>(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none
// CHECK-NEXT:  [[V0:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none: @del]([[V0]])
// CHECK-NEXT:  [[V1:%.*]] = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call_param[(!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none: @del]([[V1]])
// CHECK-NEXT:  kgen.param.constant: none
// CHECK-NEXT:  kgen.return
kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1 : !kgen.pointer<struct<(index, pointer<index>)>>) {
  %3 = kgen.closure.init(%arg0[@move, @del move], %arg1[@copy, @move, @del])<A>() -> index {
	  %0 = kgen.param.constant = <mul(C, A)>
	  kgen.return %0 : index
  } : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
  kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>>) -> !kgen.none
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

#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" escaping>> : !kgen.type

// CHECK-LABEL: kgen.struct.generator @"foo::fn"<C, D> = struct_inst<"foo::fn" memoryOnly>
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping>{
    kgen.conformance @"AnyType" {
      // CHECK: kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem) -> !kgen.none = @foo__del__fn<C, D>
      kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
    kgen.conformance @"Movable" {
      // CHECK: kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> deinit_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = @foo__move__fn<C, D>
      kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
    kgen.conformance @"closure_trait" {
      // kgen.witness "__call__" : <index>(!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> read_mem) -> index = @foo_fn<?, C, D>
      kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> read_mem) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
}

kgen.generator @foo<C, D>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1 : !kgen.pointer<struct<(index, pointer<index>)>>) {
  %3 = kgen.closure.init(%arg0[@move, @del move], %arg1[@copy, @move, @del])<A>() -> index {
	  %0 = kgen.param.constant = <mul(mul(C, A), D)>
	  kgen.return %0 : index
  } : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
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


#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" escaping>> : !kgen.type

kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Copyable" {
    kgen.witness "__copyinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> read_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<copy>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> read_mem) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}

// CHECK-LABEL: kgen.generator @foo__copy__fn<C>(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> read_mem, %arg1: !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> byref_result) -> !kgen.none
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
  %3 = kgen.closure.init(%arg0[@copy, @move, @del move], %arg1)<A>() -> index {
	%0 = kgen.param.constant = <mul(C, A)>
	kgen.return %0 : index
  } : (!kgen.pointer<struct<(index, pointer<index>)>>, index), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
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

// COM: Check that thunks are properly replaced. Requires type before attribute replacement to prevent stale cache errors in attr replacer + module wide replacement.

#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn1"<:!kgen.param_closure<@foo "fn1"> #kgen.closure<@foo "fn1">>>>, !kgen.closure<@foo, "fn1" nonescaping>> : !kgen.type
kgen.struct.generator @wrapper<T: type> = struct<(T)>

// CHECK: typevalue<#kgen.genref<@"foo::fn1">>, struct<(index) memoryOnly>
kgen.generator @thunk2(%arg0: !kgen.pack<[#kgen.genref<@wrapper<:type #type_value>>]>) {
    kgen.return
}

// CHECK: kgen.generator @thunk(%arg0: !kgen.pointer<struct<(struct<(index) memoryOnly>)>> read_mem) -> index
kgen.generator @thunk(%arg0: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> read_mem) -> index {
    %index3 = kgen.param.constant = <3>
    kgen.return %index3 : index
}

kgen.generator @foo(%arg0: index) {
    %0 = kgen.closure.init(%arg0)(%arg1: index) -> index {
      kgen.return %arg0 : index
    } : (index), !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>
    %1 = pop.stack_allocation 1 x struct<(!kgen.closure<@foo, "fn1" nonescaping>)>
    %2 = kgen.struct.gep %1[0] : <struct<(!kgen.closure<@foo, "fn1" nonescaping>)>>
    %3 = pop.load %0 : !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>
    pop.store %3, %2 : !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>
    %4 = kgen.closure.init(%1[@copy, @move, @del])(%arg1: index) -> index {
      %5 = kgen.call @thunk(%1) : (!kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> read_mem) -> index
      kgen.return %5 : index
    } : (!kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>>), !kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>>
    kgen.return
}

// COM: Everything below here is boilerplate.
kgen.struct.generator @"foo::fn1"<CAPTURES: !kgen.param_closure<@foo "fn1">> = !kgen.closure<@foo, "fn1" nonescaping>{
    kgen.conformance @closure_trait {
      kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> read_mem, index) -> index = #kgen.closure.symbol<@foo, "fn1", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn1"> CAPTURES>>
    }
    kgen.conformance @AnyType {
      kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@foo, "fn1", #kgen.closure_method<del>, <:!kgen.param_closure<@foo "fn1"> CAPTURES>>
    }
    kgen.conformance @Movable {
      kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@foo, "fn1", #kgen.closure_method<move>, <:!kgen.param_closure<@foo "fn1"> CAPTURES>>
    }
}
kgen.struct.generator @"foo::fn2"<CAPTURES: !kgen.param_closure<@foo "fn2">> = !kgen.closure<@foo, "fn2" nonescaping>{
    kgen.conformance @closure_trait {
      kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> read_mem, index) -> index = #kgen.closure.symbol<@foo, "fn2", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn2"> CAPTURES>>
    }
    kgen.conformance @AnyType {
      kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@foo, "fn2", #kgen.closure_method<del>, <:!kgen.param_closure<@foo "fn2"> CAPTURES>>
    }
    kgen.conformance @Movable {
      kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@foo, "fn2", #kgen.closure_method<move>, <:!kgen.param_closure<@foo "fn2"> CAPTURES>>
    }
}
kgen.generator @copy(%arg0: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> read_mem, %arg1: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> byref_result) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}
kgen.generator @move(%arg0: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> owned_in_mem, %arg1: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> byref_result) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}
kgen.generator @del(%arg0: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> owned_in_mem) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

// -----

module {

    // CHECK-NOT: !kgen.closure

  kgen.struct.generator @"thin::fn1"<CAPTURES: !kgen.param_closure<@thin "fn1">> = !kgen.closure<@thin, "fn1" trivial>{
    kgen.conformance @closure_trait {
      kgen.witness "__call__" : (!kgen.closure<@thin, "fn1" trivial>, index) -> index = #kgen.closure.symbol<@thin, "fn1", #kgen.closure_method<call>, <:!kgen.param_closure<@thin "fn1"> CAPTURES>>
    }
  }
  kgen.struct.generator @"thin::fn2"<CAPTURES: !kgen.param_closure<@thin "fn2">> = !kgen.closure<@thin, "fn2" trivial>{
    kgen.conformance @closure_trait {
      kgen.witness "__call__" : (!kgen.closure<@thin, "fn2" trivial>, index) -> index = #kgen.closure.symbol<@thin, "fn2", #kgen.closure_method<call>, <:!kgen.param_closure<@thin "fn2"> CAPTURES>>
    }
  }
  kgen.struct.generator @"thin::fn3"<CAPTURES: !kgen.param_closure<@thin "fn3">> = !kgen.closure<@thin, "fn3" trivial>{
    kgen.conformance @closure_trait {
      kgen.witness "__call__" : (!kgen.closure<@thin, "fn3" trivial>, index) -> index = #kgen.closure.symbol<@thin, "fn3", #kgen.closure_method<call>, <:!kgen.param_closure<@thin "fn3"> CAPTURES>>
    }
  }
  kgen.generator @thin(%arg0: i1) {
    hlcf.if %arg0 {
      %0 = kgen.closure.init()(%arg1: index) -> index {
        lit.try "try0" {
          lit.try.raise "try0"
        } except {
          %1 = kgen.closure.init()(%arg2: index) -> index {
            kgen.return %arg2 : index
          } : (), !kgen.closure<@thin, "fn3" trivial>
          kgen.return %arg1 : index
        } else {
          kgen.unreachable
        }
        kgen.return %arg1 : index
      } : (), !kgen.closure<@thin, "fn1" trivial>
      kgen.return
    } else {
      %0 = kgen.closure.init()(%arg1: index) -> index {
        kgen.return %arg1 : index
      } : (), !kgen.closure<@thin, "fn2" trivial>
      kgen.return
    }
    kgen.return
  }
}

// -----

// COM: Rely on structured metadata instead of strings for matching symbol to synthesized function


#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" escaping>> : !kgen.type
// CHECK-LABEL: kgen.struct.generator @"foo::fn"
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping> {
  kgen.conformance @"AnyType" {
    // CHECK: @foo__del__fn
    kgen.witness "weird_mangle__del__[]()" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    // CHECK: @foo__move__fn
    kgen.witness "weird_mangle__moveinit__[]()" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Copyable" {
    // CHECK: @foo__copy__fn
    kgen.witness "weird_mangle__copyinit__[]()" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> read_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<copy>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    // CHECK: @foo_fn
    kgen.witness "weird_mangle__call__[]()" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> read_mem) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}

kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1: index) {
  %3 = kgen.closure.init(%arg0[@copy, @move, @del move], %arg1)<A>() -> index {
	%0 = kgen.param.constant = <mul(C, A)>
	kgen.return %0 : index
  } : (!kgen.pointer<struct<(index, pointer<index>)>>, index), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
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

// COM: Ensure declared captured not uses are used.
// If the captured type is used in a parameter expression we need to detect that.

#type_value = #kgen.type<typevalue<#kgen.genref<@"foo::fn"<:!kgen.param_closure<@"foo" "fn"> #kgen.closure<@"foo" "fn">>>>, !kgen.closure<@"foo", "fn" nonescaping>> : !kgen.type

kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" nonescaping> {
  kgen.conformance @"AnyType" {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"Movable" {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" nonescaping>> read_mem) -> () = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}

// CHECK: kgen.generator @foo_fn<E: type, D: type>
kgen.generator @foo<D: type, E: type>(%arg0 : !kgen.pointer<struct<(E, D)>>) {
%3 = kgen.closure.init(%arg0)() {
  kgen.return
} : (!kgen.pointer<struct<(E, D)>>), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
kgen.return
}

// -----

// COM: Verify that LLVMMetadataArray and LLVMArgMetadataArray on a closure.init
// COM: are transferred to the lifted generator. LLVMArgMetadataArray indices must
// COM: be shifted by one to account for the prepended self parameter.

#type_value = #kgen.type<typevalue<#kgen.genref<@"kernel::fn">>, struct<(index)>> : !kgen.type

kgen.struct.generator @"kernel::fn"<CAPTURES: !kgen.param_closure<@"kernel" "fn">> = !kgen.closure<@"kernel", "fn" trivial> {
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.closure<@"kernel", "fn" trivial>, index) -> index = #kgen.closure.symbol<@"kernel", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"kernel" "fn"> CAPTURES>>
  }
}

kgen.generator @launch<x: type>(%arg0: !kgen.param<x>, %arg1: index) -> index {
  kgen.param.declare call: (!kgen.param<x>, index) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.param<x>, index) -> index: call](%arg0, %arg1)
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @kernel_fn
// CHECK-SAME: LLVMArgMetadataArray = {{\[}}[], ["nvvm.grid_constant", unit]]
// CHECK-SAME: LLVMMetadataArray = ["nvvm.maxntid", #pop.array<256> : !pop.array<1, i32>]

kgen.generator @kernel(%arg0 : index) {
  %3 = kgen.closure.init(%arg0)(%arg1: index) -> index {
    kgen.return %arg0 : index
  } : (index), !kgen.closure<@kernel, "fn" trivial> {LLVMMetadataArray = ["nvvm.maxntid", #pop.array<256> : !pop.array<1, i32>], LLVMArgMetadataArray = [["nvvm.grid_constant", unit]]}
  %2 = kgen.call @launch<:type #type_value>(%3, %arg0) : (!kgen.closure<@kernel, "fn" trivial>, index) -> index
  kgen.return
}

// -----

// COM: Test that captured parameters propagate transitively through nested
// COM: closures.

#type_value_inner = #kgen.type<typevalue<#kgen.genref<@"foo::fn1"<:!kgen.param_closure<@foo "fn1"> #kgen.closure<@foo "fn1">>>>, !kgen.closure<@foo, "fn1" nonescaping>> : !kgen.type

kgen.struct.generator @"foo::fn1"<CAPTURES: !kgen.param_closure<@foo "fn1">> = !kgen.closure<@foo, "fn1" nonescaping> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> read_mem) -> index = #kgen.closure.symbol<@foo, "fn1", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@foo "fn1"> CAPTURES>>
  }
  kgen.conformance @AnyType {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@foo, "fn1", #kgen.closure_method<del>, <:!kgen.param_closure<@foo "fn1"> CAPTURES>>
  }
  kgen.conformance @Movable {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@foo, "fn1", #kgen.closure_method<move>, <:!kgen.param_closure<@foo "fn1"> CAPTURES>>
  }
}

// CHECK: kgen.struct.generator @"foo::fn2"<C>
kgen.struct.generator @"foo::fn2"<CAPTURES: !kgen.param_closure<@foo "fn2">> = !kgen.closure<@foo, "fn2" nonescaping> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> read_mem) -> index = #kgen.closure.symbol<@foo, "fn2", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn2"> CAPTURES>>
  }
  kgen.conformance @AnyType {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@foo, "fn2", #kgen.closure_method<del>, <:!kgen.param_closure<@foo "fn2"> CAPTURES>>
  }
  kgen.conformance @Movable {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@foo, "fn2", #kgen.closure_method<move>, <:!kgen.param_closure<@foo "fn2"> CAPTURES>>
  }
}

kgen.generator @copy(%arg0: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> read_mem, %arg1: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @move(%arg0: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> owned_in_mem, %arg1: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> byref_result) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @del(%arg0: !kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>> owned_in_mem) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
  kgen.param.declare call: <index>(!kgen.pointer<x> read_mem) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: bind_params(:<index>(!kgen.pointer<x> read_mem) -> index call, 3)](%arg0)
  kgen.return %0 : index
}

// CHECK: kgen.generator @foo_fn1<C, A>
// CHECK: kgen.generator @foo_fn2<C>(
// CHECK-SAME: read_mem) -> index
// CHECK: kgen.call @consume
kgen.generator @foo<C>(%arg0: index) {
  %0 = kgen.closure.init(%arg0)<A>() -> index {
    %1 = kgen.param.constant = <mul(C, A)>
    kgen.return %arg0 : index
  } : (index), !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>

  %2 = pop.stack_allocation 1 x struct<(!kgen.closure<@foo, "fn1" nonescaping>)>
  %3 = kgen.struct.gep %2[0] : <struct<(!kgen.closure<@foo, "fn1" nonescaping>)>>
  %4 = pop.load %0 : !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>
  pop.store %4, %3 : !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>

  %5 = kgen.closure.init(%2[@copy, @move, @del])() -> index {
    %6 = kgen.struct.gep %2[0] : <struct<(!kgen.closure<@foo, "fn1" nonescaping>)>>
    %7 = pop.load %6 : !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>
    %8 = pop.stack_allocation 1 x !kgen.closure<@foo, "fn1" nonescaping>
    pop.store %7, %8 : !kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>>
    %9 = kgen.call @consume<:type #type_value_inner>(%8) : (!kgen.pointer<!kgen.closure<@foo, "fn1" nonescaping>> read_mem) -> index
    kgen.return %9 : index
  } : (!kgen.pointer<struct<(!kgen.closure<@foo, "fn1" nonescaping>)>>), !kgen.pointer<!kgen.closure<@foo, "fn2" nonescaping>>

  kgen.return
}

// -----

// COM: Test that locally-declared parameters are not duplicated when inflating
// COM: nested closure captures.  l1 declares R and captures C from bar; l2 is
// COM: nested inside l1 and captures R.  Without the fix l1 would get <R, C, R>.

#type_val_nested = #kgen.type<typevalue<#kgen.genref<@"bar::l1::l2"<:!kgen.param_closure<@"bar::l1" "l2"> #kgen.closure<@"bar::l1" "l2">>>>, !kgen.closure<@"bar::l1", "l2" nonescaping>> : !kgen.type

kgen.struct.generator @"bar::l1::l2"<CAPTURES: !kgen.param_closure<@"bar::l1" "l2">> = !kgen.closure<@"bar::l1", "l2" nonescaping> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>> read_mem) -> index = #kgen.closure.symbol<@"bar::l1", "l2", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@"bar::l1" "l2"> CAPTURES>>
  }
  kgen.conformance @AnyType {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"bar::l1", "l2", #kgen.closure_method<del>, <:!kgen.param_closure<@"bar::l1" "l2"> CAPTURES>>
  }
  kgen.conformance @Movable {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"bar::l1", "l2", #kgen.closure_method<move>, <:!kgen.param_closure<@"bar::l1" "l2"> CAPTURES>>
  }
}

kgen.struct.generator @"bar::l1"<CAPTURES: !kgen.param_closure<@bar "l1">> = !kgen.closure<@bar, "l1" nonescaping> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@bar, "l1" nonescaping>> read_mem) -> index = #kgen.closure.symbol<@bar, "l1", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@bar "l1"> CAPTURES>>
  }
  kgen.conformance @AnyType {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@bar, "l1" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@bar, "l1", #kgen.closure_method<del>, <:!kgen.param_closure<@bar "l1"> CAPTURES>>
  }
  kgen.conformance @Movable {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@bar, "l1" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@bar, "l1" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@bar, "l1", #kgen.closure_method<move>, <:!kgen.param_closure<@bar "l1"> CAPTURES>>
  }
}

kgen.generator @use_l2<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
  kgen.param.declare call: <index>(!kgen.pointer<x> read_mem) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: bind_params(:<index>(!kgen.pointer<x> read_mem) -> index call, 3)](%arg0)
  kgen.return %0 : index
}

// CHECK: kgen.generator @bar_l2<R: type>
// CHECK: kgen.generator @bar_l1<C: type, R: type>
kgen.generator @bar<C: type>(%arg0: index) {
  %0 = kgen.closure.init(%arg0)<R: type>() -> index {
    %1 = pop.stack_allocation 1 x C

    %2 = kgen.closure.init(%arg0)() -> index {
      %3 = pop.stack_allocation 1 x R
      kgen.return %arg0 : index
    } : (index), !kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>>

    %4 = pop.stack_allocation 1 x !kgen.closure<@"bar::l1", "l2" nonescaping>
    %5 = pop.load %2 : !kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>>
    pop.store %5, %4 : !kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>>
    %6 = kgen.call @use_l2<:type #type_val_nested>(%4) : (!kgen.pointer<!kgen.closure<@"bar::l1", "l2" nonescaping>> read_mem) -> index
    kgen.return %6 : index
  } : (index), !kgen.pointer<!kgen.closure<@bar, "l1" nonescaping>>

  kgen.return
}

// -----

// COM: Verify that repeated nested closure references hidden behind aliased
// COM: register-passable closure values do not get cached as parameterless.
// CHECK: kgen.struct.generator @"foo::_kernelWrapper3"<FuncType: type> = struct_inst<"foo::_kernelWrapper3">

#type_value_inner = #kgen.type<typevalue<#kgen.genref<@"foo::_kernel"<:!kgen.param_closure<@foo "_kernel"> #kgen.closure<@foo "_kernel">>>>, !kgen.closure<@foo, "_kernel" register_passable>> : !kgen.type
#type_value_outer = #kgen.type<typevalue<#kgen.genref<@"foo::_kernelWrapper2"<:!kgen.param_closure<@foo "_kernelWrapper2"> #kgen.closure<@foo "_kernelWrapper2">>>>, !kgen.closure<@foo, "_kernelWrapper2" register_passable>> : !kgen.type

kgen.struct.generator @"foo::_kernel"<CAPTURES: !kgen.param_closure<@foo "_kernel">> = !kgen.closure<@foo, "_kernel" register_passable> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.closure<@foo, "_kernel" register_passable>) -> index = #kgen.closure.symbol<@foo, "_kernel", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "_kernel"> CAPTURES>>
  }
}

kgen.struct.generator @"foo::_kernelWrapper2"<CAPTURES: !kgen.param_closure<@foo "_kernelWrapper2">> = !kgen.closure<@foo, "_kernelWrapper2" register_passable> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.closure<@foo, "_kernelWrapper2" register_passable>) -> index = #kgen.closure.symbol<@foo, "_kernelWrapper2", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "_kernelWrapper2"> CAPTURES>>
  }
}
kgen.struct.generator @"foo::_kernelWrapper3"<CAPTURES: !kgen.param_closure<@foo "_kernelWrapper3">> = !kgen.closure<@foo, "_kernelWrapper3" register_passable> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.closure<@foo, "_kernelWrapper3" register_passable>) -> index = #kgen.closure.symbol<@foo, "_kernelWrapper3", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "_kernelWrapper3"> CAPTURES>>
  }
}
kgen.generator @consume<x: type>(%arg0: !kgen.param<x>) -> index {
  kgen.param.declare call: (!kgen.param<x>) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.param<x>) -> index: call](%arg0)
  kgen.return %0 : index
}

kgen.generator @foo<FuncType: type, flag: i1>(%arg1: !kgen.param<FuncType>) {
  %0 = kgen.closure.init(%arg1)() -> index {
    kgen.param.declare call: (!kgen.param<FuncType>) -> index = <#kgen.get_witness<FuncType, "closure_trait", "__call__">>
    %1 = kgen.call_param[(!kgen.param<FuncType>) -> index: call](%arg1)
    kgen.return %1 : index
  } : (!kgen.param<FuncType>), !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
  %27 = pop.load %0 : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
  %28 = pop.stack_allocation 1 x !kgen.closure<@foo, "_kernel" register_passable> align 1
  pop.store %27, %28 align<1> : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
  kgen.param.if <flag> {
    %29 = pop.load %28 : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
    %2 = kgen.closure.init(%29)() -> index {
      %3 = kgen.call @consume<:type #type_value_inner>(%29) : (!kgen.closure<@foo, "_kernel" register_passable>) -> index
      kgen.return %3 : index
    } : (!kgen.closure<@foo, "_kernel" register_passable>), !kgen.pointer<!kgen.closure<@foo, "_kernelWrapper2" register_passable>>
    kgen.param.yield
  } else {
    %30 = pop.load %28 : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
    %4 = kgen.closure.init(%30)() -> index {
      %5 = kgen.call @consume<:type #type_value_inner>(%30) : (!kgen.closure<@foo, "_kernel" register_passable>) -> index
      kgen.return %5 : index
    } : (!kgen.closure<@foo, "_kernel" register_passable>), !kgen.pointer<!kgen.closure<@foo, "_kernelWrapper3" register_passable>>
    kgen.param.yield
  }

  kgen.return
}

// -----

// COM: Verify that an aliased register-passable closure captured without any
// COM: body use still contributes its transitive captured parameters.
// CHECK: kgen.struct.generator @"foo::_kernelWrapper3"<FuncType: type> = struct_inst<"foo::_kernelWrapper3">

kgen.struct.generator @"foo::_kernel"<CAPTURES: !kgen.param_closure<@foo "_kernel">> = !kgen.closure<@foo, "_kernel" register_passable> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.closure<@foo, "_kernel" register_passable>) -> index = #kgen.closure.symbol<@foo, "_kernel", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "_kernel"> CAPTURES>>
  }
}

kgen.struct.generator @"foo::_kernelWrapper2"<CAPTURES: !kgen.param_closure<@foo "_kernelWrapper2">> = !kgen.closure<@foo, "_kernelWrapper2" register_passable> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.closure<@foo, "_kernelWrapper2" register_passable>) -> index = #kgen.closure.symbol<@foo, "_kernelWrapper2", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "_kernelWrapper2"> CAPTURES>>
  }
}

kgen.struct.generator @"foo::_kernelWrapper3"<CAPTURES: !kgen.param_closure<@foo "_kernelWrapper3">> = !kgen.closure<@foo, "_kernelWrapper3" register_passable> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.closure<@foo, "_kernelWrapper3" register_passable>) -> index = #kgen.closure.symbol<@foo, "_kernelWrapper3", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "_kernelWrapper3"> CAPTURES>>
  }
}

kgen.generator @foo<FuncType: type>(%arg1: !kgen.param<FuncType>) {
  %0 = kgen.closure.init(%arg1)() -> index {
    kgen.param.declare call: (!kgen.param<FuncType>) -> index = <#kgen.get_witness<FuncType, "closure_trait", "__call__">>
    %1 = kgen.call_param[(!kgen.param<FuncType>) -> index: call](%arg1)
    kgen.return %1 : index
  } : (!kgen.param<FuncType>), !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
  %27 = pop.load %0 : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
  %28 = pop.stack_allocation 1 x !kgen.closure<@foo, "_kernel" register_passable> align 1
  pop.store %27, %28 align<1> : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
  %29 = pop.load %28 : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
  %2 = kgen.closure.init(%29)() -> index {
    %3 = kgen.param.constant: index = <0>
    %30 = pop.stack_allocation 1 x !kgen.closure<@foo, "_kernel" register_passable>
    pop.store %29, %30 align<1> : !kgen.pointer<!kgen.closure<@foo, "_kernel" register_passable>>
    kgen.return %3 : index
  } : (!kgen.closure<@foo, "_kernel" register_passable>), !kgen.pointer<!kgen.closure<@foo, "_kernelWrapper2" register_passable>>

  // COM: Test that no uses in body still picks up the capture.
  %4 = kgen.closure.init(%29)() -> index {
    %5 = kgen.param.constant: index = <1>
    kgen.return %5 : index
  } : (!kgen.closure<@foo, "_kernel" register_passable>), !kgen.pointer<!kgen.closure<@foo, "_kernelWrapper3" register_passable>>

  kgen.return
}

// -----

// COM: Test that a TypeParamAttr whose typeValue is a ClosureType is correctly
// COM: lowered in the outer struct generator's field. This exercises the
// COM: TypeParamAttr replacement added to closureTypeReplacer. The outer
// COM: closure captures an inner register-passable closure by value; since the
// COM: capture has no explicit name/type in the closure.init, it produces
// COM: TypeParamAttr(ClosureType) for the struct field. After lowering, the
// COM: field should hold a TypeParamAttr wrapping the inner closure's genref
// COM: and struct type (printed as #type_value by the attribute alias printer).

kgen.struct.generator @"foo::inner_fn"<CAPTURES: !kgen.param_closure<@foo "inner_fn">> = !kgen.closure<@foo, "inner_fn" register_passable> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.closure<@foo, "inner_fn" register_passable>) -> index = #kgen.closure.symbol<@foo, "inner_fn", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "inner_fn"> CAPTURES>>
  }
}

kgen.struct.generator @"foo::outer_fn"<CAPTURES: !kgen.param_closure<@foo "outer_fn">> = !kgen.closure<@foo, "outer_fn" nonescaping> {
  kgen.conformance @closure_trait {
    kgen.witness "__call__" : (!kgen.pointer<!kgen.closure<@foo, "outer_fn" nonescaping>> read_mem) -> index = #kgen.closure.symbol<@foo, "outer_fn", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "outer_fn"> CAPTURES>>
  }
  kgen.conformance @AnyType {
    kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "outer_fn" nonescaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@foo, "outer_fn", #kgen.closure_method<del>, <:!kgen.param_closure<@foo "outer_fn"> CAPTURES>>
  }
  kgen.conformance @Movable {
    kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@foo, "outer_fn" nonescaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@foo, "outer_fn" nonescaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@foo, "outer_fn", #kgen.closure_method<move>, <:!kgen.param_closure<@foo "outer_fn"> CAPTURES>>
  }
}

// CHECK: kgen.struct.generator @"foo::outer_fn" = struct_inst<"foo::outer_fn" memoryOnly>
kgen.generator @foo<X: index>(%val: index) {
  %0 = kgen.closure.init(%val)() -> index {
    kgen.return %val : index
  } : (index), !kgen.pointer<!kgen.closure<@foo, "inner_fn" register_passable>>
  %1 = pop.load %0 : !kgen.pointer<!kgen.closure<@foo, "inner_fn" register_passable>>
  %2 = kgen.closure.init(%1)() -> index {
    %c0 = kgen.param.constant: index = <0>
    kgen.return %c0 : index
  } : (!kgen.closure<@foo, "inner_fn" register_passable>), !kgen.pointer<!kgen.closure<@foo, "outer_fn" nonescaping>>
  kgen.return
}

// -----

// COM: Verify that hoistedCaptures on a closure.init forces the parameter into
// COM: the lifted function even when the body has no reference to it.

kgen.struct.generator @"hoisted::fn"<CAPTURES: !kgen.param_closure<@"hoisted" "fn">> = !kgen.closure<@"hoisted", "fn" trivial> {
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.closure<@"hoisted", "fn" trivial>, index) -> index = #kgen.closure.symbol<@"hoisted", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"hoisted" "fn"> CAPTURES>>
  }
}

// CHECK-LABEL: kgen.generator @hoisted_fn
// CHECK-SAME: <E1>
// CHECK-SAME: (%arg0: !kgen.struct<(index)>, %arg1: index) -> index
kgen.generator @hoisted<E: index>(%arg0: index) {
  kgen.param.declare E1: index = <E>
  %0 = kgen.closure.init(%arg0)(%arg1: index) -> index {
    kgen.return %arg0 : index
  } : (index), !kgen.closure<@hoisted, "fn" trivial> {hoistedCaptures = #kgen<param.decls[E1 : index]>}
  kgen.return
}
