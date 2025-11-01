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

// CHECK: kgen.generator @closure_types_fn<CAPTURES: none>(%arg0: !kgen.pointer<struct<(struct<(index, index)>) memoryOnly>> read_mem, %arg1: index) -> index {
// CHECK-NEXT: [[CAP:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, index)>) memoryOnly>
// CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[CAP]][0] : <struct<(index, index)>>
// CHECK-NEXT: [[FIELD:%.*]] = pop.load [[SLOT]] : !kgen.pointer<index>
// CHECK-NEXT: kgen.return [[FIELD]] : index
// CHECK-NEXT: }

// CHECK: kgen.generator @closure_types(%arg0: index, %arg1: !kgen.pointer<struct<(index, index)>>) {
// CHECK-NEXT: [[CAP:%.*]] = pop.stack_allocation 1 x struct<(struct<(index, index)>) memoryOnly> marked
// CHECK-NEXT: %1 = kgen.struct.gep %0[0] : <struct<(struct<(index, index)>) memoryOnly>>
// CHECK-NEXT: kgen.call @foo_copy(%arg1, %1) : (!kgen.pointer<struct<(index, index)>> read_mem, !kgen.pointer<struct<(index, index)>> byref_result) -> !kgen.none
// CHECK-NEXT: kgen.return
// CHECK-NEXT: }
kgen.generator @closure_types(%arg0 : index, %arg1: !kgen.pointer<struct<(index,index)>>) {
  %3 = kgen.closure.init(%arg1[@foo_copy, @foo_move, @foo_del])(%arg2: index) -> index {
    %0 = kgen.struct.gep %arg1[0] : !kgen.pointer<struct<(index,index)>>
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
    %0 = kgen.struct.gep %arg1[0] : !kgen.pointer<struct<(index,index)>>
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
  // CHECK: kgen.witness "__call__" : (!kgen.pointer<struct<(index) memoryOnly>> read_mem, index) -> index = @foo_fn<:none CAPTURES>
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

// CHECK: kgen.generator @foo_fn<CAPTURES: none>(%arg0: !kgen.pointer<struct<(index) memoryOnly>> read_mem, %arg1: index) -> index {
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
  // CHECK: kgen.witness "__call__" : (!kgen.pointer<none> read_mem, index) -> index = @thin_fn<:none CAPTURES>
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

// CHECK:  kgen.generator @thin_fn<CAPTURES: none>(%arg0: !kgen.pointer<none> read_mem, %arg1: index) -> index {
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
  // CHECK: kgen.witness "__call__" : (!kgen.struct<(index)>, index) -> index = @foo_fn<:none CAPTURES>
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.closure<@"foo", "fn" trivial>, index) -> index = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
  }
}

kgen.generator @consume<x: type>(%arg0: !kgen.param<x>, %arg1: index) -> index {
  kgen.param.declare call: (!kgen.param<x>, index) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.param<x>, index) -> index: call](%arg0, %arg1)
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @foo_fn<CAPTURES: none>(%arg0: !kgen.struct<(index)>, %arg1: index) -> index {
// CHECK-NEXT: [[V0:%.*]] = kgen.struct.extract %arg0[0] : !kgen.struct<(index)>
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
  // CHECK: kgen.witness "__call__" : (!kgen.none, index) -> index = @thin_fn<:none CAPTURES>
  kgen.conformance @"closure_trait" {
    kgen.witness "__call__" : (!kgen.closure<@"thin", "fn" trivial>, index) -> index = #kgen.closure.symbol<@"thin", "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@"thin" "fn"> CAPTURES>>
  }
}

kgen.generator @consume<x: type>(%arg0: !kgen.param<x>) -> index {
  kgen.param.declare call: (!kgen.param<x>) -> index = <#kgen.get_witness<x, "closure_trait", "__call__">>
  %0 = kgen.call_param[(!kgen.param<x>) -> index: call](%arg0)
  kgen.return %0 : index
}

// CHECK:  kgen.generator @thin_fn<CAPTURES: none>(%arg0: !kgen.none, %arg1: index) -> index {
// CHECK-NEXT:    kgen.return %arg1 : index
// CHECK-NEXT:  }

// CHECK: kgen.generator @thin()
// CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
// CHECK-NEXT: kgen.call @consume<:type #type_value>(%{{.*}}) : (!kgen.none) -> index
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
// CHECK-SAME: <A, CAPTURES>(%arg0: !kgen.pointer<struct<(index) memoryOnly>> read_mem) -> index {
// CHECK-NEXT: kgen.param.declare C = <CAPTURES>
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
// CHECK-SAME: <A, CAPTURES>(%arg0: !kgen.pointer<struct<(index) memoryOnly>> read_mem) -> index {
// CHECK: kgen.param.declare C = <CAPTURES>
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
    // CHECK: kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> owned_in_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = @foo__move__fn<CAPTURES>
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

  // CHECK: kgen.generator @foo__move__fn<CAPTURES>(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> owned_in_mem, %arg1: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none {
  // CHECK-NEXT: kgen.param.declare C = <CAPTURES>
  // CHECK-NEXT:  [[V0:%.*]] = kgen.struct.gep %arg1[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
  // CHECK-NEXT:  [[V1:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
  // CHECK-NEXT:  kgen.call @move([[V1]], [[V0]]) : (!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none
  // CHECK-NEXT:  [[V2:%.*]] = kgen.struct.gep %arg1[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
  // CHECK-NEXT:  [[V3:%.*]] = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
  // CHECK-NEXT:  kgen.call @move([[V3]], [[V2]]) : (!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none
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

// COM: Ensure that captured parameters in arguments are expressed correctly in signature and rebinds are emitted properly.

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

// COM: Signature check
// CHECK-LABEL:  kgen.generator @foo_fn<CAPTURES: struct<(type, type)>>
// CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<struct<(#kgen.struct.extract<:struct<(type, type)> CAPTURES, 0>,
// CHECK-SAME: #kgen.struct.extract<:struct<(type, type)> CAPTURES, 1>)>>) memoryOnly>> read_mem)

// COM: Unpack check (adds declarations for all references to captured params in stolen body from original nested)
// CHECK-NEXT:  kgen.param.declare E: type = <#kgen.struct.extract<:struct<(type, type)> CAPTURES, 0>>
// CHECK-NEXT:  kgen.param.declare D: type = <#kgen.struct.extract<:struct<(type, type)> CAPTURES, 1>>

// COM: Argument Rebind checks (if extractions were used in the signature, these rebinds prevent type mismatch errors)
// CHECK:  kgen.rebind %arg0 : !kgen.pointer<struct<(pointer<struct<(#kgen.struct.extract<:struct<(type, type)> CAPTURES, 0>, #kgen.struct.extract<:struct<(type, type)> CAPTURES, 1>)>>) memoryOnly>> to !kgen.pointer<struct<(pointer<struct<(E, D)>>) memoryOnly>>


kgen.generator @foo<D: type, E: type>(%arg0 : !kgen.pointer<struct<(E, D)>>) {
%3 = kgen.closure.init(%arg0)() {
  %1 = kgen.struct.gep %arg0[1] : !kgen.pointer<struct<(E, D)>>
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
      // CHECK: kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> owned_in_mem) -> !kgen.none = @foo__del__fn<CAPTURES>
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
// CHECK: kgen.generator @foo__del__fn<CAPTURES>(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> owned_in_mem) -> !kgen.none {
// CHECK-NEXT:  kgen.param.declare C = <CAPTURES>
// CHECK-NEXT:  [[V0:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call @del([[V0]]) : (!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none
// CHECK-NEXT:  [[V1:%.*]] = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>>
// CHECK-NEXT:  kgen.call @del([[V1]]) : (!kgen.pointer<struct<(index, pointer<index>)>> owned_in_mem) -> !kgen.none
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

// CHECK-LABEL: kgen.struct.generator @"foo::fn"<CAPTURES: struct<(index, index)>> = struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>
kgen.struct.generator @"foo::fn"<CAPTURES: !kgen.param_closure<@"foo" "fn">> = !kgen.closure<@"foo", "fn" escaping>{
    kgen.conformance @"AnyType" {
      // CHECK: kgen.witness "__del__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> owned_in_mem) -> !kgen.none = @foo__del__fn<:struct<(index, index)> CAPTURES>
      kgen.witness "__del__" : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>> owned_in_mem) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<del>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
    kgen.conformance @"Movable" {
      // CHECK: kgen.witness "__moveinit__" : (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> owned_in_mem, !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> byref_result) -> !kgen.none = @foo__move__fn<:struct<(index, index)> CAPTURES>
      kgen.witness "__moveinit__" : (!kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> owned_in_mem, !kgen.pointer<!kgen.closure<@"foo", "fn" escaping>> byref_result) -> !kgen.none = #kgen.closure.symbol<@"foo", "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@"foo" "fn"> CAPTURES>>
    }
    kgen.conformance @"closure_trait" {
      // kgen.witness "__call__" : <index>(!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>) memoryOnly>> read_mem) -> index = @foo_fn<?, :struct<(index, index)> CAPTURES>
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

// CHECK-LABEL: kgen.generator @foo__copy__fn<CAPTURES>(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> read_mem, %arg1: !kgen.pointer<struct<(struct<(index, pointer<index>)>, index) memoryOnly>> byref_result) -> !kgen.none {
// CHECK:   kgen.param.declare C = <CAPTURES>
// CHECK:   [[V0:%.*]] = kgen.struct.gep %arg1[0] : <struct<(struct<(index, pointer<index>)>, index) memoryOnly>>
// CHECK:   [[V1:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, index) memoryOnly>>
// CHECK:   [[V2:%.*]] = kgen.call @copy([[V1]], [[V0]]) : (!kgen.pointer<struct<(index, pointer<index>)>> read_mem, !kgen.pointer<struct<(index, pointer<index>)>> byref_result) -> !kgen.none
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

// CHECK: typevalue<#kgen.genref<@"foo::fn1"<:none #kgen.none>>>, struct<(index) memoryOnly>
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
