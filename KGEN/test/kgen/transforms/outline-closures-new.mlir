// RUN: kgen-opt %s -split-input-file -verify-parameters -outline-closures-new | FileCheck %s

// COM: Verify that the Parameter region is lifted and the closure capture and closure init is lowered away.

kgen.generator @foo_copy(%self:!kgen.pointer<struct<(index,index)>>, %src:!kgen.pointer<struct<(index,index)>>){
  kgen.return
}
kgen.generator @foo_move(%self:!kgen.pointer<struct<(index,index)>>, %src:!kgen.pointer<struct<(index,index)>>){
  kgen.return
}
kgen.generator @foo_del(%self:!kgen.pointer<struct<(index,index)>>){
  kgen.return
}

// CHECK: kgen.generator @closure_types_fn<CAPTURES: none>(%arg0: !kgen.pointer<struct<(struct<(index, index)>)>>, %arg1: index) -> index {
// CHECK-NEXT: [[CAP:%.*]] = kgen.struct.gep %arg0[0] : <struct<(struct<(index, index)>)>>
// CHECK-NEXT: [[SLOT:%.*]] = kgen.struct.gep [[CAP]][0] : <struct<(index, index)>>
// CHECK-NEXT: [[FIELD:%.*]] = pop.load [[SLOT]] : !kgen.pointer<index>
// CHECK-NEXT: kgen.return [[FIELD]] : index
// CHECK-NEXT: }

// CHECK: kgen.generator @closure_types(%arg0: index, %arg1: !kgen.pointer<struct<(index, index)>>) {
// CHECK-NEXT: [[CAP:%.*]] = pop.stack_allocation 1 x struct<(struct<(index, index)>)> marked
// CHECK-NEXT: %1 = kgen.struct.gep %0[0] : <struct<(struct<(index, index)>)>>
// CHECK-NEXT: kgen.call @foo_copy(%1, %arg1) : (!kgen.pointer<struct<(index, index)>>, !kgen.pointer<struct<(index, index)>>) -> ()
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
// CHECK: %index = kgen.param.constant = <get_sizeof(struct<(struct<(index, index)>)>, current_target())>
// CHECK-NEXT: %index_0 = kgen.param.constant = <get_alignof(struct<(struct<(index, index)>)>, current_target())>
// CHECK-NEXT: %0 = pop.aligned_alloc %index_0, %index : <struct<(struct<(index, index)>)>>
kgen.generator @closure_types_escaping(%arg0 : index, %arg1: !kgen.pointer<struct<(index,index)>>) {
  %3 = kgen.closure.init(%arg1[@foo_copy, @foo_move, @foo_del])(%arg2: index) escaping -> index {
    %0 = kgen.struct.gep %arg1[0] : !kgen.pointer<struct<(index,index)>>
    %1 = pop.load %0 : !kgen.pointer<index>
    kgen.return %1 : index
  } : (!kgen.pointer<struct<(index,index)>>), !kgen.pointer<!kgen.closure<@closure_types_escaping, "fn" escaping>>

  kgen.return
}

// -----

// COM: Verify ClosureSymbols and ClosureTypes are lowered correctly.

// CHECK: #type_value = #kgen.type<struct<(index)>, {
// CHECK-SAME: "__call__" : (!kgen.pointer<struct<(index)>>, index) -> index = @foo_fn<:none #kgen.none>}> : !kgen.type
#type_value = #kgen.type<!kgen.closure<@foo, "fn" nonescaping>,
              {"__call__" :
              (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>, index) -> index =
               #kgen.closure.symbol<@foo, "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn"> #kgen.closure<@foo "fn">> >}> : !kgen.type


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
    %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: get_vtable_entry(x, "__call__")](%arg0)
    kgen.return %0 : index
}

// CHECK: kgen.generator @foo_fn<CAPTURES: none>(%arg0: !kgen.pointer<struct<(index)>>, %arg1: index) -> index {
// CHECK-NEXT: [[V0:%.*]] = kgen.struct.gep %arg0[0] : <struct<(index)>>
// CHECK-NEXT: [[V1:%.*]] = pop.load [[V0]] : !kgen.pointer<index>
// CHECK-NEXT: kgen.return [[V1]] : index

// CHECK: kgen.generator @foo(%arg0: index) {
// CHECK-NEXT: [[V0:%.*]] = pop.stack_allocation 1 x struct<(index)> marked
// CHECK-NEXT: [[V1:%.*]] = kgen.struct.gep [[V0]][0] : <struct<(index)>>
// CHECK-NEXT: pop.store %arg0, [[V1]] : !kgen.pointer<index>
// CHECK-NEXT: [[V2:%.*]] = kgen.call @consume<:type #type_value>([[V0]]) : (!kgen.pointer<struct<(index)>> read_mem) -> index
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

// CHECK: #type_value = #kgen.type<none, {
// CHECK-SAME: "__call__" : (!kgen.pointer<none>, index) -> index = @thin_fn<:none #kgen.none>}> : !kgen.type
#type_value = #kgen.type<!kgen.closure<@thin, "fn" nonescaping>,
              { "__call__" :
              (!kgen.pointer<!kgen.closure<@thin, "fn" nonescaping>>, index) -> index =
               #kgen.closure.symbol<@thin, "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@thin "fn"> #kgen.closure<@thin "fn">> >}> : !kgen.type


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
    %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: get_vtable_entry(x, "__call__")](%arg0)
    kgen.return %0 : index
}

// CHECK:  kgen.generator @thin_fn<CAPTURES: none>(%arg0: !kgen.pointer<none>, %arg1: index) -> index {
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

// CHECK: #type_value = #kgen.type<struct<(index)>, {
// CHECK-SAME: "__call__" : (!kgen.struct<(index)>, index) -> index = @foo_fn<:none #kgen.none>}> : !kgen.type
#type_value = #kgen.type<!kgen.closure<@foo, "fn" registerpassable>,
              {
                "__call__" :
              (!kgen.closure<@foo, "fn" registerpassable>, index) -> index =
               #kgen.closure.symbol<@foo, "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn"> #kgen.closure<@foo "fn">> >}> : !kgen.type


kgen.generator @consume<x: type>(%arg0: !kgen.param<x>, %arg1: index) -> index {
    %0 = kgen.call_param[(!kgen.param<x>, index) -> index: get_vtable_entry(x, "__call__")](%arg0, %arg1)
    kgen.return %0 : index
}

// CHECK: kgen.generator @foo_fn<CAPTURES: none>(%arg0: !kgen.struct<(index)>, %arg1: index) -> index {
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
  } : (index), !kgen.closure<@foo, "fn" registerpassable>
  %2 = kgen.call @consume<:type #type_value>(%3, %arg0) : (!kgen.closure<@foo, "fn" registerpassable>, index) -> index
  kgen.return
}

// -----

// COM: Register Passable Thin closures (todo: MOCO 1702 and MOCO 1762)

// CHECK: #type_value = #kgen.type<none, {
// CHECK-SAME: "__call__" : (!kgen.none, index) -> index = @thin_fn<:none #kgen.none>}> : !kgen.type
#type_value = #kgen.type<!kgen.closure<@thin, "fn" registerpassable>,
              {"__call__" :
              (!kgen.closure<@thin, "fn" registerpassable>, index) -> index =
               #kgen.closure.symbol<@thin, "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@thin "fn"> #kgen.closure<@thin "fn">> >}> : !kgen.type


kgen.generator @consume<x: type>(%arg0: !kgen.param<x>) -> index {
    %0 = kgen.call_param[(!kgen.param<x>) -> index: get_vtable_entry(x, "__call__")](%arg0)
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
  } : (), !kgen.closure<@thin, "fn" registerpassable>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.closure<@thin, "fn" registerpassable>) -> index
  kgen.return
}

// -----

// COM: Test that a Parameterless Closure that Captures Parameters Is Lifted Correctly

// CHECK: #type_value = #kgen.type<struct<(index)>,
// CHECK-SAME: "__call__" :
// CHECK-SAME: (!kgen.pointer<struct<(index)>>) -> index =
// CHECK-SAME: @foo_fn<:struct<(index, index)> CAPTURES>}> : !kgen.type
#type_value =
  #kgen.type<!kgen.closure<@foo, "fn" nonescaping>,
    {"__call__" :
      (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index =
      #kgen.closure.symbol<@foo, "fn", #kgen.closure_method<call>, <:!kgen.param_closure<@foo "fn"> #kgen.closure<@foo "fn">> >}
  > : !kgen.type


kgen.generator @consume<x: type
                        >(%arg0: !kgen.pointer<x>) -> index {
  %0 = kgen.call_param[(!kgen.pointer<x>) -> index: get_vtable_entry(x, "__call__")](%arg0)
  kgen.return %0 : index
}

// COM: The lifted function should have a single parameter of type struct with the captured parameters as members
// CHECK:      kgen.generator @foo_fn<CAPTURES: struct<(index, index)>>(%arg0: !kgen.pointer<struct<(index)>>) -> index {
// CHECK-NEXT:   kgen.param.declare C = <#kgen.struct.extract<:struct<(index, index)> CAPTURES, 0>>
// CHECK-NEXT:   kgen.param.declare D = <#kgen.struct.extract<:struct<(index, index)> CAPTURES, 1>>
// CHECK-NEXT:   kgen.struct.gep %arg0[0]
// CHECK-NEXT:  [[V1:%.*]] = pop.load
// CHECK-NEXT:  %index = kgen.param.constant = <mul(C, D)>
// CHECK-NEXT:  kgen.return [[V1]]
// CHECK-NEXT: }


// COM: The parent generator should bind the struct attribute it created to the parameter of the consuming function
// CHECK: kgen.generator @foo<C, D>(%arg0: index) {
// CHECK-NEXT:   kgen.param.declare CAPTURES: struct<(index, index)> = <{ C, D }>
// CHECK-NEXT:   [[V0:%.*]] = pop.stack_allocation 1 x struct<(index)> marked
// CHECK-NEXT:   kgen.struct.gep
// CHECK-NEXT:   pop.store
// CHECK-NEXT:   kgen.call @consume<:type #type_value
// CHECK-SAME:   ) : (!kgen.pointer<struct<(index)>>) -> index
// CHECK-NEXT:   kgen.return
// CHECK-NEXT:   }
kgen.generator @foo<C,D>(%arg0 : index) {
  %3 = kgen.closure.init(%arg0)() -> index {
	%0 = kgen.param.constant = <mul(C, D)>
	kgen.return %arg0 : index
  } : (index), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index
  kgen.return
}

// -----

// COM: Test that a Parametric Closure that Captures Parameters Is Lifted Correctly

#type_value =
  #kgen.type<!kgen.closure<@foo, "fn" nonescaping>,
    {"__call__" :
      <index>
      (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index =
      #kgen.closure.symbol<@foo, "fn", #kgen.closure_method<call>, <:index ?, :!kgen.param_closure<@foo "fn"> #kgen.closure<@foo "fn">> >}
  > : !kgen.type


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x>) -> index {
  %0 = kgen.call_param[(!kgen.pointer<x>) -> index: bind_params(:<index>(!kgen.pointer<none>) -> index get_vtable_entry(x, "__call__"), 3)](%arg0)
  kgen.return %0 : index
}

// COM: Verify that single param capture does not result in disassembly
// CHECK: kgen.generator @foo_fn
// CHECK-SAME: <A, C>(%arg0: !kgen.pointer<struct<(index)>>) -> index {
// CHECK-NEXT: kgen.struct.gep
// CHECK-NEXT: pop.load
// CHECK-NEXT: <mul(A, C)>

// COM: Verify single capture does not result in struct.
// CHECK-NOT: CAPTURES
// CHECK: kgen.call @consume<:type #type_value>
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


// CHECK:  "__move__" :
// CHECK-SAME: (!kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>,
// CHECK-SAME: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>) -> ()
// CHECK-SAME: = @foo_move_fn<C>
#type_value =
  #kgen.type<!kgen.closure<@foo, "fn" escaping>,
    { "__move__" :
      (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>>, !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>) -> () =
      #kgen.closure.symbol<@foo, "fn", #kgen.closure_method<move>, <:!kgen.param_closure<@foo "fn"> #kgen.closure<@foo "fn">> >}
  > : !kgen.type


kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x>) -> index {
  %0 = kgen.call_param[(!kgen.pointer<x>) -> index: bind_params(:<index>(!kgen.pointer<none>) -> index get_vtable_entry(x, "__call__"), 3)](%arg0)
  kgen.return %0 : index
}

  // CHECK: kgen.generator @foo_move_fn<C>(%arg0: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>, %arg1: !kgen.pointer<struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>) {
  // CHECK-NEXT:  %0 = kgen.struct.gep %arg1[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>
  // CHECK-NEXT:  %1 = kgen.struct.gep %arg0[0] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>
  // CHECK-NEXT:  kgen.call @move(%1, %0) : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>) -> ()
  // CHECK-NEXT:  %2 = kgen.struct.gep %arg1[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>
  // CHECK-NEXT:  %3 = kgen.struct.gep %arg0[1] : <struct<(struct<(index, pointer<index>)>, struct<(index, pointer<index>)>)>>
  // CHECK-NEXT:  kgen.call @move(%3, %2) : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>) -> ()
  // CHECK-NEXT:  kgen.return
  // CHECK-NEXT:  }
kgen.generator @foo<C>(%arg0 : !kgen.pointer<struct<(index, pointer<index>)>>, %arg1 : !kgen.pointer<struct<(index, pointer<index>)>>) {
  %3 = kgen.closure.init(%arg0[@move, @del], %arg1[@copy, @move, @del])<A>() -> index {
	%0 = kgen.param.constant = <mul(C, A)>
	kgen.return %0 : index
  } : (!kgen.pointer<struct<(index, pointer<index>)>>, !kgen.pointer<struct<(index, pointer<index>)>>), !kgen.pointer<!kgen.closure<@foo, "fn" escaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" escaping>>) -> index
  kgen.return
}

kgen.generator @copy(%arg0:!kgen.pointer<struct<(index, pointer<index>)>>, %arg1:!kgen.pointer<struct<(index, pointer<index>)>>) {
    kgen.return
}
kgen.generator @move(%arg0:!kgen.pointer<struct<(index, pointer<index>)>>, %arg1:!kgen.pointer<struct<(index, pointer<index>)>>) {
    kgen.return
}
kgen.generator @del(%arg0: !kgen.pointer<struct<(index, pointer<index>)>>) {
    kgen.return
}
