// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true allow-multiple-primary-impls=true" -allow-unregistered-dialect | FileCheck %s

kgen.generator @recursive(%arg0: index) -> index {
  %idx1 = index.constant 1
  %0 = index.cmp sge(%arg0, %idx1)
  hlcf.if %0 {
    %1 = index.sub %arg0, %idx1
    %2 = kgen.call @recursive(%1) : (index) -> index
    kgen.return %2 : index
  } else {
    hlcf.yield
  }
  kgen.return %idx1 : index
}

// CHECK-LABEL: kgen.func export @recursive_return_after_call
kgen.generator export @recursive_return_after_call() {
  kgen.param.apply x = [(index) -> index: @recursive](5)
  // CHECK-NEXT: <1>
  kgen.param.constant = <x>
  kgen.return
}

// -----

kgen.generator @fma(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = index.mul %arg1, %arg2
  %1 = index.add %0, %arg0
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func export @constexpr_fma
kgen.generator export @constexpr_fma() -> index {
  // CHECK-NEXT: kgen.param.constant = <7>
  %0 = kgen.param.constant = <apply(:(index, index, index) -> index @fma, 1, 2, 3)>
  kgen.return %0 : index
}

kgen.generator @init_self(%arg0: !kgen.pointer<index> init_self, %arg1: index) {
  %idx1 = index.constant 1
  %0 = index.add %idx1, %arg1
  pop.store %0, %arg0 : !kgen.pointer<index>
  kgen.return
}

kgen.generator @byref_result(%arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<index> byref_result) {
  %0 = pop.load %arg0 : !kgen.pointer<index>
  %idx2 = index.constant 2
  %1 = index.mul %idx2, %0
  pop.store %1, %arg1 : !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: kgen.param.constant = <2048>
  kgen.param.declare value = <apply_result_slot(:(!kgen.pointer<index> init_self, index) -> () @init_self, 1023)>
  kgen.param.constant = <apply_result_slot(:(!kgen.pointer<index>, !kgen.pointer<index> byref_result) -> () @byref_result, store_to_mem(value))>
  kgen.return
}

// -----

kgen.generator @alloc_load_store(%arg0: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx3 = index.constant 3

  %p0 = pop.stack_allocation 4 x index
  pop.store %idx0, %p0 : !kgen.pointer<index>
  %p1 = pop.offset %p0[%idx1] : !kgen.pointer<index>
  pop.store %idx1, %p1 : !kgen.pointer<index>
  %p2 = pop.offset %p0[%idx2] : !kgen.pointer<index>
  pop.store %idx2, %p2 : !kgen.pointer<index>
  %p3 = pop.offset %p1[%idx2] : !kgen.pointer<index>
  pop.store %idx3, %p3 : !kgen.pointer<index>

  %ptr = pop.offset %p0[%arg0] : !kgen.pointer<index>
  %result = pop.load %ptr : !kgen.pointer<index>
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @constexpr_load_store
kgen.generator @constexpr_load_store() {
  // CHECK-NEXT: = <0>
  %0 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 0)>
  // CHECK-NEXT: = <1>
  %1 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 1)>
  // CHECK-NEXT: = <2>
  %2 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 2)>
  // CHECK-NEXT: = <3>
  %3 = kgen.param.constant = <apply(:(index) -> index @alloc_load_store, 3)>
  kgen.return
}

// -----

kgen.generator @return_it<A>() -> index {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:() -> index bind_signature(:<index>() -> index @return_it, 1))>
  // CHECK-NEXT: <2>
  kgen.param.constant = <apply(:() -> index bind_signature(:<index>() -> index @return_it, 2))>
  // CHECK-NEXT: <3>
  kgen.param.constant = <apply(:() -> index bind_signature(:<index>() -> index @return_it,
    apply(:() -> index bind_signature(:<index>() -> index @return_it, 3))))>
  kgen.return
}

// -----

kgen.generator @callee(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

kgen.generator @func(%arg0: index) -> index {
  %0 = kgen.call @callee(%arg0) : (index) -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() -> index {
  // CHECK-NEXT: <14>
  %0 = kgen.param.constant = <apply(:(index) -> index @func, 7)>
  kgen.return %0 : index
}

// -----

kgen.generator @sum(%from: index, %to: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %result = hlcf.loop (%acc = %idx0 : index, %i = %from : index) -> index {
    %cond = index.cmp sle(%i, %to)
    hlcf.if %cond {
      hlcf.yield
    } else {
      hlcf.break %acc : index
    }
    %nextI = index.add %idx1, %i
    %nextAcc = index.add %acc, %i
    hlcf.continue %nextAcc, %nextI : index, index
  }
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <55>
  kgen.param.constant = <apply(:(index, index) -> index @sum, 0, 10)>
  kgen.return
}

// -----

kgen.generator @early_return(%cond: i1) -> index {
  %idx0 = index.constant 0
  %result = hlcf.if %cond -> index {
    hlcf.yield %idx0 : index
  } else {
    %idx1 = index.constant 1
    kgen.return %idx1 : index
  }
  kgen.return %result : index
}

// CHECK-LABEL: kgen.func @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(i1) -> index @early_return, 0)>
  // CHECK-NEXT: <0>
  kgen.param.constant = <apply(:(i1) -> index @early_return, 1)>
  kgen.return
}

// CHECK-LABEL: kgen.func @"rebind_value,dtype=ui8"
kgen.generator @rebind_value<dtype: dtype>(%a: !pop.scalar<ui8>) -> !pop.scalar<dtype> {
  // CHECK-NEXT: return %arg0 : !pop.scalar<ui8>
  %result = kgen.rebind %a : !pop.scalar<ui8> to !pop.scalar<dtype>
  kgen.return %result : !pop.scalar<dtype>
}

// CHECK-LABEL: kgen.func @rebind_it
kgen.generator @rebind_it() {
  // CHECK-NEXT: constant: scalar<ui8> = <4>
  kgen.param.declare Fn: (!pop.scalar<ui8>) -> !pop.scalar<ui8> =
    <bind_signature(:<dtype>(!pop.scalar<ui8>) -> !pop.scalar<*(0,0)> @rebind_value, ui8)>
  kgen.param.constant: scalar<ui8> = <apply(:(!pop.scalar<ui8>) -> !pop.scalar<ui8> Fn, <4>)>
  kgen.return
}

// -----

kgen.generator @box(%a: index) -> !kgen.struct<(index)> {
  %0 = kgen.struct.create(%a) : !kgen.struct<(index)>
  kgen.return %0 : !kgen.struct<(index)>
}

kgen.generator @unbox(%a: !kgen.struct<(index)>) -> index {
  %0 = kgen.struct.extract %a[0] : !kgen.struct<(index)>
  kgen.return %0 : index
}

kgen.generator @callee<a: !kgen.struct<(index)>>(
    %a: !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox, a), index>) {
  kgen.return
}

// CHECK-LABEL: kgen.func @unbox_in_result_sig
kgen.generator @unbox_in_result_sig() {
  // CHECK-NEXT: kgen.create_closure[(!pop.array<2, index>) -> (): @"callee,a={ 2 }"]()
  kgen.param.declare a = <2>
  kgen.param.declare fn: <!kgen.struct<(index)>>(
    !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox, *(0,0)), index>
  ) -> () = <@callee>
  kgen.create_closure[(
    !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox,
                     apply(:(index) -> !kgen.struct<(index)> @box, a)),
               index>) -> ():
    bind_signature(:<!kgen.struct<(index)>>(
      !pop.array<apply(:(!kgen.struct<(index)>) -> index @unbox, *(0,0)), index>
     ) -> () fn, apply(:(index) -> !kgen.struct<(index)> @box, a))]()
  kgen.return
}

// -----

kgen.generator @make_one() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @parametric_const
kgen.generator @parametric_const() {
  // CHECK-NEXT: constant: variant<index, scalar<f32>> = <{1, 0}>
  kgen.param.constant: variant<index, simd<apply(:() -> index @make_one), f32>> = <{1, 0}>
  kgen.return
}

// -----

kgen.generator @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

kgen.generator @make_array<size>() -> !pop.array<apply(:(index) -> index @pass, size), i1> {
  %false = index.bool.constant false
  %0 = pop.array.repeat [%false] : !pop.array<apply(:(index) -> index @pass, size), i1>
  kgen.return %0 : !pop.array<apply(:(index) -> index @pass, size), i1>
}

// CHECK-LABEL: kgen.func @caller
kgen.generator @caller() {
  // CHECK-NEXT: array<2, i1> = <[0, 0]>
  kgen.param.constant: array<apply(:(index) -> index @pass, 2), i1> = <
    apply(:() -> !pop.array<apply(:(index) -> index @pass, 2), i1> @make_array<2>)
  >
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @some_function
kgen.generator @some_function() {
  kgen.return
}

kgen.generator @return_closure_formation() -> !kgen.signature<() -> ()> {
  %0 = kgen.create_closure[() -> (): @some_function]()
  kgen.return %0 : !kgen.signature<() -> ()>
}

// CHECK-LABEL: kgen.func export @interpret_create_closure
kgen.generator export @interpret_create_closure() {
  // CHECK-NEXT: constant: () -> () = <@some_function>
  kgen.param.constant: () -> () = <apply(:() -> !kgen.signature<() -> ()> @return_closure_formation)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @a_function
kgen.generator @a_function() {
  kgen.return
}

kgen.generator @load_store_function(%arg0: !kgen.signature<() -> ()>) -> !kgen.signature<() -> ()> {
  %0 = pop.stack_allocation 1 x !kgen.signature<() -> ()>
  pop.store %arg0, %0 : !kgen.pointer<() -> ()>
  %1 = pop.load %0 : !kgen.pointer<() -> ()>
  kgen.return %1 : !kgen.signature<() -> ()>
}

// CHECK-LABEL: kgen.func export @call_it
kgen.generator export @call_it() {
  // CHECK-NEXT: constant: () -> () = <@a_function>
  kgen.param.constant: () -> () = <apply(:(!kgen.signature<() -> ()>) -> !kgen.signature<() -> ()> @load_store_function, @a_function)>
  kgen.return
}

// -----

kgen.generator @store_variadic(%arg0: !kgen.variadic<index>, %arg1: !kgen.pointer<variadic<index>>) {
  pop.store %arg0, %arg1 : !kgen.pointer<variadic<index>>
  kgen.return
}

kgen.generator @pass_and_read_variadic(%arg0: !kgen.variadic<index>) -> !kgen.variadic<index> {
  %0 = pop.stack_allocation 1 x !kgen.variadic<index>
  kgen.call @store_variadic(%arg0, %0) : (!kgen.variadic<index>, !kgen.pointer<variadic<index>>) -> ()
  %1 = pop.load %0 : !kgen.pointer<variadic<index>>
  kgen.return %1 : !kgen.variadic<index>
}

// CHECK-LABEL: kgen.func export @persistent_variadic
kgen.generator export @persistent_variadic() {
  // CHECK-NEXT: variadic<index> = <[1, 2]>
  kgen.param.constant: variadic<index> = <apply(:(!kgen.variadic<index>) -> !kgen.variadic<index> @pass_and_read_variadic, [1, 2])>
  kgen.return
}

// -----

kgen.generator @variadic_of_pointers(%arg0: !kgen.variadic<pointer<index>>) -> index {
  %idx0 = index.constant 0
  %0 = pop.variadic.get %arg0[%idx0] : !kgen.variadic<pointer<index>>
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func export @interp_nested_pointer
kgen.generator export @interp_nested_pointer() {
  // CHECK-NEXT: <1>
  kgen.param.constant = <apply(:(!kgen.variadic<pointer<index>>) -> index @variadic_of_pointers, [store_to_mem(1)])>
  kgen.return
}

// -----

kgen.generator @size_zero_alloc() -> !kgen.pointer<index> {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = pop.aligned_alloc %idx1, %idx0 : <index>
  kgen.return %0 : !kgen.pointer<index>
}

// CHECK-LABEL: kgen.func export @malloc_nullptr
kgen.generator export @malloc_nullptr() {
  // CHECK-NEXT: pointer<index> = <0>
  %0 = kgen.param.constant: pointer<index> = <apply(:() -> !kgen.pointer<index> @size_zero_alloc)>
  kgen.return
}

// -----

// COM: This test ensures scalar data and a pointer region can co-exist
// COM: correctly in the same memory allocation.

// CHECK: [[BLOB1:#.*]] = #interp.memory_handle<8, "0x0100000000000000000000000000000020CA9A3B000000000000000000000000">
// CHECK: [[BLOB2:#.*]] = #interp.memory_handle<1, "0x00">

!ptr_t = !kgen.pointer<variant<index, pointer<none>>>

kgen.generator @fill_ptr() -> !ptr_t {
  %idx1 = index.constant 1
  %idx8 = index.constant 8
  %idx32 = index.constant 32

  // Allocate an array of 2 elements.
  %0 = pop.aligned_alloc %idx8, %idx32 : !ptr_t

  // Store 1 to the first element.
  %1 = pop.pointer.bitcast %0 : !ptr_t to !kgen.pointer<index>
  pop.store %idx1, %1 : !kgen.pointer<index>

  // Store a pointer to the second element.
  %2 = pop.offset %0[%idx1] : !ptr_t
  %3 = pop.aligned_alloc %idx1, %idx1 : !kgen.pointer<none>
  %4 = pop.pointer.bitcast %2 : !ptr_t to !kgen.pointer<pointer<none>>
  pop.store %3, %4 : !kgen.pointer<pointer<none>>

  kgen.return %0 : !ptr_t
}

// CHECK-LABEL: kgen.func export @pointer_overwrite
kgen.generator export @pointer_overwrite() {
  // CHECK-NEXT: memref<[([[BLOB1]], heap, [(16, 1, 0)]), ([[BLOB2]], heap, [])], 0, 0>>
  kgen.param.constant: !ptr_t = <apply(:() -> !ptr_t @fill_ptr)>
  kgen.return
}

// -----

kgen.generator @elif(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %0 = hlcf.elif -> index {
    %cond0 = index.cmp eq(%arg0, %idx0)
    hlcf.elif.yield %cond0 : i1
  } then {
    hlcf.yield %idx0 : index
  } {
    %cond1 = index.cmp eq(%arg0, %idx1)
    hlcf.elif.yield %cond1 : i1
  } then {
    hlcf.yield %idx1 : index
  } {
    %cond2 = index.cmp eq(%arg0, %idx2)
    hlcf.elif.yield %cond2 : i1
  } then {
    hlcf.yield %idx2 : index
  } else {
    hlcf.yield %arg1 : index
  }

  %2 = index.mul %0, %0
  kgen.return %2 : index
}

// CHECK-LABEL: kgen.func export @constexpr_elif
kgen.generator export @constexpr_elif() -> index {
  // CHECK-NEXT: kgen.param.constant = <4>
  %0 = kgen.param.constant = <apply(:(index, index) -> index @elif, 2, 3)>
  // CHECK-NEXT: kgen.param.constant = <25>
  %1 = kgen.param.constant = <apply(:(index, index) -> index @elif, 3, 5)>
  kgen.return %1 : index
}

// -----

kgen.func @elifWithArgs(%arg0: index) -> index {
  %idx3 = index.constant 3
  %idx1 = index.constant 1
  %0:2 = hlcf.elif -> index, index {
    %2 = index.add %arg0, %idx1
    %3 = index.cmp eq(%2, %idx3)
    hlcf.elif.yield %3, %2 : i1, index
  } then (%arg1 : index) {
    hlcf.yield %arg1, %arg1 : index, index
  } else (%arg1 : index) {
    hlcf.yield %idx1, %idx1 : index, index
  }
  %1 = index.add %0#1, %0#0
  kgen.return %1 : index
}

kgen.func @elifManyRegionsWithArgs(%arg0: index) -> index {
  %idx3 = index.constant 3
  %idx1 = index.constant 1
  %idx4 = index.constant 4
  %0:2 = hlcf.elif -> index, index {
    %2 = index.add %arg0, %idx1
    %3 = index.cmp eq(%2, %idx3)
    hlcf.elif.yield %3, %2 : i1, index
  } then (%arg1 : index) {
    hlcf.yield %arg1, %arg1 : index, index
  } (%arg1 : index) {
    %4 = index.cmp eq(%arg0, %idx4)
    %5 = index.add %arg1, %idx3
    hlcf.elif.yield %4, %5 : i1, index
  } then (%arg1 : index) {
    hlcf.yield %arg1, %arg1 : index, index
  } else (%arg1 : index) {
    hlcf.yield %idx1, %idx1 : index, index
  }
  %1 = index.add %0#1, %0#0
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func export @constexpr_elif
kgen.generator export @constexpr_elif() -> index {
  // CHECK: kgen.param.constant = <6>
  %0 = kgen.param.constant = <apply(:(index) -> index @elifWithArgs, 2)>
  // CHECK: kgen.param.constant = <16>
  %1 = kgen.param.constant = <apply(:(index) -> index @elifManyRegionsWithArgs, 4)>
  kgen.return %0 : index
}

// -----

kgen.func @pack_load() -> !kgen.pack<[si4, ui8]> {
  %i0 = kgen.param.constant: si4 = <-5>
  %i1 = kgen.param.constant: ui8 = <42>
  %p0 = pop.stack_allocation 1 x si4
  %p1 = pop.stack_allocation 1 x ui8
  pop.store %i0, %p0 : !kgen.pointer<si4>
  pop.store %i1, %p1 : !kgen.pointer<ui8>
  %pack = kgen.pack.create(%p0, %p1) : !kgen.pack<[pointer<si4>, pointer<ui8>]>
  %loaded_pack = kgen.pack.load %pack : !kgen.pack<[pointer<si4>, pointer<ui8>]>
  kgen.return %loaded_pack : !kgen.pack<[si4, ui8]>
}

// CHECK-LABEL: kgen.func export @interpret_pack_load
kgen.generator export @interpret_pack_load() -> !kgen.pack<[si4, ui8]> {
  // CHECK-NEXT: !kgen.pack<[si4, ui8]> = <<-5, 42>>
  %0 = kgen.param.constant: !kgen.pack<[si4, ui8]> = <apply(:() -> !kgen.pack<[si4, ui8]> @pack_load)>
  kgen.return %0 : !kgen.pack<[si4, ui8]>
}
