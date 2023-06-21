// RUN: kgen-opt %s -lower-structs | FileCheck %s
lit.struct.decl @Bar {
  lit.struct.field x : !kgen.declref<@Pointer<type: type = !kgen.declref<@Foo>>>
  lit.struct.field y : ui32
}

lit.struct.decl @Foo {
  lit.struct.field x : !kgen.declref<@Bar>
  lit.struct.field y : f32
}

lit.struct.decl @Pointer<type: type> {
  lit.struct.field address : !pop.pointer<type>
}

!bar_ref = !kgen.declref<@Bar>
!foo_ref = !kgen.declref<@Foo>
!foo_ptr_ref = !kgen.declref<@Pointer<type: type = !foo_ref>>
!null_ptr = !pop.pointer<scalar<invalid>>

// CHECK-LABEL: @gepFooFromBar
kgen.func @gepFooFromBar(%s: !pop.pointer<@Bar>) ->
!pop.pointer<@Pointer<type: type = !foo_ref>> {
  // CHECK: %0 = pop.struct.gep %arg0[0] : <struct<pointer<struct<struct<pointer<scalar<invalid>>, ui32>, f32>>, ui32>>
  // CHECK: kgen.return %0 : !pop.pointer<pointer<struct<struct<pointer<scalar<invalid>>, ui32>, f32>>>
  %0 = lit.struct.gep %s[x] : <@Pointer<type: type = !foo_ref>> from <@Bar>
  kgen.return %0 : !pop.pointer<@Pointer<type: type = !foo_ref>>
}

// CHECK-LABEL: @makeFoo
kgen.func @makeFoo(%arg0: !bar_ref, %arg1: f32) -> !foo_ref {
  // CHECK: %0 = pop.struct.create(%arg0, %arg1) : !pop.struct<struct<pointer<scalar<invalid>>, ui32>, f32>
  // CHECK: kgen.return %0 : !pop.struct<struct<pointer<scalar<invalid>>, ui32>, f32>
  %0 = lit.struct.create(x=%arg0, y=%arg1) : (!bar_ref, f32) -> !foo_ref
  kgen.return %0 : !foo_ref
}

// CHECK-LABEL: @makeBar
kgen.func @makeBar(%arg0: !foo_ptr_ref, %arg1: ui32) -> !bar_ref {
  // CHECK: %0 = pop.struct.create(%arg0, %arg1) : !pop.struct<pointer<struct<struct<pointer<scalar<invalid>>, ui32>, f32>>, ui32>
  // CHECK: kgen.return %1 : !pop.struct<pointer<scalar<invalid>>, ui32>
  %0 = lit.struct.create(x=%arg0, y=%arg1) : (!foo_ptr_ref, ui32) -> !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertUIntToBar
kgen.func @structInsertUIntToBar(%arg0: ui32, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: %0 = pop.struct.replace %arg0, %arg1[1] : !pop.struct<pointer<scalar<invalid>>, ui32>
  // CHECK: kgen.return %0 : !pop.struct<pointer<scalar<invalid>>, ui32>
  %0 = lit.struct.insert %arg0, %arg1[y] : ui32 into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertFooPtrToBar
kgen.func @structInsertFooPtrToBar(%arg0: !foo_ptr_ref, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: %0 = pop.pointer.bitcast %arg0 : !pop.pointer<struct<struct<pointer<scalar<invalid>>, ui32>, f32>> to !pop.pointer<scalar<invalid>>
  // CHECK: %1 = pop.struct.replace %0, %arg1[0] : !pop.struct<pointer<scalar<invalid>>, ui32>
  // CHECK: kgen.return %1 : !pop.struct<pointer<scalar<invalid>>, ui32>
  %0 = lit.struct.insert %arg0, %arg1[x] : !foo_ptr_ref into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertBarToFoo
kgen.func @structInsertBarToFoo(%arg0: !foo_ptr_ref, %arg1: ui32,  %arg2: !foo_ref) -> !foo_ref {
  // CHECK: %0 = kgen.call @makeBar(%arg0, %arg1) : (!pop.pointer<struct<struct<pointer<scalar<invalid>>, ui32>, f32>>, ui32) -> !pop.struct<pointer<scalar<invalid>>, ui32>
  // CHECK: %1 = pop.struct.replace %0, %arg2[0] : !pop.struct<struct<pointer<scalar<invalid>>, ui32>, f32>
  // CHECK: kgen.return %1 : !pop.struct<struct<pointer<scalar<invalid>>, ui32>, f32>
  %0 = kgen.call @makeBar(%arg0, %arg1): (!foo_ptr_ref, ui32) -> !bar_ref
  %1 = lit.struct.insert %0, %arg2[x] : !bar_ref into !foo_ref
  kgen.return %1 : !foo_ref
}

// CHECK-LABEL: @structExtractFooFromBar
kgen.func @structExtractFooFromBar(%arg0: !bar_ref) -> !foo_ptr_ref {
  // CHECK: %0 = pop.struct.extract %arg0[0] : !pop.struct<pointer<scalar<invalid>>, ui32>
  // CHECK: %1 = pop.pointer.bitcast %0 : !pop.pointer<scalar<invalid>> to !pop.pointer<struct<struct<pointer<scalar<invalid>>, ui32>, f32>>
  // CHECK: kgen.return %1 : !pop.pointer<struct<struct<pointer<scalar<invalid>>, ui32>, f32>>
  %0 = lit.struct.extract %arg0[x] : !foo_ptr_ref from !bar_ref
  kgen.return %0 : !foo_ptr_ref
}

// CHECK-LABEL: @structExtractBarFromFoo
kgen.func @structExtractBarFromFoo(%arg0: !foo_ref) -> !bar_ref {
  // CHECK: %0 = pop.struct.extract %arg0[0] : !pop.struct<struct<pointer<scalar<invalid>>, ui32>, f32>
  // CHECK: kgen.return %0 : !pop.struct<pointer<scalar<invalid>>, ui32>
  %0 = lit.struct.extract %arg0[x] : !bar_ref from !foo_ref
  kgen.return %0 : !bar_ref
}

lit.struct.decl @Recursive {
  lit.struct.field x : !pop.pointer<@Recursive>
}

// CHECK-LABEL: @thing
// CHECK: -> !kgen.declref<@Recursive>
kgen.generator @thing() -> !kgen.declref<@Recursive> {
  // CHECK: kgen.unreachable
  kgen.unreachable
}
