// RUN: kgen-opt %s -lower-kgen-list -allow-unregistered-dialect -cse | FileCheck %s

// CHECK-LABEL: @list_in_struct
// CHECK-SAME: %arg0: !pop.struct<i32, array<2, index>, i32>
// CHECK-SAME: -> !pop.array<2, index>
kgen.func @list_in_struct(%s: !pop.struct<i32, !kgen.list<index[2]>, i32>) -> !kgen.list<index[2]> {
  // CHECK-NEXT: extract %arg0
  // CHECK-NEXT: extract %arg0
  // CHECK-NEXT: extract %arg0
  %1 = pop.struct.extract %s[0] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  %2 = pop.struct.extract %s[1] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  %3 = pop.struct.extract %s[2] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: replace %{{.*}}, %{{.*}}[0] : !pop.struct<i32, array<2, index>, i32>
  %4 = pop.struct.replace %1, %s[0] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: replace %{{.*}}, %{{.*}}[1] : !pop.struct<i32, array<2, index>, i32>
  // CHECK-NEXT: replace %{{.*}}, %{{.*}}[2] : !pop.struct<i32, array<2, index>, i32>
  %5 = pop.struct.replace %2, %4[1] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  %6 = pop.struct.replace %3, %5[2] : !pop.struct<i32, !kgen.list<index[2]>, i32>

  // CHECK-NEXT: "hold"(%{{.*}}) : (!pop.struct<i32, array<2, index>, i32>)
  "hold"(%6) : (!pop.struct<i32, !kgen.list<index[2]>, i32>) -> ()

  // CHECK-NEXT: return %1 : !pop.array<2, index>
  kgen.return %2 : !kgen.list<index[2]>
}

// CHECK-LABEL: @empty_list
// CHECK-SAME: %arg0: !pop.struct<array<0, index>, i32>
// CHECK-NOT: %arg1
kgen.func @empty_list(%s: !pop.struct<!kgen.list<index[0]>, i32>) -> !kgen.list<index[0]> {
  // CHECK-NEXT: pop.struct.extract %{{.*}}[0] : !pop.struct<array<0, index>, i32>
  %1 = pop.struct.extract %s[0] : !pop.struct<!kgen.list<index[0]>, i32>
  // CHECK-NEXT: pop.struct.extract %{{.*}}[1] : !pop.struct<array<0, index>, i32>
  %2 = pop.struct.extract %s[1] : !pop.struct<!kgen.list<index[0]>, i32>
  // CHECK-NEXT: "hold"
  "hold"(%2) : (i32) -> ()
  // CHECK: return %0 : !pop.array<0, index>
  kgen.return %1 : !kgen.list<index[0]>
}

// CHECK-LABEL: @nested_list_in_struct
// CHECK-SAME: %arg0: !pop.struct<array<2, array<2, index>>>
// CHECK-SAME: %arg1: !pop.array<2, array<2, index>>
// CHECK-SAME: -> !pop.array<2, array<2, index>>
kgen.func @nested_list_in_struct(%s: !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>, %list: !kgen.list<!kgen.list<index[2]>[2]>) -> !kgen.list<!kgen.list<index[2]>[2]> {
  // CHECK-NEXT: replace %arg1, %arg0[0]
  %0 = pop.struct.replace %list, %s[0] : !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>

  // CHECK-NEXT: %{{.*}} = "foo"(%arg1) : (!pop.array<2, array<2, index>>) -> !pop.array<2, array<2, index>>
  %1 = "foo"(%list) : (!kgen.list<!kgen.list<index[2]>[2]>) -> !kgen.list<!kgen.list<index[2]>[2]>

  // CHECK-NEXT: pop.struct.extract %arg0[0] : !pop.struct<array<2, array<2, index>>>
  %2 = pop.struct.extract %s[0] : !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>

  // CHECK-NEXT: hold
  "hold"(%0) : (!pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>) -> ()

  // CHECK-NEXT: pop.struct.create(%arg1)
  %3 = pop.struct.create(%list) : !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>

  // CHECK-NEXT: hold
  "hold"(%3) : (!pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>) -> ()
  kgen.return %2 : !kgen.list<!kgen.list<index[2]>[2]>
}

// CHECK-LABEL: @nested_list_not_boundary
// CHECK-SAME: %arg0: !pop.struct<array<2, pointer<array<2, index>>>>
kgen.func @nested_list_not_boundary(%a: !pop.struct<!kgen.list<!pop.pointer<!kgen.list<index[2]>>[2]>>) {
  kgen.return
}

// CHECK-LABEL: @list_pointer
// CHECK-SAME: %arg0: !pop.pointer<array<2, index>>
// CHECK-SAME: -> !pop.array<2, index>
kgen.func @list_pointer(%ptr: !pop.pointer<!kgen.list<index[2]>>) -> !kgen.list<index[2]> {
  // CHECK: %0 = pop.load %arg0
  %0 = pop.load %ptr align 16 : !pop.pointer<!kgen.list<index[2]>>

  // CHECK-NEXT: pop.store %0, %arg0
  pop.store %0, %ptr align 16 : !pop.pointer<!kgen.list<index[2]>>

  // CHECK-NEXT: kgen.return %0
  kgen.return %0 : !kgen.list<index[2]>
}

// CHECK-LABEL: @struct_list_gep
// CHECK-SAME: %arg0: !pop.pointer<struct<array<2, index>>>
kgen.func @struct_list_gep(%ptr: !pop.pointer<struct<!kgen.list<index[2]>>>,
                           %empty: !pop.pointer<struct<!kgen.list<i1[0]>>>) -> !pop.pointer<!kgen.list<index[2]>> {
  // CHECK-NEXT: %0 = pop.struct.gep %arg0[0]
  // CHECK-NEXT: %1 = pop.struct.gep %arg1[0]
  %0 = pop.struct.gep %ptr[0] : <struct<!kgen.list<index[2]>>>
  %1 = pop.struct.gep %empty[0] : <struct<!kgen.list<i1[0]>>>
  // CHECK-NEXT: "use"(%1)
  "use"(%1) : (!pop.pointer<!kgen.list<i1[0]>>) -> ()
  // CHECK-NEXT: return %0
  kgen.return %0 : !pop.pointer<!kgen.list<index[2]>>
}

// CHECK-LABEL: @struct_list_gep_empty
// CHECK-SAME: %arg0: !pop.pointer<struct<array<0, index>>>
kgen.func @struct_list_gep_empty(%ptr: !pop.pointer<struct<!kgen.list<index[0]>>>) -> !kgen.list<index[0]> {
  // CHECK: %0 = pop.struct.gep %arg0[0]
  %0 = pop.struct.gep %ptr[0] : <struct<!kgen.list<index[0]>>>
  // CHECK: %1 = pop.load %0
  %1 = pop.load %0 : !pop.pointer<!kgen.list<index[0]>>
  // CHECK-NEXT: return %1
  kgen.return %1 : !kgen.list<index[0]>
}

// CHECK-LABEL: @do_something
// CHECK-SAME: %arg0: !pop.array<1, array<1, index>>
// CHECK-SAME: -> (!pop.array<1, array<1, index>>, i1)
kgen.func @do_something(%list: !kgen.list<!kgen.list<index[1]>[1]>) -> (!kgen.list<!kgen.list<index[1]>[1]>, i1) {
  %true = index.bool.constant true
  kgen.return %list, %true : !kgen.list<!kgen.list<index[1]>[1]>, i1
}

// CHECK-LABEL: @hlcf_scf_loops
// CHECK-SAME: %arg0: !pop.array<1, array<1, index>>
// CHECK-SAME: -> !pop.array<1, array<1, index>>
kgen.func @hlcf_scf_loops(%list: !kgen.list<!kgen.list<index[1]>[1]>, %cond: i1, %ub: index) -> !kgen.list<!kgen.list<index[1]>[1]> {
  %zero = index.constant 0
  %one = index.constant 1
  // CHECK: %{{.*}} = hlcf.loop (%{{.*}} = %arg0 : !pop.array<1, array<1, index>>) -> !pop.array<1, array<1, index>>
  %2 = hlcf.loop (%a = %list : !kgen.list<!kgen.list<index[1]>[1]>) -> !kgen.list<!kgen.list<index[1]>[1]> {
    %3, %4 = kgen.call @do_something(%a) : (!kgen.list<!kgen.list<index[1]>[1]>) -> (!kgen.list<!kgen.list<index[1]>[1]>, i1)
    hlcf.if %4 {
      // CHECK: hlcf.break %{{.*}} : !pop.array<1, array<1, index>>
      hlcf.break %3 : !kgen.list<!kgen.list<index[1]>[1]>
    } else {
      hlcf.yield
    }
    // CHECK: kgen.return %{{.*}} : !pop.array<1, array<1, index>>
    kgen.return %3 : !kgen.list<!kgen.list<index[1]>[1]>
  }
  kgen.return %2 : !kgen.list<!kgen.list<index[1]>[1]>
}

// CHECK-LABEL: @list_get_op
kgen.func @list_get_op(%list: !kgen.list<index[3]>) -> index {
  %0 = pop.list.get %list[1] : <index[3]>
  // CHECK: return %0 : index
  kgen.return %0 : index
}

// CHECK-LABEL: @list_create_op
kgen.func @list_create_op(%arg0: index, %arg1: index) -> !kgen.list<index[2]> {
  %list = pop.list.create(%arg0, %arg1) : <index[2]>
  // CHECK: return %0 : !pop.array<2, index>
  kgen.return %list : !kgen.list<index[2]>
}

// CHECK-LABEL: @list_in_coroutine
// CHECK-SAME: !pop.coroutine<() -> !pop.array<0, index>>
kgen.func @list_in_coroutine(%coro: !pop.coroutine<() -> !kgen.list<index[0]>>) {
  kgen.return
}

// CHECK-LABEL: @list_attr_in_attr
kgen.func @list_attr_in_attr() {
  // CHECK-NEXT: struct<array<2, index>, array<3, i32>> = <{ [1, 2], [5, 6, 7] }>
  %0 = kgen.param.constant: struct<!kgen.list<index[2]>, !kgen.list<i32[3]>>
    = <{ [1, 2], [5, 6, 7] }>
  // CHECK-NEXT: struct<array<2, array<2, index>>> = <{ {{.*}}[1, 2], [3, 4]] }>
  %1 = kgen.param.constant: struct<!kgen.list<!kgen.list<index[2]>[2]>> = <{ [[1, 2], [3, 4]] }>
  // CHECK-NEXT: variant<array<0, i1>, array<2, array<2, i32>>> = <#pop.variant<:array<0, i1> []>>
  %2 = kgen.param.constant: variant<!kgen.list<i1[0]>, array<2, !kgen.list<i32[2]>>> = <#pop.variant<:!kgen.list<i1[0]> []>>

  "use"(%0, %1, %2) : (
    !pop.struct<!kgen.list<index[2]>, !kgen.list<i32[3]>>,
    !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>,
    !pop.variant<!kgen.list<i1[0]>, array<2, !kgen.list<i32[2]>>>
  ) -> ()

  kgen.return
}

// CHECK-LABEL: @return_none
kgen.func @return_none() -> !kgen.list<i1[0]> {
  %list = pop.list.create() : <i1[0]>
  kgen.return %list : !kgen.list<i1[0]>
}
