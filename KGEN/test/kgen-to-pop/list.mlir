// RUN: kgen-opt %s -lower-kgen-to-pop -allow-unregistered-dialect -cse | FileCheck %s

// CHECK-LABEL: @list_in_struct
// CHECK-SAME: %arg0: i32, %arg1: index, %arg2: index
// CHECK-SAME: -> (index, index)
kgen.func @list_in_struct(%a: i32, %list: !kgen.list<index[2]>) -> !kgen.list<index[2]> {
  // CHECK-NEXT: construct(%arg0, %arg1, %arg2, %arg0)
  %0 = pop.struct.construct(%a, %list, %a) : !pop.struct<i32, !kgen.list<index[2]>, i32>

  // CHECK-NEXT: pop.struct.extract %{{.*}}[0] : !pop.struct<i32, index, index, i32>
  %1 = pop.struct.extract %0[0] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: pop.struct.extract %{{.*}}[1]
  // CHECK-NEXT: pop.struct.extract %{{.*}}[2]
  %2 = pop.struct.extract %0[1] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: pop.struct.extract %{{.*}}[3]
  %3 = pop.struct.extract %0[2] : !pop.struct<i32, !kgen.list<index[2]>, i32>

  // CHECK-NEXT: replace %{{.*}}, %{{.*}}[0] : !pop.struct<i32, index, index, i32>
  %4 = pop.struct.replace %1, %0[0] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: replace %{{.*}}, %{{.*}}[1] : !pop.struct<i32, index, index, i32>
  // CHECK-NEXT: replace %{{.*}}, %{{.*}}[2] : !pop.struct<i32, index, index, i32>
  %5 = pop.struct.replace %2, %4[1] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: replace %{{.*}}, %{{.*}}[3] : !pop.struct<i32, index, index, i32>
  %6 = pop.struct.replace %3, %5[2] : !pop.struct<i32, !kgen.list<index[2]>, i32>

  // CHECK-NEXT: "hold"(%{{.*}}) : (!pop.struct<i32, index, index, i32>)
  "hold"(%6) : (!pop.struct<i32, !kgen.list<index[2]>, i32>) -> ()

  // CHECK-NEXT: return %arg1, %arg2 : index, index
  kgen.return %list : !kgen.list<index[2]>
}

// CHECK-LABEL: @empty_list
// CHECK-SAME: %arg0: i32
// CHECK-NOT: %arg1
// CHECK-NOT: ->
kgen.func @empty_list(%a: i32, %list: !kgen.list<index[0]>) -> !kgen.list<index[0]> {
  // CHECK-NEXT: pop.struct.construct(%arg0) : !pop.struct<i32>
  %0 = pop.struct.construct(%list, %a) : !pop.struct<!kgen.list<index[0]>, i32>
  // CHECK-NEXT: pop.struct.extract %{{.*}}[0] : !pop.struct<i32>
  %1 = pop.struct.extract %0[0] : !pop.struct<!kgen.list<index[0]>, i32>
  %2 = pop.struct.extract %0[1] : !pop.struct<!kgen.list<index[0]>, i32>
  // CHECK-NEXT: "hold"
  "hold"(%2) : (i32) -> ()
  // CHECK: return
  // CHECK-NOT: %
  kgen.return %1 : !kgen.list<index[0]>
}

// CHECK-LABEL: @nested_list_in_struct
// CHECK-SAME: %arg0: !pop.struct<index, index, index, index>
// CHECK-SAME: %arg1: index, %arg2: index, %arg3: index, %arg4: index
// CHECK-SAME: -> (index, index, index, index)
kgen.func @nested_list_in_struct(%s: !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>, %list: !kgen.list<!kgen.list<index[2]>[2]>) -> !kgen.list<!kgen.list<index[2]>[2]> {
  // CHECK-NEXT: replace %arg1, %arg0[0]
  // CHECK-NEXT: replace %arg2, %0[1]
  // CHECK-NEXT: replace %arg3, %1[2]
  // CHECK-NEXT: replace %arg4, %2[3]
  %0 = pop.struct.replace %list, %s[0] : !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>

  // CHECK-NEXT: %{{.*}}:4 = "foo"(%arg1, %arg2, %arg3, %arg4) : (index, index, index, index) -> (index, index, index, index)
  %1 = "foo"(%list) : (!kgen.list<!kgen.list<index[2]>[2]>) -> !kgen.list<!kgen.list<index[2]>[2]>

  // CHECK-NEXT: pop.struct.extract %arg0[0] : !pop.struct<index, index, index, index>
  // CHECK-NEXT: pop.struct.extract %arg0[1]
  // CHECK-NEXT: pop.struct.extract %arg0[2]
  // CHECK-NEXT: pop.struct.extract %arg0[3]
  %2 = pop.struct.extract %s[0] : !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>

  // CHECK-NEXT: hold
  "hold"(%0) : (!pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>) -> ()

  // CHECK-NEXT: construct(%arg1, %arg2, %arg3, %arg4)
  %3 = pop.struct.construct(%list) : !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>

  // CHECK-NEXT: hold
  "hold"(%3) : (!pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>) -> ()
  kgen.return %2 : !kgen.list<!kgen.list<index[2]>[2]>
}

// CHECK-LABEL: @nested_list_not_boundary
// CHECK-SAME: %arg0: !pop.struct<pointer<array<2, index>>, pointer<array<2, index>>>
kgen.func @nested_list_not_boundary(%a: !pop.struct<!kgen.list<!pop.pointer<!kgen.list<index[2]>>[2]>>) {
  kgen.return
}

// CHECK-LABEL: @list_pointer
// CHECK-SAME: %arg0: !pop.pointer<array<2, index>>
// CHECK-SAME: -> (index, index)
kgen.func @list_pointer(%ptr: !pop.pointer<!kgen.list<index[2]>>) -> !kgen.list<index[2]> {
  // CHECK-DAG: %[[C0:.*]] = index.constant 0
  // CHECK-DAG: %[[C1:.*]] = index.constant 1

  // CHECK: %[[P:.*]] = pop.pointer.bitcast %arg0 : !pop.pointer<array<2, index>> to !pop.pointer<index>
  // CHECK-NEXT: %[[P0:.*]] = pop.offset %[[P]][%[[C0]]]
  // CHECK-NEXT: %[[V0:.*]] = pop.load %[[P0]] align 16
  // CHECK-NEXT: %[[P1:.*]] = pop.offset %[[P]][%[[C1]]]
  // CHECK-NEXT: %[[V1:.*]] = pop.load %[[P1]] align 16
  %0 = pop.load %ptr align 16 : !pop.pointer<!kgen.list<index[2]>>

  // CHECK-NEXT: pop.store %[[V0]], %[[P0]] align 16
  // CHECK-NEXT: pop.store %[[V1]], %[[P1]] align 16
  pop.store %0, %ptr align 16 : !pop.pointer<!kgen.list<index[2]>>

  // CHECK-NEXT: %[[V0]], %[[V1]]
  kgen.return %0 : !kgen.list<index[2]>
}

// CHECK-LABEL: @struct_list_gep
// CHECK-SAME: %arg0: !pop.pointer<struct<index, index>>
kgen.func @struct_list_gep(%ptr: !pop.pointer<struct<!kgen.list<index[2]>>>,
                           %empty: !pop.pointer<struct<!kgen.list<i1[0]>>>) -> !pop.pointer<!kgen.list<index[2]>> {
  // CHECK-NEXT: %0 = kgen.param.constant: pointer<array<0, i1>> = <#M.pointer<0>>
  // CHECK-NEXT: %1 = pop.struct.gep %arg0[0] : <struct<index, index>>
  // CHECK-NEXT: %2 = pop.pointer.bitcast %1 : !pop.pointer<index> to !pop.pointer<array<2, index>>
  %0 = pop.struct.gep %ptr[0] : <struct<!kgen.list<index[2]>>>
  %1 = pop.struct.gep %empty[0] : <struct<!kgen.list<i1[0]>>>
  // CHECK-NEXT: "use"(%0)
  "use"(%1) : (!pop.pointer<!kgen.list<i1[0]>>) -> ()
  // CHECK-NEXT: return %2
  kgen.return %0 : !pop.pointer<!kgen.list<index[2]>>
}

// CHECK-LABEL: @struct_list_gep_empty
// CHECK-SAME: %arg0: !pop.pointer<struct<>>
kgen.func @struct_list_gep_empty(%ptr: !pop.pointer<struct<!kgen.list<index[0]>>>) -> !kgen.list<index[0]> {
  // CHECK-NOT: gep
  // CHECK-NOT: load
  %0 = pop.struct.gep %ptr[0] : <struct<!kgen.list<index[0]>>>
  %1 = pop.load %0 : !pop.pointer<!kgen.list<index[0]>>
  // CHECK-NEXT: return
  // CHECK-NOT: %
  kgen.return %1 : !kgen.list<index[0]>
}

// CHECK-LABEL: @do_something
// CHECK-SAME: %arg0: index
// CHECK-SAME: -> (index, i1)
kgen.func @do_something(%list: !kgen.list<!kgen.list<index[1]>[1]>) -> (!kgen.list<!kgen.list<index[1]>[1]>, i1) {
  %true = index.bool.constant true
  kgen.return %list, %true : !kgen.list<!kgen.list<index[1]>[1]>, i1
}

// CHECK-LABEL: @hlcf_scf_loops
// CHECK-SAME: %arg0: index, %arg1: i1, %arg2: index
// CHECK-SAME: -> index
kgen.func @hlcf_scf_loops(%list: !kgen.list<!kgen.list<index[1]>[1]>, %cond: i1, %ub: index) -> !kgen.list<!kgen.list<index[1]>[1]> {
  %zero = index.constant 0
  %one = index.constant 1
  // CHECK: %{{.*}} = hlcf.loop (%{{.*}} = %arg0 : index) -> index
  %2 = hlcf.loop (%a = %list : !kgen.list<!kgen.list<index[1]>[1]>) -> !kgen.list<!kgen.list<index[1]>[1]> {
    %3, %4 = kgen.call @do_something(%a) : (!kgen.list<!kgen.list<index[1]>[1]>) -> (!kgen.list<!kgen.list<index[1]>[1]>, i1)
    hlcf.if %4 {
      // CHECK: hlcf.break %{{.*}} : index
      hlcf.break %3 : !kgen.list<!kgen.list<index[1]>[1]>
    } else {
      hlcf.yield
    }
    // CHECK: hlcf.return %{{.*}} : index
    hlcf.return %3 : !kgen.list<!kgen.list<index[1]>[1]>
  }
  kgen.return %2 : !kgen.list<!kgen.list<index[1]>[1]>
}

// CHECK-LABEL: @list_get_op
kgen.func @list_get_op(%list: !kgen.list<index[3]>) -> index {
  %0 = pop.list.get %list[1] : <index[3]>
  // CHECK: return %arg1 : index
  kgen.return %0 : index
}

// CHECK-LABEL: @list_create_op
kgen.func @list_create_op(%arg0: index, %arg1: index) -> !kgen.list<index[2]> {
  %list = pop.list.create(%arg0, %arg1) : <index[2]>
  // CHECK-NEXT: return %arg0, %arg1 : index, index
  kgen.return %list : !kgen.list<index[2]>
}

// CHECK-LABEL: @variant_of_list
kgen.func @variant_of_list(%list: !kgen.list<index[2]>, %var: !pop.variant<i1, !kgen.list<index[2]>>) -> (!kgen.list<index[2]>, !pop.variant<i1, !kgen.list<index[2]>>) {
  // CHECK-NEXT: %[[ARR:.*]] = pop.array.create [%arg0, %arg1] : !pop.array<2, index>
  // CHECK-NEXT: %[[VAR:.*]] = pop.variant.create %[[ARR]] : !pop.array<2, index> -> !pop.variant<i1, array<2, index>>
  %0 = pop.variant.create %list : !kgen.list<index[2]> -> !pop.variant<i1, !kgen.list<index[2]>>
  // CHECK-NEXT: %[[ARR:.*]] = pop.variant.get %arg2 : !pop.variant<i1, array<2, index>> as !pop.array<2, index>
  // CHECK-NEXT: %[[L0:.*]] = pop.array.get %[[ARR]][0]
  // CHECK-NEXT: %[[L1:.*]] = pop.array.get %[[ARR]][1]
  %1 = pop.variant.get %var : !pop.variant<i1, !kgen.list<index[2]>> as !kgen.list<index[2]>
  // CHECK-NEXT: return %[[L0]], %[[L1]], %[[VAR]]
  kgen.return %1, %0 : !kgen.list<index[2]>, !pop.variant<i1, !kgen.list<index[2]>>
}

// CHECK-LABEL: @variant_of_empty_list
kgen.func @variant_of_empty_list(%list: !kgen.list<i0[0]>, %var: !pop.variant<i1, !kgen.list<i0[0]>>) -> (!kgen.list<i0[0]>, !pop.variant<i1, !kgen.list<i0[0]>>) {
  // CHECK-NEXT: %[[VAR:.*]] = kgen.param.constant: variant<i1, array<0, i0>> = <#pop.variant<:array<0, i0> []>
  %0 = pop.variant.create %list : !kgen.list<i0[0]> -> !pop.variant<i1, !kgen.list<i0[0]>>
  %1 = pop.variant.get %var : !pop.variant<i1, !kgen.list<i0[0]>> as !kgen.list<i0[0]>
  // CHECK-NEXT: return %[[VAR]]
  kgen.return %1, %0 : !kgen.list<i0[0]>, !pop.variant<i1, !kgen.list<i0[0]>>
}

// CHECK-LABEL: @two_lists_in_variant
kgen.func @two_lists_in_variant(%list: !kgen.list<i1[1]>, %var: !pop.variant<!kgen.list<i1[1]>, !kgen.list<i2[1]>>) -> (!kgen.list<i2[1]>, !pop.variant<!kgen.list<i1[1]>, !kgen.list<i2[1]>>) {
  // CHECK-NEXT: %[[ARR:.*]] = pop.array.create [%arg0] : !pop.array<1, i1>
  // CHECK-NEXT: %[[VAR:.*]] = pop.variant.create %[[ARR]] : !pop.array<1, i1> -> !pop.variant<array<1, i1>, array<1, i2>>
  %0 = pop.variant.create %list : !kgen.list<i1[1]> -> !pop.variant<!kgen.list<i1[1]>, !kgen.list<i2[1]>>
  // CHECK-NEXT: %[[ARR:.*]] = pop.variant.get %arg1 : !pop.variant<array<1, i1>, array<1, i2>> as !pop.array<1, i2>
  // CHECK-NEXT: %[[L0:.*]] = pop.array.get %[[ARR]][0]
  %1 = pop.variant.get %var : !pop.variant<!kgen.list<i1[1]>, !kgen.list<i2[1]>> as !kgen.list<i2[1]>
  // CHECK-NEXT: return %[[L0]], %[[VAR]]
  kgen.return %1, %0 : !kgen.list<i2[1]>, !pop.variant<!kgen.list<i1[1]>, !kgen.list<i2[1]>>
}

// CHECK-LABEL: @list_in_coroutine
// CHECK-SAME: !pop.coroutine<() -> ()>
kgen.func @list_in_coroutine(%coro: !pop.coroutine<() -> !kgen.list<index[0]>>) {
  kgen.return
}

// CHECK-LABEL: @list_attr_in_attr
kgen.func @list_attr_in_attr() {
  // CHECK-NEXT: struct<index, index, i32, i32, i32> = <{ 1, 2, 5, 6, 7 }>
  %0 = kgen.param.constant: struct<!kgen.list<index[2]>, !kgen.list<i32[3]>>
    = <{ [1, 2], [5, 6, 7] }>
  // CHECK-NEXT: struct<index, index, index, index> = <{ 1, 2, 3, 4 }>
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

// CHECK-LABEL: @list_in_fn_type
kgen.func @list_in_fn_type() -> (() -> !kgen.list<i1[0]>) {
  // CHECK-NEXT: kgen.addressof @return_none : () -> ()
  %0 = kgen.addressof @return_none : () -> !kgen.list<i1[0]>
  // CHECK-NEXT: return %0 : () -> ()
  kgen.return %0 : () -> !kgen.list<i1[0]>
}
