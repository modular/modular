// RUN: kgen-opt %s -lower-kgen-to-pop -allow-unregistered-dialect -cse | FileCheck %s

// CHECK-LABEL: @list_in_struct
// CHECK-SAME: %arg0: i32, %arg1: index, %arg2: index
// CHECK-SAME: -> (index, index)
kgen.func @list_in_struct(%a: i32, %list: !kgen.list<index[2]>) -> !kgen.list<index[2]> {
  // CHECK-NEXT: construct(%arg0, %arg1, %arg2, %arg0)
  %0 = pop.struct.construct(%a, %list, %a) : !pop.struct<i32, !kgen.list<index[2]>, i32>

  // CHECK-NEXT: get %{{.*}}[0] : !pop.struct<i32, index, index, i32>
  %1 = pop.struct.get %0[0] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: get %{{.*}}[1]
  // CHECK-NEXT: get %{{.*}}[2]
  %2 = pop.struct.get %0[1] : !pop.struct<i32, !kgen.list<index[2]>, i32>
  // CHECK-NEXT: get %{{.*}}[3]
  %3 = pop.struct.get %0[2] : !pop.struct<i32, !kgen.list<index[2]>, i32>

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
  // CHECK-NEXT: construct(%arg0) : !pop.struct<i32>
  %0 = pop.struct.construct(%list, %a) : !pop.struct<!kgen.list<index[0]>, i32>
  // CHECK-NEXT: get %{{.*}}[0] : !pop.struct<i32>
  %1 = pop.struct.get %0[0] : !pop.struct<!kgen.list<index[0]>, i32>
  %2 = pop.struct.get %0[1] : !pop.struct<!kgen.list<index[0]>, i32>
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

  // CHECK-NEXT: get %arg0[0] : !pop.struct<index, index, index, index>
  // CHECK-NEXT: get %arg0[1]
  // CHECK-NEXT: get %arg0[2]
  // CHECK-NEXT: get %arg0[3]
  %2 = pop.struct.get %s[0] : !pop.struct<!kgen.list<!kgen.list<index[2]>[2]>>

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
kgen.func @struct_list_gep(%ptr: !pop.pointer<struct<!kgen.list<index[2]>>>) -> !pop.pointer<!kgen.list<index[2]>> {
  // CHECK-NEXT: %0 = pop.struct.gep %arg0[0] : <struct<index, index>>
  // CHECK-NEXT: %1 = pop.pointer.bitcast %0 : !pop.pointer<index> to !pop.pointer<array<2, index>>
  %0 = pop.struct.gep %ptr[0] : <struct<!kgen.list<index[2]>>>
  // CHECK-NEXT: return %1
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
  // CHECK: scf.if %arg1 -> (index)
  %0 = scf.if %cond -> !kgen.list<!kgen.list<index[1]>[1]> {
    // CHECK: kgen.call @do_something(%arg0) : (index) -> (index, i1)
    %1, %2 = kgen.call @do_something(%list) : (!kgen.list<!kgen.list<index[1]>[1]>) -> (!kgen.list<!kgen.list<index[1]>[1]>, i1)
    // CHECK-NEXT: scf.yield %{{.*}}#0 : index
    scf.yield %1 : !kgen.list<!kgen.list<index[1]>[1]>
  } else {
    // CHECK: scf.yield %{{.*}} : index
    scf.yield %list : !kgen.list<!kgen.list<index[1]>[1]>
  }
  %zero = index.constant 0
  %one = index.constant 1
  // CHECK: %{{.*}} = scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index)
  %1 = scf.for %i = %zero to %ub step %one iter_args(%a = %list) -> !kgen.list<!kgen.list<index[1]>[1]> {
    %1, %2 = kgen.call @do_something(%a) : (!kgen.list<!kgen.list<index[1]>[1]>) -> (!kgen.list<!kgen.list<index[1]>[1]>, i1)
    scf.yield %1 : !kgen.list<!kgen.list<index[1]>[1]>
  }
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

// CHECK-LABEL: @list_iterate_accum
kgen.func @list_iterate_accum() -> !pop.scalar<si32> {
  // CHECK-NEXT: constant(0 : si32)
  %cst = pop.constant(0 : si32) : !pop.scalar<si32>
  // CHECK-NEXT: %0 = kgen.param.constant: i32 = <1>
  // CHECK-NEXT: %1 = kgen.param.constant: i32 = <2>
  // CHECK-NEXT: %2 = kgen.param.constant: i32 = <3>
  %values = kgen.param.constant: list<i32[3]> = <[1, 2, 3]>
  // CHECK-NEXT: %3 = pop.cast_from_builtin %0 : i32 to !pop.scalar<si32>
  // CHECK-NEXT: %4 = pop.add %3, %cst
  // CHECK-NEXT: %5 = pop.cast_from_builtin %1 : i32 to !pop.scalar<si32>
  // CHECK-NEXT: %6 = pop.add %5, %4
  // CHECK-NEXT: %7 = pop.cast_from_builtin %2 : i32 to !pop.scalar<si32>
  // CHECK-NEXT: %8 = pop.add %7, %6
  // CHECK-NEXT: return %8
  %result = kgen.list.iterate %c in %values : list<i32[3]> [0 : (d0) -> (d0 + 1)] (%acc = %cst) -> !pop.scalar<si32> {
    %v = pop.cast_from_builtin %c : i32 to !pop.scalar<si32>
    %r = pop.add %v, %acc : !pop.scalar<si32>
    kgen.list.yield %r : !pop.scalar<si32>
  }
  kgen.return %result : !pop.scalar<si32>
}

// CHECK-LABEL: @list_is_sorted
// CHECK-SAME: %arg0: !pop.scalar<si32>, %arg1: !pop.scalar<si32>, %arg2: !pop.scalar<si32>
kgen.func @list_is_sorted(%list: !kgen.list<!pop.scalar<si32>[3]>) -> !pop.scalar<bool> {
  // CHECK-NEXT: %cst = pop.constant(true)
  %init = pop.constant(true) : !pop.scalar<bool>
  // CHECK-NEXT: %0 = pop.cmp le(%arg0, %arg1)
  // CHECK-NEXT: %1 = pop.and %0, %cst
  // CHECK-NEXT: %2 = pop.cmp le(%arg1, %arg2)
  // CHECK-NEXT: %3 = pop.and %2, %1
  // CHECK-NEXT: return %3
  %result = kgen.list.iterate (%lhs, %rhs) in %list : list<!pop.scalar<si32>[3]> [0, 1 : (d0, d1) -> (d0 + 1, d1 + 1)] (%sorted = %init) -> !pop.scalar<bool> {
    %pairwiseSorted = pop.cmp le(%lhs, %rhs) : !pop.scalar<si32>
    %stillSorted = pop.and %pairwiseSorted, %sorted : !pop.scalar<bool>
    kgen.list.yield %stillSorted : !pop.scalar<bool>
  }
  kgen.return %result : !pop.scalar<bool>
}

// CHECK-LABEL: @list_get_op
kgen.func @list_get_op(%list: !kgen.list<index[3]>) -> index {
  %0 = kgen.list.get %list[1] : <index[3]>
  // CHECK: return %arg1 : index
  kgen.return %0 : index
}

// CHECK-LABEL: @list_make_op
kgen.func @list_make_op(%arg0: index, %arg1: index) -> !kgen.list<index[2]> {
  %list = kgen.list.make(%arg0, %arg1) : <index[2]>
  // CHECK-NEXT: return %arg0, %arg1 : index, index
  kgen.return %list : !kgen.list<index[2]>
}
