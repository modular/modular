// RUN: kgen-opt --hoist-trivial-invariants -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @basic
kgen.func @basic(%arg0: !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>) {
  %struct = pop.load %arg0 : !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>

  // CHECK: %[[LOAD:.*]] = pop.load
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][2]
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][1]
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][0]

  // Loops should now be empty.
  // CHECK-NOT: pop.struct.extract

  hlcf.loop {
    %0 = pop.struct.extract %struct[0] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
    %1 = pop.struct.extract %struct[1] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
    %2 = pop.struct.extract %struct[2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @many_nests
kgen.func @many_nests(%arg0: !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>, %cond: i1) {
  %struct = pop.load %arg0 : !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>

  // CHECK: %[[LOAD:.*]] = pop.load
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][0]
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][1]
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][2]
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][2]

  // Loops should now be empty.
  // CHECK-NOT: pop.struct.extract

  hlcf.loop {
    %0 = pop.struct.extract %struct[0] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
    hlcf.loop {
      %1 = pop.struct.extract %struct[1] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
      hlcf.if %cond {
        %2 = pop.struct.extract %struct[2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
        hlcf.yield
      } else {
        %2 = pop.struct.extract %struct[2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
        hlcf.yield
      }
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @memory_ops_untouched
kgen.func @memory_ops_untouched(%input: !pop.pointer<index>, %output: !pop.pointer<index>) {
  // CHECK: hlcf.loop
  // CHECK-NEXT: hlcf.loop
  // CHECK-NEXT: pop.load
  // CHECK-NEXT: pop.store
  hlcf.loop {
    hlcf.loop {
      %load = pop.load %input : !pop.pointer<index>
      pop.store %load, %output  : !pop.pointer<index>
      hlcf.break
    }
    hlcf.break
  }
  kgen.return
}

// CHECK-LABEL: @hoist_loop_index
kgen.func @hoist_loop_index(%arg0: index, %cond: i1) {
  // CHECK-NEXT: hlcf.loop
  hlcf.loop (%arg1 = %arg0 : index) {
    // CHECK-NEXT: index.add
    // CHECK-NEXT: hlcf.if
    hlcf.if %cond {
      %0 = index.add %arg1, %arg0
      hlcf.yield
    } else {
      hlcf.yield
    }
    hlcf.break
  }
  kgen.return
}
