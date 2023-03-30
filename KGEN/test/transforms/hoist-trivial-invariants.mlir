// RUN: kgen-opt --hoist-trivial-invariants -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @basic
kgen.func @basic(%arg0: !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>) {
  %empty_return = kgen.param.constant: list<i1[0]> = <[]>

  %struct = pop.load %arg0 : !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>
  
  // CHECK: %[[LOAD:.*]] = pop.load
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][2] 
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][1] 
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][0] 

  // Loops should now be empty.
  // CHECK-NOT: pop.struct.extract

  %16 = hlcf.loop "inlined_cf_scope" () -> !kgen.list<i1[0]> {
    %0 = pop.struct.extract %struct[0] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
    %1 = pop.struct.extract %struct[1] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
    %2 = pop.struct.extract %struct[2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
    hlcf.break "inlined_cf_scope" %empty_return : !kgen.list<i1[0]>
  }

  kgen.return
}



// CHECK-LABEL: @many_nests
kgen.func @many_nests(%arg0: !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>, %cond: i1) {
  %empty_return = kgen.param.constant: list<i1[0]> = <[]>

  %struct = pop.load %arg0 : !pop.pointer<struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>>
  
  // CHECK: %[[LOAD:.*]] = pop.load
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][0] 
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][1] 
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][2] 
  // CHECK-NEXT: pop.struct.extract %[[LOAD]][2] 

  // Loops should now be empty.
  // CHECK-NOT: pop.struct.extract

  %loop1 = hlcf.loop "inlined_cf_scope" () -> !kgen.list<i1[0]> {
    %0 = pop.struct.extract %struct[0] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>

    %loop2 = hlcf.loop "inlined_cf_scope" () -> !kgen.list<i1[0]> {
      %1 = pop.struct.extract %struct[1] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
      
      hlcf.if %cond {
        %2 = pop.struct.extract %struct[2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
        hlcf.yield
      } else {
        %2 = pop.struct.extract %struct[2] : !pop.struct<struct<scalar<index>>, struct<scalar<index>>, struct<scalar<index>>>
        hlcf.yield
      }

      hlcf.break "inlined_cf_scope" %empty_return : !kgen.list<i1[0]>
    }
    hlcf.break "inlined_cf_scope" %empty_return : !kgen.list<i1[0]>
  }
  kgen.return
}


// CHECK-LABEL: @memory_ops_untouched
kgen.func @memory_ops_untouched(%input: !pop.pointer<index>, %output: !pop.pointer<index>) {

  // CHECK: hlcf.loop "inlined_cf_scope" {
  // CHECK-NEXT: hlcf.loop "inlined_cf_scope" {
  // CHECK-NEXT: pop.load
  // CHECK-NEXT: pop.store

  hlcf.loop "inlined_cf_scope" {
    hlcf.loop "inlined_cf_scope" {
      %load = pop.load %input : !pop.pointer<index>
      pop.store %load, %output  : !pop.pointer<index>
      hlcf.break "inlined_cf_scope"
    }
    hlcf.break "inlined_cf_scope"
  }

  kgen.return
}
