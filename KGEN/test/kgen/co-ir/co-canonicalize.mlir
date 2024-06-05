// RUN: kgen-opt -canonicalize -mlir-print-debuginfo %s | FileCheck %s

// COM: Check that constant are only hoisted from subprogram regions if there is
// COM: no debuginfo scope given.

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<name = <"SomeClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc3 = loc("bar.mlir":327:17)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])
#loc6 = loc(fused<#subprogram1>[#loc3])
#call_loc = #debuginfo.call_loc<#loc4>
#loc7 = loc(fused<#call_loc>[#loc2])
#loc8 = loc(fused<#subprogram1>[#loc7])

// CHECK-LABEL: kgen.func @no_hoist
kgen.func @no_hoist() -> !co.routine {
  // CHECK-NEXT: co.execute {
  %0 = co.execute {
    // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]> loc(#loc6)
    %1 = pop.stack_allocation 1 x !pop.array<1, index>  loc(#loc6)
    pop.store %array, %1 : !kgen.pointer<array<1, index>> loc(#loc6)
    kgen.return loc(#loc5)
  } loc(#loc8)
  kgen.return %0 : !co.routine loc(#loc4)
} loc(#loc4)

// CHECK-LABEL: kgen.func @hoist
kgen.func @hoist() -> !co.routine {
  // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
  // CHECK-NEXT: co.execute {
  %0 = co.execute {
    // CHECK-NOT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]>
    %1 = pop.stack_allocation 1 x !pop.array<1, index>
    pop.store %array, %1 : !kgen.pointer<array<1, index>>
    kgen.return
  }
  kgen.return %0 : !co.routine
}

// CHECK-LABEL: @no_cse_async_execute
kgen.func @no_cse_async_execute() -> (!co.routine, !co.routine) {
  // CHECK-COUNT-2: co.execute
  %0 = co.execute {
    kgen.return
  }
  %1 = co.execute {
    kgen.return
  }
  kgen.return %0, %1 : !co.routine, !co.routine
}
