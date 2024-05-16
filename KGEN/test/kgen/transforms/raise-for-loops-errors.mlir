// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen.func(raise-for-loops{warn-failure=false}, loop-unrolling, canonicalize, raise-for-loops))' -verify-diagnostics

// CHECK-LABEL: @decorated_but_cannot_unroll0
kgen.func @decorated_but_cannot_unroll0() -> () {
  %index4 = kgen.param.constant = <4>
  %index1 = kgen.param.constant = <1>
  %index2 = kgen.param.constant = <2>

  // expected-warning @below {{loop is decorated with @unroll, but compiler can't fully unroll it}}
  // expected-note @below {{loop has multiple exits}}
  hlcf.loop (%arg0 = %index1 : index, %arg1 = %index2 : index) {
    %0 = index.cmp slt(%arg0, %index4)
    hlcf.if %0 {
      hlcf.yield
    } else {
      // expected-note @below {{loop exits}}
      hlcf.break
    }
    %1 = index.cmp slt(%arg1, %index2)
    hlcf.if %1 {
      // expected-note @below {{loop exits}}
      hlcf.break
    } else {
      hlcf.yield
    }
    kgen.call @foo(%arg0) : (index) -> ()
    %3 = index.add %arg0, %index1
    %4 = index.add %arg1, %3
    hlcf.continue %3, %4 : index, index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}

// CHECK-LABEL: @can_unroll_inner_loop_2nd_run
kgen.func @can_unroll_inner_loop_2nd_run() -> () {
  // COM: no error should be raised because we can unroll the inner loop in
  // COM: the 2nd run of raise-for-loops, though the 1st run will fail.
  %index1 = kgen.param.constant = <1>
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx3 = index.constant 3
  hlcf.loop (%arg0 = %idx3 : index) {
    %0 = index.cmp sgt(%arg0, %idx0)
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.break
    }
    %1 = index.sub %arg0, %idx1
    %2 = index.sub %idx3, %arg0
    %3 = index.sub %idx3, %2
    hlcf.loop (%arg1 = %3 : index) {
      %4 = index.cmp sgt(%arg1, %idx0)
      hlcf.if %4 {
        hlcf.yield
      } else {
        hlcf.break
      }
      %5 = index.sub %arg1, %idx1
      kgen.call @foo(%5) : (index) -> ()
      hlcf.continue %5 : index
    } {unrollLevel = #hlcf<unroll_level full>}
    hlcf.continue %1 : index
  } {unrollLevel = #hlcf<unroll_level full>}
  kgen.return
}
