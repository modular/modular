// RUN: kgen-opt -split-input-file -verify-diagnostics -lower-global-pop-to-llvm %s

// Metal only lowers the `weak` form of `pop.atomic.cmpxchg`.
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "", data_layout = "">} {
  kgen.func @atomic_cmpxchg_requires_weak(%ptr: !kgen.pointer<scalar<si32>>,
                            %cmp: !kgen.scalar<si32>,
                            %val: !kgen.scalar<si32>) {
    // expected-error @below {{Apple GPU only supports `weak` compare-exchange; AIR exposes no strong compare-exchange primitive}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.cmpxchg' that was explicitly marked illegal}}
    %0 = pop.atomic.cmpxchg %ptr, %cmp, %val monotonic monotonic :
                      !kgen.pointer<scalar<si32>>
    hlcf.return
  }
}
