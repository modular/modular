// RUN: kgen-opt -split-input-file -verify-diagnostics -lower-global-pop-to-llvm %s

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  kgen.func @rmw_acq_rel_needs_metal4(%ptr: !kgen.pointer<scalar<ui32>>,
                                      %val: !kgen.scalar<ui32>) {
    // expected-error @below {{Apple GPU supports `acq_rel` atomic ordering only with a Metal 4 deployment target; use `monotonic` or raise the deployment target}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.rmw' that was explicitly marked illegal}}
    %0 = pop.atomic.rmw add(%ptr, %val) acq_rel : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  kgen.func @rmw_seq_cst_needs_metal4(%ptr: !kgen.pointer<scalar<ui32>>,
                                      %val: !kgen.scalar<ui32>) {
    // expected-error @below {{Apple GPU supports `seq_cst` atomic ordering only with a Metal 4 deployment target; use `monotonic` or raise the deployment target}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.rmw' that was explicitly marked illegal}}
    %0 = pop.atomic.rmw add(%ptr, %val) seq_cst : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  kgen.func @cmpxchg_acquire_needs_metal4(%ptr: !kgen.pointer<scalar<ui32>>,
                                          %cmp: !kgen.scalar<ui32>,
                                          %val: !kgen.scalar<ui32>) {
    // expected-error @below {{Apple GPU supports `acquire` atomic ordering only with a Metal 4 deployment target; use `monotonic` or raise the deployment target}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.cmpxchg' that was explicitly marked illegal}}
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val acquire monotonic :
                      !kgen.pointer<scalar<ui32>>
    hlcf.return
  }
}

// -----

// Guards against a substring feature check matching "-metal4_0".
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "-metal4_0", data_layout = "">} {
  kgen.func @rmw_release_metal4_explicitly_disabled(
      %ptr: !kgen.pointer<scalar<ui32>>, %val: !kgen.scalar<ui32>) {
    // expected-error @below {{Apple GPU supports `release` atomic ordering only with a Metal 4 deployment target; use `monotonic` or raise the deployment target}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.rmw' that was explicitly marked illegal}}
    %0 = pop.atomic.rmw add(%ptr, %val) release : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal4_0,+air2_8_0", data_layout = "">} {
  kgen.func @cmpxchg_acq_rel_failure_rejected(%ptr: !kgen.pointer<scalar<ui32>>,
                                              %cmp: !kgen.scalar<ui32>,
                                              %val: !kgen.scalar<ui32>) {
    // expected-error @below {{compare-exchange failure ordering cannot be `acq_rel`}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.cmpxchg' that was explicitly marked illegal}}
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val acq_rel acq_rel :
                      !kgen.pointer<scalar<ui32>>
    hlcf.return
  }
}
