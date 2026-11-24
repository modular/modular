// RUN: kgen-opt -split-input-file -verify-diagnostics -lower-global-pop-to-llvm %s

// Directional orderings require the metal4_0 deployment-floor feature.
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  kgen.func @fence_acquire_needs_metal4() {
    // expected-error @below {{Apple GPU supports `acquire` fence ordering only with a Metal 4 deployment target; use `seq_cst` or raise the deployment target}}
    // expected-error @below {{failed to legalize operation 'pop.fence' that was explicitly marked illegal}}
    pop.fence syncscope("gpu") acquire
    hlcf.return
  }
}

// -----

// Explicitly disabled ("-metal4_0") must still error: a substring feature
// check would match it and silently permit the directional ordering.
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "-metal4_0", data_layout = "">} {
  kgen.func @fence_acquire_metal41_explicitly_disabled() {
    // expected-error @below {{Apple GPU supports `acquire` fence ordering only with a Metal 4 deployment target; use `seq_cst` or raise the deployment target}}
    // expected-error @below {{failed to legalize operation 'pop.fence' that was explicitly marked illegal}}
    pop.fence syncscope("gpu") acquire
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  kgen.func @fence_system_scope_rejected() {
    // expected-error @below {{Apple GPU has no system-scope fence: Metal cannot order memory against the host}}
    // expected-error @below {{failed to legalize operation 'pop.fence' that was explicitly marked illegal}}
    pop.fence syncscope("system") seq_cst
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  kgen.func @fence_unknown_scope_rejected() {
    // expected-error @below {{unknown fence syncscope "agent" on Apple GPU; expected "thread", "warp", "block", or "gpu"}}
    // expected-error @below {{failed to legalize operation 'pop.fence' that was explicitly marked illegal}}
    pop.fence syncscope("agent") seq_cst
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  kgen.func @fence_cluster_scope_rejected() {
    // expected-error @below {{Apple GPU has no cluster scope; use syncscope "gpu" for device-wide ordering (a strict superset of cluster)}}
    // expected-error @below {{failed to legalize operation 'pop.fence' that was explicitly marked illegal}}
    pop.fence syncscope("cluster") seq_cst
    hlcf.return
  }
}
