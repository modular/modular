// RUN: kgen-opt -split-input-file -verify-diagnostics -lower-global-pop-to-llvm %s

// AIR atomics only support device (global, address space 1 or unannotated)
// and threadgroup (local, address space 3) memory. Address space 4
// (`AddressSpace.CONSTANT`) is not accepted; see MSTDL-1974.
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "", data_layout = "">} {
  kgen.func @atomic_rmw_rejects_constant_address_space(
      %ptr: !kgen.pointer<scalar<si32>, 4>, %val: !kgen.scalar<si32>) {
    // expected-error @below {{Atomic operation is only supported for local (3) and global (1) address spaces on Apple GPU, but got address space '4'}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.rmw' that was explicitly marked illegal}}
    %0 = pop.atomic.rmw add(%ptr, %val) monotonic :
                      !kgen.pointer<scalar<si32>, 4>
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "", data_layout = "">} {
  kgen.func @atomic_cmpxchg_rejects_constant_address_space(
      %ptr: !kgen.pointer<scalar<si32>, 4>, %cmp: !kgen.scalar<si32>,
      %val: !kgen.scalar<si32>) {
    // expected-error @below {{Atomic operation is only supported for local (3) and global (1) address spaces on Apple GPU, but got address space '4'}}
    // expected-error @below {{failed to legalize operation 'pop.atomic.cmpxchg' that was explicitly marked illegal}}
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val monotonic monotonic :
                      !kgen.pointer<scalar<si32>, 4>
    hlcf.return
  }
}
