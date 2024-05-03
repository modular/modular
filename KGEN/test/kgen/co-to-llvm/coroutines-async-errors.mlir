// RUN: kgen-opt -lower-coroutines-async %s -verify-diagnostics

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// expected-note @below {{should this function be marked @always_inline?}}
llvm.func @not_a_coroutine() {
  // expected-error @below {{coroutine await operation is not contained inside an async function}}
  co.await {
    co.await.end
  }
  llvm.return
}

}
