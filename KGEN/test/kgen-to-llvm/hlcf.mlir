// RUN: kgen-opt -lower-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: llvm.func internal @hlcf_and_scf
kgen.func @hlcf_and_scf(%arg0: !pop.scalar<si32>, %arg1: !pop.simd<2, si64>) -> (!pop.scalar<si32>, !pop.simd<2, si64>) {
  %c10 = kgen.param.constant: scalar<si32> = <<10>>
  %c1 = kgen.param.constant: scalar<si32> = <<1>>
  %c12 = kgen.param.constant: simd<2, si64> = <<1, 2>>
  // CHECK: llvm.br ^bb1(%arg0, %arg1 : i32, vector<2xi64>)
  %0 = hlcf.loop (%0 = %arg0 : !pop.scalar<si32>, %1 = %arg1 : !pop.simd<2, si64>) -> !pop.simd<2, si64> {
    // CHECK: ^bb1(%{{.*}}: i32, %{{.*}}: vector<2xi64>):
    %2 = pop.cmp lt(%0, %c10) : !pop.scalar<si32>
    %3 = pop.cast_to_builtin %2 : !pop.scalar<bool> to i1
    // CHECK: llvm.cond_br %{{.*}}, ^bb2, ^bb8
    %4 = hlcf.if %3 -> !pop.simd<2, si64> {
      // CHECK: ^bb2:
      %zero = index.constant 0
      %one = index.constant 1
      %nine = index.constant 9
      // CHECK: llvm.br ^bb3(%{{.*}}, %{{.*}} : i64, vector<2xi64>)
      // CHECK: ^bb3(%{{.*}}: i64, %{{.*}}: vector<2xi64>):
      // CHECK: llvm.cond_br %{{.*}}, ^bb4, ^bb7
      %lhs = scf.for %i = %zero to %nine step %one iter_args(%v = %c12) -> !pop.simd<2, si64> {
        // CHECK: ^bb4:
        // CHECK: llvm.cond_br %{{.*}}, ^bb5(%{{.*}} : vector<2xi64>), ^bb5(%{{.*}} : vector<2xi64>)
        // CHECK: ^bb5(%{{.*}}: vector<2xi64>):
        // CHECK: llvm.br ^bb6(%{{.*}} : vector<2xi64>)
        %rhs = hlcf.if %3 -> !pop.simd<2, si64> {
          hlcf.yield %v : !pop.simd<2, si64>
        } else {
          hlcf.yield %c12 : !pop.simd<2, si64>
        }
        // CHECK: ^bb6(%{{.*}}: vector<2xi64>):
        %r = pop.add %v, %rhs : !pop.simd<2, si64>
        // CHECK: llvm.br ^bb3(%{{.*}}, %{{.*}} : i64, vector<2xi64>)
        scf.yield %r : !pop.simd<2, si64>
      }
      // CHECK: ^bb7:
      %5 = pop.add %1, %lhs : !pop.simd<2, si64>
      // CHECK: llvm.br ^bb9(%{{.*}} : vector<2xi64>)
      hlcf.yield %5 : !pop.simd<2, si64>
    } else {
      // CHECK: ^bb8:
      // CHECK: llvm.br ^bb10(%{{.*}} : vector<2xi64>)
      hlcf.break %1 : !pop.simd<2, si64>
    }
    // CHECK: ^bb9(%{{.*}}: vector<2xi64>):
    %6 = pop.add %0, %c1 : !pop.scalar<si32>
    // CHECK: llvm.br ^bb1(%{{.*}}, %{{.*}} : i32, vector<2xi64>)
    hlcf.continue %6, %4 : !pop.scalar<si32>, !pop.simd<2, si64>
  }
  // CHECK: ^bb10(%{{.*}}: vector<2xi64>):
  kgen.return %c10, %0 : !pop.scalar<si32>, !pop.simd<2, si64>
}

}
