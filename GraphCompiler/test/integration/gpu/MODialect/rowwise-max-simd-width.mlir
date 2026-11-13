// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --gc-pipeline="constant-fold=false default-device-label=gpu dump-mojo-text=true" -o %t.mlir 2>&1 | FileCheck %s --check-prefix=MOJO
// RUN: mt --execute --default-device-label=gpu --input-shapes=12x32 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// Reductions whose width is limited, run on the GPU: one with a static limit,
// one whose limit is the input's run-time row count. Width inference keeps a
// limit already on a body, so both reach the kernels. The MAP IR names
// kernels the pipeline imports, so the parser can't verify it first.

// Each limit reaches `_rowwise`: the static one as a number, the dynamic one
// as the run-time divisor of its dims.
// MOJO-DAG: _rowwise[{{.*}}]({{.*}}, max_simd_width = layout.Idx[2])
// MOJO-DAG: _rowwise[{{.*}}]({{.*}}, max_simd_width = Scalar[DType.int]({{.*}}))

// 12 rows of 32 ones, summed by each kernel: at width 2 for the static limit,
// at width 4 for the dynamic one (12's largest power-of-two divisor).
// CHECK{LITERAL}: 'limits' returned tensor<12x1xf32> [3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01]
// CHECK{LITERAL}: tensor<12x1xf32> [3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01, 3.200000e+01]
mo.graph @limits<N>(%x: !mo.tensor<[N, 32], f32>) -> (!mo.tensor<[N, 1], f32>, !mo.tensor<[N, 1], f32>) {
  %c0 = mo.chain.create()
  %x_gpu, %c1 = mo.transfer[%c0] %x : !mo.tensor<[N, 32], f32> to <"gpu", 0>
  %s_gpu = map.iter.opaque
      inputs (%x_gpu : !mo.tensor<[N, 32], f32, gpu:0>)
      outputs (%sum : !mo.tensor<[N, 1], f32, gpu:0>) {
    map.iter.rowwise.body<[N, 32], axis = 1> max_simd_width(2)
        row (%row: !map.index<2>)
        capturing (%x0 = %x_gpu : !mo.tensor<[N, 32], f32, gpu:0>, %sum0 = %sum : !mo.tensor<[N, 1], f32, gpu:0>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
        %v = map.load %x0[%i] : !mo.tensor<[N, 32], f32, gpu:0>
        map.yield %v : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[N, 1], f32, gpu:0>
    }
  } {device = #M.device_ref<"gpu", 0>}
  %d_gpu = map.iter.opaque
      inputs (%x_gpu : !mo.tensor<[N, 32], f32, gpu:0>)
      outputs (%sum : !mo.tensor<[N, 1], f32, gpu:0>) {
    map.iter.rowwise.body<[N, 32], axis = 1> max_simd_width(N)
        row (%row: !map.index<2>)
        capturing (%x0 = %x_gpu : !mo.tensor<[N, 32], f32, gpu:0>, %sum0 = %sum : !mo.tensor<[N, 1], f32, gpu:0>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
        %v = map.load %x0[%i] : !mo.tensor<[N, 32], f32, gpu:0>
        map.yield %v : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[N, 1], f32, gpu:0>
    }
  } {device = #M.device_ref<"gpu", 0>}
  %s, %c2 = mo.transfer[%c1] %s_gpu : !mo.tensor<[N, 1], f32, gpu:0> to <"cpu", 0>
  %d, %c3 = mo.transfer[%c2] %d_gpu : !mo.tensor<[N, 1], f32, gpu:0> to <"cpu", 0>
  mo.output %s, %d : !mo.tensor<[N, 1], f32>, !mo.tensor<[N, 1], f32>
}
