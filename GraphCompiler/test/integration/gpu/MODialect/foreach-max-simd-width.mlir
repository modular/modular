// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --gc-pipeline="constant-fold=false default-device-label=gpu dump-mojo-text=true" -o %t.mlir 2>&1 | FileCheck %s --check-prefix=MOJO
// RUN: mt --execute --default-device-label=gpu --input-shapes=12 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

// Elementwise loops whose width is limited, run on the GPU: one with a
// static limit, one whose limit is the input's run-time length. Width
// inference keeps a limit already on a body, so both reach the kernels.
// The MAP IR names kernels the pipeline imports, so the parser can't verify
// it first.

// Each limit reaches `_foreach`: the static one as a number, the dynamic one
// as the run-time divisor of its dims.
// MOJO-DAG: _foreach[{{.*}}]({{.*}}, max_simd_width = layout.Idx[2])
// MOJO-DAG: _foreach[{{.*}}]({{.*}}, max_simd_width = Scalar[DType.int]({{.*}}))

// 12 ones, doubled by each loop: at width 2 for the static limit, at width 4
// for the dynamic one (12's largest power-of-two divisor).
// CHECK{LITERAL}: 'limits' returned tensor<12xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
// CHECK{LITERAL}: tensor<12xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]
mo.graph @limits<N>(%a: !mo.tensor<[N], f32>) -> (!mo.tensor<[N], f32>, !mo.tensor<[N], f32>) {
  %c0 = mo.chain.create()
  %a_gpu, %c1 = mo.transfer[%c0] %a : !mo.tensor<[N], f32> to <"gpu", 0>
  %s_gpu = map.iter.opaque
      inputs (%a_gpu : !mo.tensor<[N], f32, gpu:0>)
      outputs (%out : !mo.tensor<[N], f32, gpu:0>) {
    map.iter.elementwise.body<[N]> max_simd_width(2)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a_gpu : !mo.tensor<[N], f32, gpu:0>, %out0 = %out : !mo.tensor<[N], f32, gpu:0>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32, gpu:0>
      %d = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %v, %v) : !map.index<1>, f32, f32 -> f32
      map.store %out0[%idx] = %d : !mo.tensor<[N], f32, gpu:0>
    }
  } {device = #M.device_ref<"gpu", 0>}
  %d_gpu = map.iter.opaque
      inputs (%a_gpu : !mo.tensor<[N], f32, gpu:0>)
      outputs (%out : !mo.tensor<[N], f32, gpu:0>) {
    map.iter.elementwise.body<[N]> max_simd_width(N)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a_gpu : !mo.tensor<[N], f32, gpu:0>, %out0 = %out : !mo.tensor<[N], f32, gpu:0>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32, gpu:0>
      %d = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %v, %v) : !map.index<1>, f32, f32 -> f32
      map.store %out0[%idx] = %d : !mo.tensor<[N], f32, gpu:0>
    }
  } {device = #M.device_ref<"gpu", 0>}
  %s, %c2 = mo.transfer[%c1] %s_gpu : !mo.tensor<[N], f32, gpu:0> to <"cpu", 0>
  %d, %c3 = mo.transfer[%c2] %d_gpu : !mo.tensor<[N], f32, gpu:0> to <"cpu", 0>
  mo.output %s, %d : !mo.tensor<[N], f32>, !mo.tensor<[N], f32>
}
