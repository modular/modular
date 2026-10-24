// RUN: mo-opt %s --mo-pipeline --mo-to-map --map-pipeline --map-to-mogg --mogg-to-mgp -o /dev/null

// Regression test: `mo.parallel` with parameter-expression-form dims in
// tensor types.
//
// NOTE: Requires a system with at least 2 GPUs.

mo.graph @bundled_allreduce_expr_dim_inputs<D0>(
    %in0: !mo.tensor<[D0, 3], f32, gpu:0>,
    %in1: !mo.tensor<[D0, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>) {
  // Create tensors with expression-form dims (mimics VisionPatchMerger reshape).
  %r0 = mo.rebind(%in0) : !mo.tensor<[D0, 3], f32, gpu:0> -> !mo.tensor<[div(D0, 2), 3], f32, gpu:0>
  %r1 = mo.rebind(%in1) : !mo.tensor<[D0, 3], f32, gpu:1> -> !mo.tensor<[div(D0, 2), 3], f32, gpu:1>
  %dt = mo.tensor.bundle(%r0, %r1) : (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.tensor<[div(D0, 2), 3], f32, gpu:1>) -> (!mo.bundle<[div(D0, 2), 3], f32, gpu:0-1>)
  %ch = mo.chain.create()
  // Parallel inputs have div(D0, 2) -- an expression-form dim that kernel
  // stubs must provide runtime values for.
  %res_bundle, %out_ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[div(D0, 2), 3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>, %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch) -> (!mo.bundle<[div(D0, 2), 3], f32, gpu:0-1>) {
    %peer0, %peer1 = mo.bundled.expand(%arg) : !mo.tensor<[div(D0, 2), 3], f32, gpu:0> -> (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.tensor<[div(D0, 2), 3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %sig0, %sig1, %bch)
        : (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.tensor<[div(D0, 2), 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.chain
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[div(D0, 2), 3], f32, gpu:0-1>) -> (!mo.tensor<[div(D0, 2), 3], f32, gpu:0>, !mo.tensor<[div(D0, 2), 3], f32, gpu:1>)
  %h0, %ch3 = mo.transfer[%out_ch] %res_0 : !mo.tensor<[div(D0, 2), 3], f32, gpu:0> to <"cpu", 0>
  %h1, %ch4 = mo.transfer[%ch3] %res_1 : !mo.tensor<[div(D0, 2), 3], f32, gpu:1> to <"cpu", 0>
  mo.output %h0, %h1 : !mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>
}
