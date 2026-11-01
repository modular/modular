// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-data=0::negones --result-output-style=full %t.mlir | FileCheck %s

// mo.tile on a device-resident input: the kernel runs on the input's device
// and the repeats stay host-resident, so only the data moves once per copy.

// %neg is unused here: every graph carries the same input so one
// --input-data flag serves them all.
mo.graph @tile_gpu(%neg: !mo.tensor<[2], si64, gpu:0>) -> !mo.tensor<[6, 6], si32> {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5> : tensor<6xsi32>> : !mo.tensor<[2, 3], si32, gpu:0>
  %repeats = mo.constant <#M.dense_array<3, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], si32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[6, 6], si32, gpu:0>
  %ch0 = mo.chain.create()
  %res_host, %ch1 = mo.transfer[%ch0] %output : !mo.tensor<[6, 6], si32, gpu:0> to <"cpu", 0>
  mo.output %res_host : !mo.tensor<[6, 6], si32>
}
// CHECK{LITERAL}: 'tile_gpu' returned tensor<6x6xsi32> [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]

// The runtime-repeats graphs pass GPU-computed repeats through an explicit
// transfer to the host, with different computed values, proving the tile
// re-reads the transferred repeats every execution. %neg arrives as negones.
mo.graph @tile_gpu_runtime_repeats_identity<() -> N, M>(%neg: !mo.tensor<[2], si64, gpu:0>) -> !mo.tensor<[N, M], si32> {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5> : tensor<6xsi32>> : !mo.tensor<[2, 3], si32, gpu:0>
  %repeats_gpu = mo.negative(%neg) : !mo.tensor<[2], si64, gpu:0>
  %ch0 = mo.chain.create()
  %repeats, %ch1 = mo.transfer[%ch0] %repeats_gpu : !mo.tensor<[2], si64, gpu:0> to <"cpu", 0>
  %output = mo.tile<() -> D4, D5>(%input, %repeats) : (!mo.tensor<[2, 3], si32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[D4, D5], si32, gpu:0>
  %ch2 = mo.chain.create()
  %res_host, %ch3 = mo.transfer[%ch2] %output : !mo.tensor<[D4, D5], si32, gpu:0> to <"cpu", 0>
  mo.output<D4, D5> %res_host : !mo.tensor<[D4, D5], si32>
}
// CHECK{LITERAL}: 'tile_gpu_runtime_repeats_identity' returned tensor<2x3xsi32> [0, 1, 2, 3, 4, 5]

// Same transfer chain, different runtime values: scaling negones by [3, 2]
// then negating gives repeats [3, 2], so the chain must re-read the
// transferred repeats rather than reuse a baked count.
mo.graph @tile_gpu_runtime_repeats_scaled<() -> N, M>(%neg: !mo.tensor<[2], si64, gpu:0>) -> !mo.tensor<[N, M], si32> {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5> : tensor<6xsi32>> : !mo.tensor<[2, 3], si32, gpu:0>
  %scale = mo.constant <#M.dense_array<3, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64, gpu:0>
  %scaled = mo.mul(%neg, %scale) : !mo.tensor<[2], si64, gpu:0>
  %repeats_gpu = mo.negative(%scaled) : !mo.tensor<[2], si64, gpu:0>
  %ch0 = mo.chain.create()
  %repeats, %ch1 = mo.transfer[%ch0] %repeats_gpu : !mo.tensor<[2], si64, gpu:0> to <"cpu", 0>
  %output = mo.tile<() -> D4, D5>(%input, %repeats) : (!mo.tensor<[2, 3], si32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[D4, D5], si32, gpu:0>
  %ch2 = mo.chain.create()
  %res_host, %ch3 = mo.transfer[%ch2] %output : !mo.tensor<[D4, D5], si32, gpu:0> to <"cpu", 0>
  mo.output<D4, D5> %res_host : !mo.tensor<[D4, D5], si32>
}
// CHECK{LITERAL}: 'tile_gpu_runtime_repeats_scaled' returned tensor<6x6xsi32> [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]

