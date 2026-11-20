// RUN: mo-opt %s --gc-pipeline -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @random_normal() -> !mo.tensor<[1, 1, 7, 8], f32> {
    %shape = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %mean = mo.constant <#M.dense_array<3.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %variance = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %seed = mo.constant <#M.dense_array<2> : tensor<1xui64>> : !mo.tensor<[1], ui64>
    %output = mo.random.normal(%shape, %mean, %variance, %seed) : (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// CHECK-LABEL: --- Running 'random_normal':
// CHECK-NEXT: 'random_normal' returned tensor<1x1x7x8xf32> [{{((-?[[:digit:]]\.[[:digit:]]+e[+-][[:digit:]]+(, )?){56})}}

mo.graph @random_normal_f32_mean_var() -> !mo.tensor<[1, 1, 7, 8], f32> {
    %shape = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %mean = mo.constant <#M.dense_array<3.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %variance = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %seed = mo.constant <#M.dense_array<2> : tensor<1xui64>> : !mo.tensor<[1], ui64>
    %output = mo.random.normal(%shape, %mean, %variance, %seed) : (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// CHECK-LABEL: --- Running 'random_normal_f32_mean_var':
// CHECK-NEXT: 'random_normal_f32_mean_var' returned tensor<1x1x7x8xf32> [{{((-?[[:digit:]]\.[[:digit:]]+e[+-][[:digit:]]+(, )?){56})}}

mo.graph @random_fallback_shape_func<() -> D0, D1, D2, D3>(%one : !mo.tensor<[], si64>, %size : !mo.tensor<[4], si64>, %scalar : !mo.tensor<[], si64>, %seed : !mo.tensor<[1], ui64>) -> !mo.tensor<[D0, D1, D2, D3], si64> {
    mo.invoke_shape_func["mo.random.normal"]<() -> D4, D5, D6, D7>(%size, %scalar, %scalar, %seed) [dataDeptTensors = [0 : i32]] : (!mo.tensor<[4], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[1], ui64>)
    %val = mo.static.broadcast_to(%one) : !mo.tensor<[], si64> -> !mo.tensor<[D4, D5, D6, D7], si64>
    mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si64>
}

// CHECK-LABEL: --- Running 'random_fallback_shape_func':
// CHECK-NEXT: 'random_fallback_shape_func' returned tensor<1x1x1x1xsi64> [1]

mo.graph @random_uniform() -> !mo.tensor<[1, 1, 7, 8], f64> {
    %shape = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %lowerBound = mo.constant <#M.dense_array<0.0> : tensor<1xf64>> : !mo.tensor<[], f64>
    %upperBound = mo.constant <#M.dense_array<100.0> : tensor<1xf64>> : !mo.tensor<[], f64>
    %seed = mo.constant <#M.dense_array<2> : tensor<1xui64>> : !mo.tensor<[1], ui64>
    %output = mo.random.uniform(%shape, %lowerBound, %upperBound, %seed) : (!mo.tensor<[4], si64>, !mo.tensor<[], f64>, !mo.tensor<[], f64>, !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f64>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f64>
}

// CHECK-LABEL: --- Running 'random_uniform':
// CHECK-NEXT: 'random_uniform' returned tensor<1x1x7x8xf64> [{{((-?[[:digit:]]\.[[:digit:]]+e[+-][[:digit:]]+(, )?){56})}}

mo.graph @random_uniform_ints() -> !mo.tensor<[1000], si64> {
    %shape = mo.constant <#M.dense_array<1000> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %lowerBound = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %upperBound = mo.constant <#M.dense_array<12> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %seed = mo.constant <#M.dense_array<2> : tensor<1xui64>> : !mo.tensor<[1], ui64>
    %output = mo.random.uniform(%shape, %lowerBound, %upperBound, %seed) : (!mo.tensor<[1], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[1], ui64>) -> !mo.tensor<[1000], si64>
    mo.output %output : !mo.tensor<[1000], si64>
}

// CHECK-LABEL: --- Running 'random_uniform_ints':
// CHECK-NEXT: 'random_uniform_ints' returned tensor<1000xsi64> [{{.*}}]
