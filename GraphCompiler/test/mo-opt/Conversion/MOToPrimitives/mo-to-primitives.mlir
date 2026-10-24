// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s

// CHECK-LABEL:  mgp.model @constants_with_multiple_returns : mof
// CHECK:  } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context, %{{.*}}: !mgp.opaque<"weights_registry">
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<2xf32>, %[[ARG3:.*]]: !mgp.tensor<5x1xf32>
// CHECK-SAME: ) -> (!mef.chain, !mgp.tensor<2xf32>, !mgp.tensor<3xf32>, !mgp.tensor<3xf32>,
// CHECK-SAME: !mgp.tensor<5x1xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<4xsi8>) {
// CHECK-DAG:    %[[VAL0:.*]] = mgp.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040", align {{.*}}> : !M.array<12xui8>
// CHECK-DAG:    %[[VAL1:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<3xf32>
// CHECK-DAG:    %[[VAL3:.*]] = mgp.tensor.create %[[ARG0]], %[[VAL0]], %[[VAL1]] : !mgp.tensor<3xf32>
// CHECK-DAG:    %[[VAL4:.*]] = mgp.buffer.constant dense_resource<weight_buffer> : !M.aligned_bytes<4, align 64>
// CHECK-DAG:    %[[VAL5:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<4xsi8>
// CHECK:    %[[VAL7:.*]] = mgp.tensor.create %[[ARG0]], %[[VAL4]], %[[VAL5]] : !mgp.tensor<4xsi8>
// CHECK:    mef.output %[[ARG0]], %[[ARG2]], %[[VAL3]], %[[VAL3]], %[[ARG3]], %[[ARG3]], %[[VAL7]] : !mef.chain, !mgp.tensor<2xf32>, !mgp.tensor<3xf32>, !mgp.tensor<3xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<5x1xf32>, !mgp.tensor<4xsi8>
// CHECK:  }

mo.graph @constants_with_multiple_returns(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[5, 1], f32>) -> (
    !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
) {
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %1 = mo.constant <dense_resource<weight_buffer> : tensor<4xsi8>> : !mo.tensor<[4], si8>
  mo.output %arg0, %0, %0, %arg1, %arg1, %1 : !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[5, 1], f32>, !mo.tensor<[4], si8>
}

// CHECK-LABEL: mgp.model @foo : mof
// CHECK: } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<1x?xf32>, %[[ARG3:.*]]: !mgp.tensor<?xf32>) -> !mef.chain {
// CHECK:   %[[VAL0:.*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <1x?xf32>
// CHECK:   %[[VAL1:.*]] = mgp.tensor_spec.get_dim[1] %[[VAL0]] : <1x?xf32>
// CHECK:   %[[VAL2:.*]] = mgp.tensor.extract.tensor_spec %[[ARG3]] : <?xf32>
// CHECK:   %[[VAL3:.*]] = mgp.tensor_spec.get_dim[0] %[[VAL2]] : <?xf32>
// CHECK:   mef.output %[[ARG0:.*]] : !mef.chain
mo.graph @foo<N>(%arg0: !mo.tensor<[1, N], f32>, %arg1: !mo.tensor<[N], f32>) {
  mo.output
}

// CHECK-LABEL: mgp.model @scalar_arg : mof
// CHECK: } execute (
// CHECK-SAME: %[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<f32>) -> (!mef.chain, !mgp.tensor<f32>) {
// CHECK:   mef.output %[[ARG0]], %[[ARG2]] : !mef.chain, !mgp.tensor<f32>
mo.graph @scalar_arg(%arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
  mo.output %arg0 : !mo.tensor<[], f32>
}

{-#
  dialect_resources: {
    builtin: {
      weight_buffer: "0x4000000001234567"
    }
  }
#-}

// CHECK-LABEL: mgp.model @test_buffer_dependencies

// CHECK: execute (%[[CH0:.*]]: !mef.chain
// CHECK: %[[CH1:.*]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH0]],
// CHECK: %[[MERGE1:.*]] = grt.chain.create (%[[CH1]], %[[CH0]]
// CHECK: %[[CH2:.*]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[MERGE1]]
// CHECK: %[[MERGE2:.*]] = grt.chain.create (%[[CH1]], %[[CH2]], %[[CH0]]
// CHECK: %[[CH3:.*]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[MERGE2]],

mo.graph @test_buffer_dependencies(%arg0: !mo.tensor<[], si64>) -> !mo.tensor<[], si64> attributes {device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "x86_64-unknown-linux-gnu", arch = "icelake-server", features = "+adx,+aes,+avx,+avx2,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+rdpid,+rdrnd,+rdseed,+sahf,+sgx,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves">>]>} {
    %0 = mo.relu(%arg0): !mo.tensor<[], si64>
    %1 = mo.relu(%0): !mo.tensor<[], si64>
    %2 = mo.add(%0, %1): !mo.tensor<[], si64>
    mo.output %2 : !mo.tensor<[], si64>
}




// CHECK-LABEL: mgp.model @to_tensor_from_tensor
// CHECK: } execute ({{.*}}, %[[ARG:.*]]: !mgp.tensor<4xsi64>)
// CHECK:   %[[BUFFER:.*]] = mgp.tensor.extract.buffer %[[ARG]] : <4xsi64>
// CHECK:   %[[SIZE_IN_BYTES:.*]] = index.constant 8
// CHECK:   %[[C0:.*]] = index.constant 0
// CHECK:   %[[OFFSET0:.*]] = index.mul %[[C0]], %[[SIZE_IN_BYTES]]
// CHECK:   %[[SLICE0:.*]] = mgp.buffer.slice %[[BUFFER]], %[[OFFSET0]], %[[SIZE_IN_BYTES]], %{{.*}} : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
// CHECK:   %[[DIM0:.*]] = mgp.buffer.to_index %{{.*}}, %[[SLICE0]] : !mgp.buffer

// CHECK:   %[[C1:.*]] = index.constant 1
// CHECK:   %[[OFFSET1:.*]] = index.mul %[[C1]], %[[SIZE_IN_BYTES]]
// CHECK:   %[[SLICE1:.*]] = mgp.buffer.slice %[[BUFFER]], %[[OFFSET1]], %[[SIZE_IN_BYTES]], %{{.*}} : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
// CHECK:   %[[DIM1:.*]] = mgp.buffer.to_index %{{.*}}, %[[SLICE1]] : !mgp.buffer

// CHECK:   %[[C2:.*]] = index.constant 2
// CHECK:   %[[OFFSET2:.*]] = index.mul %[[C2]], %[[SIZE_IN_BYTES]]
// CHECK:   %[[SLICE2:.*]] = mgp.buffer.slice %[[BUFFER]], %[[OFFSET2]], %[[SIZE_IN_BYTES]], %{{.*}} : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
// CHECK:   %[[DIM2:.*]] = mgp.buffer.to_index %{{.*}}, %[[SLICE2]] : !mgp.buffer

// CHECK:   %[[C3:.*]] = index.constant 3
// CHECK:   %[[OFFSET3:.*]] = index.mul %[[C3]], %[[SIZE_IN_BYTES]]
// CHECK:   %[[SLICE3:.*]] = mgp.buffer.slice %[[BUFFER]], %[[OFFSET3]], %[[SIZE_IN_BYTES]], %{{.*}} : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
// CHECK:   %[[DIM3:.*]] = mgp.buffer.to_index %{{.*}}, %[[SLICE3]] : !mgp.buffer

// CHECK:   %[[SPEC:.*]] = mgp.tensor_spec.create %[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]] : !mgp.tensor_spec<?x?x?x?xsi64>

mo.graph @to_tensor_from_tensor(%arg: !mo.tensor<[4], si64>) -> !mo.tensor<[4], si32> {
  %sh = mo.shape.from_tensor(%arg) : (!mo.tensor<[4], si64>) -> !mosh.ape
  %tensor = mo.shape.to_tensor(%sh) -> !mo.tensor<[4], si32>

  mo.output %tensor : !mo.tensor<[4], si32>
}
