// RUN: mo-opt --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp %s | FileCheck %s

// CHECK-LABEL: mgp.model @call_subgraph subgraph : mof
// CHECK-SAME: devices [#M.device_ref<"cpu", 0>]
// CHECK: } execute (

mo.graph @call_subgraph(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> isSubgraph attributes {device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "x86_64-unknown-linux-gnu", arch = "icelake-server", features = "+adx,+aes,+avx,+avx2,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+rdpid,+rdrnd,+rdseed,+sahf,+sgx,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves">>]>} {
  %0 = mo.relu(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// CHECK-LABEL: mgp.model @test_call
// CHECK: } execute (
// CHECK-SAME: %[[CHAIN_MAIN:.*]]: !mef.chain, %[[CTX_MAIN:.*]]: !mgp.context, %[[WR_MAIN:.*]]: !mgp.opaque<"weights_registry">, %[[DEV_MAIN:.*]]: !mgp.device_context<<"cpu", 0>>, %[[ARG_MAIN:.*]]: !mgp.tensor<4xf32>)
// CHECK-SAME: -> (!mef.chain, !mgp.tensor<4xf32>) {
// CHECK-NEXT:   %[[CALL_OUT:.*]]:2 = mgp.call @call_subgraph(%[[CHAIN_MAIN]], %[[CTX_MAIN]], %[[WR_MAIN]], %[[DEV_MAIN]], %[[ARG_MAIN]]) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<4xf32>) -> (!mef.chain, !mgp.tensor<4xf32>)
// CHECK-NEXT:   mef.output %[[CALL_OUT]]#0, %[[CALL_OUT]]#1 : !mef.chain, !mgp.tensor<4xf32>
// CHECK-NEXT: }
mo.graph @test_call(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> attributes {device_specs = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "x86_64-unknown-linux-gnu", arch = "icelake-server", features = "+adx,+aes,+avx,+avx2,+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+avx512vpopcntdq,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+evex512,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+rdpid,+rdrnd,+rdseed,+sahf,+sgx,+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+wbnoinvd,+x87,+xsave,+xsavec,+xsaveopt,+xsaves">>]>} {
  %0 = mo.call @call_subgraph(%arg0) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}
