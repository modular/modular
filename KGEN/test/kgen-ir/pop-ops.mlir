// RUN: kgen-opt %s | FileCheck %s

// CHECK-LABEL: kgen.kernel @pop_constant() -> !meta.scalar<f32> {
kgen.kernel @pop_constant() -> !meta.scalar<f32> {
  // CHECK-NEXT: pop.constant(32 : si64) : !meta.scalar<si64>
  %0 = pop.constant(32 : si64) : !meta.scalar<si64>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f32) : !meta.scalar<f32>
  %1 = pop.constant(32.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: pop.constant(3.200000e+01 : f64) : !meta.scalar<f32>
  %2 = pop.constant(32.0 : f64) : !meta.scalar<f32>
  // CHECK-NEXT: pop.constant(32 : i64) : !meta.scalar<f32>
  %3 = pop.constant(32) : !meta.scalar<f32>
  // CHECK-NEXT: pop.constant(32 : si64) : !meta.scalar<si32>
  %4 = pop.constant(32 : si64) : !meta.scalar<si32>
  kgen.return %1 : !meta.scalar<f32>
}

// CHECK-LABEL: @pop_constant_simd
kgen.kernel @pop_constant_simd() {
  // CHECK: pop.constant(dense<[32, 64]>
  %0 = pop.constant(dense<[32, 64]> : vector<2xsi64>) : !meta.simd<2, si32>
  // CHECK: pop.constant(dense<[32, 64]>
  %1 = pop.constant(dense<[32, 64]> : vector<2xi32>) : !meta.simd<2, f64>
  // CHECK: pop.constant(dense<[32, 64]>
  %2 = pop.constant(dense<[32, 64]> : vector<2xi32>) : !meta.simd<2, ui64>
  kgen.return
}

// CHECK-LABEL: kgen.generator @pop_constant2<type: dtype>() -> !meta.scalar<type> {
kgen.generator @pop_constant2<type: dtype>() -> !meta.scalar<type> {
  // CHECK-NEXT: pop.constant(32 : i64) : !meta.scalar<type>
  %0 = pop.constant(32) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

// CHECK-LABEL: kgen.kernel @pop_abs(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_abs(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.abs %arg0 : !meta.scalar<f32>
  %0 = pop.abs %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_neg(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_neg(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.neg %arg0 : !meta.scalar<f32>
  %0 = pop.neg %arg0 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_add() -> !meta.scalar<f32> {
kgen.kernel @pop_add() -> !meta.scalar<f32> {
  // CHECK-NEXT: %[[CST:.*]] = pop.constant(4.000000e+00 : f32) : !meta.scalar<f32>
  %a = pop.constant(4.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %[[CST0:.*]] = pop.constant(6.000000e+00 : f32) : !meta.scalar<f32>
  %b = pop.constant(6.0 : f32) : !meta.scalar<f32>
  // CHECK-NEXT: %0 = pop.add %[[CST]], %[[CST0]] : !meta.scalar<f32>
  %c = pop.add %a, %b : !meta.scalar<f32>
  kgen.return %c : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.generator @pop_add2<type: dtype>(%arg0: !meta.scalar<type>, %arg1: !meta.scalar<type>) -> !meta.scalar<type> {
kgen.generator @pop_add2<type: dtype>(%a: !meta.scalar<type>, %b: !meta.scalar<type>) -> !meta.scalar<type> {
  // CHECK-NEXT: %0 = pop.add %arg0, %arg1 : !meta.scalar<type>
  %c = pop.add %a, %b : !meta.scalar<type>
  kgen.return %c : !meta.scalar<type>
}

// CHECK-LABEL: kgen.kernel @pop_add_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_add_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.add %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.add %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_sub(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_sub(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %0 = pop.sub %arg0, %arg1 : !meta.scalar<f32>
  %0 = pop.sub %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_sub_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_sub_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.sub %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.sub %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_mul(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_mul(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %0 = pop.mul %arg0, %arg1 : !meta.scalar<f32>
  %0 = pop.mul %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_mul_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_mul_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.mul %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.mul %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_div(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_div(%arg0 : !meta.scalar<f32>, %arg1 : !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK-NEXT: %0 = pop.div %arg0, %arg1 : !meta.scalar<f32>
  %0 = pop.div %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_div_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_div_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.div %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.div %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_shifts(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>) -> !meta.scalar<si32> {
kgen.kernel @pop_shifts(%arg0: !meta.scalar<si32>, %arg1: !meta.scalar<si32>) -> !meta.scalar<si32> {
  // CHECK: %0 = pop.shl %arg0, %arg1 : !meta.scalar<si32>
  %0 = pop.shl %arg0, %arg1 : !meta.scalar<si32>
  // CHECK: %1 = pop.shrs %arg0, %arg1 : !meta.scalar<si32>
  %1 = pop.shrs %arg0, %arg1 : !meta.scalar<si32>
  // CHECK: %2 = pop.shru %arg0, %arg1 : !meta.scalar<si32>
  %2 = pop.shru %arg0, %arg1 : !meta.scalar<si32>
  kgen.return %2 : !meta.scalar<si32>
}

// CHECK-LABEL: kgen.kernel @pop_shifts_simd(%arg0: !meta.simd<4, si32>, %arg1: !meta.simd<4, si32>) -> !meta.simd<4, si32> {
kgen.kernel @pop_shifts_simd(%arg0: !meta.simd<4, si32>, %arg1: !meta.simd<4, si32>) -> !meta.simd<4, si32> {
  // CHECK: %0 = pop.shl %arg0, %arg1 : !meta.simd<4, si32>
  %0 = pop.shl %arg0, %arg1 : !meta.simd<4, si32>
  // CHECK: %1 = pop.shrs %arg0, %arg1 : !meta.simd<4, si32>
  %1 = pop.shrs %arg0, %arg1 : !meta.simd<4, si32>
  // CHECK: %2 = pop.shru %arg0, %arg1 : !meta.simd<4, si32>
  %2 = pop.shru %arg0, %arg1 : !meta.simd<4, si32>
  kgen.return %2 : !meta.simd<4, si32>
}

// CHECK-LABEL: kgen.kernel @pop_copysign(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_copysign(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.copysign %arg0, %arg1 : !meta.scalar<f32>
  %0 = pop.copysign %arg0, %arg1 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_copysign_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_copysign_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.copysign %arg0, %arg1 : !meta.simd<4, f32>
  %0 = pop.copysign %arg0, %arg1 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: kgen.kernel @pop_fma(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
kgen.kernel @pop_fma(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: %0 = pop.fma %arg0, %arg1, %arg2 : !meta.scalar<f32>
  %0 = pop.fma %arg0, %arg1, %arg2: !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: kgen.kernel @pop_fma_simd(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f32>, %arg2: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
kgen.kernel @pop_fma_simd(%arg0 : !meta.simd<4, f32>, %arg1 : !meta.simd<4, f32>, %arg2 : !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  // CHECK-NEXT: %0 = pop.fma %arg0, %arg1, %arg2 : !meta.simd<4, f32>
  %0 = pop.fma %arg0, %arg1, %arg2 : !meta.simd<4, f32>
  kgen.return %0 : !meta.simd<4, f32>
}

// CHECK-LABEL: @pop_cmp
kgen.kernel @pop_cmp(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<bool> {
  // CHECK: pop.cmp ge(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ge(%arg0, %arg1) : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<bool>
}

kgen.kernel @pop_cmp_simd(
    %arg0: !meta.simd<4, si32>, %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<2, f64>, %arg3: !meta.simd<2, f64>
  ) -> (!meta.simd<4, bool>, !meta.simd<2, bool>) {
  // CHECK: pop.cmp ne(%{{.*}}, %{{.*}}) :
  %0 = pop.cmp ne(%arg0, %arg1) : !meta.simd<4, si32>
  // CHECK: pop.cmp lt(%{{.*}}, %{{.*}}) :
  %1 = pop.cmp lt(%arg2, %arg3) : !meta.simd<2, f64>
  kgen.return %0, %1 : !meta.simd<4, bool>, !meta.simd<2, bool>
}


// CHECK-LABEL: @pop_select
kgen.kernel @pop_select(%arg0 : !meta.scalar<bool>, %arg1: !meta.scalar<f32>, %arg2: !meta.scalar<f32>) -> !meta.scalar<f32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.scalar<f32>
  kgen.return %0 : !meta.scalar<f32>
}

// CHECK-LABEL: @pop_select_simd
kgen.kernel @pop_select_simd(
    %arg0: !meta.simd<4, bool>,
    %arg1: !meta.simd<4, si32>,
    %arg2: !meta.simd<4, si32>
  ) -> !meta.simd<4, si32> {
  // CHECK: pop.select %{{.*}}, %{{.*}}, %{{.*}} :
  %0 = pop.select %arg0, %arg1, %arg2 : !meta.simd<4, si32>
  kgen.return %0 : !meta.simd<4, si32>
}

// CHECK-LABEL: @scalar_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @scalar_bitcast(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f64>) -> !meta.scalar<f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !meta.scalar<f32> to !meta.scalar<si32>
  %0 = pop.bitcast %arg0 : !meta.scalar<f32> to !meta.scalar<si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !meta.scalar<f64> to !meta.scalar<f64>
  %1 = pop.bitcast %arg1 : !meta.scalar<f64> to !meta.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !meta.scalar<si32> to !meta.scalar<ui32>
  %2 = pop.bitcast %0 : !meta.scalar<si32> to !meta.scalar<ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !meta.scalar<ui32> to !meta.scalar<f32>
  %3 = pop.bitcast %2 : !meta.scalar<ui32> to !meta.scalar<f32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !meta.scalar<f32>
}

// CHECK-LABEL: @simd_bitcast
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]:
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]:
kgen.generator @simd_bitcast(%arg0: !meta.simd<4, f32>, %arg1: !meta.simd<4, f64>) -> !meta.simd<4, f32> {
  // CHECK: %[[V0:.*]] = pop.bitcast %[[ARG0]] : !meta.simd<4, f32> to !meta.simd<4, si32>
  %0 = pop.bitcast %arg0 : !meta.simd<4, f32> to !meta.simd<4, si32>
  // CHECK: %[[V1:.*]] = pop.bitcast %[[ARG1]] : !meta.simd<4, f64> to !meta.simd<4, f64>
  %1 = pop.bitcast %arg1 : !meta.simd<4, f64> to !meta.simd<4, f64>
  // CHECK: %[[V2:.*]] = pop.bitcast %[[V0]] : !meta.simd<4, si32> to !meta.simd<4, ui32>
  %2 = pop.bitcast %0 : !meta.simd<4, si32> to !meta.simd<4, ui32>
  // CHECK: %[[V3:.*]] = pop.bitcast %[[V2]] : !meta.simd<4, ui32> to !meta.simd<4, f32>
  %3 = pop.bitcast %2 : !meta.simd<4, ui32> to !meta.simd<4, f32>
  // CHECK: %[[V4:.*]] = pop.bitcast %[[V2]] : !meta.simd<4, ui32> to !meta.simd<2, f64>
  %4 = pop.bitcast %2 : !meta.simd<4, ui32> to !meta.simd<2, f64>
  // CHECK: return %[[V3]]
  kgen.return %3 : !meta.simd<4, f32>
}

// CHECK-LABEL: @scalar_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @scalar_cast<type: dtype>(%a: !meta.scalar<f32>) -> !meta.scalar<si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !meta.scalar<f32> to !meta.scalar<type>
  %0 = pop.cast %a : !meta.scalar<f32> to !meta.scalar<type>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !meta.scalar<type> to !meta.scalar<f64>
  %1 = pop.cast %0 : !meta.scalar<type> to !meta.scalar<f64>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !meta.scalar<f64> to !meta.scalar<si32>
  %2 = pop.cast %1 : !meta.scalar<f64> to !meta.scalar<si32>
  // CHECK: return %[[V2]]
  kgen.return %2 : !meta.scalar<si32>
}

// CHECK-LABEL: @simd_cast
// CHECK-SAME: %[[A:.*]]:
kgen.generator @simd_cast<size, type: dtype>(%a: !meta.simd<size, f32>) -> !meta.simd<size, si32> {
  // CHECK: %[[V0:.*]] = pop.cast %[[A]] : !meta.simd<size, f32> to !meta.simd<size, type>
  %0 = pop.cast %a : !meta.simd<size, f32> to !meta.simd<size, type>
  // CHECK: %[[V1:.*]] = pop.cast %[[V0]] : !meta.simd<size, type> to !meta.simd<size, si32>
  %1 = pop.cast %0 : !meta.simd<size, type> to !meta.simd<size, si32>
  // CHECK: %[[V2:.*]] = pop.cast %[[V1]] : !meta.simd<size, si32> to !meta.simd<size, f64>
  %2 = pop.cast %1 : !meta.simd<size, si32> to !meta.simd<size, f64>
  // CHECK: %[[V3:.*]] = pop.cast %[[V2]] : !meta.simd<size, f64> to !meta.simd<size, si32>
  %3 = pop.cast %2 : !meta.simd<size, f64> to !meta.simd<size, si32>
  // CHECK: return %[[V3]]
  kgen.return %3 : !meta.simd<size, si32>
}

// CHECK-LABEL: @pop_simd_extractelement
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_simd_extractelement<size, type: dtype>(
    %a: !meta.simd<size, type>,
    %b: !meta.simd<size, f32>,
    %c: !meta.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  arith.constant
  %idx = arith.constant 2 : index
  // CHECK: %[[U:.*]] = pop.simd.extractelement %[[A]][%[[IDX]]] : !meta.simd<size, type>
  %u = pop.simd.extractelement %a[%idx] : !meta.simd<size, type>
  // CHECK: %[[V:.*]] = pop.simd.extractelement %[[B]][%[[IDX]]] : !meta.simd<size, f32>
  %v = pop.simd.extractelement %b[%idx] : !meta.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.extractelement %[[C]][%[[IDX]]] : !meta.simd<4, si32>
  %w = pop.simd.extractelement %c[%idx] : !meta.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_insertelement
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
// CHECK-SAME: %[[V1:[a-z0-9]+]]:
// CHECK-SAME: %[[V2:[a-z0-9]+]]:
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_simd_insertelement<size, type: dtype>(
    %v0: !meta.scalar<type>,
    %v1: !meta.scalar<f32>,
    %v2: !meta.scalar<si32>,
    %a: !meta.simd<size, type>,
    %b: !meta.simd<size, f32>,
    %c: !meta.simd<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  arith.constant
  %idx = arith.constant 2 : index
  // CHECK: %[[U:.*]] = pop.simd.insertelement %[[V0]], %[[A]][%[[IDX]]] : !meta.simd<size, type>
  %u = pop.simd.insertelement %v0, %a[%idx] : !meta.simd<size, type>
  // CHECK: %[[V:.*]] = pop.simd.insertelement %[[V1]], %[[B]][%[[IDX]]] : !meta.simd<size, f32>
  %v = pop.simd.insertelement %v1, %b[%idx] : !meta.simd<size, f32>
  // CHECK: %[[w:.*]] = pop.simd.insertelement %[[V2]], %[[C]][%[[IDX]]] : !meta.simd<4, si32>
  %w = pop.simd.insertelement %v2, %c[%idx] : !meta.simd<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_simd_splat
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
kgen.generator @pop_simd_splat<size, type: dtype>(%a: !meta.scalar<f32>, %b: !meta.scalar<type>) -> (!meta.simd<4, f32>, !meta.simd<size, type>) {
  // CHECK: %[[U:.*]] = pop.simd.splat %[[A]] : !meta.simd<4, f32>
  %u = pop.simd.splat %a : !meta.simd<4, f32>
  // CHECK: %[[V:.*]] = pop.simd.splat %[[B]] : !meta.simd<size, type>
  %v = pop.simd.splat %b : !meta.simd<size, type>
  // CHECK: return %[[U]], %[[V]]
  kgen.return %u, %v : !meta.simd<4, f32>, !meta.simd<size, type>
}

// CHECK-LABEL: @pop_buffer_load
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_buffer_load<size, type: dtype>(
    %a: !meta.buffer<size, type>,
    %b: !meta.buffer<size, f32>,
    %c: !meta.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  arith.constant
  %idx = arith.constant 2 : index
  // CHECK: %[[U:.*]] = pop.buffer.load %[[A]][%[[IDX]]] : !meta.buffer<size, type>
  %u = pop.buffer.load %a[%idx] : !meta.buffer<size, type>
  // CHECK: %[[V:.*]] = pop.buffer.load %[[B]][%[[IDX]]] : !meta.buffer<size, f32>
  %v = pop.buffer.load %b[%idx] : !meta.buffer<size, f32>
  // CHECK: %[[w:.*]] = pop.buffer.load %[[C]][%[[IDX]]] : !meta.buffer<4, si32>
  %w = pop.buffer.load %c[%idx] : !meta.buffer<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_buffer_store
// CHECK-SAME: %[[V0:[a-z0-9]+]]:
// CHECK-SAME: %[[V1:[a-z0-9]+]]:
// CHECK-SAME: %[[V2:[a-z0-9]+]]:
// CHECK-SAME: %[[A:[a-z0-9]+]]:
// CHECK-SAME: %[[B:[a-z0-9]+]]:
// CHECK-SAME: %[[C:[a-z0-9]+]]:
kgen.generator @pop_buffer_store<size, type: dtype>(
    %v0: !meta.scalar<type>,
    %v1: !meta.scalar<f32>,
    %v2: !meta.scalar<si32>,
    %a: !meta.buffer<size, type>,
    %b: !meta.buffer<size, f32>,
    %c: !meta.buffer<4, si32>
  ) {
  // CHECK: %[[IDX:.*]] =  arith.constant
  %idx = arith.constant 2 : index
  // CHECK: pop.buffer.store %[[V0]], %[[A]][%[[IDX]]] : !meta.buffer<size, type>
  pop.buffer.store %v0, %a[%idx] : !meta.buffer<size, type>
  // CHECK: pop.buffer.store %[[V1]], %[[B]][%[[IDX]]] : !meta.buffer<size, f32>
  pop.buffer.store %v1, %b[%idx] : !meta.buffer<size, f32>
  // CHECK: pop.buffer.store %[[V2]], %[[C]][%[[IDX]]] : !meta.buffer<4, si32>
  pop.buffer.store %v2, %c[%idx] : !meta.buffer<4, si32>
  kgen.return
}

// CHECK-LABEL: @pop_load_store
kgen.generator @pop_load_store<type: dtype>(%p0: !meta.pointer<f32>, %p1: !meta.pointer<type>) {
  // CHECK: %[[V0:.*]] = pop.load %{{.*}} : !meta.pointer<f32>
  %0 = pop.load %p0 : !meta.pointer<f32>
  // CHECK: %[[V1:.*]] = pop.load %{{.*}} : !meta.pointer<type>
  %1 = pop.load %p1 : !meta.pointer<type>
  // CHECK: pop.store %[[V0]], %{{.*}} : !meta.pointer<f32>
  pop.store %0, %p0 : !meta.pointer<f32>
  // CHECK: pop.store %[[V1]], %{{.*}} : !meta.pointer<type>
  pop.store %1, %p1 : !meta.pointer<type>
  kgen.return
}

// CHECK-LABEL: @pop_buffer_stack_allocation
kgen.generator @pop_buffer_stack_allocation<type:dtype, size>() {
  // CHECK: pop.buffer.stack_allocation : !meta.buffer<4, f32>
  %0 = pop.buffer.stack_allocation : !meta.buffer<4, f32>
  // CHECK: pop.buffer.stack_allocation : !meta.buffer<size, f32>
  %1 = pop.buffer.stack_allocation : !meta.buffer<size, f32>
  // CHECK: pop.buffer.stack_allocation : !meta.buffer<size, type>
  %2 = pop.buffer.stack_allocation : !meta.buffer<size, type>
  kgen.return
}
