// RUN: kgen-opt -split-input-file -lower-zap-to-pop -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @zap_print
kgen.generator @zap_print(%a: !pop.scalar<f32>) {
  // CHECK: %[[FMT:.*]] = pop.global_constant(#M.dense_array<102, 111, 111{{.*}}> : !M.array<7xsi8>)
  // CHECK: %[[C_STR:.*]] = pop.pointer.bitcast %[[FMT]] : !pop.pointer<array{{.*}}> to !pop.pointer<scalar<si8>>
  // CHECK: pop.external_call @KGEN_CompilerRT_PrintFormat(%[[C_STR]], %{{.*}}) (!pop.pointer<scalar<si8>>) -> ()
  zap.print "foo %f"(%a) : !pop.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_debug_assert
kgen.generator @zap_debug_assert(%cond: !pop.scalar<bool>) {
  // CHECK: pop.global_constant(#M.dense_array
  // CHECK: pop.global_constant(#M.dense_array
  // CHECK: pop.global_constant(#M.dense_array
  // CHECK: pop.external_call @KGEN_CompilerRT_DebugAssert
  zap.debug_assert %cond, "my message" : !pop.scalar<bool>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_global_string
kgen.generator @zap_global_string() -> !pop.pointer<array<4, scalar<si8>>> {
  // CHECK: pop.global_constant
  %0 = zap.global_string "foo!"[4]
  kgen.return %0 : !pop.pointer<array<4, scalar<si8>>>
}

// -----

// CHECK-LABEL: @zap_ndbuffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<scalar<f32>>
kgen.func @zap_ndbuffer_construct(%ptr: !pop.pointer<scalar<f32>>) {
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[ONE:.*]] = index.constant 1
  // CHECK-DAG: %[[FOUR:.*]] = index.constant 4
  // CHECK-DAG: %[[ARRAY:.*]] = pop.array.create [%[[FOUR]], %[[ZERO]], %[[ZERO]], %[[ZERO]], %[[ZERO]]] : !pop.array<5, index>
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: pop.struct.construct(%[[PTR]], %[[ONE]], %[[ARRAY]], %[[DTYPE]]) : !pop.struct<pointer<scalar<f32>>, index, array<5, index>, dtype>
  %0 = zap.ndbuffer.construct %ptr : !zap.ndbuffer<[4], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<scalar<f32>>
// CHECK-SAME: %[[SIZE:.*]]: index
kgen.func @zap_ndbuffer_construct(%ptr: !pop.pointer<scalar<f32>>,
                                %size: index) {
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[THREE:.*]] = index.constant 3
  // CHECK-DAG: %[[FOUR:.*]] = index.constant 4
  // CHECK-DAG: %[[ARRAY:.*]] = pop.array.create [%[[SIZE]], %[[FOUR]], %[[SIZE]], %[[ZERO]], %[[ZERO]]] : !pop.array<5, index>
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: pop.struct.construct(%[[PTR]], %[[THREE]], %[[ARRAY]], %[[DTYPE]]) : !pop.struct<pointer<scalar<f32>>, index, array<5, index>, dtype>
  %0 = zap.ndbuffer.construct %ptr[%size, %size] : !zap.ndbuffer<[?, 4, ?], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<scalar<?>>
// CHECK-SAME: %[[SIZE0:.*]]: index
// CHECK-SAME: %[[DTYPE:.*]]: !kgen.dtype
kgen.func @zap_ndbuffer_construct(%ptr: !pop.pointer<scalar<?>>,
                                %size: index,
                                %dtype: !kgen.dtype) {
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[FOUR:.*]] = index.constant 4
  // CHECK-DAG: %[[ARRAY:.*]] = pop.array.create [%[[SIZE]], %[[SIZE]], %[[SIZE]], %[[SIZE]], %[[ZERO]]] : !pop.array<5, index>
  // CHECK: pop.struct.construct(%[[PTR]], %[[FOUR]], %[[ARRAY]], %[[DTYPE]]) : !pop.struct<pointer<scalar<?>>, index, array<5, index>, dtype>
  %0 = zap.ndbuffer.construct %ptr[%size, %size, %size, %size] of %dtype : !zap.ndbuffer<[?, ?, ?, ?], ?>
  kgen.return
}

// -----

// CHECK-LABEL: @ndbuffer_stack_allocation
kgen.generator @ndbuffer_stack_allocation<type: dtype>(%i: index) -> (
  !zap.ndbuffer<[4], f32>, !zap.ndbuffer<[42, 42], type>
) {
  // CHECK: %[[PTR0:.*]] = pop.stack_allocation 4 x !pop.scalar<f32>
  // CHECK: %[[BUF0:.*]] = pop.struct.construct(%[[PTR0]], %{{.*}}, %{{.*}}, %{{.*}}) : !pop.struct<pointer<scalar<f32>>, index, array<5, index>, dtype>
  %0 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[4], f32>

  // CHECK: %[[PTR1:.*]] = pop.stack_allocation 1764 x !pop.scalar<type>
  // CHECK: %[[BUF1:.*]] = pop.struct.construct(%[[PTR1]], %{{.*}}, %{{.*}}, %{{.*}}) : !pop.struct<pointer<scalar<type>>, index, array<5, index>, dtype>
  %1 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[42, 42], type>

  // CHECK: return %[[BUF0]], %[[BUF1]]
  kgen.return %0, %1 : !zap.ndbuffer<[4], f32>,
                       !zap.ndbuffer<[42, 42], type>
}

// -----

// CHECK-LABEL: @ndbuffer_stack_allocation_with_alignment
kgen.generator @ndbuffer_stack_allocation_with_alignment<type: dtype>() {
  // CHECK: %[[PTR0:.*]] = pop.stack_allocation 4 x !pop.scalar<f32> align 8
  // CHECK: %[[BUF0:.*]] = pop.struct.construct(%[[PTR0]], %{{.*}}, %{{.*}}, %{{.*}}) : !pop.struct<pointer<scalar<f32>>, index, array<5, index>, dtype>
  %0 = zap.ndbuffer.stack_allocation align 8 : !zap.ndbuffer<[4], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @ndbuffer_stack_allocation_parametric_size
kgen.generator @ndbuffer_stack_allocation_parametric_size<type: dtype, size>(%i: index) -> (
  !zap.ndbuffer<[size], f32>, !zap.ndbuffer<[42, size, 2], type>
) {
  // CHECK: %[[PTR0:.*]] = pop.stack_allocation size x !pop.scalar<f32>
  // CHECK: %[[SIZE:.*]] = kgen.param.constant = <size>
  // CHECK: %[[BUF0:.*]] = pop.struct.construct(%[[PTR0]], %{{.*}}, %{{.*}}, %{{.*}}) : !pop.struct<pointer<scalar<f32>>, index, array<5, index>, dtype>
  %0 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[size], f32>

  // CHECK: %[[PTR1:.*]] = pop.stack_allocation mul(size, 84) x !pop.scalar<type>
  // CHECK: %[[BUF1:.*]] = pop.struct.construct(%[[PTR1]], %{{.*}}, %{{.*}}, %{{.*}}) : !pop.struct<pointer<scalar<type>>, index, array<5, index>, dtype>
  %1 = zap.ndbuffer.stack_allocation : !zap.ndbuffer<[42, size, 2], type>

  // CHECK: return %[[BUF0]], %[[BUF1]]
  kgen.return %0, %1 : !zap.ndbuffer<[size], f32>,
                       !zap.ndbuffer<[42, size, 2], type>
}

// -----

// CHECK-LABEL: @zap_ndbuffer_dim
// CHECK-SAME: %[[NDBUFFER0:.*]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !pop.struct<pointer<scalar<si32>>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !pop.struct<pointer<scalar<?>>
kgen.func @zap_ndbuffer_dim(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?, 4, ?], si32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?], ?>) {
  // CHECK: kgen.param.constant  = <4>
  %0 = zap.ndbuffer.dim %ndbuffer0[0] : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: %[[ARRAY0:.*]] = pop.struct.get %[[NDBUFFER1]][2] : !pop.struct<
  // CHECK: pop.array.get %[[ARRAY0]][0] : !pop.array<5, index>
  %1 = zap.ndbuffer.dim %ndbuffer1[0] : !zap.ndbuffer<[?, 4, ?], si32>
  // CHECK: kgen.param.constant  = <4>
  %2 = zap.ndbuffer.dim %ndbuffer1[1] : !zap.ndbuffer<[?, 4, ?], si32>
  // CHECK: %[[ARRAY1:.*]] = pop.struct.get %[[NDBUFFER2]][2] : !pop.struct<
  // CHECK: pop.array.get %[[ARRAY1]][0] : !pop.array<5, index>
  %3 = zap.ndbuffer.dim %ndbuffer2[0] : !zap.ndbuffer<[?, ?, ?], ?>
  // CHECK: %[[ARRAY2:.*]] = pop.struct.get %[[NDBUFFER2]][2] : !pop.struct<
  // CHECK: pop.array.get %[[ARRAY2]][2] : !pop.array<5, index>
  %4 = zap.ndbuffer.dim %ndbuffer2[2] : !zap.ndbuffer<[?, ?, ?], ?>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_rank
// CHECK-SAME: %[[NDBUFFER0:.*]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !pop.struct<pointer<scalar<si32>>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !pop.struct<pointer<scalar<?>>
kgen.func @zap_ndbuffer_rank(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?], si32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?, ?], ?>) {
  // CHECK: kgen.param.constant = <3>
  %0 = zap.ndbuffer.rank %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: kgen.param.constant = <1>
  %1 = zap.ndbuffer.rank %ndbuffer1 : !zap.ndbuffer<[?], si32>
  // CHECK: kgen.param.constant = <4>
  %2 = zap.ndbuffer.rank %ndbuffer2 : !zap.ndbuffer<[?, ?, ?, ?], ?>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_address
// CHECK-SAME: %[[NDBUFFER0:.*]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !pop.struct<pointer<scalar<si32>>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !pop.struct<pointer<scalar<?>>
kgen.func @zap_ndbuffer_address(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?], si32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?, ?], ?>) {
  // CHECK: pop.struct.get %[[NDBUFFER0]][0]
  %0 = zap.ndbuffer.address %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: pop.struct.get %[[NDBUFFER1]][0]
  %1 = zap.ndbuffer.address %ndbuffer1 : !zap.ndbuffer<[?], si32>
  // CHECK: pop.struct.get %[[NDBUFFER2]][0]
  %2 = zap.ndbuffer.address %ndbuffer2 : !zap.ndbuffer<[?, ?, ?, ?], ?>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_dtype
// CHECK-SAME: %[[NDBUFFER0:.*]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !pop.struct<pointer<scalar<si32>>
// CHECK-SAME: %[[NDBUFFER2:.*]]: !pop.struct<pointer<scalar<?>>
kgen.func @zap_ndbuffer_dtype(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?], si32>,
  %ndbuffer2: !zap.ndbuffer<[?, ?, ?, ?], ?>) {
  // CHECK: kgen.param.constant: dtype = <f32>
  %0 = zap.ndbuffer.dtype %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK: kgen.param.constant: dtype = <si32>
  %1 = zap.ndbuffer.dtype %ndbuffer1 : !zap.ndbuffer<[?], si32>
  // CHECK: pop.struct.get %[[NDBUFFER2]][3]
  %2 = zap.ndbuffer.dtype %ndbuffer2 : !zap.ndbuffer<[?, ?, ?, ?], ?>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_size
// CHECK-SAME: %[[NDBUFFER0:.*]]: !pop.struct<pointer<scalar<f32>>
kgen.func @zap_ndbuffer_size(%ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>) {
  // CHECK: kgen.param.constant = <60>
  %0 = zap.ndbuffer.size %ndbuffer0 : !zap.ndbuffer<[4, 5, 3], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_size
// CHECK-SAME: %[[NDBUFFER:.*]]: !pop.struct<pointer<scalar<f32>>
kgen.func @zap_ndbuffer_size(%ndbuffer0: !zap.ndbuffer<[4, ?, 3], f32>) {
  // CHECK: %[[STRUCT:.*]] = pop.struct.get %[[NDBUFFER]][2]
  // CHECK: %[[DIM0:.*]] = index.constant 4
  // CHECK: %[[DIM1:.*]] = pop.array.get %[[STRUCT]][1] : !pop.array<5, index>
  // CHECK: %[[PARTIAL:.*]] = index.mul %[[DIM0]], %[[DIM1]]
  // CHECK: %[[DIM2:.*]] = index.constant 3
  // CHECK: %[[SIZE:.*]] = index.mul %[[PARTIAL]], %[[DIM2]]
  %0 = zap.ndbuffer.size %ndbuffer0 : !zap.ndbuffer<[4, ?, 3], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_load
// CHECK-SAME: %[[NDBUFFER0:[a-z0-9]+]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[NDBUFFER1:[a-z0-9]+]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[IDX0:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX1:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX2:.*]]: index)
kgen.func @zap_ndbuffer_load(
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?, 99, ?], f32>,
  %idx0: index,
  %idx1: index,
  %idx2: index) {
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[NDBUFFER0]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[NDBUFFER0]][0]
  // CHECK-DAG: %[[SIZE1:.*]] = index.constant 5
  // CHECK-DAG: %[[SIZE2:.*]] = index.constant 3
  // CHECK-DAG: %[[MUL1:.*]] = index.mul %[[IDX0]], %[[SIZE1]]
  // CHECK-DAG: %[[ADD1:.*]] = index.add %[[MUL1]], %[[IDX1]]
  // CHECK-DAG: %[[MUL2:.*]] = index.mul %[[ADD1]], %[[SIZE2]]
  // CHECK-DAG: %[[ADD2:.*]] = index.add %[[MUL2]], %[[IDX2]]
  // CHECK-DAG: %[[POP_OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD2]]] : !pop.pointer<scalar<f32>>
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[POP_OFFSET]]
  // CHECK: pop.load %[[SIMD_OFFSET]] : !pop.pointer<scalar<f32>>
  %0 = zap.ndbuffer.load %ndbuffer0[%idx0, %idx1, %idx2] : !zap.ndbuffer<[4, 5, 3], f32>, !pop.scalar<f32>
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[NDBUFFER1]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[NDBUFFER1]][0]
  // CHECK-DAG: %[[SIZE1:.*]] = index.constant 99
  // CHECK-DAG: %[[SIZE2:.*]] = pop.array.get %[[SHAPEARRAY]][2]
  // CHECK-DAG: %[[MUL1:.*]] = index.mul %[[IDX0]], %[[SIZE1]]
  // CHECK-DAG: %[[ADD1:.*]] = index.add %[[MUL1]], %[[IDX1]]
  // CHECK-DAG: %[[MUL2:.*]] = index.mul %[[ADD1]], %[[SIZE2]]
  // CHECK-DAG: %[[ADD2:.*]] = index.add %[[MUL2]], %[[IDX2]]
  // CHECK-DAG: %[[POP_OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD2]]] : !pop.pointer<scalar<f32>>
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[POP_OFFSET]]
  // CHECK: pop.load %[[SIMD_OFFSET]] : !pop.pointer<scalar<f32>>
  %1 = zap.ndbuffer.load %ndbuffer1[%idx0, %idx1, %idx2] : !zap.ndbuffer<[?, 99, ?], f32>, !pop.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_store
// CHECK-SAME: %[[VAL:.*]]: !pop.scalar<f32>,
// CHECK-SAME: %[[NDBUFFER0:arg[0-9]+]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[NDBUFFER1:.*]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[IDX0:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX1:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX2:.*]]: index)
kgen.func @zap_ndbuffer_store(
  %val : !pop.scalar<f32>,
  %ndbuffer0: !zap.ndbuffer<[4, 5, 3], f32>,
  %ndbuffer1: !zap.ndbuffer<[?, 99, ?], f32>,
  %idx0: index,
  %idx1: index,
  %idx2: index) {
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[NDBUFFER0]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[NDBUFFER0]][0]
  // CHECK-DAG: %[[SIZE1:.*]] = index.constant 5
  // CHECK-DAG: %[[SIZE2:.*]] = index.constant 3
  // CHECK-DAG: %[[MUL1:.*]] = index.mul %[[IDX0]], %[[SIZE1]]
  // CHECK-DAG: %[[ADD1:.*]] = index.add %[[MUL1]], %[[IDX1]]
  // CHECK-DAG: %[[MUL2:.*]] = index.mul %[[ADD1]], %[[SIZE2]]
  // CHECK-DAG: %[[ADD2:.*]] = index.add %[[MUL2]], %[[IDX2]]
  // CHECK-DAG: %[[POP_OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD2]]] : !pop.pointer<scalar<f32>>
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[POP_OFFSET]]
  // CHECK: pop.store %[[VAL]], %[[SIMD_OFFSET]] : !pop.pointer<scalar<f32>>
  zap.ndbuffer.store %val, %ndbuffer0[%idx0, %idx1, %idx2] : !pop.scalar<f32>, !zap.ndbuffer<[4, 5, 3], f32>
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[NDBUFFER1]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[NDBUFFER1]][0]
  // CHECK-DAG: %[[SIZE1:.*]] = index.constant 99
  // CHECK-DAG: %[[SIZE2:.*]] = pop.array.get %[[SHAPEARRAY]][2]
  // CHECK-DAG: %[[MUL1:.*]] = index.mul %[[IDX0]], %[[SIZE1]]
  // CHECK-DAG: %[[ADD1:.*]] = index.add %[[MUL1]], %[[IDX1]]
  // CHECK-DAG: %[[MUL2:.*]] = index.mul %[[ADD1]], %[[SIZE2]]
  // CHECK-DAG: %[[ADD2:.*]] = index.add %[[MUL2]], %[[IDX2]]
  // CHECK-DAG: %[[POP_OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD2]]] : !pop.pointer<scalar<f32>>
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[POP_OFFSET]]
  // CHECK: pop.store %[[VAL]], %[[SIMD_OFFSET]] : !pop.pointer<scalar<f32>>
  zap.ndbuffer.store %val, %ndbuffer1[%idx0, %idx1, %idx2] : !pop.scalar<f32>, !zap.ndbuffer<[?, 99, ?], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_aligned_load
// CHECK-SAME: %[[NDBUFFER0:[a-z0-9]+]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[IDX0:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX1:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX2:.*]]: index)
kgen.func @zap_ndbuffer_aligned_load(
  %ndbuffer0: !zap.ndbuffer<[?, 5, ?], f32>,
  %idx0: index,
  %idx1: index,
  %idx2: index) {
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[NDBUFFER0]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[NDBUFFER0]][0]
  // CHECK-DAG: %[[SIZE1:.*]] = index.constant 5
  // CHECK-DAG: %[[SIZE2:.*]] = pop.array.get %[[SHAPEARRAY]][2]
  // CHECK-DAG: %[[MUL1:.*]] = index.mul %[[IDX0]], %[[SIZE1]]
  // CHECK-DAG: %[[ADD1:.*]] = index.add %[[MUL1]], %[[IDX1]]
  // CHECK-DAG: %[[MUL2:.*]] = index.mul %[[ADD1]], %[[SIZE2]]
  // CHECK-DAG: %[[ADD2:.*]] = index.add %[[MUL2]], %[[IDX2]]
  // CHECK: %[[POP_OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD2]]]
  // CHECK: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[POP_OFFSET]]
  // CHECK: pop.load %[[SIMD_OFFSET]] align 64 : !pop.pointer<simd<4, f32>>
  %0 = zap.ndbuffer.load %ndbuffer0[%idx0, %idx1, %idx2] align 64 : !zap.ndbuffer<[?, 5, ?], f32>, !pop.simd<4, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_store_aligned
// CHECK-SAME: %[[VAL:.*]]: !pop.simd<4, f32>
// CHECK-SAME: %[[NDBUFFER0:[a-z0-9]+]]: !pop.struct<pointer<scalar<f32>>
// CHECK-SAME: %[[IDX0:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX1:arg[0-9]+]]: index,
// CHECK-SAME: %[[IDX2:.*]]: index)
kgen.func @zap_ndbuffer_store_aligned(
  %val : !pop.simd<4, f32>,
  %ndbuffer0: !zap.ndbuffer<[?, 5, ?], f32>,
  %idx0: index,
  %idx1: index,
  %idx2: index) {
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[NDBUFFER0]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[NDBUFFER0]][0]
  // CHECK-DAG: %[[SIZE1:.*]] = index.constant 5
  // CHECK-DAG: %[[SIZE2:.*]] = pop.array.get %[[SHAPEARRAY]][2]
  // CHECK-DAG: %[[MUL1:.*]] = index.mul %[[IDX0]], %[[SIZE1]]
  // CHECK-DAG: %[[ADD1:.*]] = index.add %[[MUL1]], %[[IDX1]]
  // CHECK-DAG: %[[MUL2:.*]] = index.mul %[[ADD1]], %[[SIZE2]]
  // CHECK-DAG: %[[ADD2:.*]] = index.add %[[MUL2]], %[[IDX2]]
  // CHECK: %[[POP_OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD2]]]
  // CHECK: %[[OFFSET:.*]] = pop.pointer.bitcast %[[POP_OFFSET]]
  // CHECK: pop.store %[[VAL]], %[[OFFSET]] align 8 : !pop.pointer<simd<4, f32>>
  zap.ndbuffer.store %val, %ndbuffer0[%idx0, %idx1, %idx2] align 8 : !pop.simd<4, f32>, !zap.ndbuffer<[?, 5, ?], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_ndbuffer_loadstore_with_param
// CHECK-SAME: %[[VAL:.*]]: !pop.scalar<type>,
// CHECK-SAME: %[[BUFFER:.*]]: !pop.struct<pointer<scalar<type>>
kgen.generator @zap_ndbuffer_loadstore_with_param<size, type: dtype>(
    %val : !pop.scalar<type>,
    %buffer: !zap.ndbuffer<[size, size], type>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[BUFFER]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[BUFFER]][0]
  // CHECK-DAG: %[[SIZE:.*]] = kgen.param.constant = <size>
  // CHECK-DAG: %[[MUL:.*]] = index.mul %[[IDX]], %[[SIZE]]
  // CHECK-DAG: %[[ADD:.*]] = index.add %[[MUL]], %[[IDX]]
  // CHECK-DAG: %[[OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD]]]
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[OFFSET]]
  // CHECK-DAG: pop.load %[[SIMD_OFFSET]]
  %u = zap.ndbuffer.load %buffer[%idx, %idx] : !zap.ndbuffer<[size, size], type>, !pop.scalar<type>
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[BUFFER]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[BUFFER]][0]
  // CHECK-DAG: %[[SIZE:.*]] = kgen.param.constant = <size>
  // CHECK-DAG: %[[MUL:.*]] = index.mul %[[IDX]], %[[SIZE]]
  // CHECK-DAG: %[[ADD:.*]] = index.add %[[MUL]], %[[IDX]]
  // CHECK-DAG: %[[OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD]]]
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[OFFSET]]
  // CHECK-DAG: pop.store %[[VAL]], %[[SIMD_OFFSET]]
  zap.ndbuffer.store %val, %buffer[%idx, %idx] : !pop.scalar<type>, !zap.ndbuffer<[size, size], type>
  kgen.return
}


// -----

// CHECK-LABEL: @zap_ndbuffer_loadstore_aligned_with_param
// CHECK-SAME: %[[VAL:.*]]: !pop.scalar<type>,
// CHECK-SAME: %[[BUFFER:.*]]: !pop.struct<pointer<scalar<type>>
kgen.generator @zap_ndbuffer_loadstore_aligned_with_param<size, type: dtype>(
    %val : !pop.scalar<type>,
    %buffer: !zap.ndbuffer<[size, size], type>
  ) {
  // CHECK: %[[IDX:.*]] =  index.constant
  %idx = index.constant 2
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[BUFFER]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[BUFFER]][0]
  // CHECK-DAG: %[[SIZE:.*]] = kgen.param.constant = <size>
  // CHECK-DAG: %[[MUL:.*]] = index.mul %[[IDX]], %[[SIZE]]
  // CHECK-DAG: %[[ADD:.*]] = index.add %[[MUL]], %[[IDX]]
  // CHECK-DAG: %[[OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD]]]
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[OFFSET]]
  // CHECK-DAG: pop.load %[[SIMD_OFFSET]] align size
  %u = zap.ndbuffer.load %buffer[%idx, %idx] align size : !zap.ndbuffer<[size, size], type>, !pop.scalar<type>
  // CHECK-DAG: %[[SHAPEARRAY:.*]] = pop.struct.get %[[BUFFER]][2]
  // CHECK-DAG: %[[BASE:.*]] = pop.struct.get %[[BUFFER]][0]
  // CHECK-DAG: %[[SIZE:.*]] = kgen.param.constant = <size>
  // CHECK-DAG: %[[MUL:.*]] = index.mul %[[IDX]], %[[SIZE]]
  // CHECK-DAG: %[[ADD:.*]] = index.add %[[MUL]], %[[IDX]]
  // CHECK-DAG: %[[OFFSET:.*]] = pop.offset %[[BASE]][%[[ADD]]]
  // CHECK-DAG: %[[SIMD_OFFSET:.*]] = pop.pointer.bitcast %[[OFFSET]]
  // CHECK-DAG: pop.store %[[VAL]], %[[SIMD_OFFSET]] align size
  zap.ndbuffer.store %val, %buffer[%idx, %idx] align size : !pop.scalar<type>, !zap.ndbuffer<[size, size], type>
  kgen.return
}

// -----

// CHECK-LABEL: @ndbuffer_bitcast
// CHECK-SAME: %[[A:.*]]:
kgen.func @ndbuffer_bitcast(%a: !zap.ndbuffer<[4], f32>) -> !zap.ndbuffer<[32], f32> {
  // CHECK-DAG: %[[PTR:.*]] = pop.struct.get %[[A]][0]
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[SIZE:.*]] = index.constant 32
  // CHECK-DAG: %[[SHAPE:.*]] = pop.array.create [%[[SIZE]], %[[ZERO]]
  // CHECK-DAG: %[[RANK:.*]] = index.constant 1
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[NDBUF:.*]] = pop.struct.construct(%[[PTR]], %[[RANK]], %[[SHAPE]], %[[DTYPE]])
  %0 = zap.ndbuffer.bitcast %a : !zap.ndbuffer<[4], f32> to !zap.ndbuffer<[32], f32>
  // CHECK: return %[[NDBUF]]
  kgen.return %0 : !zap.ndbuffer<[32], f32>
}

// -----

// CHECK-LABEL: @ndbuffer_bitcast
// CHECK-SAME: %[[A:.*]]:
kgen.func @ndbuffer_bitcast(%a: !zap.ndbuffer<[4], f32>) -> !zap.ndbuffer<[?], f64> {
  // CHECK-DAG: %[[PTR0:.*]] = pop.struct.get %[[A]][0]
  // CHECK-DAG: %[[PTR:.*]] = pop.pointer.bitcast %[[PTR0]]
  // CHECK-DAG: %[[SHAPE_ARRAY:.*]] = pop.struct.get %[[A]][2]
  // CHECK-DAG: %[[SIZE:.*]] = pop.array.get %[[SHAPE_ARRAY]][0]
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[SHAPE:.*]] = pop.array.create [%[[SIZE]], %[[ZERO]]
  // CHECK-DAG: %[[RANK:.*]] = index.constant 1
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f64>
  // CHECK: %[[NDBUF:.*]] = pop.struct.construct(%[[PTR]], %[[RANK]], %[[SHAPE]], %[[DTYPE]])
  %0 = zap.ndbuffer.bitcast %a : !zap.ndbuffer<[4], f32> to !zap.ndbuffer<[?], f64>
  // CHECK: return %[[NDBUF]]
  kgen.return %0 : !zap.ndbuffer<[?], f64>
}

// -----

// CHECK-LABEL: @ndbuffer_bitcast
// CHECK-SAME: %[[A:.*]]:
kgen.func @ndbuffer_bitcast(%a: !zap.ndbuffer<[4], f32>) -> !zap.ndbuffer<[?], ?> {
  // CHECK-DAG: %[[PTR0:.*]] = pop.struct.get %[[A]][0]
  // CHECK-DAG: %[[PTR:.*]] = pop.pointer.bitcast %[[PTR0]]
  // CHECK-DAG: %[[SHAPE_ARRAY:.*]] = pop.struct.get %[[A]][2]
  // CHECK-DAG: %[[SIZE:.*]] = pop.array.get %[[SHAPE_ARRAY]][0]
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[SHAPE:.*]] = pop.array.create [%[[SIZE]], %[[ZERO]]
  // CHECK-DAG: %[[RANK:.*]] = index.constant 1
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[NDBUF:.*]] = pop.struct.construct(%[[PTR]], %[[RANK]], %[[SHAPE]], %[[DTYPE]])
  %0 = zap.ndbuffer.bitcast %a : !zap.ndbuffer<[4], f32> to !zap.ndbuffer<[?], ?>
  // CHECK: return %[[NDBUF]]
  kgen.return %0 : !zap.ndbuffer<[?], ?>
}

// -----

// CHECK-LABEL: @ndbuffer_bitcast
// CHECK-SAME: %[[A:.*]]:
kgen.func @ndbuffer_bitcast(%a: !zap.ndbuffer<[4], ?>) -> !zap.ndbuffer<[?], f32> {
  // CHECK-DAG: %[[PTR0:.*]] = pop.struct.get %[[A]][0]
  // CHECK-DAG: %[[PTR:.*]] = pop.pointer.bitcast %[[PTR0]]
  // CHECK-DAG: %[[SHAPE_ARRAY:.*]] = pop.struct.get %[[A]][2]
  // CHECK-DAG: %[[SIZE:.*]] = pop.array.get %[[SHAPE_ARRAY]][0]
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[SHAPE:.*]] = pop.array.create [%[[SIZE]], %[[ZERO]]
  // CHECK-DAG: %[[RANK:.*]] = index.constant 1
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[NDBUF:.*]] = pop.struct.construct(%[[PTR]], %[[RANK]], %[[SHAPE]], %[[DTYPE]])
  %0 = zap.ndbuffer.bitcast %a : !zap.ndbuffer<[4], ?> to !zap.ndbuffer<[?], f32>
  // CHECK: return %[[NDBUF]]
  kgen.return %0 : !zap.ndbuffer<[?], f32>
}
