// RUN: kgen-opt -split-input-file -lower-zap-to-pop -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @buffer_size
kgen.func @buffer_size(%a: !zap.buffer<4, f32>) -> index {
  // CHECK: %[[S:.*]] = kgen.param.constant = <4>
  %0 = zap.buffer.size %a : !zap.buffer<4, f32>
  // CHECK: return %[[S]]
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @buffer_size
// CHECK-SAME: %[[A:.*]]: !pop.struct
kgen.func @buffer_size(%a: !zap.buffer<?, f32>) -> index {
  // CHECK: %[[S:.*]] = pop.struct.get %[[A]][1]
  %0 = zap.buffer.size %a : !zap.buffer<?, f32>
  // CHECK: return %[[S]] : index
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @buffer_dtype
kgen.func @buffer_dtype(%a: !zap.buffer<?, f32>) -> !kgen.dtype {
  // CHECK: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  %0 = zap.buffer.dtype %a : !zap.buffer<?, f32>
  // CHECK: return %[[DTYPE]]
  kgen.return %0 : !kgen.dtype
}

// -----

// CHECK-LABEL: @buffer_dtype
// CHECK-SAME: %[[A:.*]]: !pop.struct
kgen.func @buffer_dtype(%a: !zap.buffer<4, ?>) -> !kgen.dtype {
  // CHECK: %[[DTYPE:.*]] = pop.struct.get %[[A]][2]
  %0 = zap.buffer.dtype %a : !zap.buffer<4, ?>
  // CHECK: return %[[DTYPE]]
  kgen.return %0 : !kgen.dtype
}

// -----

// CHECK-LABEL: @buffer_address
// CHECK-SAME: %[[A:.*]]: !pop.struct
kgen.func @buffer_address(%a: !zap.buffer<4, f32>) -> !pop.pointer<!pop.scalar<f32>> {
  // CHECK: %[[PTR:.*]] = pop.struct.get %[[A]][0]
  %0 = zap.buffer.address %a : !zap.buffer<4, f32>
  // CHECK: return %[[PTR]]
  kgen.return %0 : !pop.pointer<!pop.scalar<f32>>
}

// -----

// CHECK-LABEL: @buffer_convert
// CHECK-SAME: %[[A:.*]]:
kgen.func @buffer_convert(%a: !zap.buffer<?, ?>) -> !zap.buffer<?, ?> {
  %0 = zap.buffer.bitcast %a : !zap.buffer<?, ?> to !zap.buffer<?, ?>
  // CHECK: return %[[A]]
  kgen.return %0 : !zap.buffer<?, ?>
}

// -----

// CHECK-LABEL: @buffer_convert
// CHECK-SAME: %[[A:.*]]:
kgen.func @buffer_convert(%a: !zap.buffer<4, ?>) -> !zap.buffer<32, ?> {
  // CHECK: %[[PTR:.*]] = pop.struct.get %[[A]][0]
  // CHECK: %[[SIZE:.*]] = kgen.param.constant = <32>
  // CHECK: %[[DTYPE:.*]] = pop.struct.get %[[A]][2]
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.bitcast %a : !zap.buffer<4, ?> to !zap.buffer<32, ?>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<32, ?>
}

// -----

// CHECK-LABEL: @buffer_convert
// CHECK-SAME: %[[A:.*]]:
kgen.func @buffer_convert(%a: !zap.buffer<?, f32>) -> !zap.buffer<?, f64> {
  // CHECK: %[[RAW:.*]] = pop.struct.get %[[A]][0]
  // CHECK: %[[PTR:.*]] = pop.pointer.bitcast %[[RAW]] : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.scalar<f64>>
  // CHECK: %[[SIZE:.*]] = pop.struct.get %[[A]][1]
  // CHECK: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f64>
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.bitcast %a : !zap.buffer<?, f32> to !zap.buffer<?, f64>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<?, f64>
}

// -----

// CHECK-LABEL: @buffer_convert
// CHECK-SAME: %[[A:.*]]:
kgen.func @buffer_convert(%a: !zap.buffer<?, ?>) -> !zap.buffer<4, f32> {
  // CHECK: %[[RAW:.*]] = pop.struct.get %[[A]][0]
  // CHECK: %[[PTR:.*]] = pop.pointer.bitcast %[[RAW]]
  // CHECK: %[[SIZE:.*]] = kgen.param.constant = <4>
  // CHECK: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.bitcast %a : !zap.buffer<?, ?> to !zap.buffer<4, f32>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<4, f32>
}

// -----

// CHECK-LABEL: @buffer_convert
// CHECK-SAME: %[[A:.*]]:
kgen.func @buffer_convert(%a: !zap.buffer<4, f32>) -> !zap.buffer<?, ?> {
  // CHECK: %[[RAW:.*]] = pop.struct.get %[[A]][0]
  // CHECK: %[[PTR:.*]] = pop.pointer.bitcast %[[RAW]]
  // CHECK: %[[SIZE:.*]] = kgen.param.constant = <4>
  // CHECK: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.bitcast %a : !zap.buffer<4, f32> to !zap.buffer<?, ?>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<?, ?>
}

// -----

// CHECK-LABEL: @buffer_construct
// CHECK-SAME: %[[PTR:.*]]:
kgen.func @buffer_construct(%ptr: !pop.pointer<!pop.scalar<f32>>) -> !zap.buffer<4, f32> {
  // CHECK: %[[SIZE:.*]] = kgen.param.constant = <4>
  // CHECK: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.construct %ptr : !zap.buffer<4, f32>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<4, f32>
}

// -----

// CHECK-LABEL: @buffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer
// CHECK-SAME: %[[SIZE:.*]]:
kgen.func @buffer_construct(%ptr: !pop.pointer<!pop.scalar<f32>>, %size: index) -> !zap.buffer<?, f32> {
  // CHECK: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.construct %ptr[%size] : !zap.buffer<?, f32>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<?, f32>
}

// -----

// CHECK-LABEL: @buffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer
// CHECK-SAME: %[[DTYPE:.*]]:
kgen.func @buffer_construct(%ptr: !pop.pointer<!pop.scalar<invalid>>, %dtype: !kgen.dtype) -> !zap.buffer<4, ?> {
  // CHECK: %[[SIZE:.*]] = kgen.param.constant = <4>
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.construct %ptr of %dtype : !zap.buffer<4, ?>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<4, ?>
}

// -----

// CHECK-LABEL: @buffer_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer
// CHECK-SAME: %[[SIZE:.*]]: index, %[[DTYPE:.*]]:
kgen.func @buffer_construct(%ptr: !pop.pointer<!pop.scalar<invalid>>, %size: index, %dtype: !kgen.dtype) -> !zap.buffer<?, ?> {
  // CHECK: %[[BUF:.*]] = pop.struct.construct(%[[PTR]], %[[SIZE]], %[[DTYPE]])
  %0 = zap.buffer.construct %ptr[%size] of %dtype : !zap.buffer<?, ?>
  // CHECK: return %[[BUF]]
  kgen.return %0 : !zap.buffer<?, ?>
}

// -----

!pop_struct0 = !pop.struct<index, !pop.pointer<!pop.scalar<f32>>, !kgen.dtype>
!pop_struct1 = !pop.struct<index, !pop.pointer<!pop.scalar<type>>, !kgen.dtype>

// CHECK-LABEL: @buffer_stack_allocation
kgen.generator @buffer_stack_allocation<size, type: dtype>(%i: index) -> (
  !zap.buffer<4, f32>, !zap.buffer<size, type>
) {
  // CHECK: %[[PTR0:.*]] = pop.stack_allocation 4 : !pop.scalar<f32>
  // CHECK: %[[BUF0:.*]] = pop.struct.construct(%[[PTR0]], %{{.*}}, %{{.*}}) : !pop.struct<!pop.pointer<!pop.scalar<f32>>, index, !kgen.dtype>
  %0 = zap.buffer.stack_allocation : !zap.buffer<4, f32>

  // CHECK: %[[PTR1:.*]] = pop.stack_allocation size : !pop.scalar<type>
  // CHECK: %[[BUF1:.*]] = pop.struct.construct(%[[PTR1]], %{{.*}}, %{{.*}}) : !pop.struct<!pop.pointer<!pop.scalar<type>>, index, !kgen.dtype>
  %1 = zap.buffer.stack_allocation : !zap.buffer<size, type>

  // CHECK: return %[[BUF0]], %[[BUF1]]
  kgen.return %0, %1 : !zap.buffer<4, f32>, !zap.buffer<size, type>
}

// -----

// CHECK-LABEL: @zap_buffer_constant
kgen.generator @zap_buffer_constant<type: dtype>(%i: index) -> (
  !zap.buffer<3, f32>, !zap.buffer<2, type>
) {
  // CHECK: %[[ARR0:.*]] = pop.global_constant(#M.dense_array<1.{{0+}}e+01, 1.2{{0+}}e+01, -2.{{0+}}e+00>
  // CHECK: %[[PTR0:.*]] = pop.pointer.bitcast %[[ARR0]] : !pop.pointer<!pop.array<3, !pop.scalar<f32>>> to !pop.pointer<!pop.scalar<f32>>
  // CHECK: %[[BUF0:.*]] = pop.struct.construct(%[[PTR0]],
  // CHECK: %[[ARR1:.*]] = pop.global_constant(#M.dense_array<2, 3>
  // CHECK: %[[PTR1:.*]] = pop.pointer.bitcast %[[ARR1]] : !pop.pointer<!pop.array<2, !pop.scalar<type>>> to !pop.pointer<!pop.scalar<type>>
  // CHECK: %[[BUF1:.*]] = pop.struct.construct(%[[PTR1]],
  %0 = zap.buffer.constant(#M.dense_array<10.0, 12.0, -2.0> : !M.array<3xf32>) : f32
  %1 = zap.buffer.constant(#M.dense_array<2, 3> : !M.array<2xui8>) : type

  kgen.return %0, %1 : !zap.buffer<3, f32>, !zap.buffer<2, type>
}

// -----

// CHECK-LABEL: @buffer_load
// CHECK-SAME: %[[BUF:.*]]: !pop.struct
// CHECK-SAME: %[[IDX:.*]]: index
kgen.generator @buffer_load(%buf: !zap.buffer<4, f32>, %idx: index) -> !pop.scalar<f32> {
  // CHECK: %[[BASE:.*]] = pop.struct.get %[[BUF]][0]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: %[[VAL:.*]] = pop.load %[[PTR]]
  // CHECK: return %[[VAL]]
  %0 = zap.buffer.load %buf[%idx] : !zap.buffer<4, f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

// CHECK-LABEL: @buffer_store
// CHECK-SAME: %[[VAL:.*]]: !pop.scalar
// CHECK-SAME: %[[BUF:.*]]: !pop.struct
// CHECK-SAME: %[[IDX:.*]]: index
kgen.generator @buffer_store(%val: !pop.scalar<f32>, %buf: !zap.buffer<4, f32>, %idx: index) -> () {
  // CHECK: %[[BASE:.*]] = pop.struct.get %[[BUF]][0]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: pop.store %[[VAL]], %[[PTR]]
  zap.buffer.store %val, %buf[%idx] : !zap.buffer<4, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @simd_load
// CHECK-SAME: %[[BUF:.*]]: !pop.struct
// CHECK-SAME: %[[IDX:.*]]: index
kgen.generator @simd_load(%buf: !zap.buffer<4, f32>, %idx: index) -> !pop.simd<4, f32> {
  // CHECK: %[[BASE:.*]] = pop.struct.get %[[BUF]][0]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: %[[BPTR:.*]] = pop.pointer.bitcast %[[PTR]] : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.simd<4, f32>>
  // CHECK: %[[VAL:.*]] = pop.load %[[BPTR]]
  %0 = zap.buffer.simd_load %buf[%idx] : !zap.buffer<4, f32>, !pop.simd<4, f32>
  kgen.return %0 : !pop.simd<4, f32>
}

// -----

// CHECK-LABEL: @simd_store
// CHECK-SAME: %[[VAL:.*]]: !pop.simd
// CHECK-SAME: %[[BUF:.*]]: !pop.struct
// CHECK-SAME: %[[IDX:.*]]: index
kgen.generator @simd_store(%val : !pop.simd<4, f32>, %buf: !zap.buffer<4, f32>, %idx: index) {
  // CHECK: %[[BASE:.*]] = pop.struct.get %[[BUF]][0]
  // CHECK: %[[PTR:.*]] = pop.offset %[[BASE]][%[[IDX]]]
  // CHECK: %[[BPTR:.*]] = pop.pointer.bitcast %[[PTR]] : !pop.pointer<!pop.scalar<f32>> to !pop.pointer<!pop.simd<4, f32>>
  // CHECK: pop.store %[[VAL]], %[[BPTR]]
  zap.buffer.simd_store %val, %buf[%idx] : !pop.simd<4, f32>, !zap.buffer<4, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_print
kgen.generator @zap_print(%a: !pop.scalar<f32>) {
  // CHECK: %[[FMT:.*]] = pop.global_constant(#M.dense_array<102, 111, 111{{.*}}> : !M.array<7xsi8>)
  // CHECK: %[[C_STR:.*]] = pop.pointer.bitcast %[[FMT]] : !pop.pointer<!pop.array{{.*}}> to !pop.pointer<!pop.scalar<si8>>
  // CHECK: pop.external_call @printf(%[[C_STR]], %{{.*}}) (!pop.pointer<!pop.scalar<si8>>) -> ()
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

// CHECK-LABEL: @simd_store
// CHECK-SAME: !pop.simd
// CHECK-SAME: !pop.struct
// CHECK-SAME: index
kgen.precompiled.llvm @simd_store(%val : !pop.simd<4, f32>, %buf: !zap.buffer<4, f32>, %idx: index) attributes {
  compiledFor = #kgen.target<"darwin-arm64-unknown", "generic", "">,
  llvm = "hash key for LLVM IR for @symbol",
  ir = "hash key for @precompiled"
}

// -----

// CHECK-LABEL: @buffer
// CHECK-SAME: !pop.struct
kgen.precompiled.object @buffer(%a: !zap.buffer<5, f32>) attributes {
  compiledFor = #kgen.target<"darwin-arm64-unknown", "generic", "">,
  object = "hash key for @symbol object",
  llvm = "hash key for @llvm_precompiled"
}

// -----

// CHECK-LABEL: @zap_tensor_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<!pop.scalar<f32>>
kgen.func @zap_tensor_construct(%ptr: !pop.pointer<!pop.scalar<f32>>) {
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[ONE:.*]] = index.constant 1
  // CHECK-DAG: %[[FOUR:.*]] = index.constant 4
  // CHECK-DAG: %[[ARRAY:.*]] = pop.array.create [%[[FOUR]], %[[ZERO]], %[[ZERO]], %[[ZERO]], %[[ZERO]]] : !pop.array<5, index>
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: pop.struct.construct(%[[PTR]], %[[ONE]], %[[ARRAY]], %[[DTYPE]]) : !pop.struct<!pop.pointer<!pop.scalar<f32>>, index, !pop.array<5, index>, !kgen.dtype>
  %0 = zap.tensor.construct %ptr : !zap.tensor<[4], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_tensor_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<!pop.scalar<f32>>
// CHECK-SAME: %[[SIZE:.*]]: index
kgen.func @zap_tensor_construct(%ptr: !pop.pointer<!pop.scalar<f32>>,
                                %size: index) {
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[THREE:.*]] = index.constant 3
  // CHECK-DAG: %[[FOUR:.*]] = index.constant 4
  // CHECK-DAG: %[[ARRAY:.*]] = pop.array.create [%[[SIZE]], %[[FOUR]], %[[SIZE]], %[[ZERO]], %[[ZERO]]] : !pop.array<5, index>
  // CHECK-DAG: %[[DTYPE:.*]] = kgen.param.constant: dtype = <f32>
  // CHECK: pop.struct.construct(%[[PTR]], %[[THREE]], %[[ARRAY]], %[[DTYPE]]) : !pop.struct<!pop.pointer<!pop.scalar<f32>>, index, !pop.array<5, index>, !kgen.dtype>
  %0 = zap.tensor.construct %ptr[%size, %size] : !zap.tensor<[?, 4, ?], f32>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_tensor_construct
// CHECK-SAME: %[[PTR:.*]]: !pop.pointer<!pop.scalar<invalid>>
// CHECK-SAME: %[[SIZE0:.*]]: index
// CHECK-SAME: %[[DTYPE:.*]]: !kgen.dtype
kgen.func @zap_tensor_construct(%ptr: !pop.pointer<!pop.scalar<invalid>>,
                                %size: index,
                                %dtype: !kgen.dtype) {
  // CHECK-DAG: %[[ZERO:.*]] = index.constant 0
  // CHECK-DAG: %[[FOUR:.*]] = index.constant 4
  // CHECK-DAG: %[[ARRAY:.*]] = pop.array.create [%[[SIZE]], %[[SIZE]], %[[SIZE]], %[[SIZE]], %[[ZERO]]] : !pop.array<5, index>
  // CHECK: pop.struct.construct(%[[PTR]], %[[FOUR]], %[[ARRAY]], %[[DTYPE]]) : !pop.struct<!pop.pointer<!pop.scalar<invalid>>, index, !pop.array<5, index>, !kgen.dtype>
  %0 = zap.tensor.construct %ptr[%size, %size, %size, %size] of %dtype : !zap.tensor<[?, ?, ?, ?], ?>
  kgen.return
}

// -----

// CHECK-LABEL: @zap_tensor_dim
// CHECK-SAME: %[[TENSOR0:.*]]: !pop.struct<!pop.pointer<!pop.scalar<f32>>
// CHECK-SAME: %[[TENSOR1:.*]]: !pop.struct<!pop.pointer<!pop.scalar<si32>>
// CHECK-SAME: %[[TENSOR2:.*]]: !pop.struct<!pop.pointer<!pop.scalar<invalid>>
kgen.func @zap_tensor_dim(
  %tensor0: !zap.tensor<[4, 5, 3], f32>,
  %tensor1: !zap.tensor<[?, 4, ?], si32>,
  %tensor2: !zap.tensor<[?, ?, ?], ?>) {
  // CHECK: kgen.param.constant  = <4>
  %0 = zap.tensor.dim %tensor0[0] : !zap.tensor<[4, 5, 3], f32>
  // CHECK: %[[ARRAY0:.*]] = pop.struct.get %[[TENSOR1]][2] : !pop.struct<
  // CHECK: pop.array.get %[[ARRAY0]][0] : !pop.array<5, index>
  %1 = zap.tensor.dim %tensor1[0] : !zap.tensor<[?, 4, ?], si32>
  // CHECK: kgen.param.constant  = <4>
  %2 = zap.tensor.dim %tensor1[1] : !zap.tensor<[?, 4, ?], si32>
  // CHECK: %[[ARRAY1:.*]] = pop.struct.get %[[TENSOR2]][2] : !pop.struct<
  // CHECK: pop.array.get %[[ARRAY1]][0] : !pop.array<5, index>
  %3 = zap.tensor.dim %tensor2[0] : !zap.tensor<[?, ?, ?], ?>
  // CHECK: %[[ARRAY2:.*]] = pop.struct.get %[[TENSOR2]][2] : !pop.struct<
  // CHECK: pop.array.get %[[ARRAY2]][2] : !pop.array<5, index>
  %4 = zap.tensor.dim %tensor2[2] : !zap.tensor<[?, ?, ?], ?>
  kgen.return
}
