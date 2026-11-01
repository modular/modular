// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-to-mogg --mogg-conform-call-operands | FileCheck %s

// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t MAX_GC_USE_ADV_FUSION=1 mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false -o /dev/null
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=MOJO

// `trait_tensor_tuple_add` takes `outputs: TensorTuple[*Outs]`, `bias`, and
// `inputs: TensorTuple[*Ins]`. ConformCallOperands wraps every tuple element in
// its own `mogg.trait.create`, as it would a single trait argument, then packs
// each run into a `mogg.tuple.create` with the `tensor_tuple` ABI.

// CHECK-LABEL: mo.graph @trait_tensor_tuple_add_graph
// CHECK: %[[BIAS:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[IN0:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[IN1:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = input, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[INS:.*]] = mogg.tuple.create(%[[IN0]], %[[IN1]]) variadic_abi = <tensor_tuple> : !mogg.tensor, !mogg.tensor
// CHECK: %[[OUT0:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = output, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[OUT1:.*]] = mogg.trait.create(%{{.*}}) contract = <{role = output, {{.*}}}>, device = {{.*}} : (!mogg.tensor) -> !mogg.tensor
// CHECK: %[[OUTS:.*]] = mogg.tuple.create(%[[OUT0]], %[[OUT1]]) variadic_abi = <tensor_tuple> : !mogg.tensor, !mogg.tensor
// CHECK: mogg.call.opaque @{{.*}}TraitTensorTupleAdd inputs(%[[BIAS]], %[[INS]] : !mogg.tensor, !mogg.tuple) outputs(%[[OUTS]] : !mogg.tuple)

// Emit builds a `TensorTuple` holding each run's trait arguments and passes
// the tuples positionally. The `TypeList` parameters are infer-only, so the
// call leaves them to Mojo.
// MOJO: var [[BIAS:[a-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.INPUT
// MOJO: var [[IN0:[a-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.INPUT
// MOJO: var [[IN1:[a-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.INPUT
// MOJO: var [[INS:[a-z0-9_]+]] = extensibility.TensorTuple[type_of([[IN0]]), type_of([[IN1]])]([[IN0]], [[IN1]])
// MOJO: var [[OUT0:[a-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.OUTPUT
// MOJO: var [[OUT1:[a-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.OUTPUT
// MOJO: var [[OUTS:[a-z0-9_]+]] = extensibility.TensorTuple[type_of([[OUT0]]), type_of([[OUT1]])]([[OUT0]], [[OUT1]])
// MOJO: TraitTensorTupleAdd.execute[target = "cpu"]([[OUTS]], [[BIAS]], [[INS]], {{[a-z0-9_]+}})
mo.graph @trait_tensor_tuple_add_graph(%bias: !mo.tensor<[1], f32>, %a: !mo.tensor<[2], si32>, %b: !mo.tensor<[3], f64>) -> (!mo.tensor<[2], f32>, !mo.tensor<[3], si32>) {
  %z0, %z1 = mo.custom [symbol = "trait_tensor_tuple_add", device = #M.device_ref<"cpu", 0>] (%bias, %a, %b) : (
    !mo.tensor<[1], f32>,
    !mo.tensor<[2], si32>,
    !mo.tensor<[3], f64>
  ) -> (!mo.tensor<[2], f32>, !mo.tensor<[3], si32>)
  mo.output %z0, %z1 : !mo.tensor<[2], f32>, !mo.tensor<[3], si32>
}
