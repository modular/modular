// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%mts_invalid_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s

// `execute` takes tensor trait arguments. The verifier rejects each
// `ManagedTensorSlice` argument, scalar or variadic, at its Mojo location.

// CHECK-DAG: mts_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: execute functions take tensor trait arguments; argument at position 0 is a 'ManagedTensorSlice'
// CHECK-DAG: mts_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: execute functions take tensor trait arguments; argument at position 1 is a 'ManagedTensorSlice'
// CHECK-DAG: mts_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: execute functions take tensor trait arguments; argument at position 1 is a 'ManagedTensorSlice'
// CHECK-DAG: mts_invalid_kernels.mojo:{{[0-9]+}}:{{[0-9]+}}: error: execute functions take tensor trait arguments; argument at position 1 is a 'VariadicTensors'; use a trait bound such as 'Some[Input & Tensor]', or 'TensorTuple' for variadic operands

module {}
