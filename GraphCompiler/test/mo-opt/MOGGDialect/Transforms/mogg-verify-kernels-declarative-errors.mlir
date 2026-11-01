// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%declarative_invalid_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s --implicit-check-not="entry point named execute"

// Each kernel in the declarative_invalid_stubs package breaks one rule for a
// declarative kernel, and its error names it. A kernel has either `execute` or
// `execute_device`.
// `execute_device` and `execute_configuration` come as a pair over the same
// operands, trait-bound tensors and scalars that match in kind, and
// `execute_device`, which runs on the device, neither raises nor returns. The
// verifier reports every violation rather than stopping at the first.

// CHECK-DAG: error: kernel 'declarative_execute_configuration_only': a declarative kernel needs both 'execute_device' and 'execute_configuration'
// CHECK-DAG: error: kernel 'declarative_execute_configuration_arity': 'execute_configuration' takes 1 argument(s) but 'execute_device' takes 2; both take the kernel's operands
// CHECK-DAG: error: kernel 'declarative_execute_device_raises': 'execute_device' runs on the device and must not raise
// CHECK-DAG: error: kernel 'declarative_execute_device_returns': 'execute_device' must not return a value
// CHECK-DAG: error: kernel 'declarative_legacy_tensor': 'execute_device' operand 0 must be a trait-bound tensor (not fused or a mutable input) or a scalar
// CHECK-DAG: error: kernel 'declarative_fused_operand': 'execute_device' operand 0 must be a trait-bound tensor (not fused or a mutable input) or a scalar
// CHECK-DAG: error: kernel 'declarative_operand_kinds': operand 1 is a scalar in 'execute_device' but a tensor in 'execute_configuration'
// CHECK-DAG: error: kernel 'declarative_execute_and_execute_device': a kernel has either 'execute' or 'execute_device'

module {}
