// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%workspace_pkg" --mogg-verify-kernels | FileCheck %s
// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%workspace_invalid_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s --check-prefix=INVALID

// A kernel requests a workspace buffer with a `Workspace` argument on
// `execute` or `execute_device`, and sizes it with `workspace_size`, which the
// decl records.

// CHECK-DAG: mogg.kernel.decl @{{.*}}WorkspaceExecute registration = "workspace_execute" execute = {{.*}} workspace_size =
// CHECK-DAG: mogg.kernel.decl @{{.*}}WorkspaceExecuteDevice registration = "workspace_execute_device" {{.*}} workspace_size =
// CHECK-DAG: mogg.kernel.decl @{{.*}}WorkspaceVariadic registration = "workspace_variadic" execute = {{.*}} workspace_size =

// Each kernel in the workspace_invalid_stubs package breaks one rule. The
// `Workspace` argument and `workspace_size` come as a pair. There is one
// `Workspace` argument, after the operands and before the context, which stays
// last. A kernel that takes one has no `build_graph`. `workspace_size` returns
// an `Int` and takes each tensor input of the entry point by `Tensor`. The verifier reports
// every violation rather than stopping at the first.

// INVALID-DAG: error: kernel 'workspace_without_workspace_size': the execute function takes a Workspace argument, but the kernel declares no workspace_size function to size it
// INVALID-DAG: error: kernel 'workspace_size_without_workspace': the kernel declares workspace_size, but its execute function takes no Workspace argument
// INVALID-DAG: error: kernel 'workspace_twice': there can be at most one Workspace argument
// INVALID-DAG: error: kernel 'workspace_before_operand': the Workspace argument must come after every operand, but it comes before 'a'
// INVALID-DAG: error: Context argument must be the last argument in the function signature
// INVALID-DAG: error: kernel 'workspace_with_build_graph': Workspace arguments are not supported on a kernel with a build_graph function
// INVALID-DAG: error: kernel 'workspace_size_returns_float': workspace_size must return the size of the workspace buffer in bytes, as an Int
// INVALID-DAG: error: kernel 'workspace_size_arity': workspace_size takes 2 argument(s) but execute has 1 tensor input(s); workspace_size takes one argument per tensor input
// INVALID-DAG: error: kernel 'workspace_size_reads_data': workspace_size argument at position 0 must be bound by Tensor alone, as in 'Some[Tensor]'; the workspace size depends on the shapes of the inputs and not on their data

module {}
