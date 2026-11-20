// The built-in kernels alone must never make kernel selection ambiguous, on
// any device. User packages can only shadow built-ins, so this check holds
// whatever a user loads.
// RUN: mo-opt %s --mlir-print-op-on-diagnostic=false --mogg-import-packages --mogg-validate-kernel-registry="builtins-only=true" -o /dev/null

module {
}
