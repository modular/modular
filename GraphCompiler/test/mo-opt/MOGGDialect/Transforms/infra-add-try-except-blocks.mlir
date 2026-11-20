// RUN: mo-opt %s --mogg-import-packages -mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-pipeline -mogg-to-mgp -canonicalize -mogg-outline-async-routines -mogg-add-try-except-blocks 2>&1 | FileCheck %s

// Verify that the three ops introduced by the MOGG ABI lowering are present
// after mogg-outline-async-routines and mogg-add-try-except-blocks:
//
//   mogg.with_trace  -- created by mogg-outline-async-routines (createKernelStub)
//                       to wrap the kernel terminator in a Trace context
//   mogg.try / mogg.raise -- created by mogg-add-try-except-blocks for MOGG
//                            ABI stubs; the except block re-raises via mogg.raise

// The MGP ABI stub (@region_0) wraps the call to the MOGG stub in a
// mogg.try/except that returns an async error value instead of raising.
// CHECK-LABEL: mogg.func @{{__[A-Za-z0-9_]+_async}} private
// CHECK:         mogg.try {
// CHECK-NOT:       mogg.raise
// CHECK:         } except(%{{[^)]+}}) {

// The MOGG ABI stub (@relu_mogg) has mogg.with_trace around the kernel
// terminator and mogg.try/mogg.raise for error propagation.
// CHECK-LABEL: mogg.func @__relu private
// CHECK:         mogg.try {
// CHECK:           mogg.with_trace
// CHECK:             mogg.call.foreach
// CHECK:           }
// CHECK:           mogg.yield
// CHECK:         } except(%{{[^)]+}}) {
// CHECK:           mogg.raise

mo.graph @relu(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %result = mo.relu(%x) : !mo.tensor<[4], f32>
  mo.output %result : !mo.tensor<[4], f32>
}
