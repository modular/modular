// RUN: kgen-opt %s -split-input-file -inline | FileCheck %s

kgen.func @callee() force_inline -> index {
  %0 = kgen.param.constant = <13>
  kgen.return %0 : index
}

// CHECK-LABEL: @caller() -> index
// CHECK-NEXT: kgen.param.constant
// CHECK-NEXT: kgen.return
kgen.func @caller() -> index{
  %0 = kgen.call @callee() : () force_inline -> index
  kgen.return %0 : index
}

// -----

// COM: index.constant cannot be inlined.
kgen.func @callee() force_inline -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: @caller() -> index
// CHECK-NEXT: kgen.call @callee
// CHECK-NEXT: kgen.return
kgen.func @caller() -> index{
  %0 = kgen.call @callee() : () force_inline -> index
  kgen.return %0 : index
}
