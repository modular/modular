// RUN: kgen-opt %s -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library-test.mlir"

kgen.generator.interface @param_call<fn: signature<() -> ()>>()

// CHECK-LABEL: kgen.func @"simple_param_call,fn=@body"
kgen.generator @body() {
  kgen.return
}

kgen.generator public @test_param_call() {
  // CHECK: @"simple_param_call,fn=@body"
  kgen.call @param_call<fn: signature<() -> ()> = @body>() : () -> ()
  kgen.return
}
