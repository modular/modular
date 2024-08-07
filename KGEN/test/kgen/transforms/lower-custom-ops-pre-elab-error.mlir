// RUN: kgen-opt %s -lower-custom-ops-pre-elab -verify-diagnostics -split-input-file

module {
  kgen.generator @my_impl(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    kgen.return %arg0 : !pop.scalar<si32>
  }
  kgen.generator @main(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    // expected-error @below {{no 'kgen.custom.op_impls' op found at the top-level module}}
    %res = "custom.arith.neg"(%arg0) : (!pop.scalar<si32>) -> !pop.scalar<si32>
    kgen.return %res : !pop.scalar<si32>
  }
}

// -----

module {
  kgen.custom.op_impls @__CustomOpImplSymbol []
  kgen.generator @my_impl(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    kgen.return %arg0 : !pop.scalar<si32>
  }
  kgen.generator @main(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    // expected-error @below {{no implementation found for custom op 'custom.arith.neg'}}
    %res = "custom.arith.neg"(%arg0) : (!pop.scalar<si32>) -> !pop.scalar<si32>
    kgen.return %res : !pop.scalar<si32>
  }
}

// -----

module {
  kgen.custom.op_impls @__CustomOpImplSymbol [<"custom.a", impl: :!kgen.signature<(!pop.scalar<si32>) -> !pop.scalar<si32>> @my_impl>,
                                              <"custom.c", impl: :!kgen.signature<(!pop.scalar<si32>) -> !pop.scalar<si32>> @my_impl>]
  kgen.generator @my_impl(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    kgen.return %arg0 : !pop.scalar<si32>
  }
  kgen.generator @main(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
    // expected-error @below {{no implementation found for custom op 'custom.b'}}
    %res = "custom.b"(%arg0) : (!pop.scalar<si32>) -> !pop.scalar<si32>
    kgen.return %res : !pop.scalar<si32>
  }
}
