// RUN: kgen-opt -ensure-no-parameters -verify-diagnostics -split-input-file -allow-unregistered-dialect %s

kgen.func @"identity,T=index"(%arg0: index) -> index {
  hlcf.return %arg0 : index
}

kgen.func @legal_func() {
  %0 = kgen.param.constant : (index) -> index = <@"identity,T=index">
  hlcf.return
}

kgen.func @parameterized_signature() {
  // expected-error @below {{parameterized functions cannot be used at runtime}}
  %0 = kgen.param.constant : <type>(!kgen.param<*(0,0)>) -> !kgen.param<*(0,0)> = <@"identity">
  hlcf.return
}

