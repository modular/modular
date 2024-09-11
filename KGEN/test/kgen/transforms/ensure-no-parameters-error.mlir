// RUN: kgen-opt -ensure-no-parameters -verify-diagnostics %s

kgen.func @"identity,T=index"(%arg0: index) -> index {
  kgen.return %arg0 : index
}

kgen.func @legal_func() {
  %0 = kgen.param.constant : (index) -> index = <@"identity,T=index">
  kgen.return
}

kgen.func @parameterized_signature() {
  // expected-error @below {{parameterized functions cannot be used at runtime}}
  %0 = kgen.param.constant : <type>(!kgen.paramref<*(0,0)>) -> !kgen.paramref<*(0,0)> = <@"identity">
  kgen.return
}
