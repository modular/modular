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

// -----

kgen.func @capturing_fn(%arg0: index) capturing -> index {
  hlcf.return %arg0 : index
}

kgen.func @capturing_fn_reference() {
  // expected-error @below {{capturing closures cannot be materialized at runtime}}
  %0 = kgen.param.constant : (index) capturing -> index = <@"capturing_fn">
  hlcf.return
}

// -----

kgen.func @capturing_fn(%arg0: index) capturing -> index {
  hlcf.return %arg0 : index
}

kgen.func @create_closure(%arg0: index) {
  // expected-error @below {{capturing closures cannot be materialized at runtime}}
  %0 = kgen.create_closure[(index) capturing -> index: @capturing_fn](%arg0)
  hlcf.return
}

