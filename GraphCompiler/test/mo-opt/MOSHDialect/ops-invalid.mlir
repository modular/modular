// RUN: mo-opt %s -verify-diagnostics

// expected-note @below {{prior use here}}
kgen.func @param_from_value(%sh0: !mosh.ape) {
  // expected-error @below {{use of value '%sh0' expects different type than prior uses: 'index' vs '!mosh.ape'}}
  mosh.param.from_value SH : index = %sh0
  hlcf.return
}
