// RUN: kgen %s -elaborate -S -o - -D bar -D foo | FileCheck %s
// RUN: kgen %s -elaborate -S -o - -D bar | FileCheck %s --check-prefix=UNDEF

kgen.export @main

kgen.generator @main() -> i1 {
  // CHECK: constant: i1 = <1>
  // UNDEF: constant: i1 = <0>
  %0 = kgen.param.constant: i1 = <get_env("foo")>
  kgen.return %0 : i1
}
