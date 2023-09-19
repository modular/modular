// RUN: kgen-opt %s -verify-parameters | FileCheck %s

// CHECK: lit.func @positional_args(%a: index, %b: index) numPosArgs(1)
lit.func @positional_args(%a: index, %b: index) numPosArgs(1) {
  // CHECK: self: (index, "b": index) -> () = <@positional_args>
  kgen.param.declare self: (index, "b": index) -> () = <@positional_args>
  kgen.return
}
