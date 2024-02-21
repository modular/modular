// RUN: kgen-opt -canonicalize -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// CHECK-LABEL: @int_literal_to_float_literal
kgen.func @int_literal_to_float_literal() -> !kgen.float_literal {
  %fl = kgen.param.constant: !kgen.int_literal = <#kgen.int_literal<5>>
  // CHECK: #kgen.float_literal<normal (5|1)>
  %il = kgen.int_literal.to_float_literal %fl
  kgen.return %il : !kgen.float_literal
}
