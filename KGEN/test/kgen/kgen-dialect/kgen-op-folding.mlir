// RUN: kgen-opt -canonicalize -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// CHECK-LABEL: @int_literal_to_float_literal
kgen.func @int_literal_to_float_literal() -> !kgen.float_literal {
  %fl = kgen.param.constant: !kgen.int_literal = <#kgen.int_literal<5>>
  // CHECK: #kgen.float_literal<normal (5|1)>
  %il = kgen.int_literal.to_float_literal %fl
  kgen.return %il : !kgen.float_literal
}

// -----

// CHECK-LABEL: @float_literal_cmp_normal_diff
kgen.func @float_literal_cmp_normal_diff() -> (i1, i1, i1, i1, i1, i1) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (5|3)>>
  %fb = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (8|3)>>
  // CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // Test normal cases with different normal numbers
  // CHECK-SAME: [[FALSE]]
  %b1 = kgen.float_literal.cmp eq(%fa, %fb)
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.float_literal.cmp ne(%fa, %fb)
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.float_literal.cmp lt(%fa, %fb)
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.float_literal.cmp le(%fa, %fb)
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.float_literal.cmp gt(%fa, %fb)
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.float_literal.cmp ge(%fa, %fb)

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6 : i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: @float_literal_cmp_normal_same
kgen.func @float_literal_cmp_normal_same() -> (i1, i1, i1, i1, i1, i1) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (5|3)>>
  // CHECK: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: kgen.return

  // Test normal cases with the same normal number
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.float_literal.cmp eq(%fa, %fa)
  // CHECK-SAME: [[FALSE]]
  %b2 = kgen.float_literal.cmp ne(%fa, %fa)
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.float_literal.cmp lt(%fa, %fa)
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.float_literal.cmp le(%fa, %fa)
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.float_literal.cmp gt(%fa, %fa)
  // CHECK-SAME: [[TRUE]]
  %b6 = kgen.float_literal.cmp ge(%fa, %fa)

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6: i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: @float_literal_cmp_neg_zero
kgen.func @float_literal_cmp_neg_zero() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (5|3)>>
  %fna = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (-5|3)>>
  %f0 = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (0|1)>>
  %nz = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_zero>>

  // CHECK: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: kgen.return

  // Test negative zero cases
  // Note that in Python, -0 = 0
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.float_literal.cmp eq(%nz, %f0)
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.float_literal.cmp eq(%nz, %nz)
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.float_literal.cmp ne(%nz, %f0)
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.float_literal.cmp lt(%nz, %f0)
  // CHECK-SAME: [[TRUE]]
  %b5 = kgen.float_literal.cmp le(%nz, %f0)
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.float_literal.cmp gt(%f0, %nz)
  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.float_literal.cmp ge(%f0, %nz)
  // CHECK-SAME: [[TRUE]]
  %b8 = kgen.float_literal.cmp gt(%fa, %nz)
  // CHECK-SAME: [[TRUE]]
  %b9 = kgen.float_literal.cmp lt(%fna, %nz)

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8, %b9:
    i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: @float_literal_cmp_inf
kgen.func @float_literal_cmp_inf() -> (i1, i1, i1, i1, i1, i1, i1, i1) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (5|3)>>
  %f0 = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<normal (0|1)>>
  %nz = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_zero>>
  %inf = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<inf>>
  %ninf = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_inf>>
  %nan = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<nan>>

  // CHECK: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>

  // CHECK: kgen.return
  // Some infinity cases
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.float_literal.cmp eq(%inf, %inf)
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.float_literal.cmp eq(%ninf, %ninf)
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.float_literal.cmp lt(%ninf, %fa)
  // CHECK-SAME: [[TRUE]]
  %b4= kgen.float_literal.cmp gt(%inf, %fa)
  // CHECK-SAME: [[FALSE]]
  %b5= kgen.float_literal.cmp gt(%fa, %inf)
  // CHECK-SAME: [[TRUE]]
  %b6= kgen.float_literal.cmp gt(%inf, %f0)
  // CHECK-SAME: [[TRUE]]
  %b7= kgen.float_literal.cmp gt(%inf, %nz)
  // CHECK-SAME: [[FALSE]]
  %b8= kgen.float_literal.cmp gt(%inf, %nan)

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8:
    i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: @float_literal_cmp_nan
kgen.func @float_literal_cmp_nan() -> (i1, i1, i1, i1, i1, i1) {
  %nan = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<nan>>
  // CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // Some NAN cases
  // CHECK-SAME: [[FALSE]]
  %b1 = kgen.float_literal.cmp eq(%nan, %nan)
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.float_literal.cmp ne(%nan, %nan)
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.float_literal.cmp lt(%nan, %nan)
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.float_literal.cmp le(%nan, %nan)
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.float_literal.cmp gt(%nan, %nan)
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.float_literal.cmp ge(%nan, %nan)

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6: i1, i1, i1, i1, i1, i1
}
