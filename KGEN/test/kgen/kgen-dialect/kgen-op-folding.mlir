// RUN: kgen-opt -canonicalize -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// CHECK-LABEL: @int_literal_bit_width
kgen.func @int_literal_bit_width() -> (!kgen.int_literal, !kgen.int_literal) {
  %x1 = kgen.param.constant: !kgen.int_literal = <255>
  // Note that int literals are signed.
  // CHECK: kgen.int_literal = <9>
  %width1 = kgen.int_literal.bit_width %x1
  %x2 = kgen.param.constant: !kgen.int_literal = <170141183460469231731687303715884105728>
  // Note that int literals are signed.
  // CHECK: kgen.int_literal = <129>
  %width2 = kgen.int_literal.bit_width %x2
  kgen.return %width1, %width2 : !kgen.int_literal, !kgen.int_literal
}

// -----

// CHECK-LABEL: @int_literal_to_float_literal
kgen.func @int_literal_to_float_literal() -> !kgen.float_literal {
  %il = kgen.param.constant: !kgen.int_literal = <5>
  // CHECK: #kgen.float_literal<5|1>
  %fl = kgen.int_literal.to_float_literal %il
  kgen.return %fl : !kgen.float_literal
}

// -----

// CHECK-LABEL: @float_literal_cmp_normal_diff
kgen.func @float_literal_cmp_normal_diff() -> (i1, i1, i1, i1, i1, i1) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  %fb = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<8|3>>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
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
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
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
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  %fna = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<-5|3>>
  %f0 = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<0|1>>
  %nz = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_zero>>

  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
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
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  %f0 = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<0|1>>
  %nz = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_zero>>
  %inf = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<inf>>
  %ninf = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_inf>>
  %nan = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<nan>>

  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>

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
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
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

// -----

// CHECK-LABEL: @float_literal_binop_nan
kgen.func @float_literal_binop_nan() -> (
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal
  ) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  %nan = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<nan>>
  // CHECK: [[NAN:%.*]] = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<nan>>
  // CHECK: kgen.return

  // Nan always results in Nan
  // CHECK-SAME: [[NAN]]
  %r1 = kgen.float_literal.binop add(%nan, %fa)
  // CHECK-SAME: [[NAN]]
  %r2 = kgen.float_literal.binop sub(%nan, %fa)
  // CHECK-SAME: [[NAN]]
  %r3 = kgen.float_literal.binop mul(%nan, %fa)
  // CHECK-SAME: [[NAN]]
  %r4 = kgen.float_literal.binop truediv(%nan, %fa)
  // CHECK-SAME: [[NAN]]
  %r5 = kgen.float_literal.binop add(%nan, %fa)
  // CHECK-SAME: [[NAN]]
  %r6 = kgen.float_literal.binop add(%fa, %nan)
  // CHECK-SAME: [[NAN]]
  %r7 = kgen.float_literal.binop sub(%fa, %nan)
  // CHECK-SAME: [[NAN]]
  %r8 = kgen.float_literal.binop mul(%fa, %nan)
  // CHECK-SAME: [[NAN]]
  %r9 = kgen.float_literal.binop truediv(%fa, %nan)
  // CHECK-SAME: [[NAN]]
  %r10 = kgen.float_literal.binop add(%fa, %nan)

  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10
  :
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal
}

// -----

// CHECK-LABEL: @float_literal_binop_uniques
kgen.func @float_literal_binop_uniques() ->
  (!kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  %fna = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<-5|3>>

  // CHECK: <0|1>
  %r1 = kgen.float_literal.binop add(%fa, %fna)
  // CHECK: <-25|9>
  %r2 = kgen.float_literal.binop mul(%fa, %fna)
  // CHECK: <-1|1>
  %r3 = kgen.float_literal.binop truediv(%fa, %fna)
  // CHECK: <10|3>
  %r4 = kgen.float_literal.binop sub(%fa, %fna)

  kgen.return %r1, %r2, %r3, %r4 :
    !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
    !kgen.float_literal
}

// -----

// CHECK-LABEL: @float_literal_convert
kgen.func @float_literal_convert()
  -> (f64, f64, f64, f64, f64, f64, f64) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  %fna = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<-5|3>>
  %fb = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<8|3>>
  %fnb = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<-8|3>>
  %f0 = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<0|1>>
  %nz = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_zero>>
  %inf = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<inf>>
  %ninf = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_inf>>
  %nan = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<nan>>

  // CHECK: kgen.param.constant: f64 = <1.666666{{.*}}>
  %r1 = kgen.float_literal.convert %fa : to f64
  // CHECK: kgen.param.constant: f64 = <-1.666666{{.*}}>
  %r2 = kgen.float_literal.convert %fna : to f64
  // CHECK: kgen.param.constant: f64 = <0.000{{.*}}>
  %r3 = kgen.float_literal.convert %f0 : to f64
  // CHECK: kgen.param.constant: f64 = <-0.000{{.*}}>
  %r4 = kgen.float_literal.convert %nz : to f64
  // CHECK: kgen.param.constant: f64 = <0x7FF0000000000000>
  %r5 = kgen.float_literal.convert %inf : to f64
  // CHECK: kgen.param.constant: f64 = <0xFFF0000000000000>
  %r6 = kgen.float_literal.convert %ninf : to f64
  // CHECK: kgen.param.constant: f64 = <0x7FF8000000000000>
  %r7 = kgen.float_literal.convert %nan : to f64

  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7
    : f64, f64, f64, f64, f64, f64, f64
}

// -----

// CHECK-LABEL: @float_literal_to_int_literal
kgen.func @float_literal_to_int_literal() ->
  (!kgen.int_literal, !kgen.int_literal, !kgen.int_literal, !kgen.int_literal,
   !kgen.int_literal) {
  %fa = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<5|3>>
  %fna = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<-5|3>>
  %fb = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<8|3>>
  %fnb = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<-8|3>>
  %nz = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<neg_zero>>

  // CHECK: kgen.param.constant: !kgen.int_literal = <1>
  %r1 = kgen.float_literal.to_int_literal %fa
  // CHECK: kgen.param.constant: !kgen.int_literal = <2>
  %r2 = kgen.float_literal.to_int_literal %fb
  // CHECK: kgen.param.constant: !kgen.int_literal = <-1>
  %r3 = kgen.float_literal.to_int_literal %fna
  // CHECK: kgen.param.constant: !kgen.int_literal = <-2>
  %r4 = kgen.float_literal.to_int_literal %fnb
  // CHECK: kgen.param.constant: !kgen.int_literal = <0>
  %r5 = kgen.float_literal.to_int_literal %nz

  kgen.return %r1, %r2, %r3, %r4, %r5 : !kgen.int_literal, !kgen.int_literal,
    !kgen.int_literal, !kgen.int_literal, !kgen.int_literal
}
