// RUN: kgen-opt -canonicalize -mlir-print-debuginfo %s | FileCheck %s

// CHECK-LABEL: @int_literal_to_float_literal
kgen.func @int_literal_to_float_literal() -> !kgen.float_literal {
  %il = kgen.param.constant: !kgen.float_literal = <#kgen<int_to_float_literal<5>>>
  // CHECK: #kgen.float_literal<5|1>
  kgen.return %il : !kgen.float_literal
}


// CHECK-LABEL: @float_literal_isa
kgen.func @float_literal_isa() -> (
  i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
) {
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#kgen<float_literal_isa<normal #kgen.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b2 = kgen.param.constant: i1 = <#kgen<float_literal_isa<neg_zero #kgen.float_literal<0|1>>>>

  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#kgen<float_literal_isa<neg_zero #kgen.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#kgen<float_literal_isa<normal #kgen.float_literal<neg_zero>>>>

  // CHECK-SAME: [[TRUE]]
  %b5 = kgen.param.constant: i1 = <#kgen<float_literal_isa<inf #kgen.float_literal<inf>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#kgen<float_literal_isa<normal #kgen.float_literal<inf>>>>

  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.param.constant: i1 = <#kgen<float_literal_isa<neg_inf #kgen.float_literal<neg_inf>>>>
  // CHECK-SAME: [[FALSE]]
  %b8 = kgen.param.constant: i1 = <#kgen<float_literal_isa<normal #kgen.float_literal<neg_inf>>>>

  // CHECK-SAME: [[TRUE]]
  %b9 = kgen.param.constant: i1 = <#kgen<float_literal_isa<nan #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b10 = kgen.param.constant: i1 = <#kgen<float_literal_isa<normal #kgen.float_literal<nan>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8, %b9, %b10
    : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_normal_diff
kgen.func @float_literal_cmp_normal_diff() -> (i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // Test normal cases with different normal numbers
  // CHECK-SAME: [[FALSE]]
  %b1 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<eq #kgen.float_literal<5|3>, #kgen.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ne #kgen.float_literal<5|3>, #kgen.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<lt #kgen.float_literal<5|3>, #kgen.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<le #kgen.float_literal<5|3>, #kgen.float_literal<8|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<5|3>, #kgen.float_literal<8|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ge #kgen.float_literal<5|3>, #kgen.float_literal<8|3>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_normal_same
kgen.func @float_literal_cmp_normal_same() -> (i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: kgen.return

  // Test normal cases with the same normal number
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<eq #kgen.float_literal<5|3>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b2 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ne #kgen.float_literal<5|3>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<lt #kgen.float_literal<5|3>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<le #kgen.float_literal<5|3>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<5|3>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b6 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ge #kgen.float_literal<5|3>, #kgen.float_literal<5|3>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6: i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_neg_zero
kgen.func @float_literal_cmp_neg_zero() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: kgen.return

  // Test negative zero cases
  // Note that in Python, -0 = 0
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<eq #kgen.float_literal<neg_zero>, #kgen.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<eq #kgen.float_literal<neg_zero>, #kgen.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ne #kgen.float_literal<neg_zero>, #kgen.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<lt #kgen.float_literal<neg_zero>, #kgen.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b5 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<le #kgen.float_literal<neg_zero>, #kgen.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<0|1>, #kgen.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ge #kgen.float_literal<0|1>, #kgen.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b8 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<5|3>, #kgen.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b9 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<lt #kgen.float_literal<-5|3>, #kgen.float_literal<neg_zero>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8, %b9:
    i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_inf
kgen.func @float_literal_cmp_inf() -> (i1, i1, i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>

  // CHECK: kgen.return
  // Some infinity cases
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<eq #kgen.float_literal<inf>, #kgen.float_literal<inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<eq #kgen.float_literal<neg_inf>, #kgen.float_literal<neg_inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<lt #kgen.float_literal<neg_inf>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4= kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<inf>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5= kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<5|3>, #kgen.float_literal<inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b6= kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<inf>, #kgen.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b7= kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<inf>, #kgen.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b8= kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<inf>, #kgen.float_literal<nan>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6, %b7, %b8:
    i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_nan
kgen.func @float_literal_cmp_nan() -> (i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // Some NAN cases
  // CHECK-SAME: [[FALSE]]
  %b1 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<eq #kgen.float_literal<nan>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ne #kgen.float_literal<nan>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<lt #kgen.float_literal<nan>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<le #kgen.float_literal<nan>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<gt #kgen.float_literal<nan>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#kgen<float_literal_cmp<ge #kgen.float_literal<nan>, #kgen.float_literal<nan>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6: i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_binop_nan
kgen.func @float_literal_binop_nan() -> (
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal
  ) {
  // CHECK: [[NAN:%.*]] = kgen.param.constant: !kgen.float_literal = <#kgen.float_literal<nan>>
  // CHECK: kgen.return

  // Nan always results in Nan
  // CHECK-SAME: [[NAN]]
  %r1 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<add #kgen.float_literal<nan>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r2 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<sub #kgen.float_literal<nan>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r3 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<mul #kgen.float_literal<nan>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r4 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<truediv #kgen.float_literal<nan>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r5 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<add #kgen.float_literal<nan>, #kgen.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r6 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<add #kgen.float_literal<5|3>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r7 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<sub #kgen.float_literal<5|3>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r8 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<mul #kgen.float_literal<5|3>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r9 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<truediv #kgen.float_literal<5|3>, #kgen.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r10 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<add #kgen.float_literal<5|3>, #kgen.float_literal<nan>>>>

  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10
  :
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal
}

// CHECK-LABEL: @float_literal_binop_uniques
kgen.func @float_literal_binop_uniques() ->
  (!kgen.float_literal, !kgen.float_literal,
  !kgen.float_literal, !kgen.float_literal) {

  // CHECK: <0|1>
  %r1 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<add #kgen.float_literal<5|3>, #kgen.float_literal<-5|3>>>>
  // CHECK: <-25|9>
  %r2 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<mul #kgen.float_literal<5|3>, #kgen.float_literal<-5|3>>>>
  // CHECK: <-1|1>
  %r3 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<truediv #kgen.float_literal<5|3>, #kgen.float_literal<-5|3>>>>
  // CHECK: <10|3>
  %r4 = kgen.param.constant: !kgen.float_literal = <#kgen<float_literal_bin<sub #kgen.float_literal<5|3>, #kgen.float_literal<-5|3>>>>

  kgen.return %r1, %r2, %r3, %r4 :
    !kgen.float_literal, !kgen.float_literal, !kgen.float_literal,
    !kgen.float_literal
}

// CHECK-LABEL: @float_literal_convert
kgen.func @float_literal_convert()
  -> (f64, f64, f64, f64, f64, f64, f64) {
  // CHECK: kgen.param.constant: f64 = <1.666666{{.*}}>
  %r1 = kgen.param.constant: f64 = <#kgen<float_literal_convert<#kgen.float_literal<5|3>>>>
  // CHECK: kgen.param.constant: f64 = <-1.666666{{.*}}>
  %r2 = kgen.param.constant: f64 = <#kgen<float_literal_convert<#kgen.float_literal<-5|3>>>>
  // CHECK: kgen.param.constant: f64 = <0.000{{.*}}>
  %r3 = kgen.param.constant: f64 = <#kgen<float_literal_convert<#kgen.float_literal<0|1>>>>
  // CHECK: kgen.param.constant: f64 = <-0.000{{.*}}>
  %r4 = kgen.param.constant: f64 = <#kgen<float_literal_convert<#kgen.float_literal<neg_zero>>>>
  // CHECK: kgen.param.constant: f64 = <0x7FF0000000000000>
  %r5 = kgen.param.constant: f64 = <#kgen<float_literal_convert<#kgen.float_literal<inf>>>>
  // CHECK: kgen.param.constant: f64 = <0xFFF0000000000000>
  %r6 = kgen.param.constant: f64 = <#kgen<float_literal_convert<#kgen.float_literal<neg_inf>>>>
  // CHECK: kgen.param.constant: f64 = <0x7FFFFFFFFFFFFFFF>
  %r7 = kgen.param.constant: f64 = <#kgen<float_literal_convert<#kgen.float_literal<nan>>>>

  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7
    : f64, f64, f64, f64, f64, f64, f64
}

// CHECK-LABEL: @float_literal_to_int_literal
kgen.func @float_literal_to_int_literal() ->
  (!kgen.int_literal, !kgen.int_literal, !kgen.int_literal, !kgen.int_literal,
   !kgen.int_literal) {
  // CHECK: kgen.param.constant: !kgen.int_literal = <1>
  %r1 = kgen.param.constant: !kgen.int_literal = <#kgen<float_to_int_literal<#kgen.float_literal<5|3>>>>
  // CHECK: kgen.param.constant: !kgen.int_literal = <2>
  %r2 = kgen.param.constant: !kgen.int_literal = <#kgen<float_to_int_literal<#kgen.float_literal<8|3>>>>
  // CHECK: kgen.param.constant: !kgen.int_literal = <-1>
  %r3 = kgen.param.constant: !kgen.int_literal = <#kgen<float_to_int_literal<#kgen.float_literal<-5|3>>>>
  // CHECK: kgen.param.constant: !kgen.int_literal = <-2>
  %r4 = kgen.param.constant: !kgen.int_literal = <#kgen<float_to_int_literal<#kgen.float_literal<-8|3>>>>
  // CHECK: kgen.param.constant: !kgen.int_literal = <0>
  %r5 = kgen.param.constant: !kgen.int_literal = <#kgen<float_to_int_literal<#kgen.float_literal<neg_zero>>>>

  kgen.return %r1, %r2, %r3, %r4, %r5 : !kgen.int_literal, !kgen.int_literal,
    !kgen.int_literal, !kgen.int_literal, !kgen.int_literal
}
