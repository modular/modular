// RUN: kgen-opt -canonicalize -mlir-print-debuginfo %s | FileCheck %s

// CHECK-LABEL: @int_literal_to_float_literal
kgen.func @int_literal_to_float_literal() -> !pop.float_literal {
  %il = kgen.param.constant: !pop.float_literal = <#pop<int_to_float_literal<5>>>
  // CHECK: #pop.float_literal<5|1>
  kgen.return %il : !pop.float_literal
}


// CHECK-LABEL: @float_literal_isa
kgen.func @float_literal_isa() -> (
  i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
) {
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK: kgen.return

  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_isa<neg_zero #pop.float_literal<0|1>>>>

  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_isa<neg_zero #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<neg_zero>>>>

  // CHECK-SAME: [[TRUE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_isa<inf #pop.float_literal<inf>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<inf>>>>

  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.param.constant: i1 = <#pop<float_literal_isa<neg_inf #pop.float_literal<neg_inf>>>>
  // CHECK-SAME: [[FALSE]]
  %b8 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<neg_inf>>>>

  // CHECK-SAME: [[TRUE]]
  %b9 = kgen.param.constant: i1 = <#pop<float_literal_isa<nan #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b10 = kgen.param.constant: i1 = <#pop<float_literal_isa<normal #pop.float_literal<nan>>>>

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
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<5|3>, #pop.float_literal<8|3>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_cmp_normal_same
kgen.func @float_literal_cmp_normal_same() -> (i1, i1, i1, i1, i1, i1) {
  // CHECK-DAG: [[TRUE:%.*]] = kgen.param.constant: i1 = <1>
  // CHECK-DAG: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
  // CHECK: kgen.return

  // Test normal cases with the same normal number
  // CHECK-SAME: [[TRUE]]
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<5|3>, #pop.float_literal<5|3>>>>

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
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<neg_zero>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<neg_zero>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<0|1>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b7 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<0|1>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b8 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[TRUE]]
  %b9 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<-5|3>, #pop.float_literal<neg_zero>>>>

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
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<inf>, #pop.float_literal<inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<neg_inf>, #pop.float_literal<neg_inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<neg_inf>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[TRUE]]
  %b4= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[FALSE]]
  %b5= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<5|3>, #pop.float_literal<inf>>>>
  // CHECK-SAME: [[TRUE]]
  %b6= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<0|1>>>>
  // CHECK-SAME: [[TRUE]]
  %b7= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<neg_zero>>>>
  // CHECK-SAME: [[FALSE]]
  %b8= kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<inf>, #pop.float_literal<nan>>>>

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
  %b1 = kgen.param.constant: i1 = <#pop<float_literal_cmp<eq #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[TRUE]]
  %b2 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ne #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b3 = kgen.param.constant: i1 = <#pop<float_literal_cmp<lt #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b4 = kgen.param.constant: i1 = <#pop<float_literal_cmp<le #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b5 = kgen.param.constant: i1 = <#pop<float_literal_cmp<gt #pop.float_literal<nan>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[FALSE]]
  %b6 = kgen.param.constant: i1 = <#pop<float_literal_cmp<ge #pop.float_literal<nan>, #pop.float_literal<nan>>>>

  kgen.return %b1, %b2, %b3, %b4, %b5, %b6: i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @float_literal_binop_nan
kgen.func @float_literal_binop_nan() -> (
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal
  ) {
  // CHECK: [[NAN:%.*]] = kgen.param.constant: !pop.float_literal = <#pop.float_literal<nan>>
  // CHECK: kgen.return

  // Nan always results in Nan
  // CHECK-SAME: [[NAN]]
  %r1 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r2 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<sub #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r3 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<mul #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r4 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<truediv #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r5 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<nan>, #pop.float_literal<5|3>>>>
  // CHECK-SAME: [[NAN]]
  %r6 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r7 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<sub #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r8 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<mul #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r9 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<truediv #pop.float_literal<5|3>, #pop.float_literal<nan>>>>
  // CHECK-SAME: [[NAN]]
  %r10 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<5|3>, #pop.float_literal<nan>>>>

  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10
  :
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal, !pop.float_literal,
  !pop.float_literal
}

// CHECK-LABEL: @float_literal_binop_uniques
kgen.func @float_literal_binop_uniques() ->
  (!pop.float_literal, !pop.float_literal,
  !pop.float_literal, !pop.float_literal) {

  // CHECK: <0|1>
  %r1 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<add #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>
  // CHECK: <-25|9>
  %r2 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<mul #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>
  // CHECK: <-1|1>
  %r3 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<truediv #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>
  // CHECK: <10|3>
  %r4 = kgen.param.constant: !pop.float_literal = <#pop<float_literal_bin<sub #pop.float_literal<5|3>, #pop.float_literal<-5|3>>>>

  kgen.return %r1, %r2, %r3, %r4 :
    !pop.float_literal, !pop.float_literal, !pop.float_literal,
    !pop.float_literal
}

// CHECK-LABEL: @float_literal_convert
kgen.func @float_literal_convert()
  -> (!pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>) {
  // CHECK: kgen.param.constant: scalar<f64> = <"1.666666{{.*}}">
  %r1 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<5|3>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"-1.666666{{.*}}">
  %r2 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<-5|3>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"0">
  %r3 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<0|1>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"-0">
  %r4 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<neg_zero>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"+Inf">
  %r5 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<inf>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"-Inf">
  %r6 = kgen.param.constant: scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<neg_inf>>>>
  // CHECK: kgen.param.constant: scalar<f64> = <"NaN">
  %r7 = kgen.param.constant: !pop.scalar<f64> = <#pop<float_literal_convert<#pop.float_literal<nan>>>>

  kgen.return %r1, %r2, %r3, %r4, %r5, %r6, %r7
    : !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>, !pop.scalar<f64>
}

// CHECK-LABEL: @float_literal_to_int_literal
kgen.func @float_literal_to_int_literal() ->
  (!pop.int_literal, !pop.int_literal, !pop.int_literal, !pop.int_literal,
   !pop.int_literal) {
  // CHECK: kgen.param.constant: !pop.int_literal = <1>
  %r1 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<5|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <2>
  %r2 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<8|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <-1>
  %r3 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<-5|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <-2>
  %r4 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<-8|3>>>>
  // CHECK: kgen.param.constant: !pop.int_literal = <0>
  %r5 = kgen.param.constant: !pop.int_literal = <#pop<float_to_int_literal<#pop.float_literal<neg_zero>>>>

  kgen.return %r1, %r2, %r3, %r4, %r5 : !pop.int_literal, !pop.int_literal,
    !pop.int_literal, !pop.int_literal, !pop.int_literal
}
