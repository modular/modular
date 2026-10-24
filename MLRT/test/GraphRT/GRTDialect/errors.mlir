// RUN: graphrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

// CHECK-LABEL: --- Running 'trivial':
mef.callable @trivial() {
  // expected-error @+1 {{'grt.invoke' op '@foo' does not reference a valid invokable}}
  grt.invoke @foo() : () -> ()
  mef.output
}

// -----

// CHECK-LABEL: --- Running 'controlflow_if_invalid1':
mef.callable @controlflow_if_invalid1(%cond: i1, %x: si32) {
  // expected-error @+1 {{custom op 'grt.if' has duplicate SSA values in its operand list which have not been renamed apart by 'as' clauses.}}
  %res = grt.if %cond (%x: si32, %x: si32) -> si32 {
    mef.output %x : si32
  } else {
    mef.output %x : si32
  }
  mef.output %res : si32
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid1':
mef.callable @controlflow_while_invalid1() {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 4
  // expected-error @+1 {{'grt.while' op expected 'cond' block to return additional i1 result for loop condition}}
  %res:2 = grt.while (%i: si32, %n: si32) {
    %bogus = grt.constant.si32 42
    mef.output %i, %n, %bogus: si32, si32, si32
  } do {
    mef.output %i, %n: si32, si32
  }
  mef.output %res#0, %res#1: si32, si32
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid2':
mef.callable @controlflow_while_invalid2() {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 4
  // expected-error @+1 {{'grt.while' op expected 2 mef.output operands in 'body' block entries, but given 1}}
  %res:2 = grt.while (%i: si32, %n: si32) {
    %cond = grt.lessequal.si32 %i, %n
    mef.output %i, %n, %cond: si32, si32, i1
  } do {
    mef.output %i: si32
  }
  mef.output %res#0, %res#1: si32, si32
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid3':
mef.callable @controlflow_while_invalid3() {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 4
  // expected-error @+1 {{'grt.while' op actual mef.output operands in 'body' block #1 has type 'i1' but expected 'si32'}}
  %res:2 = grt.while (%i: si32, %n: si32) {
    %cond = grt.lessequal.si32 %i, %n
    mef.output %i, %n, %cond: si32, si32, i1
  } do {
    %bogus = grt.lessequal.si32 %i, %n
    mef.output %i, %bogus: si32, i1
  }
  mef.output %res#0, %res#1: si32, si32
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid4':
mef.callable @controlflow_while_invalid4() {
  %i = grt.constant.si32 0
  %bogus = grt.constant.si32 42
  %res = grt.while (%i: si32) {
    %n = grt.constant.si32 4
    %cond = grt.lessequal.si32 %i, %n
    mef.output %i, %cond: si32, i1
  } do {
    // expected-error @+1 {{use of undeclared SSA value name}}
    %new_i = grt.add.si32 %i, %bogus
    mef.output %new_i: si32
  }
  mef.output %res: si32
}
