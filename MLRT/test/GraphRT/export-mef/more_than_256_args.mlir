// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef | FileCheck %s

// This test is a little ugly because of we need to check for more than 256
// arguments to a function type. This function has 257 arguments. On failure
// we get a "wrapping" behavior, where MEF thinks it has only one argument. So
// if the output looks like this:
//
// Type attribute: (si32) -> si32
//
// that indicates the failure case.

grt.testcase @test_function_type_with_257_args{
  %ch0 = grt.chain.create ()
  // Look for 257 input types listed for this function.
  // CHECK: Type attribute:
  // CHECK:(
  // CHECK-COUNT-257: si32
  // CHECK:)
  %ch1 = "test.print_type"(%ch0) {t = (
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32) -> (si32) } : (!mef.chain) -> (!mef.chain)
  mef.output
}

