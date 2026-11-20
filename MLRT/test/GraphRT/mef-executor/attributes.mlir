// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef | FileCheck %s

// CHECK-LABEL: --- Running 'multiline_string'
// CHECK-NEXT: line1
// CHECK-NEXT: line2
// CHECK-NEXT: line3
// CHECK-NEXT: 'multiline_string' returned chain
grt.testcase @multiline_string {
  %ch0 = grt.chain.create ()
  %ch1 = "test.print_string_attribute"(%ch0)
         { string=#M<multiline["line1", "line2", "line3"]> }
         : (!mef.chain) -> !mef.chain
  mef.output %ch1 : !mef.chain
}
