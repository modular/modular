// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #kgen.conventions<[byref, byval], throws>
"someop"() {conventions = #kgen.conventions<[byref, byval], throws>} : () -> ()

// CHECL: #kgen.conventions<[], none>
"someop"() {conventions = #kgen.conventions<[], none>} : () -> ()
