// RUN: support-dialect-opt -allow-unregistered-dialect %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: !M.array<32xf32>
"M"() {m = !M.array<32xf32>} : () -> ()

// CHECK: !M.array<256xf64>
"M"() {m = !M.array<256xf64>} : () -> ()
