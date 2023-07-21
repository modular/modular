// RUN: kgen -march armv8.2-a %s -elaborate -mcpu=neoverse-n1 -S -o - | FileCheck %s
// REQUIRES: aarch64-registered-target

// CHECK: triple = "aarch64-a-{{.*}}", cpu = "neoverse-n1", features = "+aes,+crc,+dotprod,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+spe,+ssbs", data_layout = "e-m:o-i64:64-i128:128-n32:64-S128"
kgen.generator export @main() {
  kgen.return
}
