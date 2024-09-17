// RUN: kgen -march armv8.2-a %s -elaborate -mcpu=neoverse-n1 -S -o - | FileCheck %s

// CHECK: triple = "aarch64-{{.*}}", arch = "neoverse-n1", features = "+aes,+crc,+dotprod,+fp-armv8,+fullfp16,+lse,+neon,+perfmon,+ras,+rcpc,+rdm,+sha2,+spe,+ssbs", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32",
kgen.generator export @main() {
  kgen.return
}
