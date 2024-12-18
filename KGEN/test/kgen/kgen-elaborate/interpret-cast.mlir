// RUN: kgen-opt %s -elaborate-generators -o - | FileCheck %s

// COM: Compilation should succeed.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64">} {
  kgen.generator @cast_index_to_si64_large() -> !pop.scalar<si64> {
    %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si64>
    kgen.return %1 : !pop.scalar<si64>
  }

  kgen.generator @cast_index_to_si32_large() -> !pop.scalar<si32> {
    %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
    %1 = pop.cast %0 : !pop.scalar<index> to !pop.scalar<si32>
    kgen.return %1 : !pop.scalar<si32>
  }
  // CHECK-LABEL: @main
  kgen.generator export @main() {
    kgen.param.apply x = [() -> !pop.scalar<si64> : @cast_index_to_si64_large]()
    kgen.param.apply y = [() -> !pop.scalar<si32> : @cast_index_to_si32_large]()
    // CHECK-NEXT: kgen.param.constant: scalar<si64> = <-8664705627211539068>
    %0 = kgen.param.constant: !pop.scalar<si64> = <x>
    // CHECK-NEXT:kgen.param.constant: scalar<si32> = <-1095082620>
    %1 = kgen.param.constant: !pop.scalar<si32> = <y>
    kgen.return
  }
}
