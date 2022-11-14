// RUN: kgen-opt %s -elaborate-generators="search-path=%S enable-search=false" | FileCheck %s

/// This evaluator returns a constant index 0.
kgen.generator @first<FN:type>(%funcs: !pop.pointer<FN>, %size: index) -> index {
  %0 = kgen.param.constant = <0>
  kgen.return %0 : index
}

/// Always pick the second implementation of this interface (because of defaultImpl).
kgen.generator.interface @pickFirst()
  evaluator (!pop.pointer<() -> ()>, index) -> index = @first<FN:type=()->()>
  defaultImpl () -> () = @pickFirstB

kgen.generator @pickFirstA() implements @pickFirst {
  kgen.return
}

kgen.generator @pickFirstB() implements @pickFirst {
  kgen.return
}

// CHECK-LABEL: @test
kgen.generator @test() {
  // CHECK-NEXT: kgen.call @pickFirstB
  kgen.call @pickFirst() : () -> ()
  kgen.return
}
