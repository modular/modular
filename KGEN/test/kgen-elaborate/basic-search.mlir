// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true" -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: kgen.func public @"even_only,param=72"()
// CHECK-NOT: @"even_only,
// CHECK-LABEL: kgen.func public @"even_only,param=16"() {
// CHECK-NOT: @"even_only,
kgen.generator public @even_only<param>() {
  kgen.param.assert <eq(and(param, 1), 0)>, "the param shalt be even!"
  kgen.return
}

// This should turn into two variants exactly, not a duplicate for 72.
// CHECK-LABEL: kgen.func public @find_even
// CHECK-LABEL: kgen.func public @find_even
// CHECK-NOT: kgen.func public @find_even
kgen.generator public @find_even() {
  kgen.param.search seventy_two = <72>
  kgen.param.search value = <3, 16, 1, 72, seventy_two>
  kgen.call @even_only<param=value>() : () -> ()
  kgen.return
}

// -----

/// This evaluator returns a constant index.
kgen.generator public @simpleEvaluator<N, FN:type>(%funcs: !pop.pointer<FN>, %size: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

/// Always pick the first implementation of this interface.
kgen.generator.interface @pickFirst()
  evaluator (!pop.pointer<() -> ()>, index) -> index = @simpleEvaluator<N=0, FN:type=()->()>

/// Always pick the second implementation of this interface.
kgen.generator.interface @pickSecond()
  evaluator (!pop.pointer<() -> ()>, index) -> index = @simpleEvaluator<N=1, FN:type=()->()>

kgen.generator @pickFirstA() implements @pickFirst {
  kgen.return
}

kgen.generator @pickFirstB() implements @pickFirst {
  kgen.return
}

kgen.generator @pickSecondA() implements @pickSecond {
  kgen.return
}

kgen.generator @pickSecondB() implements @pickSecond {
  kgen.return
}

// CHECK-LABEL: @test
kgen.generator public @test() {
  // CHECK-NEXT: kgen.call @pickFirstA
  kgen.call @pickFirst() : () -> ()
  // CHECK-NEXT: kgen.call @pickSecondB
  kgen.call @pickSecond() : () -> ()
  kgen.return
}
