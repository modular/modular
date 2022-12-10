// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true" -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: kgen.func @"even_only,param=72"()
// CHECK-NOT: @"even_only,
// CHECK-LABEL: kgen.func @"even_only,param=16"() {
// CHECK-NOT: @"even_only,
kgen.generator @even_only<param>() {
  kgen.param.assert <eq(and(param, 1), 0)>, "the param shalt be even!"
  kgen.return
}

// This should turn into two variants exactly, not a duplicate for 72.
// CHECK-LABEL: kgen.func @find_even
// CHECK-LABEL: kgen.func @find_even
// CHECK-NOT: kgen.func @find_even
kgen.generator @find_even() {
  kgen.param.search seventy_two = <72>
  kgen.param.search value = <3, 16, 1, 72, seventy_two>
  kgen.call @even_only<param=value>() : () -> ()
  kgen.return
}

// -----

/// This evaluator returns a constant index.
kgen.generator @simpleEvaluator<N, FN:type>(%funcs: !pop.pointer<FN>, %size: index) -> index {
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
kgen.generator @test() {
  // CHECK-NEXT: kgen.call @pickFirstA
  kgen.call @pickFirst() : () -> ()
  // CHECK-NEXT: kgen.call @pickSecondB
  kgen.call @pickSecond() : () -> ()
  kgen.return
}

// -----

kgen.generator @pick<T: type, N>(%fns: !pop.pointer<(!kgen.paramref<T>) -> !kgen.paramref<T>>, %size: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

kgen.generator.interface @paramItf<T: type>(!kgen.paramref<T>) -> !kgen.paramref<T>
  evaluator (!pop.pointer<(!kgen.paramref<T>) -> !kgen.paramref<T>>, index) -> index = @pick<T: type = T, N = 1>

kgen.generator @impl1<T: type>(%val: !kgen.paramref<T>) -> !kgen.paramref<T>
    implements @paramItf {
  kgen.return %val : !kgen.paramref<T>
}

// CHECK-LABEL: kgen.func @"impl2,T=index"
kgen.generator @impl2<T: type>(%val: !kgen.paramref<T>) -> !kgen.paramref<T>
    implements @paramItf {
  kgen.return %val : !kgen.paramref<T>
}

// CHECK-LABEL: kgen.func @entry
kgen.generator @entry(%val: index) {
  // CHECK-NEXT: kgen.call @"impl2,T=index"
  %0 = kgen.call @paramItf<T: type = index>(%val) : (index) -> index
  kgen.return
}
