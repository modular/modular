// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true" -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.func @"even_only,param=16"() {
// CHECK-NOT: @"even_only,
// CHECK-LABEL: kgen.func @"even_only,param=72"()
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
  kgen.param.fork seventy_two = <[72]>
  kgen.param.fork value = <[3, 16, 1, 72, seventy_two]>
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

// -----

// COM: This is an example that tests using the generator interface replacement.

/// This evaluator returns a constant index.
kgen.generator @simpleEvaluator<N, FN:type>(%funcs: !pop.pointer<FN>, %num: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

/// Always pick the first implementation of this interface.
// CHECK-LABEL: @pickFirst
kgen.generator @pickFirst() {
  kgen.param.declare evaluator: (!pop.pointer<!kgen.signature<() -> ()>>, index) -> index
    = <bind_signature(:<N, FN:type>(!pop.pointer<FN>, index) -> index @simpleEvaluator, 0, !kgen.signature<()->()>)>
  kgen.param.declare chosenImpl : () -> () = <evaluate(:() -> () @pickFirstA, @pickFirstB,
                                                       :(!pop.pointer<!kgen.signature<() -> ()>>, index) -> index evaluator)>
  // CHECK-NEXT: kgen.call @pickFirstA
  kgen.call_param[() -> (): chosenImpl]()
  kgen.return
}

/// Always pick the second implementation of this interface.
// CHECK-LABEL: @pickSecond()
kgen.generator @pickSecond() {
  kgen.param.declare evaluator: (!pop.pointer<!kgen.signature<() -> ()>>, index) -> index
    = <bind_signature(:<N, FN:type>(!pop.pointer<FN>, index) -> index @simpleEvaluator, 1, !kgen.signature<()->()>)>
  kgen.param.declare chosenImpl : () -> () = <evaluate(:() -> () @pickSecondA, @pickSecondB,
                                                       :(!pop.pointer<!kgen.signature<() -> ()>>, index) -> index evaluator)>
  // CHECK-NEXT: kgen.call @pickSecondB
  kgen.call_param[() -> (): chosenImpl]()
  kgen.return
}

kgen.generator @pickFirstA() {
  kgen.return
}

kgen.generator @pickFirstB() {
  kgen.return
}

kgen.generator @pickSecondA() {
  kgen.return
}

kgen.generator @pickSecondB() {
  kgen.return
}

// CHECK-LABEL: @test
kgen.generator @test() {
  // CHECK-NEXT: kgen.call @pickFirst()
  kgen.call @pickFirst() : () -> ()
  // CHECK-NEXT: kgen.call @pickSecond()
  kgen.call @pickSecond() : () -> ()
  kgen.return
}

// -----

/// This evaluator returns a constant index.
// CHECK-LABEL: @"simpleEvaluator,N=1,FN=!kgen.signature<() -> index>"
kgen.generator @simpleEvaluator<N, FN:type>(%funcs: !pop.pointer<FN>, %num: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

/// Always pick the second implementation of this interface.
// CHECK-LABEL: @pickSecond()
kgen.generator @pickSecond() -> index {
  kgen.param.declare evaluator: (!pop.pointer<!kgen.signature<() -> index>>, index) -> index
    = <bind_signature(:<N, FN:type>(!pop.pointer<FN>, index) -> index @simpleEvaluator, 1, !kgen.signature<()->index>)>
  kgen.param.declare chosenImpl : () -> index = <evaluate(:() -> index @pickSecondA, @pickSecondB,
                                                          :(!pop.pointer<!kgen.signature<() -> index>>, index) -> index evaluator)>
  // COM: This is actually not one of the direct options, it's an expansion of one of them.
  // CHECK-NEXT: kgen.call @pickSecondA_concrete_1()
  %0 = kgen.call_param[() -> index: chosenImpl]()
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @pickSecondA_concrete_1()
// CHECK-NEXT: kgen.param.constant = <2>

// CHECK-LABEL: kgen.func @pickSecondA()
// CHECK-NEXT: kgen.param.constant = <1>

kgen.generator @pickSecondA() -> index {
  kgen.param.fork f = <[1, 2]>
  %0 = kgen.param.constant = <f>
  kgen.return %0 : index
}

// CHECK-LABEL: @pickSecondB()
// CHECK-NEXT: kgen.param.constant = <0>
kgen.generator @pickSecondB() -> index {
  %0 = kgen.param.constant = <0>
  kgen.return %0 : index
}

// CHECK-LABEL: @test
kgen.generator @test() {
  // CHECK-NEXT: kgen.call @pickSecond
  %0 = kgen.call @pickSecond() : () -> index
  kgen.return
}
