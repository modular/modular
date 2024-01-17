// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true allow-multiple-primary-impls=true" -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.func @"even_only,param=16"()
// CHECK-NOT: @"even_only,
// CHECK-LABEL: kgen.func @"even_only,param=72"()
// CHECK-NOT: @"even_only,
kgen.generator @even_only<param>() {
  kgen.param.assert <eq(and(param, 1), 0)>, "the param shalt be even!"
  kgen.return
}

// This should turn into two variants exactly, not a duplicate for 72.
// CHECK-LABEL: kgen.func @find_even()
// CHECK-LABEL: kgen.func @"find_even,value=72"
// CHECK-NOT: find_even
kgen.generator @find_even() {
  kgen.param.fork seventy_two = <[72]>
  kgen.param.fork value = <[3, 16, 1, 72, seventy_two]>
  kgen.call @even_only<value>() : () -> ()
  kgen.return
}

// -----

// COM: This is an example that tests using the generator interface replacement.

/// This evaluator returns a constant index.
kgen.generator @simpleEvaluator<N, FN: type>(%funcs: !kgen.pointer<FN>, %num: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

/// Always pick the first implementation of this interface.
// CHECK-LABEL: @pickFirst
kgen.generator @pickFirst() {
  kgen.param.declare evaluator: (!kgen.pointer<!kgen.signature<() -> ()>>, index) -> index
    = <bind_signature(:<index, type>(!kgen.pointer<*(0,1)>, index) -> index @simpleEvaluator, 0, !kgen.signature<()->()>)>
  kgen.param.evaluate chosenImpl : () -> () = [@pickFirstA, @pickFirstB]
    with [(!kgen.pointer<!kgen.signature<() -> ()>>, index) -> index: evaluator]
  // CHECK-NEXT: kgen.call @pickFirstA
  kgen.call_param[() -> (): chosenImpl]()
  kgen.return
}

/// Always pick the second implementation of this interface.
// CHECK-LABEL: @pickSecond()
kgen.generator @pickSecond() {
  kgen.param.declare evaluator: (!kgen.pointer<!kgen.signature<() -> ()>>, index) -> index
    = <bind_signature(:<index, type>(!kgen.pointer<*(0,1)>, index) -> index @simpleEvaluator, 1, !kgen.signature<()->()>)>
  kgen.param.evaluate chosenImpl : () -> () = [@pickSecondA, @pickSecondB]
    with [(!kgen.pointer<!kgen.signature<() -> ()>>, index) -> index: evaluator]
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
// CHECK-LABEL: @"simpleEvaluator,N=1,FN=() -> index"
kgen.generator @simpleEvaluator<N, FN:type>(%funcs: !kgen.pointer<FN>, %num: index) -> index {
  %0 = kgen.param.constant = <N>
  kgen.return %0 : index
}

/// Always pick the second implementation of this interface.
// CHECK-LABEL: @pickSecond()
kgen.generator @pickSecond() -> index {
  kgen.param.declare evaluator: (!kgen.pointer<!kgen.signature<() -> index>>, index) -> index
    = <bind_signature(:<index, type>(!kgen.pointer<*(0,1)>, index) -> index @simpleEvaluator, 1, !kgen.signature<()->index>)>
  kgen.param.evaluate chosenImpl : () -> index = [@pickSecondA, @pickSecondB]
    with [(!kgen.pointer<!kgen.signature<() -> index>>, index) -> index: evaluator]
  // COM: This is actually not one of the direct options, it's an expansion of one of them.
  // CHECK-NEXT: kgen.call @"pickSecondA,f=2"()
  %0 = kgen.call_param[() -> index: chosenImpl]()
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @"pickSecondA,f=2"()
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

// CHECK-LABEL: kgen.func export @test
kgen.generator export @test() {
  // CHECK-NEXT: kgen.call @pickSecond
  %0 = kgen.call @pickSecond() : () -> index
  kgen.return
}

// -----

// We should generate three versions of this function.
// CHECK-LABEL: kgen.func @"checkGetAllImpls,oneImpl=@\22multipleImplsFn,p=2\22"
// CHECK: kgen.call @"multipleImplsFn,p=2"
// CHECK-LABEL: kgen.func @"checkGetAllImpls,oneImpl=@\22multipleImplsFn,p=3\22"
// CHECK: kgen.call @"multipleImplsFn,p=3"
// CHECK-LABEL: kgen.func @checkGetAllImpls(
// CHECK: kgen.call @multipleImplsFn(
kgen.generator @checkGetAllImpls() -> index {
  kgen.param.declare impls: !kgen.variadic<!kgen.signature<() -> index>> = <get_all_impls(@multipleImplsFn)>
  // `impls` should be a list containing three implementations of `@multipleImplsFn`
  //
  // When we `kgen.param.fork` on that list, we should generate three versions
  // of this function, each version calling one of the implementations from the
  // list.

  kgen.param.fork oneImpl: !kgen.signature<() -> index> = <impls>
  %ret = kgen.call_param[() -> index: oneImpl]()
  kgen.return %ret : index
}

// This generator should also produce three versions.
// CHECK-LABEL: kgen.func @"multipleImplsFn,p=2"
// CHECK-LABEL: kgen.func @"multipleImplsFn,p=3"
// CHECK-LABEL: kgen.func @multipleImplsFn(
kgen.generator @multipleImplsFn() -> index {
  kgen.param.fork p : index = <[1, 2, 3]>
  %ret = kgen.param.constant = <p>
  kgen.return %ret : index
}
