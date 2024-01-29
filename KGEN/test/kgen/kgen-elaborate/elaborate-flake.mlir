// RUN: kgen-opt --llcl-single-thread --mlir-disable-threading %s -split-input-file -elaborate-generators="enable-search=true allow-multiple-primary-impls=true" -allow-unregistered-dialect | FileCheck %s


// FIXME(#29331): These tests are flaky in multi-threaded execution for
// reasons that aren't entirely obvious. Since `kgen.param.fork` is deprecated,
// just cordone the tests into this file until it is redesigned/removed.


//===----------------------------------------------------------------------===//

// CHECK-LABEL: @"genItf2,x=0"()
kgen.generator @genItf2<x>() {
  // CHECK-NEXT: kgen.call @"genItf2_impl0,x=0"
  kgen.param.fork impl : () -> () = <[@genItf2_impl0<x>, @genItf2_impl1<x>]>
  kgen.call_param[() -> () : impl]()
  kgen.return
}

// CHECK-NOT: kgen.func @"genItf2_impl0,x=1_1"() {
// CHECK-LABEL: kgen.func @"genItf2_impl0,x=0"() {
// CHECK-NEXT:   "impl.0"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.func @"genItf2_impl0,x=1"() {
kgen.generator @genItf2_impl0<x>() {
  kgen.param.assert <eq(x, 0)>, "x must be zero"
  "impl.0"() : () -> ()
  kgen.return
}

// CHECK-NOT: kgen.func @"genItf2_impl1,x=0"()
// CHECK-LABEL: kgen.func @"genItf2_impl1,x=1"() {
// CHECK-NEXT:   "impl.1"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NOT: kgen.func @"genItf2_impl1,x=0"()
kgen.generator @genItf2_impl1<x>() {
  kgen.param.assert <eq(x, 1)>, "x must be 1"
  "impl.1"() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @use_Itf2zero() {
// CHECK-NEXT:   kgen.call @"genItf2,x=0"() : () -> ()
// CHECK-NEXT:   kgen.return
kgen.generator @use_Itf2zero() {
  kgen.call @genItf2<0>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @use_Itf2one() {
// CHECK-NEXT:   kgen.call @"genItf2,x=1,impl=@genItf2_impl1<1>"() : () -> ()
// CHECK-NEXT:   kgen.return
// CHECK-NEXT: }
kgen.generator @use_Itf2one() {
  kgen.call @genItf2<1>() : () -> ()
  kgen.return
}

// -----

// COM: First instantiation of `@fwd` is inside an assert.

kgen.generator @fwd(%a: i1) -> i1 {
  kgen.return %a : i1
}

kgen.generator @f() {
  kgen.param.assert <apply(:(i1) -> i1 @fwd, 1)>, "true"
  kgen.return
}

// CHECK-LABEL: kgen.func export @top
kgen.generator export @top() {
  // CHECK-NEXT: call @f
  kgen.call @f() : () -> ()
  kgen.return
}
