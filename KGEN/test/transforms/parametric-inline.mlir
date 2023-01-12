// RUN: kgen-opt -test-parametric-inline='parent=parent callee=callee' -split-input-file -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() -> index {
  // CHECK: %[[RES:.*]] = hlcf.loop "inlined_cf_scope" () -> index
    // CHECK-NEXT: index.constant 0
    // CHECK-NEXT: hlcf.break "inlined_cf_scope" %idx0 : index
  // CHECK-NEXT: }
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index
  // CHECK: return %[[RES]]
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: hlcf.loop
  // CHECK: hlcf.break "inlined_cf_scope" %idx1
  // CHECK: hlcf.break "inlined_cf_scope" %idx0
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() -> index {
  %cond = "some.cond"() : () -> i1
  hlcf.if %cond {
    %0 = index.constant 1
    hlcf.return %0 : index
  } else {
    hlcf.yield
  }
  %0 = index.constant 0
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: hlcf.loop
  // CHECK: %[[V:.*]] = "some.producer"
  // CHECK: %[[R0:.*]] = kgen.rebind %[[V]] : !kgen.paramref<T> to index
  // CHECK-NEXT: hlcf.break "inlined_cf_scope" %[[R0]]
  // CHECK: %[[R1:.*]] = kgen.rebind %[[V]] : !kgen.paramref<T> to index
  // CHECK-NEXT: hlcf.break "inlined_cf_scope" %[[R1]]
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<T: type = index>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<T: type>() -> !kgen.paramref<T> {
  %0 = "some.producer"() : () -> !kgen.paramref<T>
  %cond = "some.cond"() : () -> i1
  hlcf.if %cond {
    hlcf.return %0 : !kgen.paramref<T>
  } else {
    hlcf.yield
  }
  kgen.return %0 : !kgen.paramref<T>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare A0 = <1>
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<A = 1>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A: i64>() {
  // CHECK: declare B: i64 = <A>
  kgen.param.declare B: i64 = <A>
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare A0: i32 = <1>
  // CHECK-NEXT: constant: i32 = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<A: i32 = 1>() : () -> i32
  // CHECK: declare C: i64 = <A>
  kgen.param.declare C: i64 = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A: i32 >() -> i32 {
  %0 = kgen.param.constant: i32 = <A>
  kgen.return %0 : i32
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() -> index {
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare A0 = <A>
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<A = A>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region B = <A>() {
    // CHECK-NEXT: hlcf.loop
    // CHECK-NEXT: declare A0 = <A>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<A = A>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() {
  kgen.param.constant = <A>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region B = <C>() {
    // CHECK-NEXT: hlcf.loop
    // CHECK-NEXT: declare A0 = <C>
    // CHECK-NEXT: declare C0 = <A>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NEXT: constant = <C0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<A = C, C = A>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, C>() {
  kgen.param.constant = <A>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region B = <C>() {
    // CHECK-NEXT: hlcf.loop
    // CHECK-NEXT: declare A0 = <A>
    // CHECK-NEXT: declare C0 = <C>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NEXT: constant = <C0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<A = A, C = C>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, C>() {
  kgen.param.constant = <A>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A, B, C>() {
  // CHECK: kgen.param.declare.region F
  kgen.param.declare.region F = <A>() {
    // CHECK-NEXT: kgen.param.declare.region G
    kgen.param.declare.region G = <B>() {
      // CHECK-NEXT: hlcf.loop
      // CHECK-NEXT: declare A0 = <A>
      // CHECK-NEXT: declare B0 = <B>
      // CHECK-NEXT: declare C0 = <C>
      // CHECK-NEXT: constant = <A0>
      // CHECK-NEXT: constant = <B0>
      // CHECK-NEXT: constant = <C0>
      // CHECK-NOT: kgen.call @callee
      kgen.call @callee<A = A, B = B, C = C>() : () -> ()
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, B, C>() {
  kgen.param.constant = <A>
  kgen.param.constant = <B>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
// CHECK: hlcf.loop
// CHECK-NOT: kgen.call @callee
  kgen.call @callee<B = 1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() {
  kgen.param.declare A = <B>
  kgen.param.constant = <A>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare B = <1>
  // CHECK-NEXT: declare.region A0 = ()
  // CHECK: call_param[() -> (): A0]()
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B = 1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() {
  kgen.param.declare.region A = () -> () {
    kgen.return
  }
  kgen.call_param[() -> (): A]()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare B = <A>
  // CHECK-NEXT: call @result_params<() -> A0>()
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B = A>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() {
  kgen.call @result_params<() -> A>() : () -> ()
  kgen.param.constant = <A>
  kgen.return
}

kgen.generator @result_params<() -> index>() {
  kgen.return<0>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare A = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B -> A>() : () -> ()
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A -> index>() {
  kgen.return<A>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <A, B>
  // CHECK-NEXT: constant = <A>
  // CHECK-NEXT: constant = <B>
  // CHECK: declare A = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B -> A>() : () -> ()
  // CHECK: constant = <A>
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A -> index>() {
  kgen.param.declare.region F = <A, B>() {
    kgen.param.constant = <A>
    kgen.param.constant = <B>
    kgen.return
  }
  kgen.return<A>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <B>
  // CHECK-NEXT: declare A = <A0>
  // CHECK-NEXT: constant = <A>
  // CHECK-NEXT: constant = <B>
  // CHECK: declare A = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B -> A>() : () -> ()
  // CHECK: constant = <A>
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A -> index>() {
  kgen.param.declare.region F = <B>() {
    kgen.param.constant = <A>
    kgen.param.constant = <B>
    kgen.return
  }
  kgen.return<A>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A = <B>
  kgen.param.declare A = <B>
  // CHECK: hlcf.loop
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <B>
  // CHECK-NEXT: declare A = <A0>
  // CHECK-NEXT: constant = <A>
  // CHECK-NEXT: constant = <B>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B>() : () -> ()
  // CHECK: constant = <A>
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() {
  kgen.param.declare.region F = <B>() {
    kgen.param.constant = <A>
    kgen.param.constant = <B>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK-NEXT: hlcf.loop "inlined_cf_scope_0"
  // CHECK: hlcf.break "inlined_cf_scope_0"
  kgen.call @callee() : () -> ()
  // CHECK: }
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() {
  hlcf.loop "inlined_cf_scope" {
    hlcf.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A, B, C>() {
  // CHECK: declare A0: <A, C>(index, index) -> index = <#kgen.expr.func<(B, D) -> add(B, D)>>
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() {
  kgen.param.declare A: <A, C>(index, index) -> index = <#kgen.expr.func<(B, D) -> add(B, D)>>
  kgen.return
}

// -----


// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: hlcf.break "inlined_cf_scope"
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() {
  kgen.param.declare.region F = () {
    hlcf.loop "inlined_cf_scope" {
      hlcf.continue
    }
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: declare A = <1>
  kgen.param.declare A = <1>
  // CHECK-NEXT: hlcf.loop
  // CHECK-NEXT: declare A0 = <2>
  // CHECK-NEXT: declare.region F
    // CHECK-NEXT: declare A = <A0>
    // CHECK-NEXT: constant = <A>
    // CHECK-NEXT: declare.region F
    // CHECK-NEXT: constant = <A>
    // CHECK-NOT: declare A = <A0>
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() {
  kgen.param.declare A = <2>
  kgen.param.declare.region F = () {
    kgen.param.constant = <A>
    kgen.param.declare.region F = () {
      kgen.param.constant = <A>
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK-NEXT: hlcf.loop
  // CHECK: declare.region F
    // CHECK-NEXT: hlcf.if
      // CHECK-NEXT: hlcf.return
  // CHECK: hlcf.if
  // CHECK-NEXT: hlcf.break "inlined_cf_scope"
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() {
  %cond = "some.cond"() : () -> i1
  kgen.param.declare.region F = () {
    hlcf.if %cond {
      hlcf.return
    } else {
      hlcf.yield
    }
    kgen.return
  }
  hlcf.if %cond {
    hlcf.return
  } else {
    hlcf.yield
  }
  kgen.return
}
