// RUN: kgen-opt -inline-param=optimization-level=3 -verify-parameters -split-input-file -allow-unregistered-dialect %s -mlir-print-debuginfo | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"parent">> : !debuginfo.subroutine<() -> (index): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<name = <"callee">> : !debuginfo.subroutine<() -> (index): DW_CC_normal>
#loc = loc("foo.mlir":1:1)
#loc1 = loc("foo.mlir":10:10)
#loc2 = loc("foo.mlir":20:20)
#loc3 = loc("foo.mlir":30:30)
#loc4 = loc("foo.mlir":40:40)
#loc5 = loc("foo.mlir":50:50)
#loc6 = loc("foo.mlir":60:60)
#loc7 = loc("foo.mlir":70:70)
#loc8 = loc("foo.mlir":80:80)
#parentLoc = loc(fused<#subprogram>[#loc])
#callOpLoc = loc(fused<#subprogram>[#loc1])
#parentRetLoc = loc(fused<#subprogram>[#loc2])
#calleeLoc = loc(fused<#subprogram1>[#loc3])
#ret0 = loc(fused<#subprogram1>[#loc4])
#ret1 = loc(fused<#subprogram1>[#loc5])
#closure = loc(fused<#subprogram1>[#loc6])
#closureRet = loc(fused<#subprogram1>[#loc7])
#calleeMisc = loc(fused<#subprogram1>[#loc8])

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() -> index {
  // CHECK: %[[RES:.*]] = hlcf.loop "[[LABEL:.*]]" () -> index
    // CHECK-NEXT: index.constant 0 loc(#[[CONST_LOC:.*]])
    // CHECK: hlcf.if
      // CHECK-NEXT: hlcf.break "[[LABEL]]" %idx0 : index loc(#[[BREAK_LOC0:.*]])
    // CHECK: kgen.param.declare.region SomeClosure = () {
      // CHECK-NEXT: kgen.return loc(#[[CL_RET_LOC:.*]])
    // CHECK-NEXT: } {{.*}} loc(#[[CL_LOC:.*]])
    // CHECK: hlcf.break "[[LABEL]]" %idx0 : index loc(#[[BREAK_LOC1:.*]])
  // CHECK-NEXT: } loc(#[[CALL_LOC:.*]])
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index loc(#callOpLoc)
  // CHECK: return %[[RES]]
  kgen.return %0 : index loc(#parentRetLoc)
} loc(#parentLoc)

// CHECK: kgen.generator @callee
kgen.generator @callee() -> index always_inline {
  // CHECK: index.constant 0 loc(#[[CALLEE_LOC:.*]])
  %0 = index.constant 0 loc(#calleeMisc)
  %false = index.bool.constant false loc(#calleeMisc)
  hlcf.if %false {
    // CHECK: kgen.return %idx0 : index loc(#[[RET_LOC0:.*]])
    kgen.return %0 : index loc(#ret0)
  } else {
    hlcf.yield loc(#calleeMisc)
  } loc(#calleeMisc)
  // CHECK: kgen.param.declare.region SomeClosure = () {
    // CHECK-NEXT: kgen.return loc(#[[CL_RET_LOC]])
  // CHECK-NEXT: } {{.*}} loc(#[[CL_LOC]])
  kgen.param.declare.region SomeClosure = () -> () {
    kgen.return loc(#closureRet)
  } loc(#closure)
  // CHECK: kgen.return %idx0 : index loc(#[[RET_LOC1:.*]])
  kgen.return %0 : index loc(#ret1)
} loc(#calleeLoc)

// CHECK: #[[CONST_LOC]] = loc(unknown)
// CHECK: #[[BREAK_LOC0]] = loc(callsite(#[[RET_LOC0]] at #[[CALL_LOC]]))
// CHECK: #[[BREAK_LOC1]] = loc(callsite(#[[RET_LOC1]] at #[[CALL_LOC]]))

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: hlcf.loop
  // CHECK: hlcf.break "[[LABEL:.*]]" %idx1
  // CHECK: hlcf.break "[[LABEL]]" %idx0
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() -> index always_inline {
  %cond = "some.cond"() : () -> i1
  hlcf.if %cond {
    %0 = index.constant 1
    kgen.return %0 : index
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
  // CHECK-NEXT: hlcf.break "[[LABEL:.*]]" %[[R0]]
  // CHECK: %[[R1:.*]] = kgen.rebind %[[V]] : !kgen.paramref<T> to index
  // CHECK-NEXT: hlcf.break "[[LABEL]]" %[[R1]]
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<:type index>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<T: type>() -> !kgen.paramref<T> always_inline {
  %0 = "some.producer"() : () -> !kgen.paramref<T>
  %cond = "some.cond"() : () -> i1
  hlcf.if %cond {
    kgen.return %0 : !kgen.paramref<T>
  } else {
    hlcf.yield
  }
  kgen.return %0 : !kgen.paramref<T>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK-NEXT: declare A0 = <1>
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<1>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index always_inline {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A: i64>() {
  // CHECK: declare B: i64 = <A>
  kgen.param.declare B: i64 = <A>
  // CHECK-NEXT: declare A0: i32 = <1>
  // CHECK-NEXT: constant: i32 = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<:i32 1>() : () -> i32
  // CHECK: declare C: i64 = <A>
  kgen.param.declare C: i64 = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A: i32 >() -> i32 always_inline {
  %0 = kgen.param.constant: i32 = <A>
  kgen.return %0 : i32
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() -> index {
  // CHECK-NEXT: declare A0 = <A>
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<A>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index always_inline {
  %0 = kgen.param.constant = <A>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region C = <B>() {
    // CHECK-NEXT: declare A0 = <B>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<B>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
  kgen.param.constant = <A>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region B = <C>() {
    // CHECK-NEXT: declare A0 = <C>
    // CHECK-NEXT: declare C0 = <A>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NEXT: constant = <C0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<C, A>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, C>() always_inline {
  kgen.param.constant = <A>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region B = <C>() {
    // CHECK-NEXT: declare A0 = <A>
    // CHECK-NEXT: declare C0 = <C>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NEXT: constant = <C0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<A, C>() : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, C>() always_inline {
  kgen.param.constant = <A>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A, B, C>() {
  // CHECK: kgen.param.declare.region D
  kgen.param.declare.region D = <E>() {
    // CHECK-NEXT: kgen.param.declare.region F
    kgen.param.declare.region F = <G>() {
      // CHECK-NEXT: declare A0 = <A>
      // CHECK-NEXT: declare B0 = <B>
      // CHECK-NEXT: declare C0 = <C>
      // CHECK-NEXT: constant = <A0>
      // CHECK-NEXT: constant = <B0>
      // CHECK-NEXT: constant = <C0>
      // CHECK-NOT: kgen.call @callee
      kgen.call @callee<A, B, C>() : () -> ()
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, B, C>() always_inline {
  kgen.param.constant = <A>
  kgen.param.constant = <B>
  kgen.param.constant = <C>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
// CHECK-NOT: kgen.call @callee
  kgen.call @callee<1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.param.declare A = <B>
  kgen.param.constant = <A>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK-NEXT: declare B0 = <1>
  // CHECK-NEXT: declare.region A0 = ()
  // CHECK: call_param[() -> (): A0]()
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.param.declare.region A = () -> () {
    kgen.return
  }
  kgen.call_param[() -> (): A]()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A>() {
  // CHECK-NEXT: declare B0 = <A>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A>() : () -> ()
  kgen.param.declare B = <0>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare B0 = <B>
  kgen.call @callee<B>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <C, D>
  // CHECK-NEXT: constant = <C>
  // CHECK-NEXT: constant = <D>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B>() : () -> ()
  kgen.param.declare A = <0>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
  kgen.param.declare.region F = <C, D>() {
    kgen.param.constant = <C>
    kgen.param.constant = <D>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <C>
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   constant = <C>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B>() : () -> ()
  kgen.param.declare A = <0>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
  kgen.param.declare.region F = <C>() {
    kgen.param.constant = <A>
    kgen.param.constant = <C>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A = <B>
  kgen.param.declare A = <B>
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F0 = <B0>
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   constant = <B0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B>() : () -> ()
  // CHECK: constant = <A>
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline {
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
  // CHECK: declare A = <1>
  kgen.param.declare A = <1>
  // CHECK-NEXT: declare A0 = <2>
  // CHECK-NEXT: declare.region F
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   declare.region G
  // CHECK-NEXT:     constant = <A0>
  // CHECK-NOT: declare A = <A0>
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  kgen.param.declare A = <2>
  kgen.param.declare.region F = () {
    kgen.param.constant = <A>
    kgen.param.declare.region G = () {
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
  // CHECK-NEXT: kgen.param.declare.region F
  kgen.param.declare.region F = () {
    // CHECK-NEXT: kgen.param.declare.region G
    kgen.param.declare.region G = () {
      // CHECK: kgen.param.declare A = <0>
      kgen.call @callee() : () -> ()
      kgen.return
    }
    // CHECK: kgen.param.declare A0 = <0>
    kgen.call @callee() : () -> ()
    kgen.return
  }
  // CHECK: kgen.param.declare A1 = <0>
  kgen.call @callee() : () -> ()
  kgen.return
}

kgen.generator @callee() always_inline {
  kgen.param.declare A = <0>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent() {
kgen.generator @parent() {
  // CHECK-NEXT: kgen.param.if <1> {
  // CHECK-NEXT:   kgen.param.declare A0 = <0>

  // CHECK:        kgen.param.if <0> {
  // CHECK-NEXT:     kgen.param.yield
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     kgen.param.declare A0 = <1>
  kgen.call @callee() : () -> ()
  // CHECK-NOT: kgen.call @callee
  // CHECK: kgen.param.declare A = <0>
  kgen.param.declare A = <0>
  kgen.return
}

// CHECK: kgen.generator @callee
kgen.generator @callee() always_inline {
  kgen.param.if <1> {
    kgen.param.declare A = <0>
    kgen.param.yield
  } else {
    kgen.param.if <0> {
      kgen.param.yield
    } else {
      kgen.param.declare A = <1>
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK-NEXT: kgen.param.declare A: i1 = <0>
  kgen.param.declare A: i1 = <0>
  // CHECK-NEXT: kgen.param.declare A1: i1 = <1>
  // CHECK-NEXT: kgen.param.if <A1> {
  // CHECK-NEXT:   kgen.param.declare A2 = <2>
  // CHECK-NEXT:   kgen.param.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   kgen.param.declare A2: i1 = <A1>
  // CHECK-NEXT:   kgen.param.if <A2> {
  // CHECK-NEXT:     kgen.param.declare B0: i1 = <A1>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK: kgen.generator @callee
kgen.generator @callee() always_inline {
  kgen.param.declare A: i1 = <1>
  kgen.param.if <A> {
    kgen.param.declare A0 = <2>
    kgen.param.yield
  } else {
    kgen.param.declare A0: i1 = <A>
    kgen.param.if <A0> {
      kgen.param.declare B: i1 = <A>
      kgen.param.yield
    } else {
      kgen.param.yield
    } {elseIsolated, thenIsolated}
    kgen.param.yield
  } {elseIsolated, thenIsolated}
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: declare.region F
    // CHECK-NEXT: hlcf.if
      // CHECK-NEXT: kgen.return
  // CHECK: hlcf.if
  // CHECK-NEXT: hlcf.break "[[LABEL:.*]]"
  kgen.call @callee() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  %cond = "some.cond"() : () -> i1
  kgen.param.declare.region F = () {
    hlcf.if %cond {
      kgen.return
    } else {
      hlcf.yield
    }
    kgen.return
  }
  hlcf.if %cond {
    kgen.return
  } else {
    hlcf.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A: dtype>() {
  %0 = kgen.call @callee<1>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() -> index always_inline {
  kgen.param.declare B = <A>
  %0 = kgen.param.constant = <B>
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @inline_call_in_if
kgen.generator @inline_call_in_if(%cond: i1) {
  // CHECK-NEXT: hlcf.if
  hlcf.if %cond {
    // CHECK: inlined.a
    kgen.call @callee() : () -> ()
    hlcf.yield
  } else {
    hlcf.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  "inlined.a"() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @inline_call_in_param_if
kgen.generator @inline_call_in_param_if<cond: i1>() {
  // CHECK-NEXT: kgen.param.if
  kgen.param.if <cond> {
    // CHECK: inlined.a
    kgen.call @callee() : () -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee() always_inline {
  "inlined.a"() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @rebind_call_operands
kgen.generator @rebind_call_operands(%arg0: !pop.scalar<f32>) {
  // CHECK: kgen.param.declare DT: dtype = <f32>
  // CHECK-NEXT: %0 = kgen.rebind %arg0 : !pop.scalar<f32> to !pop.scalar<DT>
  // CHECK: pop.simd.extractelement %0[%idx0] : !pop.scalar<DT>
  kgen.call @callee<:dtype f32>(%arg0) : (!pop.scalar<f32>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<DT: dtype>(%arg0: !pop.scalar<DT>) always_inline {
  %idx0 = index.constant 0
  %0 = pop.simd.extractelement %arg0[%idx0] : !pop.scalar<DT>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @rebind_mangled_types
kgen.generator @rebind_mangled_types<DT: dtype>(%arg0: !pop.scalar<DT>) {
  // CHECK: kgen.param.declare DT0: dtype = <DT>
  // CHECK-NEXT: %0 = kgen.rebind %arg0 : !pop.scalar<DT> to !pop.scalar<DT0>
  // CHECK: %1 = pop.simd.extractelement %0[%idx0] : !pop.scalar<DT0>
  // CHECK: %2 = kgen.rebind %1 : !pop.scalar<DT0> to !pop.scalar<DT>
  %0 = kgen.call @callee<:dtype DT>(%arg0) : (!pop.scalar<DT>) -> !pop.scalar<DT>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<DT: dtype>(%arg0: !pop.scalar<DT>) -> !pop.scalar<DT> always_inline {
  %idx0 = index.constant 0
  %0 = pop.simd.extractelement %arg0[%idx0] : !pop.scalar<DT>
  kgen.return %0 : !pop.scalar<DT>
}

// -----

// CHECK-LABEL: kgen.generator @replace_in_signature_with_shadow
kgen.generator @replace_in_signature_with_shadow<width>() {
  // CHECK: kgen.param.declare width0 = <width>
  // CHECK-NEXT: kgen.param.declare fn: <index>(!pop.simd<*(0,0), bool>) -> () = <@param_arg>
  // CHECK-NEXT: kgen.param.declare bound: (!pop.simd<width0, bool>) -> ()
  // CHECK-SAME: = <bind_signature(:<index>(!pop.simd<*(0,0), bool>) -> () fn, width0)>
  kgen.call @callee<width>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_arg
kgen.generator @param_arg<width>(%arg0: !pop.simd<width, bool>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<width>() always_inline {
  kgen.param.declare fn: <index>(!pop.simd<*(0,0), bool>) -> () = <@param_arg>
  kgen.param.declare bound: (!pop.simd<width, bool>) -> () =
    <bind_signature(:<index>(!pop.simd<*(0,0), bool>) -> () fn, width)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @dependent_types
kgen.generator @dependent_types() {
  // CHECK-NEXT: declare rank = <4>
  kgen.param.declare rank = <4>
  // CHECK: declare rank1 = <1>
  // CHECK-NEXT: declare shape1: array<rank1, index> = <rebind(:array<1, index> [2])>
  // CHECK: declare rank0 = <rank1>
  // CHECK-NEXT: declare shape0: array<rank0, index> = <rebind(:array<rank1, index> shape1)>
  kgen.call @callee<1, :array<1, index> [2]>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_me
kgen.generator @call_me<rank, shape: array<rank, index>>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<rank, shape: array<rank, index>>() always_inline {
  kgen.call @call_me<rank, :array<rank, index> shape>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @struct_extract
kgen.generator @struct_extract(%arg0: !kgen.struct<(simd<2, f32>)>) {
  kgen.param.declare size = <1>
  kgen.param.declare type: dtype = <si32>
  // CHECK: kgen.struct.extract %0[0] : !kgen.struct<(simd<size0, type0>)>
  kgen.call @callee<2, :dtype f32>(%arg0) : (!kgen.struct<(simd<2, f32>)>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<size, type: dtype>(%arg0: !kgen.struct<(simd<size, type>)>) always_inline {
  kgen.param.declare cond: i1 = <1>
  kgen.param.if <cond> {
    %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(simd<size, type>)>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @only_mangle_mangled_captures
kgen.generator @only_mangle_mangled_captures() {
  kgen.param.declare A = <0>
  // CHECK: constant = <A0>
  // CHECK-NEXT: constant = <B0>
  kgen.call @callee<1, 1>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A, B>() always_inline {
  kgen.param.declare.region F = () {
    kgen.param.constant = <A>
    kgen.param.constant = <B>
    kgen.return
  }
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<rank, shape: array<rank, index>>() {
  // CHECK: declare another: array<rank0, index> = <shape0>
  kgen.call @mid<rank, :array<rank, index> shape>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @mid
kgen.generator @mid<rank, shape: array<rank, index>>() always_inline {
  kgen.param.declare another: array<rank, index> = <shape>
  kgen.return
}

// -----

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<(!debuginfo.unresolved<!kgen.paramref<T>>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<scope = #subprogram, name = "foo"> : !debuginfo.unresolved<!kgen.paramref<T>>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#subprogram>[#fileLoc])

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<T: type>(%arg0: index) {
  // CHECK: kgen.param.declare T0: type = <index> loc(#[[CALL_LOC:.*]])
  // CHECK-NEXT: kgen.rebind %arg0 : index to !kgen.paramref<T0> loc(#[[CALL_LOC]])
  // CHECK-NEXT: kgen.return
  kgen.call @nodebug_inline_me<:type index>(%arg0) : (index) -> () loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// CHECK-LABEL: kgen.generator @nodebug_inline_me
kgen.generator @nodebug_inline_me<T: type>(%arg0: !kgen.paramref<T>) always_inline_no_debug {
  kgen.return loc(#loc)
} loc(#loc)

// -----

// COM: https://github.com/modularml/modular/issues/8586

kgen.generator @unroll<func: <index>() -> ()>() always_inline {
  kgen.param.constant: () -> () = <bind_signature(:<index>() -> () func, 1)>
  kgen.return
}

kgen.generator @nested_func_call<func: () -> ()>() always_inline {
  kgen.param.declare.region func_wrapper = () {
    kgen.param.declare.region nested_func = <idx>() {
      kgen.call_param[() -> (): func]()
      kgen.return
    }
    kgen.call @unroll<:<index>() -> () nested_func>() : () -> ()
    kgen.return
  }
  kgen.call_param[() -> (): func_wrapper]()
  kgen.return
}

kgen.generator @pass_it() always_inline {
  kgen.param.declare.region id = () {
    kgen.return
  }
  kgen.call @nested_func_call<:() -> () id>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  // CHECK: kgen.param.declare.region id
  // CHECK: kgen.param.declare func: () -> () = <id>
  // CHECK: kgen.param.declare.region func_wrapper = () {
    // CHECK: kgen.param.declare.region nested_func = <idx>() {
      // CHECK: kgen.call_param[() -> (): func]
    // CHECK: kgen.param.declare func0: <index>() -> () = <nested_func>
    // CHECK: kgen.param.constant: () -> () = <bind_signature(:<index>() -> () func0, 1)>
  // CHECK: kgen.call_param[() -> (): func_wrapper]
  kgen.call @pass_it() : () -> ()

  // CHECK: kgen.param.declare.region id0
  // CHECK: kgen.param.declare func1: () -> () = <id0>
  // CHECK: kgen.param.declare.region func_wrapper0 = () {
    // CHECK: kgen.param.declare.region nested_func0 = <idx0>() {
      // CHECK: kgen.call_param[() -> (): func1]
    // CHECK: kgen.param.declare func2: <index>() -> () = <nested_func0>
    // CHECK: kgen.param.constant: () -> () = <bind_signature(:<index>() -> () func2, 1)>
  // CHECK: kgen.call_param[() -> (): func_wrapper0]
  kgen.call @pass_it() : () -> ()
  kgen.return
}

// -----

// COM: Give up on recursive elaboration instead of emitting an error.

kgen.generator @passthrough<cond: i1>() always_inline {
  kgen.call @recursive<:i1 cond>() : () -> ()
  kgen.return
}

kgen.generator @recursive<cond: i1>() always_inline {
  kgen.param.if <cond> {
    kgen.call @passthrough<:i1 0>() : () -> ()
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @root
kgen.generator @root() {
  // CHECK-NEXT: kgen.call @recursive
  kgen.call @recursive<:i1 1>() : () -> ()
  kgen.return
}

// -----

// COM: This is testing a DenseMap invalidation for nested parameter scopes.
// COM: https://github.com/modularml/modular/issues/10174

kgen.generator @deeply_nested_paramif<value>() always_inline {
  kgen.param.if<eq(value, 0)> {
    kgen.param.yield
  } else {
    kgen.param.if<eq(value, 1)> {
      kgen.param.yield
    } else {
      kgen.param.if<eq(value, 2)> {
        kgen.param.yield
      } else {
        kgen.param.if<eq(value, 3)> {
          kgen.param.yield
        } else {
          kgen.param.if<eq(value, 4)> {
            kgen.param.yield
          } else {
            kgen.param.if<eq(value, 5)> {
              kgen.param.yield
            } else {
              kgen.param.if<eq(value, 6)> {
                kgen.call @deeply_nested_paramif_0<1>() : () -> ()
                kgen.param.yield
              } else {
                kgen.param.if<eq(value, 7)> {
                  kgen.param.yield
                } else {
                  kgen.param.if<eq(value, 8)> {
                    kgen.param.yield
                  } else {
                    kgen.param.if<eq(value, 9)> {
                      kgen.param.yield
                    } else {
                      kgen.param.if<eq(value, 10)> {
                        kgen.param.yield
                      } else {
                        kgen.param.if<eq(value, 11)> {
                          kgen.param.yield
                        } else {
                          kgen.param.yield
                        }
                        kgen.param.yield
                      }
                      kgen.param.yield
                    }
                    kgen.param.yield
                  }
                  kgen.param.yield
                }
                kgen.param.yield
              }
              kgen.param.yield
            }
            kgen.param.yield
          }
          kgen.param.yield
        }
        kgen.param.yield
      }
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

kgen.generator @deeply_nested_paramif_0<value>() always_inline {
  kgen.param.if<eq(value, 0)> {
    kgen.param.yield
  } else {
    kgen.param.if<eq(value, 1)> {
      kgen.param.yield
    } else {
      kgen.param.if<eq(value, 2)> {
        kgen.param.yield
      } else {
        kgen.param.if<eq(value, 3)> {
          kgen.param.yield
        } else {
          kgen.param.if<eq(value, 4)> {
            kgen.param.yield
          } else {
            kgen.param.if<eq(value, 5)> {
              kgen.param.yield
            } else {
              kgen.param.if<eq(value, 6)> {
                kgen.param.yield
              } else {
                kgen.param.if<eq(value, 7)> {
                  kgen.param.yield
                } else {
                  kgen.param.if<eq(value, 8)> {
                    kgen.param.yield
                  } else {
                    kgen.param.if<eq(value, 9)> {
                      kgen.param.yield
                    } else {
                      kgen.param.if<eq(value, 10)> {
                        kgen.param.yield
                      } else {
                        kgen.param.if<eq(value, 11)> {
                          kgen.param.yield
                        } else {
                          kgen.param.yield
                        }
                        kgen.param.yield
                      }
                      kgen.param.yield
                    }
                    kgen.param.yield
                  }
                  kgen.param.yield
                }
                kgen.param.yield
              }
              kgen.param.yield
            }
            kgen.param.yield
          }
          kgen.param.yield
        }
        kgen.param.yield
      }
      kgen.param.yield
    }
    kgen.param.yield
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_it
kgen.generator @call_it() {
  // CHECK: kgen.param.if
  kgen.call @deeply_nested_paramif<10>() : () -> ()
  kgen.return
}

// -----

kgen.generator @pass_it() always_inline {
  kgen.param.declare value: i32 = <1>
  kgen.param.declare.region f = () {
    kgen.param.declare.region g = () {
      kgen.param.constant: i32 = <value>
      kgen.return
    }
    kgen.param.declare value0 = <1>
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @main
kgen.generator @main() {
  // CHECK: declare value: f32
  kgen.param.declare value: f32 = <1.0>
  // CHECK: declare value1: i32 = <1>
  // CHECK: kgen.param.constant: i32 = <value1>
  // CHECK: declare value0 = <1>
  kgen.call @pass_it() : () -> ()

  kgen.return
}

// -----

// COM: This test case tricks a simple counter uniquer into mangling two
// COM: parameter decls into the same name.

kgen.generator @inline_me() always_inline {
  kgen.param.declare value = <1>
  kgen.param.declare value0 = <1>
  kgen.param.declare value1 = <1>
  kgen.param.declare value2 = <1>
  kgen.param.declare value3 = <1>
  kgen.param.declare value4 = <1>
  kgen.param.declare value5 = <1>
  kgen.param.declare value6 = <1>
  kgen.param.declare value7 = <1>
  kgen.param.declare value8 = <1>
  kgen.param.declare value9 = <1>
  kgen.return
}

// CHECK-LABEL: kgen.generator @entry
kgen.generator @entry() {
  // CHECK: declare value10 =
  // CHECK: declare value11 =
  kgen.param.declare value1 = <1>
  kgen.param.declare value = <1>
  kgen.call @inline_me() : () -> ()
  kgen.return
}

// -----

kgen.generator @unreachable_and_early_ret() always_inline {
  %true = index.bool.constant true
  hlcf.if %true {
    kgen.return
  } else {
    hlcf.yield
  }
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @call_it
kgen.generator @call_it() {
  // CHECK-NEXT: hlcf.loop
    // CHECK: hlcf.if
      // CHECK-NEXT: hlcf.break
    // CHECK: kgen.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: kgen.return
  kgen.call @unreachable_and_early_ret() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.func @dontinlineme() -> index
kgen.func @dontinlineme() -> index always_inline {
  // CHECK-NEXT: index.constant
  %0 = index.constant 3
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @caller
kgen.generator @caller() -> index {
  // CHECK-NEXT: kgen.call @dontinlineme
  %0 = kgen.call @dontinlineme() : () -> index
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator @foo
kgen.generator @foo() {
  // CHECK: kgen.param.declare.region SomeClosure = <DT: dtype, N>(%[[ARG:.*]]: !pop.simd<N, DT>
  // CHECK-NEXT: kgen.param.declare A = <1> loc(#[[LOC:.*]])
  // CHECK-NEXT: kgen.return %[[ARG]]
  kgen.call @bar() : () -> ()
  // CHECK: kgen.param.declare.region SomeClosure0 = <DT0: dtype, N0>(%[[ARG0:.*]]: !pop.simd<N0, DT0>
  // CHECK-NEXT: kgen.param.declare A0 = <1> loc(#[[LOC0:.*]])
  // CHECK-NEXT: kgen.return %[[ARG0]] : !pop.simd<N0, DT0> loc(#[[LOC0]])
  kgen.call @bar() : () -> ()
  kgen.return
}
kgen.generator @bar() always_inline {
  kgen.param.declare.region SomeClosure = <DT: dtype, N>(%arg0: !pop.simd<N, DT>) capturing -> !pop.simd<N, DT> {
    kgen.param.declare A = <1> loc(#loc)
    kgen.return %arg0 : !pop.simd<N, DT> loc(#loc)
  } loc(#loc)
  kgen.return
}

// CHECK-DAG: ![[M:.*]] = !debuginfo.member<value: !pop.simd<N, DT>>
// CHECK-DAG: ![[M0:.*]] = !debuginfo.member<value: !pop.simd<N0, DT0>>
// CHECK-DAG: ![[STR:.*]] = !debuginfo.struct<"builtin::$simd::SIMD"(![[M]])>
// CHECK-DAG: ![[STR0:.*]] = !debuginfo.struct<"builtin::$simd::SIMD"(![[M0]])>
// CHECK-DAG: ![[SR:.*]] = !debuginfo.subroutine<(![[STR]]) -> (![[STR]]): DW_CC_normal>
// CHECK-DAG: ![[SR0:.*]] = !debuginfo.subroutine<(![[STR0]]) -> (![[STR0]]): DW_CC_normal>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<{{.*}}, name = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : ![[SR]]
// CHECK-DAG: #[[SP0:.*]] = #debuginfo.subprogram<{{.*}}, name = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : ![[SR0]]

!struct = !debuginfo.struct<"builtin::$simd::SIMD"(!debuginfo.member<value: !pop.simd<N, DT>>)>
#file = #debuginfo.file<"foo.mlir" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
#subprogram2 = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"SomeClosure">, linkageName = "SomeClosure", file = #file, line = 1314, scopeLine = 1314, subprogramFlags = "Definition|Optimized"> : !debuginfo.subroutine<(!struct) -> (!struct): DW_CC_normal>

// CHECK-DAG: #[[LOC_ORI:.*]] = loc("foo.mlir":1317:13)
// CHECK-DAG: #[[LOC]] = loc(fused<#[[SP]]>[#[[LOC_ORI]]])
// CHECK-DAG: #[[LOC0]] = loc(fused<#[[SP0]]>[#[[LOC_ORI]]])
#loc = loc(fused<#subprogram2>["foo.mlir":1317:13])

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>
#subprogram = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"foo">, linkageName = "foo", file = #file, line = 10, scopeLine = 10, subprogramFlags = Definition> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.generator @no_debuginfo() -> index always_inline {
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// CHECK-LABEL: kgen.generator @has_debuginfo
kgen.generator @has_debuginfo() {
  // CHECK: index.constant 0 loc([[LOC:#.*]])
  kgen.call @no_debuginfo() : () -> index loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)

// CHECK: [[LOC]] = loc(unknown)

// -----

kgen.generator @recursive() always_inline_no_debug {
  kgen.call @recursive() : () -> ()
  kgen.return
}

kgen.generator @trivial() always_inline_no_debug {
  kgen.return
}

// CHECK-LABEL: kgen.generator @top
kgen.generator @top() {
  // CHECK-NEXT: call @recursive
  kgen.call @recursive() : () -> ()
  // CHECK-NOT: call @trivial
  kgen.call @trivial() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @inline_heuristic
kgen.generator @inline_heuristic<A>() {
  // CHECK: %[[V:.*]] = "some.producer"
  // CHECK: %[[R0:.*]] = kgen.rebind %[[V]] : !kgen.paramref<T> to index
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<:type index>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<T: type>() -> !kgen.paramref<T> always_inline {
  %0 = "some.producer"() : () -> !kgen.paramref<T>
  kgen.return %0 : !kgen.paramref<T>
}
