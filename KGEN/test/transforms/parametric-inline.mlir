// RUN: kgen-opt -always-inline-param -split-input-file -allow-unregistered-dialect %s -mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() -> index {
  // CHECK: %[[RES:.*]] = hlcf.loop "[[LABEL:.*]]" () -> index
    // CHECK-NEXT: index.constant 0 loc(#[[INLINED_LOC:.*]])
    // CHECK: hlcf.if
      // CHECK-NEXT: hlcf.break "[[LABEL]]" %idx0 : index
    // CHECK: hlcf.break "[[LABEL]]" %idx0 : index
  // CHECK-NEXT: } loc(#[[CALL_LOC:.*]])
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee() : () -> index
  // CHECK: return %[[RES]]
  kgen.return %0 : index
}

// CHECK: kgen.generator @callee
kgen.generator @callee() -> index always_inline {
  // CHECK: index.constant 0 loc(#[[CALLEE_LOC:.*]])
  %0 = index.constant 0
  %false = index.bool.constant false
  hlcf.if %false {
    hlcf.return %0 : index
  } else {
    hlcf.yield
  }
  kgen.return %0 : index
}

// CHECK: #[[INLINED_LOC]] = loc(callsite(#[[CALLEE_LOC]] at #[[CALL_LOC]]))

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
  // CHECK-NEXT: hlcf.break "[[LABEL:.*]]" %[[R0]]
  // CHECK: %[[R1:.*]] = kgen.rebind %[[V]] : !kgen.paramref<T> to index
  // CHECK-NEXT: hlcf.break "[[LABEL]]" %[[R1]]
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<T: type = index>() : () -> index
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<T: type>() -> !kgen.paramref<T> always_inline {
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
  // CHECK-NEXT: declare A0 = <1>
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  %0 = kgen.call @callee<A = 1>() : () -> index
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
  %0 = kgen.call @callee<A: i32 = 1>() : () -> i32
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
  %0 = kgen.call @callee<A = A>() : () -> index
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
  kgen.param.declare.region B = <A>() {
    // CHECK-NEXT: declare A0 = <A>
    // CHECK-NEXT: constant = <A0>
    // CHECK-NOT: kgen.call @callee
    kgen.call @callee<A = A>() : () -> ()
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
    kgen.call @callee<A = C, C = A>() : () -> ()
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
    kgen.call @callee<A = A, C = C>() : () -> ()
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
  // CHECK: kgen.param.declare.region F
  kgen.param.declare.region F = <A>() {
    // CHECK-NEXT: kgen.param.declare.region G
    kgen.param.declare.region G = <B>() {
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
  kgen.call @callee<B = 1>() : () -> ()
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
  // CHECK-NEXT: declare B = <1>
  // CHECK-NEXT: declare.region A0 = ()
  // CHECK: call_param[() -> (): A0]()
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B = 1>() : () -> ()
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
  // CHECK-NEXT: declare B = <A>
  // CHECK-NEXT: call @result_params<() -> A0 = A>()
  // CHECK-NEXT: constant = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<B = A>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<B>() always_inline {
  kgen.call @result_params<() -> A = A>() : () -> ()
  kgen.param.constant = <A>
  kgen.return
}

kgen.generator @result_params<() -> A>() {
  kgen.return<0>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare A = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B -> A = B>() : () -> ()
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A -> B>() always_inline {
  kgen.return<A>
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<B>() {
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <A, B>
  // CHECK-NEXT: constant = <A>
  // CHECK-NEXT: constant = <B>
  // CHECK: declare A = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B -> A = B>() : () -> ()
  // CHECK: constant = <A>
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A -> B>() always_inline {
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
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <B>
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   constant = <B>
  // CHECK: declare A = <A0>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B -> A = B>() : () -> ()
  // CHECK: constant = <A>
  kgen.param.constant = <A>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A -> B>() always_inline {
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
  // CHECK-NEXT: declare A0 = <B>
  // CHECK-NEXT: declare.region F = <B>
  // CHECK-NEXT:   constant = <A0>
  // CHECK-NEXT:   constant = <B>
  // CHECK-NOT: kgen.call @callee
  kgen.call @callee<A = B>() : () -> ()
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
  // CHECK-NEXT:   declare.region F
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
  // CHECK: declare.region F
    // CHECK-NEXT: hlcf.if
      // CHECK-NEXT: hlcf.return
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

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<A: dtype>() {
  %0 = kgen.call @callee<A = 1>() : () -> index
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
  // CHECK: kgen.param.declare type: dtype = <f32>
  // CHECK-NEXT: %0 = kgen.rebind %arg0 : !pop.scalar<f32> to !pop.scalar<type>
  // CHECK: pop.simd.extractelement %0[%idx0] : !pop.scalar<type>
  kgen.call @callee<type: dtype = f32>(%arg0) : (!pop.scalar<f32>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<type: dtype>(%arg0: !pop.scalar<type>) always_inline {
  %idx0 = index.constant 0
  %0 = pop.simd.extractelement %arg0[%idx0] : !pop.scalar<type>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @rebind_mangled_types
kgen.generator @rebind_mangled_types<type: dtype>(%arg0: !pop.scalar<type>) {
  // CHECK: kgen.param.declare type0: dtype = <type>
  // CHECK-NEXT: %0 = kgen.rebind %arg0 : !pop.scalar<type> to !pop.scalar<type0>
  // CHECK: %1 = pop.simd.extractelement %0[%idx0] : !pop.scalar<type0>
  // CHECK: %2 = kgen.rebind %1 : !pop.scalar<type0> to !pop.scalar<type>
  %0 = kgen.call @callee<type: dtype = type>(%arg0) : (!pop.scalar<type>) -> !pop.scalar<type>
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<type: dtype>(%arg0: !pop.scalar<type>) -> !pop.scalar<type> always_inline {
  %idx0 = index.constant 0
  %0 = pop.simd.extractelement %arg0[%idx0] : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

// -----

// CHECK-LABEL: kgen.generator @replace_in_signature_with_shadow
kgen.generator @replace_in_signature_with_shadow<width>() {
  // CHECK: kgen.param.declare width0 = <width>
  // CHECK-NEXT: kgen.param.declare fn: <width>(!pop.simd<width, bool>) -> () = <@param_arg>
  // CHECK-NEXT: kgen.param.declare bound: (!pop.simd<width0, bool>) -> ()
  // CHECK-SAME: = <bind_signature(:<width>(!pop.simd<width, bool>) -> () fn, width0)>
  kgen.call @callee<width = width>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_arg
kgen.generator @param_arg<width>(%arg0: !pop.simd<width, bool>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<width>() always_inline {
  kgen.param.declare fn: <width>(!pop.simd<width, bool>) -> () = <@param_arg>
  kgen.param.declare bound: (!pop.simd<width, bool>) -> () =
    <bind_signature(:<width>(!pop.simd<width, bool>) -> () fn, width)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @dependent_types
kgen.generator @dependent_types() {
  // CHECK-NEXT: declare rank = <4>
  kgen.param.declare rank = <4>
  // CHECK: declare rank0 = <1>
  // CHECK-NEXT: declare shape: list<index[rank0]> = <rebind(:list<index[1]> [2])>
  // CHECK-NEXT: call @call_me<rank = rank0, shape: list<index[rank0]> = shape>
  // CHECK-NEXT: declare output: list<index[1]> = <rebind(:list<index[rank0]> shape)>
  kgen.call @callee<rank = 1, shape: list<index[1]> = [2] -> output = output: list<index[1]>>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_me
kgen.generator @call_me<rank, shape: list<index[rank]>>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<rank, shape: list<index[rank]> -> output: list<index[rank]>>() always_inline {
  kgen.call @call_me<rank = rank, shape: list<index[rank]> = shape>() : () -> ()
  kgen.return<:list<index[rank]> shape>
}

// -----

// CHECK-LABEL: kgen.generator @struct_extract
kgen.generator @struct_extract(%arg0: !pop.struct<simd<2, f32>>) {
  kgen.param.declare size = <1>
  kgen.param.declare type: dtype = <si32>
  // CHECK: pop.struct.extract %0[0] : !pop.struct<simd<size0, type0>>
  kgen.call @callee<size = 2, type: dtype = f32>(%arg0) : (!pop.struct<simd<2, f32>>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<size, type: dtype>(%arg0: !pop.struct<simd<size, type>>) always_inline {
  kgen.param.declare cond: i1 = <1>
  kgen.param.if <cond> {
    %0 = pop.struct.extract %arg0[0] : !pop.struct<simd<size, type>>
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
  // CHECK-NEXT: constant = <B>
  kgen.call @callee<A = 1, B = 1>() : () -> ()
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
kgen.generator @parent<rank, shape: list<index[rank]>>() {
  // CHECK: declare another: list<index[rank0]> = <shape0>
  kgen.call @mid<rank = rank, shape: list<index[rank]> = shape>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @mid
kgen.generator @mid<rank, shape: list<index[rank]>>() always_inline {
  kgen.param.declare another: list<index[rank]> = <shape>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent() {
  // CHECK: declare A = <2>
  // CHECK-NEXT: assert <eq(A, 1)>, "A == 1"
  kgen.call @callee<A = 2>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @callee
kgen.generator @callee<A>() always_inline constraints <[eq(A, 1), "A == 1"]> {
  kgen.return
}

// -----

#file = #debuginfo.file<"foo.c" in "/mlir/">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_C,
  file = #file,
  producer = "MLIR",
  isOptimized = true,
  emissionKind = Full
>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
  // COM: `debuginfo.value` has a parameter usage in its attributes.
> : !debuginfo.subroutine<(!debuginfo.unresolved<!kgen.paramref<T>>) -> (): DW_CC_normal>
#local_variable = #debuginfo.local_variable<
  scope = #subprogram,
  name = "foo",
  file = #file,
  line = 10,
  arg = 1
> : !debuginfo.unresolved<index>

// CHECK-LABEL: kgen.generator @parent
kgen.generator @parent<T: type>(%arg0: index) {
  // CHECK: kgen.param.declare T0: type = <index> loc(#[[CALL_LOC:.*]])
  // CHECK-NEXT: kgen.rebind %arg0 : index to !kgen.paramref<T0> loc(#[[CALL_LOC]])
  // CHECK-NEXT: kgen.return
  kgen.call @nodebug_inline_me<T: type = index>(%arg0) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @nodebug_inline_me
kgen.generator @nodebug_inline_me<T: type>(%arg0: !kgen.paramref<T>) always_inline_no_debug {
  debuginfo.value #local_variable = %arg0 : !kgen.paramref<T>
  kgen.return
}
