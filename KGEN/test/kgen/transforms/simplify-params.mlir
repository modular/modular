// RUN: kgen-opt %s -split-input-file -mlir-print-debuginfo -allow-unregistered-dialect -verify-parameters=simplify=true -verify-parameters | FileCheck %s

kgen.generator @unbound_fn<p0>() {
  kgen.return
}

kgen.generator @callee_simd<size, dt: dtype, value: simd<size, dt>>() {
  kgen.return
}

kgen.generator @callee_fn<f: () -> ()>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_prop
kgen.generator @param_prop<p0, p1 -> p2>() {
  // CHECK-NOT: declare a0 = <p0>
  kgen.param.declare a0 = <p0>
  // CHECK-NOT: declare a2 = <2>
  kgen.param.declare a2 = <2>
  // CHECK-NEXT: declare a1 = <add(p0, 1)>
  kgen.param.declare a1 = <add(p0, 1)>
  // CHECK-NEXT: simd<p0, si8>
  "user"() : () -> !pop.simd<a0, si8>
  // CHECK-NEXT: simd<a1, si8>
  "user"() : () -> !pop.simd<a1, si8>
  // CHECK-NEXT: simd<2, si8>
  "user"() : () -> !pop.simd<a2, si8>

  // CHECK-NEXT: F1 = <value: simd<2, si8>>() -> !pop.simd<2, si8>
  kgen.param.declare.region F1 = <value: simd<a2, si8>>() -> !pop.simd<a2, si8>{
    // CHECK-NOT: declare b0 = <a1>
    kgen.param.declare b0 = <a1>
    // CHECK-NOT: declare b1
    kgen.param.declare b1 = <add(2, a2)>
    // CHECK-NEXT: simd<2, si8>
    "user"() : () -> !pop.simd<a2, si8>
    // CHECK-NEXT: simd<a1, si8>
    "user"() : () -> !pop.simd<b0, si8>
    // CHECK-NEXT: simd<4, si8>
    "user"() : () -> !pop.simd<b1, si8>
    // CHECK-NOT: declare type_change0: simd<2, si8> = <value>
    kgen.param.declare type_change0: simd<a2, si8> = <value>
    // CHECK-NEXT: declare type_change1: simd<a1, si8> = <rebind(:simd<2, si8> value)>
    kgen.param.declare type_change1: simd<b0, si8> = <rebind(:simd<a2, si8> value)>
    // CHECK-MEXT: simd<2, si8> = <type_change0>
    %0 = kgen.param.constant: simd<a2, si8> = <type_change0>
    // CHECK-MEXT: simd<a1, si8> = <type_change1>
    %1 = kgen.param.constant: simd<b0, si8> = <type_change1>
    "user"(%1) : (!pop.simd<b0, si8>) -> ()
    kgen.return %0 : !pop.simd<a2, si8>
  }

  kgen.param.declare dt: dtype = <si8>
  // CHECK: F2 = <c0, c1:  simd<c0, si8>>(%arg0: !pop.simd<c0, si8> loc({{.*}}))
  kgen.param.declare.region F2 = <c0, c1: simd<c0, dt>>(%arg0: !pop.simd<c0, dt>) {
    // CHECK-NEXT: <c0, :dtype si8, :simd<c0, si8> c1>
    kgen.call @callee_simd<c0, :dtype dt, :simd<c0, dt> c1>() : () -> ()
    kgen.return
  }

  // CHECK: :() -> () @unbound_fn<2>
  kgen.call @callee_fn<:() -> () @unbound_fn<a2>>() : () -> ()

  // CHECK: result_bind<2>
  kgen.param.result_bind<a2>
  kgen.return
}

kgen.generator @result_slot(%arg0: index, %arg1: !kgen.pointer<index> byref_result) -> !kgen.none {
  pop.store %arg0, %arg1 : !kgen.pointer<index>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: @interpret_result_slot
kgen.generator @interpret_result_slot() {
  // CHECK-NEXT: constant = <42>
  kgen.param.constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @result_slot, 42)>
  kgen.return
}

// -----

kgen.generator @interpret_me(%arg0: index) -> index {
  %0 = index.add %arg0, %arg0
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @simplify
kgen.generator @simplify() {
  kgen.param.declare x = <2>
  kgen.param.declare y = <apply(:(index) -> index @interpret_me, x)>
  // CHECK-NEXT: constant = <4>
  kgen.param.constant = <y>
  kgen.return
}

kgen.extern.generator @extern() -> index

kgen.generator @extern_apply() {
  kgen.param.constant = <apply(:() -> index @extern)>
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @foo
kgen.generator @foo() {
  kgen.param.declare N = <2> loc(#locFoo)
  // CHECK: kgen.param.declare.region SomeClosure
  kgen.param.declare.region SomeClosure = () capturing {
    // CHECK-NEXT: kgen.param.constant: array<1, i1> = <[1]> loc(#[[LOC_CL:.*]])
    %array = kgen.param.constant: array<1, i1> = <[1]> loc(#locClosure)
    // CHECK-NEXT: kgen.return loc(#[[LOC_CL]])
    kgen.return loc(#locClosure)
  // CHECK-NEXT: } {isolated} loc(#[[LOC_CL]])
  } loc(#locClosure)

  // CHECK: kgen.param.declare.region OtherClosure
  kgen.param.declare.region OtherClosure = <K>(%arg1: !pop.array<K, index>) {
    // CHECK-NEXT: kgen.return loc(#[[LOC_OTHER:.*]])
    kgen.return loc(#locOther)
  // CHECK-NEXT: } {isolated} loc(#[[LOC_OTHER]])
  } loc(#locOther)

  // CHECK-NEXT: kgen.return loc(#[[LOC_FOO:.*]])
  kgen.return loc(#locFoo)
// CHECK-NEXT: } loc(#[[LOC_FOO]])
} loc(#locFoo)

#file = #debuginfo.file<"foo.mojo" in "/">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>

// CHECK-DAG: ![[CL_SP_TYPE:.*]] = !debuginfo.subroutine<(!kgen.pointer<scalar<#kgen.struct.extract<2, 1>>>) -> (): DW_CC_normal>
// CHECK-DAG: ![[OTHER_SP_TYPE:.*]] = !debuginfo.subroutine<(!pop.array<K, index>) -> (): DW_CC_normal>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<name = <"foo">
// CHECK-DAG: #[[CL_SP:.*]] = #debuginfo.subprogram<name = <"SomeClosure">
// CHECK-DAG: #[[OTHER_SP:.*]] = #debuginfo.subprogram<name = <"OtherClosure">
#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<name = <"SomeClosure">> : !debuginfo.subroutine<(!kgen.pointer<scalar<#kgen.struct.extract<N, 1>>>) -> (): DW_CC_normal>
#subprogram2 = #debuginfo.subprogram<name = <"OtherClosure">> : !debuginfo.subroutine<(!pop.array<K, index>) -> (): DW_CC_normal>

// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mojo":25:1)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mojo":183:5)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mlir":56:5)
// CHECK-DAG: #[[LOC_FOO]] = loc(fused<#[[SP]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CL]] = loc(fused<#[[CL_SP]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_OTHER]] = loc(fused<#[[OTHER_SP]]>[#[[LOC3]]])
#locFoo = loc(fused<#subprogram>["foo.mojo":25:1])
#locClosure = loc(fused<#subprogram1>["foo.mojo":183:5])
#locOther = loc(fused<#subprogram2>["foo.mlir":56:5])

// -----

// CHECK-LABEL: kgen.generator @func
kgen.generator @func() {
  kgen.param.declare rank = <2> loc(#loc3)
  kgen.param.declare.region region = () {
    hlcf.loop {
      // CHECK: hlcf.break loc([[BREAK_LOC:#.*]])
      hlcf.break loc(#loc4)
    } loc(#loc4)
    kgen.return loc(#loc4)
  } loc(#loc4)
  kgen.return loc(#loc3)
} loc(#loc3)

#subprogram = #debuginfo.subprogram<name = <"test_stencil_avg_pool">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<name = <"map_fn">> : !debuginfo.subroutine<(!pop.simd<rank, f32>) -> (): DW_CC_normal>
// CHECK: [[SR_TYPE:!.*]] = !debuginfo.subroutine<(!pop.simd<2, f32>) -> (): DW_CC_normal>
// CHECK: [[SP:#.*]] = #debuginfo.subprogram<{{.*}}> : [[SR_TYPE]]
// CHECK: [[BREAK_LOC]] = loc(fused<[[SP]]>
#loc3 = loc(fused<#subprogram>["a":0:0])
#loc4 = loc(fused<#subprogram1>["a":0:0])
