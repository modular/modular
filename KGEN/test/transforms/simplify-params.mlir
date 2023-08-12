// RUN: kgen-opt %s -split-input-file -mlir-print-debuginfo -allow-unregistered-dialect -verify-parameters=simplify=true -verify-parameters | FileCheck %s

kgen.generator @unbound_fn<p0>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_prop
kgen.generator @param_prop<p0, p1 -> p2>() {
  // CHECK-NEXT: declare a0 = <p0>
  kgen.param.declare a0 = <p0>
  // CHECK-NOT: declare a2 = <2>
  kgen.param.declare a2 = <2>
  // CHECK-NEXT: declare a1 = <add(p0, 1)>
  kgen.param.declare a1 = <add(p0, 1)>
  // CHECK-NEXT: simd<a0, si8>
  "user"() : () -> !pop.simd<a0, si8>
  // CHECK-NEXT: simd<a1, si8>
  "user"() : () -> !pop.simd<a1, si8>
  // CHECK-NEXT: simd<2, si8>
  "user"() : () -> !pop.simd<a2, si8>

  // CHECK-NEXT: F1 = <value: simd<2, si8>>() -> !pop.simd<2, si8>
  kgen.param.declare.region F1 = <value: simd<a2, si8>>() -> !pop.simd<a2, si8>{
    // CHECK-NEXT: declare b0 = <a1>
    kgen.param.declare b0 = <a1>
    // CHECK-NOT: declare b1
    kgen.param.declare b1 = <add(2, a2)>
    // CHECK-NEXT: simd<2, si8>
    "user"() : () -> !pop.simd<a2, si8>
    // CHECK-NEXT: simd<b0, si8>
    "user"() : () -> !pop.simd<b0, si8>
    // CHECK-NEXT: simd<4, si8>
    "user"() : () -> !pop.simd<b1, si8>
    // CHECK-NEXT: declare type_change0: simd<2, si8> = <value>
    kgen.param.declare type_change0: simd<a2, si8> = <value>
    // CHECK-NEXT: declare type_change1: simd<b0, si8> = <rebind(:simd<2, si8> value)>
    kgen.param.declare type_change1: simd<b0, si8> = <rebind(:simd<a2, si8> value)>
    // CHECK-MEXT: simd<2, si8> = <type_change0>
    %0 = kgen.param.constant: simd<a2, si8> = <type_change0>
    // CHECK-MEXT: simd<b0, si8> = <type_change1>
    %1 = kgen.param.constant: simd<b0, si8> = <type_change1>
    "user"(%1) : (!pop.simd<b0, si8>) -> ()
    kgen.return %0 : !pop.simd<a2, si8>
  }

  kgen.param.declare dt: dtype = <si8>
  // CHECK: F2 = <c0, c1:  simd<c0, si8>>(%arg0: !pop.simd<c0, si8> loc({{.*}}))
  kgen.param.declare.region F2 = <c0, c1: simd<c0, dt>>(%arg0: !pop.simd<c0, dt>) {
    // CHECK-NEXT: fork f0: simd<c0, si8> = <[c1]>
    kgen.param.fork f0: simd<c0, dt> = <[c1]>
    kgen.return
  }

  // CHECK: bound: () -> () = <[@unbound_fn<2>]>
  kgen.param.fork bound: () -> () = <[@unbound_fn<a2>]>

  // CHECK: result_bind<2>
  kgen.param.result_bind<a2>
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
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>

// CHECK-DAG: ![[CL_SP_TYPE:.*]] = !debuginfo.subroutine<(!pop.pointer<scalar<#pop.struct.extract<2, 1>>>) -> (): DW_CC_normal>
// CHECK-DAG: ![[OTHER_SP_TYPE:.*]] = !debuginfo.subroutine<(!pop.array<K, index>) -> (): DW_CC_normal>
// CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<{{.*}}, name = "foo", linkageName = "foo"
// CHECK-DAG: #[[CL_SP:.*]] = #debuginfo.subprogram<{{.*}}, name = "SomeClosure", linkageName = "SomeClosure", {{.*}}> : ![[CL_SP_TYPE]]
// CHECK-DAG: #[[OTHER_SP:.*]] = #debuginfo.subprogram<{{.*}}, name = "OtherClosure", linkageName = "OtherClosure", {{.*}}> : ![[OTHER_SP_TYPE]]
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "foo", linkageName = "foo", file = #file, line = 25, scopeLine = 25, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "SomeClosure", linkageName = "SomeClosure", file = #file, line = 183, scopeLine = 183, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<(!pop.pointer<scalar<#pop.struct.extract<N, 1>>>) -> (): DW_CC_normal>
#subprogram2 = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, name = "OtherClosure", linkageName = "OtherClosure", file = #file, line = 56, scopeLine = 56, subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<(!pop.array<K, index>) -> (): DW_CC_normal>

// CHECK-DAG: #[[LOC1:.*]] = loc("foo.mojo":25:1)
// CHECK-DAG: #[[LOC2:.*]] = loc("foo.mojo":183:5)
// CHECK-DAG: #[[LOC3:.*]] = loc("foo.mlir":56:5)
// CHECK-DAG: #[[LOC_FOO]] = loc(fused<#[[SP]]>[#[[LOC1]]])
// CHECK-DAG: #[[LOC_CL]] = loc(fused<#[[CL_SP]]>[#[[LOC2]]])
// CHECK-DAG: #[[LOC_OTHER]] = loc(fused<#[[OTHER_SP]]>[#[[LOC3]]])
#locFoo = loc(fused<#subprogram>["foo.mojo":25:1])
#locClosure = loc(fused<#subprogram1>["foo.mojo":183:5])
#locOther = loc(fused<#subprogram2>["foo.mlir":56:5])
