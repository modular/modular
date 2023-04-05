// RUN: kgen-opt %s -allow-unregistered-dialect -split-input-file -verify-parameters | FileCheck %s

// CHECK-LABEL: kgen.generator @parameterIsolatedRegions
kgen.generator @parameterIsolatedRegions<A>() {
  // CHECK: kgen.param.declare.region
  kgen.param.declare.region Fn = <B>() {
    kgen.param.constant = <B>
    kgen.return
  }
  // CHECK: {isolated}

  // CHECK: kgen.param.if
  kgen.param.if <lt(A, 1)> {
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  // CHECK: {elseIsolated, thenIsolated}
  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @struct_of_simd
// CHECK-SAME: -> !pop.struct<simd<size, type>>
kgen.generator @struct_of_simd<size, type: dtype>(%arg0: !pop.simd<size, type>) -> !pop.struct<simd<size, type>> {
  %1 = pop.struct.create(%arg0) : !pop.struct<simd<size, type>>
  kgen.return %1 : !pop.struct<simd<size, type>>
}

// CHECK-LABEL: kgen.generator @call_it
kgen.generator @call_it<size, type: dtype, target: dtype>(%arg0: !pop.struct<simd<size, type>>) -> !pop.struct<simd<size, target>> {
  %1 = pop.struct.extract %arg0[0] : !pop.struct<simd<size, type>>
  %3 = pop.cast %1 : !pop.simd<size, type> to !pop.simd<size, target>
  // CHECK: kgen.call @struct_of_simd<size, :dtype target>
  // CHECK-SAME: (!pop.simd<size, target>) -> !pop.struct<simd<size, target>>
  %4 = kgen.call @struct_of_simd<size, :dtype target>(%3) : (!pop.simd<size, target>) -> !pop.struct<simd<size, target>>
  kgen.return %4 : !pop.struct<simd<size, target>>
}

// -----

lit.struct.decl @TakeArrayStruct<t, a: !pop.array<t, i1>> {}

kgen.generator @pass_index<t>(%arg0: index) -> !pop.array<t, i1> {
  %0 = "foo.op"() : () -> !pop.array<t, i1>
  kgen.return %0 : !pop.array<t, i1>
}

// CHECK-LABEL: kgen.generator @apply_result
// CHECK-SAME: @TakeArrayStruct<t = t, a: array<t, i1> = apply(:(index) -> !pop.array<t, i1> @pass_index<t>, t)>
kgen.generator @apply_result<t>(
  %arg0: !kgen.declref<@TakeArrayStruct<
    t = t,
    a: array<t, i1> = apply(:(index) -> !pop.array<t, i1> @pass_index<t>, t)
  >>
) {
  kgen.return
}

// -----

lit.struct.decl @Int {}

kgen.generator @make(%arg0: index) -> !kgen.declref<@Int> {
  kgen.unreachable
}

lit.struct.decl @List<l: @Int> {}

kgen.generator @create<l: @Int>() -> !kgen.declref<@List<l: @Int = l>> {
  kgen.unreachable
}

lit.struct.decl @Buf<r: @Int, s: @List<l: @Int = r>> {}

// CHECK-LABEL: kgen.generator @buffer
kgen.generator @buffer<rank>() ->
  !kgen.declref<@Buf<
    r: @Int = apply(:(index) -> !kgen.declref<@Int> @make, rank),
    s: @List<l: @Int = apply(:(index) -> !kgen.declref<@Int> @make, rank)> =
      apply(:() -> !kgen.declref<@List<l: @Int = apply(:(index) -> !kgen.declref<@Int> @make, rank)>>
              @create<:@Int apply(:(index) -> !kgen.declref<@Int> @make, rank)>)>> {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @ref_it
kgen.generator @ref_it() {
  // CHECK-NEXT: = apply(:() -> !kgen.declref<@List<l: @Int = apply(:(index) -> !kgen.declref<@Int> @make, *(1,0))
  kgen.param.declare fn: <index>() ->
     !kgen.declref<@Buf<
       r: @Int = apply(:(index) -> !kgen.declref<@Int> @make, *(0,0)),
       s: @List<l: @Int = apply(:(index) -> !kgen.declref<@Int> @make, *(0,0))> =
         apply(:() -> !kgen.declref<@List<l: @Int = apply(:(index) -> !kgen.declref<@Int> @make, *(1,0))>>
                 @create<:@Int apply(:(index) -> !kgen.declref<@Int> @make, *(0,0))>)>>
    = <@buffer>
  kgen.return
}
