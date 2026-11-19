// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -o - | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -o - | FileCheck %s

// COM: Compilation should succeed.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64">} {
  kgen.generator @cast_index_to_si64_large() -> !kgen.scalar<si64> {
    %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
    %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<si64>
    hlcf.return %1 : !kgen.scalar<si64>
  }

  kgen.generator @cast_index_to_si32_large() -> !kgen.scalar<si32> {
    %0 = kgen.param.constant: scalar<index> = <-8664705627211539068>
    %1 = pop.cast %0 : !kgen.scalar<index> to !kgen.scalar<si32>
    hlcf.return %1 : !kgen.scalar<si32>
  }
  // CHECK-LABEL: @main
  kgen.generator export @main() {
    kgen.param.apply x = [() -> !kgen.scalar<si64> : @cast_index_to_si64_large]()
    kgen.param.apply y = [() -> !kgen.scalar<si32> : @cast_index_to_si32_large]()
    // CHECK-NEXT: kgen.param.constant: scalar<si64> = <-8664705627211539068>
    %0 = kgen.param.constant: !kgen.scalar<si64> = <x>
    // CHECK-NEXT:kgen.param.constant: scalar<si32> = <-1095082620>
    %1 = kgen.param.constant: !kgen.scalar<si32> = <y>

    // CHECK-NEXT: kgen.param.constant: simd<4, si64> = <9223372036854775807>
    %2 = kgen.param.constant: !kgen.simd<4, si64> = <#kgen.simd_splat<#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si64>>: !kgen.simd<4, si64>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui64> = <9223372036854775807>
    %3 = kgen.param.constant: !kgen.scalar<ui64> = <#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui64>>
    // CHECK-NEXT: kgen.param.constant: scalar<si64> = <2147483647>
    %4 = kgen.param.constant: !kgen.scalar<si64> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si64>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui64> = <2147483647>
    %5 = kgen.param.constant: !kgen.scalar<ui64> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui64>>
    // CHECK-NEXT: kgen.param.constant: scalar<si64> = <4294967295>
    %6 = kgen.param.constant: !kgen.scalar<si64> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si64>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui64> = <4294967295>
    %7 = kgen.param.constant: !kgen.scalar<ui64> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui64>>

    // CHECK-NEXT: kgen.param.constant: scalar<si32> = <-1>
    %8 = kgen.param.constant: !kgen.scalar<si32> = <#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si32>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui32> = <4294967295>
    %9 = kgen.param.constant: !kgen.scalar<ui32> = <#pop.cast<#kgen<simd 0x7FFFFFFFFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui32>>
    // CHECK-NEXT: kgen.param.constant: scalar<si32> = <2147483647>
    %10 = kgen.param.constant: !kgen.scalar<si32> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si32>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui32> = <2147483647>
    %11 = kgen.param.constant: !kgen.scalar<ui32> = <#pop.cast<#kgen<simd 0x7FFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui32>>
    // CHECK-NEXT: kgen.param.constant: scalar<si32> = <-1>
    %12 = kgen.param.constant: !kgen.scalar<si32> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<si32>>
    // CHECK-NEXT: kgen.param.constant: scalar<ui32> = <4294967295>
    %13 = kgen.param.constant: !kgen.scalar<ui32> = <#pop.cast<#kgen<simd 0xFFFFFFFF> : !kgen.scalar<index>>: !kgen.scalar<ui32>>

    // CHECK-NEXT: kgen.param.constant: scalar<f64> = <"9007199254740992">
    %14 = kgen.param.constant: !kgen.scalar<f64> = <#pop.cast<#kgen<simd 9007199254740992> : !kgen.scalar<index>> : !kgen.scalar<f64>>
    // CHECK-NEXT: kgen.param.constant: scalar<f32> = <"9.00719925E+15">
    %15 = kgen.param.constant: !kgen.scalar<f32> = <#pop.cast<#kgen<simd 9007199254740992> : !kgen.scalar<index>> : !kgen.scalar<f32>>

    // CHECK-NEXT: kgen.param.constant: scalar<si128> = <-9223372036854775808>
    %16 = kgen.param.constant : !kgen.scalar<si128> = <#pop.cast<#kgen<simd -9223372036854775808> : !kgen.scalar<index>> : !kgen.scalar<si128>>
    // CHECK-NEXT: kgen.param.constant: scalar<si128> = <-2147483648>
    %17 = kgen.param.constant : !kgen.scalar<si128> = <#pop.cast<#kgen<simd -2147483648> : !kgen.scalar<index>> : !kgen.scalar<si128>>

    // Casting to `f8e8m0fnu` takes the magnitude, rounds it up to a power of
    // two, and saturates the top binade rather than reaching the NaN encoding.
    // In order: NaN, 2**12, 2**1, 2**-127, 2**127.
    // CHECK-NEXT: kgen.param.constant: scalar<f8e8m0fnu> = <"NaN">
    %18 = kgen.param.constant : !kgen.scalar<f8e8m0fnu> = <#pop.cast<#kgen<simd "-Inf"> : !kgen.scalar<f32>> : !kgen.scalar<f8e8m0fnu>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e8m0fnu> = <"4.1E+3">
    %19 = kgen.param.constant : !kgen.scalar<f8e8m0fnu> = <#pop.cast<#kgen<simd "-4096.0"> : !kgen.scalar<f32>> : !kgen.scalar<f8e8m0fnu>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e8m0fnu> = <"2">
    %20 = kgen.param.constant : !kgen.scalar<f8e8m0fnu> = <#pop.cast<#kgen<simd "1.1"> : !kgen.scalar<f32>> : !kgen.scalar<f8e8m0fnu>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e8m0fnu> = <"5.9E-39">
    %21 = kgen.param.constant : !kgen.scalar<f8e8m0fnu> = <#pop.cast<#kgen<simd "0.0"> : !kgen.scalar<f32>> : !kgen.scalar<f8e8m0fnu>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e8m0fnu> = <"1.7E+38">
    %22 = kgen.param.constant : !kgen.scalar<f8e8m0fnu> = <#pop.cast<#kgen<simd "2.55E+38"> : !kgen.scalar<f32>> : !kgen.scalar<f8e8m0fnu>>

    // The other float8 types saturate: a folded cast must clamp an overflow,
    // and an infinity, to the largest finite value and keep the sign, which is
    // what the conversion these lower to does. `f8e4m3fn` and the `fnuz` pair
    // have no infinity, so converting without the clamp reaches their NaN
    // encoding instead.
    // CHECK-NEXT: kgen.param.constant: scalar<f8e4m3fn> = <"448">
    %23 = kgen.param.constant : !kgen.scalar<f8e4m3fn> = <#pop.cast<#kgen<simd "1.0E+30"> : !kgen.scalar<f32>> : !kgen.scalar<f8e4m3fn>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e4m3fn> = <"-448">
    %24 = kgen.param.constant : !kgen.scalar<f8e4m3fn> = <#pop.cast<#kgen<simd "-Inf"> : !kgen.scalar<f32>> : !kgen.scalar<f8e4m3fn>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e5m2> = <"-5.7E+4">
    %26 = kgen.param.constant : !kgen.scalar<f8e5m2> = <#pop.cast<#kgen<simd "-Inf"> : !kgen.scalar<f32>> : !kgen.scalar<f8e5m2>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e4m3fnuz> = <"240">
    %27 = kgen.param.constant : !kgen.scalar<f8e4m3fnuz> = <#pop.cast<#kgen<simd "1.0E+30"> : !kgen.scalar<f32>> : !kgen.scalar<f8e4m3fnuz>>
    // CHECK-NEXT: kgen.param.constant: scalar<f8e5m2fnuz> = <"5.7E+4">
    %28 = kgen.param.constant : !kgen.scalar<f8e5m2fnuz> = <#pop.cast<#kgen<simd "1.0E+30"> : !kgen.scalar<f32>> : !kgen.scalar<f8e5m2fnuz>>

    hlcf.return
  }
}
