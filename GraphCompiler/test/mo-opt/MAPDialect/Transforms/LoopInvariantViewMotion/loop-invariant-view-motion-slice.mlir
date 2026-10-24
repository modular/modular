// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map-livm, map.iter.opaque(cse,canonicalize)))' | FileCheck %s

// LoopInvariantViewMotion hoists a map.index.slice chain feeding a map.load
// out of the loop body into a map.index.slice.stride op in the enclosing
// map.iter.opaque, rewriting the load to use the loop's own (untransformed)
// index directly. Covers: static and symbolic-dim shapes, a load with no
// transform at all (must be left untouched), an arbitrary number of
// independent views/loads in one body, a load mixed with an untransformed
// one, two loads sharing the exact same hoisted view (dedup), a chain of two
// index-transform hops feeding a single load, and a load whose target is a
// buffer.
//
// The pass itself hoists every load unconditionally, without deduping (see
// LoopInvariantViewMotion.cpp) -- two loads sharing one chain each get their
// own capture of a structurally-identical stride op. Running `cse` *before*
// `canonicalize` -- the reverse of this file's usual canonicalize-then-cse
// order, and the same reversal GraphCompiler.cpp applies right after
// `MAP::createLoopInvariantViewMotion()` in `buildMapPipeline` -- unifies
// the two identical stride ops into one shared value first, so
// `canonicalize`'s `DedupeCapturingClosureCaptures` can then fold the
// resulting duplicate capture down to one.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

// map.call.elementwise verifies its callee resolves to a mogg.kernel.decl
// with an `elementwise` function, so this stub must be present.
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// A single view feeding a single load: the common case.

// CHECK-LABEL: mo.graph @sliceStatic
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT:%[^ :,]+]]: !mo.tensor<[10, 64], f32>, [[STARTS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STOPS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STEPS:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 32], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> ([[INPUT]], [[STARTS]], [[STOPS]], [[STEPS]])
// CHECK-NEXT:      map.iter.elementwise.body<[3, 32]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}}[[OUT0:%[^ ,]+]] = [[OUT]] : !mo.tensor<[3, 32], f32>, [[NEWCAP:%[^ ]+]] = [[STRIDED]] : !mo.tensor<[3, 32], f32>} {
// CHECK-NOT:       map.index.slice
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[NEWCAP]]{{\[}}[[IDX]]] : !mo.tensor<[3, 32], f32>
// CHECK-NEXT:      map.store [[OUT0]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 32], f32>
// CHECK-NEXT:    }
mo.graph @sliceStatic(%input: !mo.tensor<[10, 64], f32>,
                      %starts: !mo.tensor<[2], si64>,
                      %stops: !mo.tensor<[2], si64>,
                      %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[10, 64], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 32], f32>) {
    map.iter.elementwise.body<[3, 32]>
        index (%idx: !map.index<2>)
        capturing {%input0 = %input : !mo.tensor<[10, 64], f32>, %starts0 = %starts : !mo.tensor<[2], si64>, %stops0 = %stops : !mo.tensor<[2], si64>, %steps0 = %steps : !mo.tensor<[2], si64>, %out0 = %out : !mo.tensor<[3, 32], f32>} {
      %idx_ = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%idx, %starts0, %stops0, %steps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %input0[%idx_] : !mo.tensor<[10, 64], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[3, 32], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[3, 32], f32>
}

// Symbolic-dim shapes and dynamic (`?`) bounds: the hoisting logic doesn't
// care whether the spec is static or dynamic.

// CHECK-LABEL: mo.graph @sliceDynamic
// CHECK:         map.iter.opaque
// CHECK:           map.index.slice.stride<#map.index.slice.spec<[N, 64] -> [M, 32], starts = [?, ?], stops = [?, ?], steps = [?, ?]>>
// CHECK:           map.iter.elementwise.body
// CHECK-NOT:         map.index.slice
mo.graph @sliceDynamic<N, M>(
    %input: !mo.tensor<[N, 64], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[M, 32], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[N, 64], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[M, 32], f32>) {
    map.iter.elementwise.body<[M, 32]>
        index (%idx: !map.index<2>)
        capturing {%input0 = %input : !mo.tensor<[N, 64], f32>, %starts0 = %starts : !mo.tensor<[2], si64>, %stops0 = %stops : !mo.tensor<[2], si64>, %steps0 = %steps : !mo.tensor<[2], si64>, %out0 = %out : !mo.tensor<[M, 32], f32>} {
      %idx_ = map.index.slice<#map.index.slice.spec<[N, 64] -> [M, 32], starts = [?, ?], stops = [?, ?], steps = [?, ?]>> (%idx, %starts0, %stops0, %steps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %input0[%idx_] : !mo.tensor<[N, 64], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[M, 32], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[M, 32], f32>
}

// A load with no index-transform at all (already indexes with the loop's own
// index, as an elementwise op's loads would): left completely untouched, no
// map.index.slice.stride is introduced.

// CHECK-LABEL: mo.graph @noTransform
// CHECK:         [[RESULT3:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X3:%[^ :,]+]]: !mo.tensor<[8, 8], f32>)
// CHECK-NEXT:    outputs ([[OUT3:%[^ :,]+]] : !mo.tensor<[8, 8], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[8, 8]>
// CHECK-NEXT:        index   ([[IDX3:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {[[X30:%[^ ,]+]] = [[X3]] : !mo.tensor<[8, 8], f32>, [[OUT30:%[^ ,]+]] = [[OUT3]] : !mo.tensor<[8, 8], f32>} {
// CHECK-NEXT:      [[VAL3:%[^ ]+]] = map.load [[X30]]{{\[}}[[IDX3]]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:      map.store [[OUT30]]{{\[}}[[IDX3]]] = [[VAL3]] : !mo.tensor<[8, 8], f32>
// CHECK-NEXT:    }
// CHECK-NOT:   map.index.slice.stride
mo.graph @noTransform(%x: !mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %output = map.iter.opaque
      inputs (%x : !mo.tensor<[8, 8], f32>)
      outputs (%out : !mo.tensor<[8, 8], f32>) {
    map.iter.elementwise.body<[8, 8]>
        index (%idx: !map.index<2>)
        capturing {%x0 = %x : !mo.tensor<[8, 8], f32>, %out0 = %out : !mo.tensor<[8, 8], f32>} {
      %val = map.load %x0[%idx] : !mo.tensor<[8, 8], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 8], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 8], f32>
}

// Two independent views feeding two independent loads in the same body
// (arbitrary number of views / loads): both get hoisted, each to its own
// map.index.slice.stride, each rewritten independently.

// CHECK-LABEL: mo.graph @twoIndependentViews
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs
// CHECK-NEXT:    outputs ([[OUT4:%[^ :,]+]] : !mo.tensor<[3, 3], f32>) {
// CHECK-NEXT:      [[STRIDEDA:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>>
// CHECK-NEXT:      [[STRIDEDB:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[20, 20] -> [3, 3], starts = [5, 5], stops = [8, 8], steps = [1, 1]>>
// CHECK-NEXT:      map.iter.elementwise.body<[3, 3]>
// CHECK-NEXT:        index   ([[IDX4:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}}[[OUT40:%[^ ,]+]] = [[OUT4]] : !mo.tensor<[3, 3], f32>, [[NEWCAPA:%[^ ,]+]] = [[STRIDEDA]] : !mo.tensor<[3, 3], f32>, [[NEWCAPB:%[^ ]+]] = [[STRIDEDB]] : !mo.tensor<[3, 3], f32>} {
// CHECK-NOT:       map.index.slice
// CHECK:           [[AV:%[^ ]+]] = map.load [[NEWCAPA]]{{\[}}[[IDX4]]]
// CHECK:           [[BV:%[^ ]+]] = map.load [[NEWCAPB]]{{\[}}[[IDX4]]]
// CHECK:           [[SUM:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Add([[IDX4]], [[AV]], [[BV]])
// CHECK:           map.store [[OUT40]]{{\[}}[[IDX4]]] = [[SUM]]
mo.graph @twoIndependentViews(
    %a: !mo.tensor<[10, 10], f32>,
    %aStarts: !mo.tensor<[2], si64>,
    %aStops: !mo.tensor<[2], si64>,
    %aSteps: !mo.tensor<[2], si64>,
    %b: !mo.tensor<[20, 20], f32>,
    %bStarts: !mo.tensor<[2], si64>,
    %bStops: !mo.tensor<[2], si64>,
    %bSteps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%a : !mo.tensor<[10, 10], f32>, %aStarts : !mo.tensor<[2], si64>, %aStops : !mo.tensor<[2], si64>, %aSteps : !mo.tensor<[2], si64>,
              %b : !mo.tensor<[20, 20], f32>, %bStarts : !mo.tensor<[2], si64>, %bStops : !mo.tensor<[2], si64>, %bSteps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.elementwise.body<[3, 3]>
        index (%idx: !map.index<2>)
        capturing {%a0 = %a : !mo.tensor<[10, 10], f32>, %aStarts0 = %aStarts : !mo.tensor<[2], si64>, %aStops0 = %aStops : !mo.tensor<[2], si64>, %aSteps0 = %aSteps : !mo.tensor<[2], si64>,
                   %b0 = %b : !mo.tensor<[20, 20], f32>, %bStarts0 = %bStarts : !mo.tensor<[2], si64>, %bStops0 = %bStops : !mo.tensor<[2], si64>, %bSteps0 = %bSteps : !mo.tensor<[2], si64>,
                   %out0 = %out : !mo.tensor<[3, 3], f32>} {
      %aIdx = map.index.slice<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %aStarts0, %aStops0, %aSteps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %aVal = map.load %a0[%aIdx] : !mo.tensor<[10, 10], f32>
      %bIdx = map.index.slice<#map.index.slice.spec<[20, 20] -> [3, 3], starts = [5, 5], stops = [8, 8], steps = [1, 1]>> (%idx, %bStarts0, %bStops0, %bSteps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %bVal = map.load %b0[%bIdx] : !mo.tensor<[20, 20], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %aVal, %bVal)
                : !map.index<2>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[3, 3], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[3, 3], f32>
}

// A view mixed with a plain (untransformed) load: only the view's load is
// rewritten; the other is untouched.

// CHECK-LABEL: mo.graph @mixedTransformedAndPlain
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X5:%[^ :,]+]]: !mo.tensor<[3, 3], f32>, [[A5:%[^ :,]+]]: !mo.tensor<[10, 10], f32>,
// CHECK:         [[STRIDED5:%[^ ]+]] = map.index.slice.stride
// CHECK-NEXT:    map.iter.elementwise.body<[3, 3]>
// CHECK-NEXT:      index   ([[IDX5:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:      capturing {[[X50:%[^ ,]+]] = [[X5]] : !mo.tensor<[3, 3], f32>, {{.*}}[[NEWCAP5:%[^ ]+]] = [[STRIDED5]] : !mo.tensor<[3, 3], f32>} {
// CHECK-NOT:     map.index.slice
// CHECK:         [[XV5:%[^ ]+]] = map.load [[X50]]{{\[}}[[IDX5]]] : !mo.tensor<[3, 3], f32>
// CHECK:         [[AV5:%[^ ]+]] = map.load [[NEWCAP5]]{{\[}}[[IDX5]]]
mo.graph @mixedTransformedAndPlain(
    %x: !mo.tensor<[3, 3], f32>,
    %a: !mo.tensor<[10, 10], f32>,
    %aStarts: !mo.tensor<[2], si64>,
    %aStops: !mo.tensor<[2], si64>,
    %aSteps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%x : !mo.tensor<[3, 3], f32>, %a : !mo.tensor<[10, 10], f32>, %aStarts : !mo.tensor<[2], si64>, %aStops : !mo.tensor<[2], si64>, %aSteps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.elementwise.body<[3, 3]>
        index (%idx: !map.index<2>)
        capturing {%x0 = %x : !mo.tensor<[3, 3], f32>, %a0 = %a : !mo.tensor<[10, 10], f32>, %aStarts0 = %aStarts : !mo.tensor<[2], si64>, %aStops0 = %aStops : !mo.tensor<[2], si64>, %aSteps0 = %aSteps : !mo.tensor<[2], si64>, %out0 = %out : !mo.tensor<[3, 3], f32>} {
      %xVal = map.load %x0[%idx] : !mo.tensor<[3, 3], f32>
      %aIdx = map.index.slice<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %aStarts0, %aStops0, %aSteps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %aVal = map.load %a0[%aIdx] : !mo.tensor<[10, 10], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xVal, %aVal)
                : !map.index<2>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[3, 3], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[3, 3], f32>
}

// Two loads of the *same* tensor sharing the exact same map.index.slice SSA
// value: the pass itself hoists each independently (no dedup cache), so each
// gets its own capture of a structurally-identical stride op. Running `cse`
// *before* `canonicalize` (GraphCompiler.cpp swaps the usual
// canonicalize-then-cse order right after LoopInvariantViewMotion for
// exactly this reason) unifies the two identical stride ops into one shared
// SSA value first, so by the time `DedupeCapturingClosureCaptures` runs, the
// two captures already alias the same value and get folded into one -- both
// loads end up reading the one map.index.slice.stride SSA value through a
// single shared capture.

// CHECK-LABEL: mo.graph @sharedChain
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs
// CHECK-NEXT:    outputs ([[OUT6:%[^ :,]+]] : !mo.tensor<[3, 3], f32>) {
// CHECK-NEXT:      [[STRIDED6:%[^ ]+]] = map.index.slice.stride
// CHECK-NEXT:      map.iter.elementwise.body<[3, 3]>
// CHECK-NEXT:        index   ([[IDX6:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}}[[NEWCAP6:%[^ ]+]] = [[STRIDED6]] : !mo.tensor<[3, 3], f32>} {
// CHECK-NOT:       map.index.slice
// CHECK-NOT:       map.index.slice.stride
// CHECK:           [[V0:%[^ ]+]] = map.load [[NEWCAP6]]{{\[}}[[IDX6]]]
// CHECK:           [[V1:%[^ ]+]] = map.load [[NEWCAP6]]{{\[}}[[IDX6]]]
mo.graph @sharedChain(
    %a: !mo.tensor<[10, 10], f32>,
    %aStarts: !mo.tensor<[2], si64>,
    %aStops: !mo.tensor<[2], si64>,
    %aSteps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%a : !mo.tensor<[10, 10], f32>, %aStarts : !mo.tensor<[2], si64>, %aStops : !mo.tensor<[2], si64>, %aSteps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.elementwise.body<[3, 3]>
        index (%idx: !map.index<2>)
        capturing {%a0 = %a : !mo.tensor<[10, 10], f32>, %aStarts0 = %aStarts : !mo.tensor<[2], si64>, %aStops0 = %aStops : !mo.tensor<[2], si64>, %aSteps0 = %aSteps : !mo.tensor<[2], si64>, %out0 = %out : !mo.tensor<[3, 3], f32>} {
      %aIdx = map.index.slice<#map.index.slice.spec<[10, 10] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %aStarts0, %aStops0, %aSteps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %v0 = map.load %a0[%aIdx] : !mo.tensor<[10, 10], f32>
      %v1 = map.load %a0[%aIdx] : !mo.tensor<[10, 10], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %v0, %v1)
                : !map.index<2>, f32, f32 -> f32
      map.store %out0[%idx] = %sum : !mo.tensor<[3, 3], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[3, 3], f32>
}

// A chain of two index-transform hops (a view feeding another view's index):
// both hops get hoisted into their own map.index.slice.stride, composed
// hop-by-hop starting from the original tensor, and the load addresses the
// fully-composed result directly with the outermost hop's own
// (untransformed) index.

// CHECK-LABEL: mo.graph @chainedSliceIndex
// CHECK:         [[RESULT7:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT7:%[^ :,]+]]: !mo.tensor<[10, 10], f32>, [[STARTS7:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STOPS7:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STEPS7:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUT7:%[^ :,]+]] : !mo.tensor<[3, 3], f32>) {
// CHECK-NEXT:      [[INNER7:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 10] -> [6, 6], starts = [0, 0], stops = [6, 6], steps = [1, 1]>> ([[INPUT7]], [[STARTS7]], [[STOPS7]], [[STEPS7]])
// CHECK-NEXT:      [[OUTER7:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[6, 6] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> ([[INNER7]], [[STARTS7]], [[STOPS7]], [[STEPS7]])
// CHECK-NEXT:      map.iter.elementwise.body<[3, 3]>
// CHECK-NEXT:        index   ([[IDX7:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}}[[NEWCAP7:%[^ ]+]] = [[OUTER7]] : !mo.tensor<[3, 3], f32>} {
// CHECK-NOT:       map.index.slice
// CHECK-NEXT:      [[VAL7:%[^ ]+]] = map.load [[NEWCAP7]]{{\[}}[[IDX7]]] : !mo.tensor<[3, 3], f32>
// CHECK-NEXT:      map.store {{%[^ ,]+}}{{\[}}[[IDX7]]] = [[VAL7]] : !mo.tensor<[3, 3], f32>
// CHECK-NEXT:    }
mo.graph @chainedSliceIndex(
    %input: !mo.tensor<[10, 10], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 3], f32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[10, 10], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 3], f32>) {
    map.iter.elementwise.body<[3, 3]>
        index (%idx: !map.index<2>)
        capturing {%input0 = %input : !mo.tensor<[10, 10], f32>, %starts0 = %starts : !mo.tensor<[2], si64>, %stops0 = %stops : !mo.tensor<[2], si64>, %steps0 = %steps : !mo.tensor<[2], si64>, %out0 = %out : !mo.tensor<[3, 3], f32>} {
      %mid = map.index.slice<#map.index.slice.spec<[6, 6] -> [3, 3], starts = [0, 0], stops = [3, 3], steps = [1, 1]>> (%idx, %starts0, %stops0, %steps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %inner = map.index.slice<#map.index.slice.spec<[10, 10] -> [6, 6], starts = [0, 0], stops = [6, 6], steps = [1, 1]>> (%mid, %starts0, %stops0, %steps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %input0[%inner] : !mo.tensor<[10, 10], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[3, 3], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[3, 3], f32>
}


// The load target is an in-place kernel's fused buffer (`FuseMutableLoads`
// retypes the block argument a `mo.mutable.load` used to feed), so the
// hoisted view reads a `!mo.buffer` and hands back a `!mo.tensor`: the view
// is only ever loaded from.

// CHECK-LABEL: mo.graph @sliceOfFusedBuffer
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[BUF8:%[^ :,]+]]: !mo.buffer<[10, 64], f32>,
// CHECK:           [[STRIDED8:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> ([[BUF8]],
// CHECK-SAME:        (!mo.buffer<[10, 64], f32>, {{.*}}) -> !mo.tensor<[3, 32], f32>
// CHECK-NEXT:      map.iter.elementwise.body<[3, 32]>
// CHECK-NEXT:        index   ([[IDX8:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {{{.*}}[[NEWCAP8:%[^ ]+]] = [[STRIDED8]] : !mo.tensor<[3, 32], f32>} {
// CHECK-NOT:       map.index.slice
// CHECK-NEXT:      [[VAL8:%[^ ]+]] = map.load [[NEWCAP8]]{{\[}}[[IDX8]]] : !mo.tensor<[3, 32], f32>
mo.graph @sliceOfFusedBuffer(%buf: !mo.buffer<[10, 64], f32>,
                             %starts: !mo.tensor<[2], si64>,
                             %stops: !mo.tensor<[2], si64>,
                             %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %output = map.iter.opaque
      inputs (%buf : !mo.buffer<[10, 64], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 32], f32>) {
    map.iter.elementwise.body<[3, 32]>
        index (%idx: !map.index<2>)
        capturing {%buf0 = %buf : !mo.buffer<[10, 64], f32>, %starts0 = %starts : !mo.tensor<[2], si64>, %stops0 = %stops : !mo.tensor<[2], si64>, %steps0 = %steps : !mo.tensor<[2], si64>, %out0 = %out : !mo.tensor<[3, 32], f32>} {
      %idx_ = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%idx, %starts0, %stops0, %steps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %val = map.load %buf0[%idx_] : !mo.buffer<[10, 64], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[3, 32], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[3, 32], f32>
}

}
