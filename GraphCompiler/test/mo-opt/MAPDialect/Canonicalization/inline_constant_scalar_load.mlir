// RUN: mo-opt %s --canonicalize | FileCheck %s

// A load whose source is a one-element compile-time constant becomes a
// `map.constant.scalar` literal in the closure body, and the constant stops
// being a kernel operand at all: the capture and the `map.iter.opaque` input
// are shed by the capturing-closure / opaque-input canonicalizers, and the
// dead view and `mo.constant` by DCE. MAP-lane equivalent of
// `MOGGDialect/Transforms/fusion-inline-broadcast-constant-scalars.mlir`.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// `map.call.elementwise` verifies its callee resolves to a `mogg.kernel.decl`
// with an `elementwise` function, so this stub must be present.
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// The shape the real pipeline produces: the broadcast is still on the INDEX
// (this canonicalizer runs before LoopInvariantViewMotion hoists it into a
// stride view), so the pattern never sees a view at all -- it keys off the
// source having one element, which makes the index irrelevant.

// CHECK-LABEL: mo.graph @broadcastIndexOfConstant
// CHECK-NOT:   mo.constant
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ()
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[8, 64], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[8, 64]>
// CHECK-NEXT:        index   ([[IDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {[[OUT0:%[^ ,]+]] = [[OUT]] : !mo.tensor<[8, 64], f32>} {
// CHECK-NEXT:      [[CST:%[^ ]+]] = map.constant.scalar 2.000000e+00 : f32 : f32
// CHECK-NOT:       map.load
// CHECK:           map.store [[OUT0]]{{\[}}[[IDX]]] = [[CST]] : !mo.tensor<[8, 64], f32>
mo.graph @broadcastIndexOfConstant() -> !mo.tensor<[8, 64], f32> {
  %cst = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %output = map.iter.opaque
      inputs (%cst : !mo.tensor<[], f32>)
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    map.iter.elementwise.body<[8, 64]>
        index (%idx: !map.index<2>)
        capturing {%cst0 = %cst : !mo.tensor<[], f32>, %out0 = %out : !mo.tensor<[8, 64], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [8, 64]>> (%idx)
          : !map.index<2> -> !map.index<0>
      %val = map.load %cst0[%idx_] : !mo.tensor<[], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 64], f32>
}

// The other spelling of the same thing, and the one the host pipeline
// actually produces: `FuseConstantIterOpaqueInputs` has already rewritten the
// operand into a `map.constant` in the body, so there is no `mo.constant` left
// to walk out to. Only a device kernel, which that pattern refuses to fold
// into (GEX-3839), still arrives in the form above.

// CHECK-LABEL: mo.graph @foldedMapConstant
// CHECK-NOT:   map.constant
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ()
// CHECK-NEXT:    outputs ([[FOUT:%[^ :,]+]] : !mo.tensor<[8, 64], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[8, 64]>
// CHECK-NEXT:        index   ([[FIDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {[[FOUT0:%[^ ,]+]] = [[FOUT]] : !mo.tensor<[8, 64], f32>} {
// CHECK-NEXT:      [[FCST:%[^ ]+]] = map.constant.scalar 6.250000e-02 : f32 : f32
// CHECK-NOT:       map.load
// CHECK:           map.store [[FOUT0]]{{\[}}[[FIDX]]] = [[FCST]] : !mo.tensor<[8, 64], f32>
mo.graph @foldedMapConstant() -> !mo.tensor<[8, 64], f32> {
  %output = map.iter.opaque
      inputs ()
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    %cst = map.constant #M.dense_array<6.250000e-02> : tensor<f32> : !mo.tensor<[], f32>
    map.iter.elementwise.body<[8, 64]>
        index (%idx: !map.index<2>)
        capturing {%cst0 = %cst : !mo.tensor<[], f32>, %out0 = %out : !mo.tensor<[8, 64], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [8, 64]>> (%idx)
          : !map.index<2> -> !map.index<0>
      %val = map.load %cst0[%idx_] : !mo.tensor<[], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 64], f32>
}

// A chained kernel. The chain shadows a block argument of its own, trailing
// the inputs, so the walk outward from a load has to stop at the `$inputs`
// prefix rather than indexing `$inputs` by raw argument number. The constant
// still inlines, and `DropDeadIterOpaqueInputs` keeps the chain (it sequences
// the kernel at the graph level even with nothing in the body consuming it)
// while the constant operand goes.

// CHECK-LABEL: mo.graph @chainedKernel
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[CH:%[^ :,]+]]: !mo.chain)
// CHECK-NEXT:    outputs ([[COUT:%[^ :,]+]] : !mo.tensor<[8, 64], f32>) {
// CHECK-NEXT:      map.iter.elementwise.body<[8, 64]>
// CHECK-NEXT:        index   ([[CIDX:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing {[[COUT0:%[^ ,]+]] = [[COUT]] : !mo.tensor<[8, 64], f32>} {
// CHECK-NEXT:      [[CCST:%[^ ]+]] = map.constant.scalar 2.000000e+00 : f32 : f32
// CHECK-NOT:       map.load
// CHECK:           map.store [[COUT0]]{{\[}}[[CIDX]]] = [[CCST]] : !mo.tensor<[8, 64], f32>
mo.graph @chainedKernel(%chIn: !mo.chain) -> !mo.tensor<[8, 64], f32> {
  %cst = mo.constant <#M.dense_array<2.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %output = map.iter.opaque
      inputs (%cst : !mo.tensor<[], f32>, %chIn : !mo.chain)
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    map.iter.elementwise.body<[8, 64]>
        index (%idx: !map.index<2>)
        capturing {%cst0 = %cst : !mo.tensor<[], f32>, %out0 = %out : !mo.tensor<[8, 64], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [8, 64]>> (%idx)
          : !map.index<2> -> !map.index<0>
      %val = map.load %cst0[%idx_] : !mo.tensor<[], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 64], f32>
}

// The already-hoisted spelling: a stride view of the constant, addressed by
// the loop's own index. A view of a one-element tensor is still that one
// element, so the view is walked through and dropped.

// CHECK-LABEL: mo.graph @broadcastStrideOfConstant
// CHECK-NOT:   mo.constant
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ()
// CHECK-NOT:     map.index.broadcast.stride
// CHECK:           map.constant.scalar 3.500000e+00 : f32 : f32
// CHECK-NOT:       map.load
mo.graph @broadcastStrideOfConstant() -> !mo.tensor<[8, 64], f32> {
  %cst = mo.constant <#M.dense_array<3.500000e+00> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %output = map.iter.opaque
      inputs (%cst : !mo.tensor<[1], f32>)
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    %strided = map.index.broadcast.stride<#map.index.broadcast.spec<[1] -> [8, 64]>> (%cst)
        : !mo.tensor<[1], f32> -> !mo.tensor<[8, 64], f32>
    map.iter.elementwise.body<[8, 64]>
        index (%idx: !map.index<2>)
        capturing {%out0 = %out : !mo.tensor<[8, 64], f32>, %strided0 = %strided : !mo.tensor<[8, 64], f32>} {
      %val = map.load %strided0[%idx] : !mo.tensor<[8, 64], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 64], f32>
}

// An integer constant inlines the same way; only float and integer scalars
// have a literal to inline.

// CHECK-LABEL: mo.graph @integerConstant
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ()
// CHECK:           map.constant.scalar 7 : si32 : si32
mo.graph @integerConstant() -> !mo.tensor<[6], si32> {
  %cst = mo.constant <#M.dense_array<7> : tensor<si32>> : !mo.tensor<[], si32>
  %output = map.iter.opaque
      inputs (%cst : !mo.tensor<[], si32>)
      outputs (%out : !mo.tensor<[6], si32>) {
    map.iter.elementwise.body<[6]>
        index (%idx: !map.index<1>)
        capturing {%cst0 = %cst : !mo.tensor<[], si32>, %out0 = %out : !mo.tensor<[6], si32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [6]>> (%idx)
          : !map.index<1> -> !map.index<0>
      %val = map.load %cst0[%idx_] : !mo.tensor<[], si32>
      map.store %out0[%idx] = %val : !mo.tensor<[6], si32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[6], si32>
}

// `ops.constant(...).to(device)` reaches the kernel through an `mo.transfer`,
// which the constant-identity chain does not see through. Inlining a literal
// creates nothing device-resident, so the transfer is walked past and the
// device operand disappears with it.

// CHECK-LABEL: mo.graph @transferredConstant
// CHECK-NOT:   mo.transfer
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ()
// CHECK:           map.constant.scalar 2.500000e-01 : f32 : f32
mo.graph @transferredConstant() -> !mo.tensor<[4], f32, gpu:0> {
  %cst = mo.constant <#M.dense_array<2.500000e-01> : tensor<f32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], f32>
  %ch0 = mo.chain.create()
  %gpu_cst, %ch1 = mo.transfer[%ch0] %cst : !mo.tensor<[], f32> to <"gpu", 0>
  %output = map.iter.opaque
      inputs (%gpu_cst : !mo.tensor<[], f32, gpu:0>)
      outputs (%out : !mo.tensor<[4], f32, gpu:0>) {
    map.iter.elementwise.body<[4]>
        index (%idx: !map.index<1>)
        capturing {%cst0 = %gpu_cst : !mo.tensor<[], f32, gpu:0>, %out0 = %out : !mo.tensor<[4], f32, gpu:0>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [4]>> (%idx)
          : !map.index<1> -> !map.index<0>
      %val = map.load %cst0[%idx_] : !mo.tensor<[], f32, gpu:0>
      map.store %out0[%idx] = %val : !mo.tensor<[4], f32, gpu:0>
    }
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %output : !mo.tensor<[4], f32, gpu:0>
}

// A load inside a prologue reaches the constant through two capture
// boundaries (the prologue's own, then the opaque's inputs), which the
// outward walk crosses generically over `CapturingFusionClosure`.

// CHECK-LABEL: mo.graph @prologueCapturedConstant
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ()
// CHECK:           map.iter.prologue (%{{[^ :]+}} : !map.index<1>) capturing {} {
// CHECK-NEXT:        [[CST:%[^ ]+]] = map.constant.scalar 5.000000e-01 : f32 : f32
// CHECK-NEXT:        map.yield [[CST]]
mo.graph @prologueCapturedConstant() -> !mo.tensor<[4], f32> {
  %cst = mo.constant <#M.dense_array<5.000000e-01> : tensor<f32>> : !mo.tensor<[], f32>
  %output = map.iter.opaque
      inputs (%cst : !mo.tensor<[], f32>)
      outputs (%out : !mo.tensor<[4], f32>) {
    %pro = map.iter.prologue (%pidx : !map.index<1>) capturing {%pcap = %cst : !mo.tensor<[], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [4]>> (%pidx)
          : !map.index<1> -> !map.index<0>
      %pval = map.load %pcap[%idx_] : !mo.tensor<[], f32>
      map.yield %pval : f32
    } : !mo.tensor<[4], f32>
    map.call.opaque @kernels.relu[] inputs (%pro) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[4], f32>
}

// Same for an epilogue, whose captures sit behind its leading value/index
// block args -- the load-bearing offset in the outward walk.

// CHECK-LABEL: mo.graph @epilogueCapturedConstant
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X:%[^ :,]+]]: !mo.tensor<[4], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[4], f32>) {
// CHECK:           map.iter.epilogue ({{.*}}) capturing {[[OCAP:%[^ =]+]] = [[OUT]] : !mo.tensor<[4], f32>} {
// CHECK-NEXT:        map.constant.scalar 1.500000e+00 : f32 : f32
mo.graph @epilogueCapturedConstant(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %cst = mo.constant <#M.dense_array<1.500000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %output = map.iter.opaque
      inputs (%x : !mo.tensor<[4], f32>, %cst : !mo.tensor<[], f32>)
      outputs (%out : !mo.tensor<[4], f32>) {
    %epi = map.iter.epilogue (%eval : f32, %eidx : !map.index<1>)
        capturing {%ocap = %out : !mo.tensor<[4], f32>, %ccap = %cst : !mo.tensor<[], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [4]>> (%eidx)
          : !map.index<1> -> !map.index<0>
      %bias = map.load %ccap[%idx_] : !mo.tensor<[], f32>
      %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%eidx, %eval, %bias)
          : !map.index<1>, f32, f32 -> f32
      map.store %ocap[%eidx] = %sum : !mo.tensor<[4], f32>
    } : !mo.tensor<[4], f32>
    map.call.opaque @kernels.relu[] inputs (%x) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[4], f32>
}

// A multi-element constant is a partial (vector) broadcast: the load's index
// selects among its elements, so there is no single literal to inline.

// CHECK-LABEL: mo.graph @multiElementConstant
// CHECK-NOT:   map.constant.scalar
// CHECK:         map.load
mo.graph @multiElementConstant() -> !mo.tensor<[8, 2], f32> {
  %cst = mo.constant <#M.dense_array<1.000000e+00, 2.000000e+00> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %output = map.iter.opaque
      inputs (%cst : !mo.tensor<[2], f32>)
      outputs (%out : !mo.tensor<[8, 2], f32>) {
    map.iter.elementwise.body<[8, 2]>
        index (%idx: !map.index<2>)
        capturing {%cst0 = %cst : !mo.tensor<[2], f32>, %out0 = %out : !mo.tensor<[8, 2], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[2] -> [8, 2]>> (%idx)
          : !map.index<2> -> !map.index<1>
      %val = map.load %cst0[%idx_] : !mo.tensor<[2], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 2], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 2], f32>
}

// A one-element runtime input is not a compile-time constant.

// CHECK-LABEL: mo.graph @oneElementRuntimeInput
// CHECK-NOT:   map.constant.scalar
// CHECK:         map.load
mo.graph @oneElementRuntimeInput(%scalar: !mo.tensor<[], f32>) -> !mo.tensor<[8, 64], f32> {
  %output = map.iter.opaque
      inputs (%scalar : !mo.tensor<[], f32>)
      outputs (%out : !mo.tensor<[8, 64], f32>) {
    map.iter.elementwise.body<[8, 64]>
        index (%idx: !map.index<2>)
        capturing {%scalar0 = %scalar : !mo.tensor<[], f32>, %out0 = %out : !mo.tensor<[8, 64], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[] -> [8, 64]>> (%idx)
          : !map.index<2> -> !map.index<0>
      %val = map.load %scalar0[%idx_] : !mo.tensor<[], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, 64], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, 64], f32>
}

// A dynamically-shaped source has no known element count, so it can never be
// proven to hold exactly one.

// CHECK-LABEL: mo.graph @dynamicSource
// CHECK-NOT:   map.constant.scalar
// CHECK:         map.load
mo.graph @dynamicSource<D0>(%dyn: !mo.tensor<[D0], f32>) -> !mo.tensor<[8, D0], f32> {
  %output = map.iter.opaque
      inputs (%dyn : !mo.tensor<[D0], f32>)
      outputs (%out : !mo.tensor<[8, D0], f32>) {
    map.iter.elementwise.body<[8, D0]>
        index (%idx: !map.index<2>)
        capturing {%dyn0 = %dyn : !mo.tensor<[D0], f32>, %out0 = %out : !mo.tensor<[8, D0], f32>} {
      %idx_ = map.index.broadcast<#map.index.broadcast.spec<[D0] -> [8, D0]>> (%idx)
          : !map.index<2> -> !map.index<1>
      %val = map.load %dyn0[%idx_] : !mo.tensor<[D0], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[8, D0], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[8, D0], f32>
}

}
