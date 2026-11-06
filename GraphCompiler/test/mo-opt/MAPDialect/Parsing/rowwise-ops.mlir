// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip map.iter.rowwise, map.iter.rowwise.body, map.iter.reduce,
// map.iter.map, map.iter.cache and map.iter.cache.load.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

module {

// map.iter.reduce verifies its monoid resolves to a mogg.kernel.decl with a
// `reduction`, and map.call.elementwise that its callee has an `elementwise`
// function, so these stubs must be present.
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.ArgMax registration = "reduction.arg_max.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc_values", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"acc_indices", #lit.struct<{_mlir_value: dtype = si64}> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "reduction.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Mul registration = "mo.mul" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// A reduce-shaped kernel: one fold over the inner axis, a row-scalar op on its
// result, then a per-row store at the row index.
// CHECK-LABEL: mo.graph @sumOfSquares
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:    inputs  ([[X:%[^ :,]+]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]]: !mo.tensor<[8, 1], f32>)
// CHECK-NEXT:    row     ([[ROW:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
// CHECK-NEXT:   %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index ([[I:%[^ :,]+]]: !map.index<2>) -> f32 {
// CHECK-NEXT:     [[XV:%[^ ]+]] = map.load [[X]][[[I]]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:     [[SQ:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Mul([[I]], [[XV]], [[XV]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:     map.yield [[SQ]] : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   [[ACC2:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Mul([[ROW]], %acc, %acc) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:   map.store [[OUT]][[[ROW]]] = [[ACC2]] : !mo.tensor<[8, 1], f32>
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT]] : !mo.tensor<[8, 1], f32>
mo.graph @sumOfSquares(%x: !mo.tensor<[8, 128], f32>)
    -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>)
      -> !mo.tensor<[8, 1], f32>
  {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %xv)
                : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %acc2 = map.call.elementwise @builtin_kernels.elementwise.Mul(%row, %acc, %acc)
                : !map.index<2>, f32, f32 -> f32
    map.store %out[%row] = %acc2 : !mo.tensor<[8, 1], f32>
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// A non-inner axis with symbolic dims, and multi-result folds: ArgMax's
// outputs (one of them a fixed int64 dtype) and Welford's three outputs.
// CHECK-LABEL: mo.graph @multiResultFolds
// CHECK:         {{.*}} = map.iter.rowwise<[M, N], axis = 0>
// CHECK-NEXT:    inputs  ([[X2:%[^ :,]+]]: !mo.tensor<[M, N], f32>)
// CHECK-NEXT:    outputs ([[OV:%[^ :,]+]]: !mo.tensor<[1, N], f32>, [[OI:%[^ :,]+]]: !mo.tensor<[1, N], si64>, [[OM:%[^ :,]+]]: !mo.tensor<[1, N], f32>)
// CHECK-NEXT:    row     ([[ROW2:%[^ :,]+]]: !map.index<2>) -> (!mo.tensor<[1, N], f32>, !mo.tensor<[1, N], si64>, !mo.tensor<[1, N], f32>) {
// CHECK-NEXT:   %acc_values, %acc_indices = map.iter.reduce @builtin_kernels.monoids.ArgMax index ([[I2:%[^ :,]+]]: !map.index<2>) -> (f32, si64) {
// CHECK-NEXT:     [[XV2:%[^ ]+]] = map.load [[X2]][[[I2]]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:     map.yield [[XV2]] : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   %count, %mean, %M2 = map.iter.reduce @builtin_kernels.monoids.Welford index ({{.*}}: !map.index<2>) -> (f32, f32, f32) {
// CHECK:        map.store [[OV]][[[ROW2]]] = %acc_values : !mo.tensor<[1, N], f32>
// CHECK-NEXT:   map.store [[OI]][[[ROW2]]] = %acc_indices : !mo.tensor<[1, N], si64>
// CHECK-NEXT:   map.store [[OM]][[[ROW2]]] = %mean : !mo.tensor<[1, N], f32>
mo.graph @multiResultFolds<M, N>(%x: !mo.tensor<[M, N], f32>)
    -> (!mo.tensor<[1, N], f32>, !mo.tensor<[1, N], si64>,
        !mo.tensor<[1, N], f32>) {
  %v, %idx, %mean = map.iter.rowwise<[M, N], axis = 0>
      inputs  (%x: !mo.tensor<[M, N], f32>)
      outputs (%ov: !mo.tensor<[1, N], f32>, %oi: !mo.tensor<[1, N], si64>,
               %om: !mo.tensor<[1, N], f32>)
      row     (%row: !map.index<2>)
      -> (!mo.tensor<[1, N], f32>, !mo.tensor<[1, N], si64>,
          !mo.tensor<[1, N], f32>)
  {
    %values, %indices = map.iter.reduce @builtin_kernels.monoids.ArgMax
        index (%i: !map.index<2>) -> (f32, si64) {
      %xv = map.load %x[%i] : !mo.tensor<[M, N], f32>
      map.yield %xv : f32
    }
    %count, %m, %m2 = map.iter.reduce @builtin_kernels.monoids.Welford
        index (%i: !map.index<2>) -> (f32, f32, f32) {
      %xv = map.load %x[%i] : !mo.tensor<[M, N], f32>
      map.yield %xv : f32
    }
    map.store %ov[%row] = %values : !mo.tensor<[1, N], f32>
    map.store %oi[%row] = %indices : !mo.tensor<[1, N], si64>
    map.store %om[%row] = %m : !mo.tensor<[1, N], f32>
  }
  mo.output %v, %idx, %mean
      : !mo.tensor<[1, N], f32>, !mo.tensor<[1, N], si64>,
        !mo.tensor<[1, N], f32>
}

// The outlined form.
// CHECK-LABEL: mo.graph @rowwiseBody
// CHECK:         [[RESULT3:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X3:%[^ :,]+]]: !mo.tensor<[M, N], f32>)
// CHECK-NEXT:    outputs ([[OUT3:%[^ :,]+]] : !mo.tensor<[M, 1], f32>) {
// CHECK-NEXT:      map.iter.rowwise.body<[M, N], axis = 1>
// CHECK-NEXT:        row ([[ROW3:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing ([[X30:%[^ ,]+]] = [[X3]] : !mo.tensor<[M, N], f32>, [[OUT30:%[^ ,]+]] = [[OUT3]] : !mo.tensor<[M, 1], f32>) {
// CHECK-NEXT:      [[ACC3:%[^ ]+]] = map.iter.reduce @builtin_kernels.monoids.ReduceSum index ([[I3:%[^ :,]+]]: !map.index<2>) -> f32 {
// CHECK-NEXT:        [[XV3:%[^ ]+]] = map.load [[X30]][[[I3]]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:        map.yield [[XV3]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      map.store [[OUT30]][[[ROW3]]] = [[ACC3]] : !mo.tensor<[M, 1], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @rowwiseBody<M, N>(%x: !mo.tensor<[M, N], f32>)
    -> !mo.tensor<[M, 1], f32> {
  %r = map.iter.opaque
      inputs (%x : !mo.tensor<[M, N], f32>)
      outputs (%out : !mo.tensor<[M, 1], f32>) {
    map.iter.rowwise.body<[M, N], axis = 1>
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[M, N], f32>,
                   %out0 = %out : !mo.tensor<[M, 1], f32>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
          index (%i: !map.index<2>) -> f32 {
        %xv = map.load %x0[%i] : !mo.tensor<[M, N], f32>
        map.yield %xv : f32
      }
      map.store %out0[%row] = %acc : !mo.tensor<[M, 1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[M, 1], f32>
}

// An in-place kernel: the fold's result is stored into a buffer, and the chain
// orders it like `map.iter.elementwise`'s. The chain has no block argument and is
// the only result; `dispatch(immediate)` prints because it isn't the default.
// CHECK-LABEL: mo.graph @rowwiseChain
// CHECK:         [[OUTCHAIN:%[^ ]+]] = map.iter.rowwise<[8, 128], axis = 1> dispatch(immediate)
// CHECK-NEXT:    inputs  ([[X4:%[^ :,]+]]: !mo.tensor<[8, 128], f32>, [[BUF:%[^ :,]+]]: !mo.buffer<[8, 1], f32>)
// CHECK-NEXT:    outputs ()
// CHECK-NEXT:    row     ([[ROW4:%[^ :,]+]]: !map.index<2>) chain([[INCHAIN:%[^)]+]]) -> !mo.chain {
// CHECK-NEXT:   %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index ([[I4:%[^ :,]+]]: !map.index<2>) -> f32 {
// CHECK-NEXT:     [[XV4:%[^ ]+]] = map.load [[X4]][[[I4]]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:     map.yield [[XV4]] : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   map.store [[BUF]][[[ROW4]]] = %acc : !mo.buffer<[8, 1], f32>
// CHECK-NEXT: } {hasWriteEffect = true}
mo.graph @rowwiseChain(%x: !mo.tensor<[8, 128], f32>, %buf: !mo.buffer<[8, 1], f32>) {
  %c = mo.chain.create()
  %oc = map.iter.rowwise<[8, 128], axis = 1> dispatch(immediate)
      inputs  (%x: !mo.tensor<[8, 128], f32>, %buf: !mo.buffer<[8, 1], f32>)
      outputs ()
      row     (%row: !map.index<2>)
      chain   (%c)
      -> !mo.chain
  {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
    map.store %buf[%row] = %acc : !mo.buffer<[8, 1], f32>
  } {hasWriteEffect = true}
  mo.output
}

// A normalize-shaped kernel: a fold, a row-scalar op on its result, then a
// map phase that scales each element by it and by `gamma`, read through a
// broadcast, and stores per element.
// CHECK-LABEL: mo.graph @mapPhase
// CHECK:         [[RESULT5:%[^ ]+]] = map.iter.rowwise<[8, 128], axis = 1>
// CHECK-NEXT:    inputs  ([[X5:%[^ :,]+]]: !mo.tensor<[8, 128], f32>, [[G5:%[^ :,]+]]: !mo.tensor<[128], f32>)
// CHECK-NEXT:    outputs ([[OUT5:%[^ :,]+]]: !mo.tensor<[8, 128], f32>)
// CHECK-NEXT:    row     ([[ROW5:%[^ :,]+]]: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
// CHECK-NEXT:   %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index ([[I5:%[^ :,]+]]: !map.index<2>) -> f32 {
// CHECK:        }
// CHECK-NEXT:   [[S5:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Mul([[ROW5]], %acc, %acc) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:   map.iter.map index ([[J5:%[^ :,]+]]: !map.index<2>) {
// CHECK-NEXT:     [[XV5:%[^ ]+]] = map.load [[X5]][[[J5]]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:     [[GI5:%[^ ]+]] = map.index.broadcast<#map.index.broadcast.spec<[128] -> [8, 128]>> ([[J5]]) : !map.index<2> -> !map.index<1>
// CHECK-NEXT:     [[GV5:%[^ ]+]] = map.load [[G5]][[[GI5]]] : !mo.tensor<[128], f32>
// CHECK-NEXT:     [[N5:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Mul([[J5]], [[XV5]], [[S5]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:     [[Y5:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Mul([[J5]], [[N5]], [[GV5]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:     map.store [[OUT5]][[[J5]]] = [[Y5]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT:   mo.output [[RESULT5]] : !mo.tensor<[8, 128], f32>
mo.graph @mapPhase(%x: !mo.tensor<[8, 128], f32>, %gamma: !mo.tensor<[128], f32>)
    -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>, %gamma: !mo.tensor<[128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>)
      -> !mo.tensor<[8, 128], f32>
  {
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %sq = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %xv)
                : !map.index<2>, f32, f32 -> f32
      map.yield %sq : f32
    }
    %s = map.call.elementwise @builtin_kernels.elementwise.Mul(%row, %acc, %acc)
             : !map.index<2>, f32, f32 -> f32
    map.iter.map index (%i: !map.index<2>) {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      %gi = map.index.broadcast<#map.index.broadcast.spec<[128] -> [8, 128]>> (%i)
                : !map.index<2> -> !map.index<1>
      %gv = map.load %gamma[%gi] : !mo.tensor<[128], f32>
      %n = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %s)
               : !map.index<2>, f32, f32 -> f32
      %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %n, %gv)
               : !map.index<2>, f32, f32 -> f32
      map.store %out[%i] = %y : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// A map phase on a non-inner symbolic axis, next to a per-row output, in the
// outlined form.
// CHECK-LABEL: mo.graph @mapPhaseBody
// CHECK:           map.iter.rowwise.body<[M, N], axis = 0>
// CHECK-NEXT:        row ([[ROW6:%[^ :,]+]]: !map.index<2>)
// CHECK-NEXT:        capturing ([[X60:%[^ ,]+]] = {{.*}} : !mo.tensor<[M, N], f32>, [[OUT60:%[^ ,]+]] = {{.*}} : !mo.tensor<[M, N], f32>, [[SUM60:%[^ ,]+]] = {{.*}} : !mo.tensor<[1, N], f32>) {
// CHECK-NEXT:      [[ACC6:%[^ ]+]] = map.iter.reduce @builtin_kernels.monoids.ReduceSum
// CHECK:           map.store [[SUM60]][[[ROW6]]] = [[ACC6]] : !mo.tensor<[1, N], f32>
// CHECK-NEXT:      map.iter.map index ([[J6:%[^ :,]+]]: !map.index<2>) {
// CHECK-NEXT:        [[XV6:%[^ ]+]] = map.load [[X60]][[[J6]]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:        [[Y6:%[^ ]+]] = map.call.elementwise @builtin_kernels.elementwise.Mul([[J6]], [[XV6]], [[ACC6]]) : !map.index<2>, f32, f32 -> f32
// CHECK-NEXT:        map.store [[OUT60]][[[J6]]] = [[Y6]] : !mo.tensor<[M, N], f32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
mo.graph @mapPhaseBody<M, N>(%x: !mo.tensor<[M, N], f32>)
    -> (!mo.tensor<[M, N], f32>, !mo.tensor<[1, N], f32>) {
  %r, %sum = map.iter.opaque
      inputs (%x : !mo.tensor<[M, N], f32>)
      outputs (%out : !mo.tensor<[M, N], f32>,
               %outSum : !mo.tensor<[1, N], f32>) {
    map.iter.rowwise.body<[M, N], axis = 0>
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[M, N], f32>,
                   %out0 = %out : !mo.tensor<[M, N], f32>,
                   %sum0 = %outSum : !mo.tensor<[1, N], f32>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
          index (%i: !map.index<2>) -> f32 {
        %xv = map.load %x0[%i] : !mo.tensor<[M, N], f32>
        map.yield %xv : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[1, N], f32>
      map.iter.map index (%i: !map.index<2>) {
        %xv = map.load %x0[%i] : !mo.tensor<[M, N], f32>
        %y = map.call.elementwise @builtin_kernels.elementwise.Mul(%i, %xv, %acc)
                 : !map.index<2>, f32, f32 -> f32
        map.store %out0[%i] = %y : !mo.tensor<[M, N], f32>
      }
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r, %sum : !mo.tensor<[M, N], f32>, !mo.tensor<[1, N], f32>
}

// A cache, read by a map phase.
// CHECK-LABEL: mo.graph @cache
// CHECK:         [[P7:%[^ ]+]] = map.iter.cache index ([[K7:%[^ :,]+]]: !map.index<2>) -> f32 {
// CHECK-NEXT:      [[XV7:%[^ ]+]] = map.load %{{.*}}[[[K7]]] : !mo.tensor<[8, 128], f32>
// CHECK-NEXT:      map.yield [[XV7]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    map.iter.map index ([[J7:%[^ :,]+]]: !map.index<2>) {
// CHECK-NEXT:      [[V7:%[^ ]+]] = map.iter.cache.load [[P7]][[[J7]]] : !map.index<2>, f32
// CHECK-NEXT:      map.store %{{.*}}[[[J7]]] = [[V7]] : !mo.tensor<[8, 128], f32>
mo.graph @cache(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %p = map.iter.cache index (%k: !map.index<2>) -> f32 {
      %xv = map.load %x[%k] : !mo.tensor<[8, 128], f32>
      map.yield %xv : f32
    }
    map.iter.map index (%i: !map.index<2>) {
      %v = map.iter.cache.load %p[%i] : !map.index<2>, f32
      map.store %out[%i] = %v : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

}
