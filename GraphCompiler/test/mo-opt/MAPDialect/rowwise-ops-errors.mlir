// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @axisOutOfRange(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  // expected-error @below{{axis 2 is out of range for rank 2}}
  %r = map.iter.rowwise<[8, 128], axis = 2>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

mo.graph @rowRank(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  // expected-error @below{{row index rank 1 does not match shape rank 2}}
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<1>) -> !mo.tensor<[8, 1], f32> {
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

// A result is per row or per element; reducing the wrong axis is neither.
mo.graph @resultShape(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[1, 128], f32> {
  // expected-error @below{{result 0 must have the rowwise shape, or that shape with dimension 1 set to 1; got '!mo.tensor<[1, 128], f32>'}}
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[1, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[1, 128], f32> {
  }
  mo.output %r : !mo.tensor<[1, 128], f32>
}

// -----

mo.graph @bodyAxis(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.opaque
      inputs (%x : !mo.tensor<[8, 128], f32>)
      outputs (%out : !mo.tensor<[8, 1], f32>) {
    // expected-error @below{{axis -1 is out of range for rank 2}}
    map.iter.rowwise.body<[8, 128], axis = -1>
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[8, 128], f32>) {
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// A fold belongs to a rowwise.
mo.graph @reduceParent(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.elementwise<[8, 128]>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      index   (%idx: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    // expected-error @below{{expects parent op to be one of 'map.iter.rowwise, map.iter.rowwise.body'}}
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

mo.graph @reduceNoResult(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    // expected-error @below{{must have at least one result}}
    map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

mo.graph @reduceIndexRank(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    // expected-error @below{{index rank 3 does not match the rowwise shape rank 2}}
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<3>) -> f32 {
      %c = map.constant.scalar 1.000000e+00 : f32 : f32
      map.yield %c : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

mo.graph @reduceNoContribution(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    // expected-error @below{{must yield at least one contribution}}
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      map.yield
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

mo.graph @reduceStores(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    // expected-error @below{{must not store; a fold only reads the row}}
    %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.store %out[%i] = %v : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @reduceUnknownMonoid(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    // expected-error @below{{'builtin_kernels.monoids.Missing' does not refer to a kernel.decl op}}
    %acc = map.iter.reduce @builtin_kernels.monoids.Missing
        index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.elementwise.Neg registration = "mo.negative" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("x": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

mo.graph @reduceNotAMonoid(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    // expected-error @below{{kernel 'builtin_kernels.elementwise.Neg' is not a reduction}}
    %acc = map.iter.reduce @builtin_kernels.elementwise.Neg
        index (%i: !map.index<2>) -> f32 {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

mo.graph @reduceResultCount(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    // expected-error @below{{monoid 'builtin_kernels.monoids.ReduceSum' has 1 output fields, but the fold has 2 results}}
    %a, %b = map.iter.reduce @builtin_kernels.monoids.ReduceSum
        index (%i: !map.index<2>) -> (f32, f32) {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>
mogg.kernel.decl @builtin_kernels.monoids.Welford registration = "reduction.mean_var.tmp" device_info = <"", "", "", ""> reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"count", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"mean", #kgen.param.decl.ref<"dtype"> : !DType>, #mogg.kernel_reduction_output<"M2", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// Welford's three outputs share the `dtype` parameter.
mo.graph @reduceParamTypes(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    // expected-error @below{{results 0 and 2 are both of dtype parameter "dtype" but have types 'f32' and 'f16'}}
    %count, %mean, %m2 = map.iter.reduce @builtin_kernels.monoids.Welford
        index (%i: !map.index<2>) -> (f32, f32, f16) {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.yield %v : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

// A chain result must come last.
mo.graph @chainNotLast(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %c = mo.chain.create()
  // expected-error @below{{only the last result may be a chain}}
  %oc, %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>)
      chain   (%c)
      -> (!mo.chain, !mo.tensor<[8, 1], f32>) {
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

// A map phase belongs to a rowwise.
mo.graph @mapParent(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.elementwise<[8, 128]>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      index   (%idx: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    // expected-error @below{{expects parent op to be one of 'map.iter.rowwise, map.iter.rowwise.body'}}
    map.iter.map index (%i: !map.index<2>) {
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @mapIndexRank(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    // expected-error @below{{index rank 3 does not match the rowwise shape rank 2}}
    map.iter.map index (%i: !map.index<3>) {
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @mapStoreIndex(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    // expected-error @below{{must store at its own index}}
    map.iter.map index (%i: !map.index<2>) {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.store %out[%row] = %v : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

// A map phase writes the rowwise shape, not a per-row output.
mo.graph @mapStoresPerRow(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 1], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 1], f32> {
    map.iter.map index (%i: !map.index<2>) {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      // expected-error @below{{in a map phase must write a tensor of the rowwise shape}}
      map.store %out[%i] = %v : !mo.tensor<[8, 1], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// -----

// A per-element output is only written in a map phase.
mo.graph @perElementOutsideMap(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %v = map.load %x[%row] : !mo.tensor<[8, 128], f32>
    // expected-error @below{{of a tensor of the rowwise shape must be in a map phase}}
    map.store %out[%row] = %v : !mo.tensor<[8, 128], f32>
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

// Outside a map phase, a store writes a per-row tensor; this buffer is neither.
mo.graph @storeNotPerRow(%x: !mo.tensor<[8, 128], f32>, %buf: !mo.buffer<[1, 128], f32>) {
  %c = mo.chain.create()
  %oc = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>, %buf: !mo.buffer<[1, 128], f32>)
      outputs ()
      row     (%row: !map.index<2>)
      chain   (%c)
      -> !mo.chain {
    %v = map.load %x[%row] : !mo.tensor<[8, 128], f32>
    // expected-error @below{{outside a map phase must write a per-row tensor at the row index}}
    map.store %buf[%row] = %v : !mo.buffer<[1, 128], f32>
  } {hasWriteEffect = true}
  mo.output
}

// -----

// The outlined form follows the same rules.
mo.graph @bodyPerElementOutsideMap(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.opaque
      inputs (%x : !mo.tensor<[8, 128], f32>)
      outputs (%out : !mo.tensor<[8, 128], f32>) {
    map.iter.rowwise.body<[8, 128], axis = 1>
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[8, 128], f32>,
                   %out0 = %out : !mo.tensor<[8, 128], f32>) {
      %v = map.load %x0[%row] : !mo.tensor<[8, 128], f32>
      // expected-error @below{{of a tensor of the rowwise shape must be in a map phase}}
      map.store %out0[%row] = %v : !mo.tensor<[8, 128], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @cacheNotFirst(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    map.iter.map index (%i: !map.index<2>) {
      %v = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      map.store %out[%i] = %v : !mo.tensor<[8, 128], f32>
    }
    // expected-error @below{{must come first in the row, after constants only}}
    %p = map.iter.cache index (%k: !map.index<2>) -> f32 {
      %xv = map.load %x[%k] : !mo.tensor<[8, 128], f32>
      map.yield %xv : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @cacheYieldType(%x: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], bf16>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    // expected-error @below{{body must yield one value of the result type}}
    %p = map.iter.cache index (%k: !map.index<2>) -> f32 {
      %xv = map.load %x[%k] : !mo.tensor<[8, 128], bf16>
      map.yield %xv : bf16
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @cacheLoadOfNonCache(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    map.iter.map index (%i: !map.index<2>) {
      %xv = map.load %x[%i] : !mo.tensor<[8, 128], f32>
      // expected-error @below{{reads a value that isn't a map.iter.cache}}
      %v = map.iter.cache.load %xv[%i] : !map.index<2>, f32
      map.store %out[%i] = %v : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @cacheStores(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    // expected-error @below{{must not store; it only reads the row}}
    %p = map.iter.cache index (%k: !map.index<2>) -> f32 {
      %xv = map.load %x[%k] : !mo.tensor<[8, 128], f32>
      map.store %out[%k] = %xv : !mo.tensor<[8, 128], f32>
      map.yield %xv : f32
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}

// -----

mo.graph @cacheLoadAtRow(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %r = map.iter.rowwise<[8, 128], axis = 1>
      inputs  (%x: !mo.tensor<[8, 128], f32>)
      outputs (%out: !mo.tensor<[8, 128], f32>)
      row     (%row: !map.index<2>) -> !mo.tensor<[8, 128], f32> {
    %p = map.iter.cache index (%k: !map.index<2>) -> f32 {
      %xv = map.load %x[%k] : !mo.tensor<[8, 128], f32>
      map.yield %xv : f32
    }
    map.iter.map index (%i: !map.index<2>) {
      // expected-error @below{{must read at its phase's index}}
      %v = map.iter.cache.load %p[%row] : !map.index<2>, f32
      map.store %out[%i] = %v : !mo.tensor<[8, 128], f32>
    }
  }
  mo.output %r : !mo.tensor<[8, 128], f32>
}
