// RUN: mo-opt %s --map-fuse-kernels | FileCheck %s

// A reshape changes rank, so `LoopInvariantViewMotion` can't hoist a view fused
// atop a reshaped load into a stride view -- the rank change drops the stride.
// A reshape is therefore the terminal (most-upstream) link of the load chain it
// sits on: the view feeding that load stays materialized, while the reshape and
// everything downstream of it fuse. The barrier is per-load, not per-body -- a
// reshape on one load of a kernel never blocks a fusion into a different load of
// the same kernel. Runs the full `--map-fuse-kernels` fixed point.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

mogg.kernel.decl @kernels.consumer registration = "kernels.consumer" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.producer registration = "kernels.producer" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// `slice -> reshape -> transpose -> opaque(prologue)`: the reshape and the
// transpose below it fuse INTO the opaque's prologue (reshape lives in a
// prologue now); the slice, which feeds the reshape, stays its own materialized
// kernel and is the opaque's plain input.

// CHECK-LABEL: mo.graph @sliceReshapeTransposeIntoPrologue
// CHECK:       %[[SLICE:.*]] = map.iter.elementwise<[3, 32]>
// CHECK:         map.index.slice<
// CHECK:       %[[RES:.*]] = map.iter.opaque
// CHECK:       inputs ({{.*}}%[[SLICE]]: !mo.tensor<[3, 32], f32>)
// CHECK:       map.iter.prologue
// CHECK:         map.index.transpose<
// CHECK:         map.index.reshape<
// CHECK:         map.load %{{.*}}[
// CHECK:       map.call.opaque
// CHECK:       mo.output %[[RES]]
mo.graph @sliceReshapeTransposeIntoPrologue(
    %input: !mo.tensor<[10, 64], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>,
    %perm: !mo.tensor<[2], si64>
) -> !mo.tensor<[6, 16], f32> {
  %sliced = map.iter.elementwise<[3, 32]>
      inputs  (%input: !mo.tensor<[10, 64], f32>, %starts: !mo.tensor<[2], si64>,
               %stops: !mo.tensor<[2], si64>, %steps: !mo.tensor<[2], si64>)
      outputs (%so: !mo.tensor<[3, 32], f32>)
      index   (%i: !map.index<2>)
      -> !mo.tensor<[3, 32], f32>
  {
    %si = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%i, %starts, %stops, %steps)
        : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
    %v = map.load %input[%si] : !mo.tensor<[10, 64], f32>
    map.store %so[%i] = %v : !mo.tensor<[3, 32], f32>
  }
  %reshaped = map.iter.elementwise<[16, 6]>
      inputs  (%sliced: !mo.tensor<[3, 32], f32>)
      outputs (%ro: !mo.tensor<[16, 6], f32>)
      index   (%j: !map.index<2>)
      -> !mo.tensor<[16, 6], f32>
  {
    %ri = map.index.reshape<#map.index.reshape.spec<[3, 32] -> [16, 6]>> (%j)
        : !map.index<2> -> !map.index<2>
    %v = map.load %sliced[%ri] : !mo.tensor<[3, 32], f32>
    map.store %ro[%j] = %v : !mo.tensor<[16, 6], f32>
  }
  %transposed = map.iter.elementwise<[6, 16]>
      inputs  (%reshaped: !mo.tensor<[16, 6], f32>, %perm: !mo.tensor<[2], si64>)
      outputs (%to: !mo.tensor<[6, 16], f32>)
      index   (%k: !map.index<2>)
      -> !mo.tensor<[6, 16], f32>
  {
    %ti = map.index.transpose<#map.index.transpose.spec<[16, 6] -> [6, 16], perm = [1, 0]>> (%k, %perm)
        : (!map.index<2>, !mo.tensor<[2], si64>) -> !map.index<2>
    %v = map.load %reshaped[%ti] : !mo.tensor<[16, 6], f32>
    map.store %to[%k] = %v : !mo.tensor<[6, 16], f32>
  }
  %res = map.iter.opaque inputs (%transposed : !mo.tensor<[6, 16], f32>) outputs (%o : !mo.tensor<[6, 16], f32>) {
    %pro = map.iter.prologue (%pi : !map.index<2>) capturing {%pc = %transposed : !mo.tensor<[6, 16], f32>} {
      %pv = map.load %pc[%pi] : !mo.tensor<[6, 16], f32>
      map.yield %pv : f32
    } : !mo.tensor<[6, 16], f32>
    map.call.opaque @kernels.consumer[] inputs (%pro) outputs (%o) kernelName = "kernels.consumer" : (!mo.tensor<[6, 16], f32>) -> (!mo.tensor<[6, 16], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[6, 16], f32>
}

// Per-load precision: `slice -> add <- reshape`. The reshape indexes one load of
// the add, the slice the other. Both fuse into the single add kernel -- the
// reshape on one load must not block the slice fusing into the other. (The
// coarse "any reshape in the body" check this replaced would wrongly keep the
// slice out.)

// CHECK-LABEL: mo.graph @viewAndReshapeFuseIntoAdd
// CHECK:       %[[SUM:.*]] = map.iter.elementwise<[3, 32]>
// CHECK-DAG:     map.index.reshape<
// CHECK-DAG:     map.index.slice<
// CHECK-DAG:     map.call.elementwise @{{.*}}Add
// CHECK:       mo.output %[[SUM]]
mo.graph @viewAndReshapeFuseIntoAdd(
    %a: !mo.tensor<[10, 64], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>,
    %b: !mo.tensor<[96], f32>
) -> !mo.tensor<[3, 32], f32> {
  %sliced = map.iter.elementwise<[3, 32]>
      inputs  (%a: !mo.tensor<[10, 64], f32>, %starts: !mo.tensor<[2], si64>,
               %stops: !mo.tensor<[2], si64>, %steps: !mo.tensor<[2], si64>)
      outputs (%so: !mo.tensor<[3, 32], f32>)
      index   (%i: !map.index<2>)
      -> !mo.tensor<[3, 32], f32>
  {
    %si = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%i, %starts, %stops, %steps)
        : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
    %v = map.load %a[%si] : !mo.tensor<[10, 64], f32>
    map.store %so[%i] = %v : !mo.tensor<[3, 32], f32>
  }
  %reshaped = map.iter.elementwise<[3, 32]>
      inputs  (%b: !mo.tensor<[96], f32>)
      outputs (%ro: !mo.tensor<[3, 32], f32>)
      index   (%j: !map.index<2>)
      -> !mo.tensor<[3, 32], f32>
  {
    %ri = map.index.reshape<#map.index.reshape.spec<[96] -> [3, 32]>> (%j)
        : !map.index<2> -> !map.index<1>
    %v = map.load %b[%ri] : !mo.tensor<[96], f32>
    map.store %ro[%j] = %v : !mo.tensor<[3, 32], f32>
  }
  %sum = map.iter.elementwise<[3, 32]>
      inputs  (%sliced: !mo.tensor<[3, 32], f32>, %reshaped: !mo.tensor<[3, 32], f32>)
      outputs (%ao: !mo.tensor<[3, 32], f32>)
      index   (%k: !map.index<2>)
      -> !mo.tensor<[3, 32], f32>
  {
    %lv = map.load %sliced[%k] : !mo.tensor<[3, 32], f32>
    %rv = map.load %reshaped[%k] : !mo.tensor<[3, 32], f32>
    %s = map.call.elementwise @builtin_kernels.elementwise.Add(%k, %lv, %rv)
            : !map.index<2>, f32, f32 -> f32
    map.store %ao[%k] = %s : !mo.tensor<[3, 32], f32>
  }
  mo.output %sum : !mo.tensor<[3, 32], f32>
}

}
