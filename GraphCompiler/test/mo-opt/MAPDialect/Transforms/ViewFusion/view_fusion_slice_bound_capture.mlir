// RUN: mo-opt %s --map-fuse-views | FileCheck %s

// A view producer whose result is a slice *bound* vector, captured into an
// existing prologue where it feeds a map.index.slice rather than a map.load.
// The capture has a single use, but that use is an index transform, so there is
// no load site to splice the producer into. ViewFuser must leave it alone
// rather than assume the capture's sole user is a load (GEX-4229).

mogg.kernel.decl @kernels.consumer registration = "kernels.consumer" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @sliceBoundViewIntoPrologue_noFuse
// The bound-vector view producer survives -- it is not fused into the prologue.
// CHECK:       %[[STOPS:.*]] = map.iter.elementwise<[1]>
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ({{.*}}%[[STOPS]]: !mo.tensor<[1], si64>{{.*}})
// CHECK:       map.iter.prologue
// CHECK:       map.index.slice
// CHECK:       map.load
// CHECK:       map.yield
mo.graph @sliceBoundViewIntoPrologue_noFuse(
    %data: !mo.tensor<[8], si64>,
    %starts: !mo.tensor<[1], si64>,
    %stopScalar: !mo.tensor<[], si64>,
    %steps: !mo.tensor<[1], si64>
) -> !mo.tensor<[1], si64> {
  // View producer: reshape a scalar into a [1] bound vector.
  %stops = map.iter.elementwise<[1]>
      inputs  (%stopScalar: !mo.tensor<[], si64>)
      outputs (%so: !mo.tensor<[1], si64>)
      index   (%i: !map.index<1>)
      -> !mo.tensor<[1], si64>
  {
    %ri = map.index.reshape<#map.index.reshape.spec<[] -> [1]>> (%i) : !map.index<1> -> !map.index<0>
    %v = map.load %stopScalar[%ri] : !mo.tensor<[], si64>
    map.store %so[%i] = %v : !mo.tensor<[1], si64>
  }
  %res = map.iter.opaque
      inputs (%data: !mo.tensor<[8], si64>, %starts: !mo.tensor<[1], si64>,
              %stops: !mo.tensor<[1], si64>, %steps: !mo.tensor<[1], si64>)
      outputs (%o : !mo.tensor<[1], si64>) {
    %pro = map.iter.prologue (%pi : !map.index<1>)
        capturing {%cdata = %data : !mo.tensor<[8], si64>,
                   %cstarts = %starts : !mo.tensor<[1], si64>,
                   %cstops = %stops : !mo.tensor<[1], si64>,
                   %csteps = %steps : !mo.tensor<[1], si64>} {
      %si = map.index.slice<#map.index.slice.spec<[8] -> [1], starts = [?], stops = [?], steps = [1]>> (%pi, %cstarts, %cstops, %csteps)
          : (!map.index<1>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !map.index<1>
      %v = map.load %cdata[%si] : !mo.tensor<[8], si64>
      map.yield %v : si64
    } : !mo.tensor<[1], si64>
    map.call.opaque @kernels.consumer[] inputs (%pro) outputs (%o) kernelName = "kernels.consumer" : (!mo.tensor<[1], si64>) -> (!mo.tensor<[1], si64>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[1], si64>
}
