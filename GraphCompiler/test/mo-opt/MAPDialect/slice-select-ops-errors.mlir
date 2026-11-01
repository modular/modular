// RUN: mo-opt %s --split-input-file --verify-diagnostics

// map.index.slice.select's verifier: the limits of the vectorized store its
// body runs in (static, positive steps; unsliced inner dim; an untransformed
// index), agreement with its spec on both sides of the slice, the one index
// its body takes, and the kernels it may sit in.

// A runtime start has nowhere to come from: the select takes no bounds
// operands.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[4, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[4, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      // expected-error @below{{dim 0: start and step must be static}}
      map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [?, 0], stops = [?, 16], steps = [1, 1]>> (%idx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[4, 16], f32>
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// -----

// A reversed slice is not supported, as by the fused concat kernel before it.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[4, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[4, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      // expected-error @below{{dim 0: step must be positive, got -2}}
      map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [7, 0], stops = [-1, 16], steps = [-2, 1]>> (%idx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[4, 16], f32>
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// -----

// The body runs once per stored vector, which spans the innermost dim, so that
// dim cannot be sliced.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[8, 8], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[8, 8], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[8, 8], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      // expected-error @below{{the innermost dim must not be sliced}}
      map.index.slice.select<#map.index.slice.spec<[8, 16] -> [8, 8], starts = [0, 8], stops = [8, 16], steps = [1, 1]>> (%idx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[8, 8], f32>
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[8, 8], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[8, 8], f32>
}

// -----

// At the epilogue's own index, the spec must slice the shape the kernel hands
// the epilogue.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[4, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[4, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      // expected-error @below{{spec's in-shape #mosh<ape[9, 16]> : !mosh.ape does not match the shape #mosh<ape[8, 16]> : !mosh.ape its index addresses}}
      map.index.slice.select<#map.index.slice.spec<[9, 16] -> [4, 16], starts = [1, 0], stops = [9, 16], steps = [2, 1]>> (%idx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[4, 16], f32>
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// -----

// The body runs at each stored vector's first index, and the kernel's vectors
// run along its own inner dim, so a reshaped index can carry one vector across
// rows the slice tells apart.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[32, 4], f32>, !mo.tensor<[16, 4], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[32, 4], f32>, %slice : !mo.tensor<[16, 4], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[32, 4], f32>, %caps = %slice : !mo.tensor<[16, 4], f32>} {
      %ridx = map.index.reshape<#map.index.reshape.spec<[32, 4] -> [8, 16]>> (%idx) : !map.index<2> -> !map.index<2>
      map.store %capf[%ridx] = %val : !mo.tensor<[32, 4], f32>
      // expected-error @below{{index must be its kernel's loop index or an enclosing select's argument}}
      map.index.slice.select<#map.index.slice.spec<[32, 4] -> [16, 4], starts = [0, 0], stops = [32, 4], steps = [2, 1]>> (%ridx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[16, 4], f32>
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[32, 4], f32>, !mo.tensor<[16, 4], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[32, 4], f32>, !mo.tensor<[16, 4], f32>
}

// -----

// Nested, the inner spec must slice the outer one's output.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[2, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[2, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[2, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>> (%idx) -> (%mid) : !map.index<2> {
        // expected-error @below{{spec's in-shape #mosh<ape[8, 16]> : !mosh.ape does not match the shape #mosh<ape[4, 16]> : !mosh.ape its index addresses}}
        map.index.slice.select<#map.index.slice.spec<[8, 16] -> [2, 16], starts = [1, 0], stops = [5, 16], steps = [2, 1]>> (%mid) -> (%iout) : !map.index<2> {
          map.store %caps[%iout] = %val : !mo.tensor<[2, 16], f32>
        }
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[2, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[2, 16], f32>
}

// -----

// A store at the body's index must have the shape the spec slices to.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[4, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[4, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      // expected-error @below{{tensor shape #mosh<ape[4, 16]> : !mosh.ape at the body's index does not match the slice's out-shape #mosh<ape[3, 16]> : !mosh.ape}}
      map.index.slice.select<#map.index.slice.spec<[8, 16] -> [3, 16], starts = [2, 0], stops = [8, 16], steps = [2, 1]>> (%idx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[4, 16], f32>
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// -----

// The index must have the spec's rank.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[2, 8, 16], f32>) -> !mo.tensor<[2, 8, 16], f32> {
  %y = map.iter.opaque inputs (%X : !mo.tensor<[2, 8, 16], f32>) outputs (%full : !mo.tensor<[2, 8, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<3>) capturing {%capf = %full : !mo.tensor<[2, 8, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[2, 8, 16], f32>
      // expected-error @below{{index rank does not match spec's rank 2}}
      map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>> (%idx) -> (%iout) : !map.index<3> {
      }
    } : !mo.tensor<[2, 8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[2, 8, 16], f32>) -> (!mo.tensor<[2, 8, 16], f32>)
    map.yield %full : !mo.tensor<[2, 8, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y : !mo.tensor<[2, 8, 16], f32>
}

// -----

// A rank-0 tensor has no dims to slice.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>, !mo.tensor<[], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[], f32>) outputs (%full : !mo.tensor<[], f32>, %slice : !mo.tensor<[], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<0>) capturing {%capf = %full : !mo.tensor<[], f32>, %caps = %slice : !mo.tensor<[], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[], f32>
      // expected-error @below{{cannot select from a slice of a rank-0 tensor}}
      map.index.slice.select<#map.index.slice.spec<[] -> [], starts = [], stops = [], steps = []>> (%idx) -> (%iout) : !map.index<0> {
        map.store %caps[%iout] = %val : !mo.tensor<[], f32>
      }
    } : !mo.tensor<[], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[], f32>) -> (!mo.tensor<[], f32>)
    map.yield %full, %slice : !mo.tensor<[], f32>, !mo.tensor<[], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[], f32>, !mo.tensor<[], f32>
}

// -----

// The body takes the one output-space index, of the spec's rank. The custom
// syntax can't spell another argument, so this uses the generic form.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[4, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[4, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      // expected-error @below{{body must take one index of the spec's rank}}
      "map.index.slice.select"(%idx) <{spec = #map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>}> ({
      ^bb0(%iout: !map.index<2>, %extra: !map.index<2>):
        map.store %caps[%iout] = %val : !mo.tensor<[4, 16], f32>
      }) : (!map.index<2>) -> ()
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// -----

// In a foreach, the spec must slice the foreach's loop shape.
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.elementwise<[8, 16]> inputs (%X: !mo.tensor<[8, 16], f32>) outputs (%full: !mo.tensor<[8, 16], f32>, %out: !mo.tensor<[4, 16], f32>) index (%idx: !map.index<2>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
    %v = map.load %X[%idx] : !mo.tensor<[8, 16], f32>
    map.store %full[%idx] = %v : !mo.tensor<[8, 16], f32>
    // expected-error @below{{spec's in-shape #mosh<ape[9, 16]> : !mosh.ape does not match the shape #mosh<ape[8, 16]> : !mosh.ape its index addresses}}
    map.index.slice.select<#map.index.slice.spec<[9, 16] -> [4, 16], starts = [1, 0], stops = [9, 16], steps = [2, 1]>> (%idx) -> (%iout) : !map.index<2> {
      map.store %out[%iout] = %v : !mo.tensor<[4, 16], f32>
    }
  }
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// -----

// A prologue's loads may be recomputed anywhere, so it holds no select.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
func.func @f(%X: !mo.tensor<[8, 16], f32>) -> !mo.tensor<[8, 16], f32> {
  %y = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>) {
    %pro = map.iter.prologue (%idx : !map.index<2>) capturing {%cap = %X : !mo.tensor<[8, 16], f32>} {
      %v = map.load %cap[%idx] : !mo.tensor<[8, 16], f32>
      // expected-error @below{{expects parent op to be one of 'map.iter.epilogue, map.iter.elementwise, map.iter.elementwise.body, map.index.slice.select'}}
      map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>> (%idx) -> (%iout) : !map.index<2> {
      }
      map.yield %v : f32
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%pro) outputs (%full) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full : !mo.tensor<[8, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y : !mo.tensor<[8, 16], f32>
}
