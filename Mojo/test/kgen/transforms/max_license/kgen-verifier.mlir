// RUN: kgen-opt --split-input-file -kgen-verifier %s --verify-diagnostics

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch="">} {
  // expected-error @below {{heap allocation is not supported on GPU}}
  kgen.struct.instance @foo = struct_inst<""[mut, dtype, layout, origin, address_space, element_layout, layout_int_type, linear_idx_type, masked, alignment]<:i1 1, :dtype f32, :struct<(struct<(struct<(pointer<none>, index)>) memoryOnly>, struct<(struct<(pointer<none>, index)>) memoryOnly>) memoryOnly> { { { #interp.memref<{[(#interp.memory_handle<64, "0x01000000000000000100FFFFFFFFFFFF01000000000000000100000000000000">, heap, [], [])], []}, 0, 0>, 4 } }, { { #interp.memref<{[(#interp.memory_handle<64, "0x01000000000000000100FFFFFFFFFFFF01000000000000000100000000000000">, persistent, [], [])], []}, 0, 0>, 4 } } }, :struct<()> {  }, 3, :struct<(struct<(struct<(pointer<none>, index)>) memoryOnly>, struct<(struct<(pointer<none>, index)>) memoryOnly>) memoryOnly> { { { #interp.memref<{[(#interp.memory_handle<8, "0x00CA9A3B000000000200000000000000">, stack, [(0, 0, 0)], [])], []}, 0, 0>, 2 } }, { { #interp.memref<{[(#interp.memory_handle<8, "0x00CA9A3B000000000200000000000000">, stack, [(0, 0, 0)], [])], []}, 0, 0>, 2 } } }, :dtype si32, :dtype si32, :i1 0, 4>(ptr: typevalue<#kgen.instref<@unsafe_ptr>>, runtime_layout: typevalue<#kgen.instref<@runtime_layout>>, runtime_element_layout: typevalue<#kgen.instref<@runtime_layout>>)>

  kgen.func @aligned_free(%arg0: !kgen.pointer<index>) {
    // expected-error @below {{heap deallocation is not supported on GPU}}
    pop.aligned_free %arg0 : <index>
    hlcf.return
  }

  kgen.func @aligned_alloc(%arg0: index, %arg1: index) {
    // expected-error @below {{heap allocation is not supported on GPU}}
    %0 = pop.aligned_alloc %arg0, %arg1 : <index>
    hlcf.return
  }
}

// -----

// `pop.atomic.cmpxchg` used to be unconditionally rejected on Apple GPU here;
// this checks it now verifies cleanly (MSTDL-2890).
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "", data_layout = "">} {
  kgen.func @atomic_cmpxchg_verifies(%ptr: !kgen.pointer<scalar<index>>,
                            %cmp: !kgen.scalar<index>,
                            %val: !kgen.scalar<index>) {
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val monotonic monotonic :
                      !kgen.pointer<scalar<index>>
    hlcf.return
  }
}
