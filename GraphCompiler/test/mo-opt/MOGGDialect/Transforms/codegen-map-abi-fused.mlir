// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=NOPARAM
// RUN: cat %t/*.mojo | FileCheck %s --check-prefix=COMPUTE

// End-to-end Mojo emission for MAP-ABI fused (store-epilogue) outputs
// (TensorCreateEpilogueOp::emitMojo plus the CallOpaqueOp fused branch). A
// `map.iter.epilogue` store lowers directly to a `mogg.tensor.create.epilogue`
// holding its own epilogue body (no separate lambda op); emission builds an
// explicit-capture `EpilogueStoreFunctor` conforming to `OutputFusion`
// (managed_tensor_slice.mojo) by recursively lowering the epilogue body's own
// ops (its `mogg.tensor.store` becomes `store`'s own store statement), then
// binds it directly onto the store target's own (unfused) ManagedTensorSlice
// via `._bind_to_fused_output(...)` -- the same mechanism the MOGG-ABI
// `TensorFusedOp::emitMojo` non-variadic path uses. The whole path is
// `@__parameter`-free (NOPARAM below). None of these graphs fuses anything
// INTO its epilogue, so the execute call reads `has_epilogue_fusion = False`:
// the binding happens regardless, because every fusible output is bound.
//
// The pipeline runs through `mogg-to-mgp` (so the region consumes the outlined
// stub), then emits; `mogg-emit-mojo` still erases the stub, so
// `-verify-each=false` tolerates the dangling `mogg.call`. `mogg-to-mgp`
// names the `ir-output-dir` dump after the graphs (not `graph.mojo`), so the
// checks read `%t/*.mojo`.

// One fused output: the store target (capture 0) is captured as a bare
// pointer + row-major layout, the functor is built over that pointer, and it
// binds onto a fresh ManagedTensorSlice reconstructed over the same pointer
// (the epilogue's `$outputPtr`). The bound tensor is the DPS output of the
// execute call.
// CHECK: var [[CAP0:.*]] = builtin_primitives.primitives.mogg_tensor_init[{{.*}}, True, extensibility.IO.Output, {{.*}}]
// CHECK: var [[PTR0:.*]] = [[CAP0]].unsafe_ptr()
// CHECK: var [[F0:.*]] = EpilogueStoreFunctor{{[0-9_]*}}[{{.*}}]([[PTR0]], {{.*}})
// CHECK: var [[BASE0:.*]] = builtin_primitives.primitives.mogg_tensor_init[{{.*}}, False, extensibility.IO.Input, {{.*}}]({{.*}})
// CHECK: var [[W0:.*]] = [[BASE0]]._bind_to_fused_output([[F0]])
// CHECK: {{.*}}.MapLowerOneFusedOutput.execute[target = "cpu", has_epilogue_fusion = False]([[W0]], {{.*}})

mo.graph @one_fused(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c : !mo.tensor<[2, 2], f32>
}

// Two fused outputs: one functor + bind per output, both bound tensors
// passed to a single fused execute call.
// CHECK-DAG: {{.*}}.MapLowerTwoFusedOutputs.execute[target = "cpu", has_epilogue_fusion = False]
// CHECK-DAG: {{.*}}.MapLowerPartialFusedOutputs.execute[target = "cpu", has_epilogue_fusion = False]

mo.graph @two_fused(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_two_fused_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Partial: res0 fused (store epilogue), res1 plain output. The plain output is a
// bare `mogg_tensor_init`; only res0 is bound.
mo.graph @partial_fused(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %c:2 = mo.custom [symbol = "map_lower_partial_fused_outputs", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %c#0, %c#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// Dynamic dim: the `!mogg.dim` handle arrives as a trailing `Int` argument
// (mirroring the plain-stub handling in codegen-map-abi-stub.mlir) and
// feeds a runtime `Scalar[DType.int]` Coord component instead of an
// all-compile-time `layout.Idx` Coord; the fused-output functor/bind/
// execute shape is otherwise identical to the static case above.
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[DIM:.*]]: Int, {{.*}}: {{.*}}DeviceContext) raises:
// CHECK:   var {{.*}} = layout.Coord(Scalar[DType.int]([[DIM]]), layout.Idx[2])
// CHECK:   var [[CAPD:.*]] = builtin_primitives.primitives.mogg_tensor_init[{{.*}}, True, extensibility.IO.Output, {{.*}}]
// CHECK:   var [[PTRD:.*]] = [[CAPD]].unsafe_ptr()
// CHECK:   var [[FD:.*]] = EpilogueStoreFunctor{{[0-9_]*}}[{{.*}}]([[PTRD]], {{.*}})
// CHECK:   var [[BASED:.*]] = builtin_primitives.primitives.mogg_tensor_init[{{.*}}, False, extensibility.IO.Input, {{.*}}]({{.*}})
// CHECK:   var [[WD:.*]] = [[BASED]]._bind_to_fused_output([[FD]])
// CHECK:   {{.*}}.MapLowerOneFusedOutput.execute[target = "cpu", has_epilogue_fusion = False]([[WD]], {{.*}})

mo.graph @one_fused_dynamic(%a: !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32>
  mo.output %c : !mo.tensor<[batch, 2], f32>
}

// Real-body kernel (not a signature-only `map_lower_*` stub) with a real
// downstream op fused into its epilogue, dynamic shape. Confirms the fused
// elementwise add gets inlined into the functor's own `store` body (a
// second capture loaded and passed to `AddElementwise.elementwise`) exactly
// like the static case, and that a real (non-empty) kernel body reaches
// `execute` the same way `map_lower_one_fused_output` does above. Unlike the
// text-golden-only checks above, this exact scenario also has real
// end-to-end compile+run coverage: see
// add-fuse-output-epilogue-fusion-dynamic.mlir
// (GraphCompiler/test/integration/cpu/E2E/NewFusion/). All `def ..._mogg(...)`
// functions are emitted before any struct definitions, so this
// function-level check must run before the generic functor-struct-shape
// check below (which lands in the later struct section); its own struct-body
// check (the fused elementwise-add specific one) is placed after that check
// instead, further down.
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], {{.*}}: OpaquePointer[MutAnyOrigin], [[DIM2:.*]]: Int, {{.*}}: {{.*}}DeviceContext) raises:
// CHECK:   var {{.*}} = layout.Coord(Scalar[DType.int]([[DIM2]]), layout.Idx[2])
// CHECK:   var [[CAPR:.*]] = builtin_primitives.primitives.mogg_tensor_init[{{.*}}, True, extensibility.IO.Output, {{.*}}]
// CHECK:   var [[PTRR:.*]] = [[CAPR]].unsafe_ptr()
// CHECK:   var [[PTRB:.*]] = {{.*}}.unsafe_ptr()
// CHECK:   var [[FR:.*]] = EpilogueStoreFunctor{{[0-9_]*}}[{{.*}}, {{.*}}]([[PTRR]], {{.*}}, [[PTRB]], {{.*}})
// CHECK:   var [[BASER:.*]] = builtin_primitives.primitives.mogg_tensor_init[{{.*}}, False, extensibility.IO.Input, {{.*}}]({{.*}})
// CHECK:   var [[WR:.*]] = [[BASER]]._bind_to_fused_output([[FR]])
// CHECK:   {{.*}}.AddFuseOutput.execute[target = "cpu"]([[WR]], {{.*}})

mo.graph @real_kernel_fused_dynamic(%a: !mo.tensor<[batch, 2], f32>, %b: !mo.tensor<[batch, 2], f32>, %bias: !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32> {
  %sum = mo.custom [symbol = "imposter_add_fuse_output", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[batch, 2], f32>, !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32>
  %out = mo.custom [symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>](%sum, %bias) : (!mo.tensor<[batch, 2], f32>, !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32>
  mo.output %out : !mo.tensor<[batch, 2], f32>
}

// The functor struct is built via the scoped `structDecl` overload, with the
// `store` body produced by recursively lowering the epilogue's own body ops
// (`emitLegacyManagedTensorSliceEpilogue` / `TensorStoreOp::emitMojo`) instead
// of substituted into a fixed Mojo template. Each capture's `dtype` / `mut`
// are concrete literals read from its own defining `mogg.tensor.create` and
// baked directly into its field types; only `LayoutType` is a struct param
// (one per capture), since its value is a `type_of(<enclosing-function
// local>)` expression a struct can't close over. Bare (ptr, layout) fields,
// and an `@inline(.always)
// store` that stores through a rebuilt `TileTensor`. `_bind_to_fused_output`
// requires an `OutputFusion`-conforming functor, which uses `IndexList` (not
// `Coord`) and a `store` method (not `__call__`) -- `idx` gets converted once
// via `Coord`'s own `IndexList` constructor, and `val` gets rebound to the
// literal dtype (not the method's own still-symbolic `dtype` param) so the
// store type-checks against a concrete type. NO `@__parameter`, NO
// `ManagedTensorSlice` field.
// CHECK:      struct EpilogueStoreFunctor{{[0-9_]*}}[[[SPEC:[a-zA-Z0-9_]+]]: {{.*}}StaticTensorSpec[DType.float32, _, ...]](managed_tensor_slice.OutputFusion, TrivialRegisterPassable):
// CHECK-NEXT:     @__allow_legacy_any_origin_fields
// CHECK-NEXT:     var [[PTRF:[a-zA-Z0-9_]+]]: Pointer[Scalar[DType.float32], MutAnyOrigin]
// CHECK-NEXT:     var [[SHF:[a-zA-Z0-9_]+]]: std.utils.IndexList[Self.[[SPEC]].rank]
// CHECK-NEXT:     var [[STF:[a-zA-Z0-9_]+]]: std.utils.IndexList[Self.[[SPEC]].rank]
// CHECK:          @inline(.always)
// CHECK-NEXT:     def store[dtype : DType, rank : Int, simd_width : SIMDLength, element_alignment : Int = 1](self, idx: std.utils.IndexList[rank], value: SIMD[dtype, simd_width]):
// CHECK-NEXT:         var [[COORDIDX:.*]] = layout.Coord(idx)
// CHECK-NEXT:         var [[REBOUND:.*]] = rebind[SIMD[DType.float32, simd_width]](value)
// CHECK-NEXT:         {{.*}}simd_store_into_tensor_pointer[static_spec = Self.[[SPEC]], simd_width = simd_width, element_alignment = element_alignment](self.[[PTRF]], self.[[SHF]], self.[[STF]], {{.*}}, rebind[SIMD[{{.*}}, simd_width]]([[REBOUND]]))

// The advanced-fusion emit path must be entirely `@__parameter`-free.
// NOPARAM-NOT: @__parameter

// `real_kernel_fused_dynamic`'s own struct-body check, placed here (after the
// generic one-capture struct-shape check above) since struct definitions
// follow all `def ..._mogg(...)` functions in the emitted file. Its functor is
// the LAST of the six emitted, and neither the struct header (every functor
// now takes a `LayoutType` and an alignment param per capture) nor
// the `def store[...]` signature discriminates it on its own -- and FileCheck
// does not backtrack a `CHECK-NEXT` chain onto a later candidate. The generic
// check above already consumed the first functor, so walk past the four
// single-capture ones that follow it by count and anchor on the last, which is
// the only two-`LayoutType`-param functor. Note the second capture (the fused
// `bias` input, only ever loaded) carries its OWN `mut = False` rather than
// inheriting the store target's mutability: each capture's dtype/mut come from
// its own defining `mogg.tensor.create`.
// CHECK-COUNT-4: struct EpilogueStoreFunctor{{[0-9_]*}}[{{[a-zA-Z0-9_]+}}: {{.*}}StaticTensorSpec[DType.float32, _, ...]](managed_tensor_slice.OutputFusion, TrivialRegisterPassable):
// CHECK:      struct EpilogueStoreFunctor{{[0-9_]*}}[[[SPEC0:[a-zA-Z0-9_]+]]: {{.*}}StaticTensorSpec[DType.float32, _, ...], [[SPEC1:[a-zA-Z0-9_]+]]: {{.*}}StaticTensorSpec[DType.float32, _, ...]](managed_tensor_slice.OutputFusion, TrivialRegisterPassable):
// CHECK-NEXT:     @__allow_legacy_any_origin_fields
// CHECK-NEXT:     var {{.*}}: Pointer[Scalar[DType.float32], MutAnyOrigin]
// CHECK-NEXT:     var {{.*}}: std.utils.IndexList[Self.[[SPEC0]].rank]
// CHECK-NEXT:     var {{.*}}: std.utils.IndexList[Self.[[SPEC0]].rank]
// CHECK-NEXT:     @__allow_legacy_any_origin_fields
// CHECK-NEXT:     var {{.*}}: Pointer[Scalar[DType.float32], MutAnyOrigin]
// CHECK-NEXT:     var {{.*}}: std.utils.IndexList[Self.[[SPEC1]].rank]
// CHECK-NEXT:     var {{.*}}: std.utils.IndexList[Self.[[SPEC1]].rank]
// CHECK:          def store[dtype : DType, rank : Int, simd_width : SIMDLength, element_alignment : Int = 1](self, idx: std.utils.IndexList[rank], value: SIMD[dtype, simd_width]):
// CHECK-NEXT:         var [[COORDIDX2:.*]] = layout.Coord(idx)
// CHECK-NEXT:         var [[REBOUND2:.*]] = rebind[SIMD[DType.float32, simd_width]](value)
// CHECK-NEXT:         var [[LOADED:.*]] = {{.*}}simd_load_from_tensor_pointer[static_spec = Self.[[SPEC1]], simd_width = simd_width, element_alignment = element_alignment](self.{{.*}}, self.{{.*}}, self.{{.*}}, {{.*}})
// CHECK-NEXT:         var [[SUM:.*]] = testing_kernels.testing_kernels.AddElementwise.elementwise[dtype = DType.float32]([[REBOUND2]], [[LOADED]])
// CHECK-NEXT:         {{.*}}simd_store_into_tensor_pointer[static_spec = Self.[[SPEC0]], simd_width = simd_width, element_alignment = element_alignment](self.{{.*}}, self.{{.*}}, self.{{.*}}, {{.*}}, rebind[SIMD[{{.*}}, simd_width]]([[SUM]]))

// A kernel declaring `_FusedComputeOutputTensor` gets a COMPUTE functor: it
// conforms to `ComputeOutputFusion`, its method is `compute` and RETURNS the
// transformed value rather than storing it, and it binds through
// `_bind_to_fused_compute_output`. The primary kernel performs the one store.
// All `def ..._mogg(...)` functions are emitted before any struct definitions,
// so these three land in different sections; match them order-independently.
// COMPUTE-DAG: _bind_to_fused_compute_output
// COMPUTE-DAG: struct EpilogueComputeFunctor{{[0-9_]*}}[{{.*}}](managed_tensor_slice.ComputeOutputFusion, TrivialRegisterPassable):
// COMPUTE-DAG: def compute[{{.*}}](self, idx: {{.*}}, value: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:

mo.graph @compute_lambda(%a: !mo.tensor<[2, 2], f32>, %b: !mo.tensor<[2, 2], f32>, %bias: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %sum = mo.custom [symbol = "imposter_add_compute_fusion", device = #M.device_ref<"cpu", 0>](%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %out = mo.add(%sum, %bias) : !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}
