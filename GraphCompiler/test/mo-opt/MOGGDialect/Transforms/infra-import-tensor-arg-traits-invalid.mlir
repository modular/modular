// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_invalid_pkg" 2>&1 | FileCheck %s

// Each kernel in the tensor_arg_trait_invalid_stubs package has a trait-bound
// tensor argument that violates exactly one validation rule. Import-time
// validation must reject every one of them and report them all (not abort on
// the first), so all of these diagnostics are present in one run. This is the
// rejection half of the trait-lane distill matrix (the valid half is the
// `IMPORT-DAG` set in infra-lower-tensor-arg-traits.mlir); together they replace the
// deleted `TestTensorArgContract.cpp` gtest.

// === Role errors ========================================================== //

// `Some[TileTensorable]` (kernel `trait_no_role`) -- no role trait. Also covers
// the empty-role boundary: a tensor arg with metadata but no Input/Output/
// MutableInput.
// CHECK-DAG: error: trait-bound tensor argument has no role trait

// `Input & Output & TileTensorable` (kernel `trait_two_roles`) -- two role traits.
// CHECK-DAG: error: trait-bound tensor argument has more than one role trait

// `Input & MutableInput & TileTensorable` (kernel `trait_input_and_mutable_input`)
// -- the Input/MutableInput pair specifically. Exact-leaf matching means
// `MutableInput` does not subsume `Input`, so both roles count (a second
// "more than one role trait" diagnostic in the same run).
// CHECK-DAG: error: trait-bound tensor argument has more than one role trait

// === Access-form errors =================================================== //

// `Output & Fused & TileTensorable` (kernel `trait_fused_no_access`) -- fused with
// no access form.
// CHECK-DAG: error: a Fused tensor argument must declare exactly one access form

// `MutableInput & Fused & Loadable & Storable`
// (kernel `trait_multi_access_forms`) -- a FUSED arg with two access forms
// (load + store). Multiple access forms are now allowed on a NON-fused arg, but
// a fused arg must declare exactly one, so this is still rejected.
// CHECK-DAG: error: a Fused tensor argument must declare exactly one access form

// === Role / access compatibility ========================================== //

// `Output & Fused & Loadable` (kernel `trait_load_on_output`) -- load access on
// an Output.
// CHECK-DAG: error: load access is not valid on an Output argument

// `Input & Fused & Storable` (kernel `trait_store_on_input`) -- store access on
// a read-only Input.
// CHECK-DAG: error: store access is not valid on a read-only Input argument

// `Input & LoadAccess & StoreAccess`
// (kernel `trait_multi_access_markers_nonfused`) -- two access MARKERS on a
// non-fused arg. Multiple access forms are now allowed, but `store` is still
// invalid on a read-only Input, so the per-form role/access check rejects it
// (a second "store access is not valid on a read-only Input argument"
// diagnostic in the same run).
// CHECK-DAG: error: store access is not valid on a read-only Input argument

// `Input & Fused & _LegacyComputeOutput` (kernel `trait_transform_on_input`)
// -- transform access on a read-only Input.
// CHECK-DAG: error: transform access requires an Output argument

// `MutableInput & Fused & _LegacyComputeOutput`
// (kernel `trait_transform_on_mutable_input`) -- transform access on a mutable
// input; transform requires an Output (a second "transform access requires an
// Output argument" diagnostic in the same run).
// CHECK-DAG: error: transform access requires an Output argument

// `Output & Fused & _LegacyComputable` (kernel `trait_computable_on_signature`)
// -- the compute capability on a signature. The graph compiler may deliver a
// `_LegacyComputeOutput` slot as a store fusion, so it can't promise
// `_LegacyComputable` up front.
// CHECK-DAG: error: declare _LegacyComputeOutput on the kernel signature and refine to _LegacyComputable in the body

// `Output & Fused & _LegacyComputeOutput & TileTensorable` (kernel
// `trait_compute_output_tile_on_signature`) -- `TileTensorable` on the
// signature, which the store-fusion form can't satisfy.
// CHECK-DAG: error: a _LegacyComputeOutput argument cannot require TileTensorable on the kernel signature

// === Foreign trait ======================================================== //

// `Output & TileTensorable & Copyable` (kernel `trait_foreign`) -- a foreign
// (stdlib) trait in the bound.
// CHECK-DAG: references 'Copyable', which is not a tensor-argument trait

// `TensorTuple[*Ts]` with `Ts: TypeList[Trait=Input & TileTensorable & Movable, ...]`
// (kernel `trait_tensor_tuple_movable`). The tuple puts no bound on its
// elements, so `Movable` is as foreign in its list bound as anywhere else.
// CHECK-DAG: references 'Movable', which is not a tensor-argument trait

// `*Ts: Input & TileTensorable` (kernel `trait_variadic_pack`). A variadic pack of
// tensor arguments is not a trait-lane spelling; `TensorTuple` is.
// CHECK-DAG: error: variadic packs of tensor-trait arguments are not supported; take a `TensorTuple` instead

// === Shape-function lane disallows ======================================== //
//
// A shape-function tensor argument may conform to `Tensor` or
// `TileTensorable` (host execution makes data access safe), but the
// shape-fn distill lane still rejects io-role markers, `Fused`, and the
// fusion access traits (`LoadAccess`/`Loadable`, `StoreAccess`/`Storable`,
// `_LegacyComputeOutput`/`_LegacyComputable`) -- those stay execute-lane-only
// fusion machinery.

// `Input & TileTensorable` (shape fn `trait_shape_fn_with_role`) -- an io-role on
// a shape-fn arg.
// CHECK-DAG: error: shape-function tensor argument carries an io-role marker

// `Fused & TileTensorable` (shape fn `trait_shape_fn_fused`) -- Fused on a
// shape-fn arg.
// CHECK-DAG: error: Fused is not allowed on a shape-function tensor argument

// `Loadable & TileTensorable` (shape fn `trait_shape_fn_loadable`) -- a fusion
// access capability on a shape-fn arg.
// CHECK-DAG: error: LoadAccess, Loadable, StoreAccess, Storable, _LegacyComputeOutput, and _LegacyComputable are fusion machinery

module {}

// === Stable storage marker ================================================ //

// `Output & TileTensorable & Stable` (kernel `trait_stable_output`).
// CHECK-DAG: error: Stable is not valid on an Output argument

// `Input & Fused & Loadable & Stable` (kernel `trait_stable_fused`).
// CHECK-DAG: error: a Stable tensor argument cannot also be Fused

// `Stable & TileTensorable` on a shape function (`trait_shape_fn_stable`).
// CHECK-DAG: error: Stable is not allowed on a shape-function tensor argument


// Legacy structs and aliases are rejected before IO parameter decoding.
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of 'ManagedTensorSlice'; use 'Some[Tensor]' for shapes, 'Some[TileTensorable]' for element values, or 'TensorTuple' for variadic operands
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of 'ManagedTensorSlice'
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of 'VariadicTensors'
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of 'VariadicTensors'
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of '_FusedInputVariadicTensors'
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of '_FusedOutputVariadicTensors'
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of 'StableTensor'
// CHECK-DAG: error: shape-function tensor arguments must use tensor traits instead of 'StableVariadicTensors'
