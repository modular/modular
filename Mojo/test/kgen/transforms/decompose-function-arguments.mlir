// `decompose-function-arguments`'s default policy flattens every
// `kgen.struct` argument unconditionally, so these tests exercise the
// pass's mechanics rather than any real target's ABI. The policy is
// pointer-unaware: a pointer-typed field becomes a leaf as-is rather than
// being dereferenced, same as any other non-struct field type.
// RUN: kgen-opt -decompose-function-arguments -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// A struct's one field is a pointer: it becomes the new argument unchanged
// (the outer struct disappears; the pointer itself isn't touched), so the
// existing load + extract chain into its pointee carries on operating on
// the new argument directly.
// CHECK-LABEL: kgen.func @read_single
// CHECK-SAME:    (%[[A:[^:]+]]: !kgen.pointer<struct<(memref<128xf32>)>>)
// CHECK-SAME:    kgen.decomposed_arg_map = [array<i32: 0, 1, 1>]
// CHECK:         %[[LOADED:.+]] = pop.load %[[A]]
// CHECK:         %[[FIELD:.+]] = kgen.struct.extract %[[LOADED]][0]
// CHECK:         "use"(%[[FIELD]])
kgen.func @read_single(%arg0: !kgen.struct<(pointer<struct<(memref<128xf32>)>>)>) {
  %0 = kgen.struct.extract %arg0[0] : <(pointer<struct<(memref<128xf32>)>>)>
  %1 = pop.load %0 : !kgen.pointer<struct<(memref<128xf32>)>>
  %2 = kgen.struct.extract %1[0] : <(memref<128xf32>)>
  "use"(%2) : (memref<128xf32>) -> ()
  hlcf.return
}

// -----

// Struct-by-value with two fields flattens to two leaf args.
// CHECK-LABEL: kgen.func @two_fields
// CHECK-SAME:    (%[[A:[^:]+]]: memref<128xf32>, %[[B:[^:]+]]: i32)
// CHECK-SAME:    kgen.decomposed_arg_map = [array<i32: 0, 2, 1>]
// CHECK:         "use"(%[[A]], %[[B]])
kgen.func @two_fields(%arg0: !kgen.struct<(memref<128xf32>, i32)>) {
  %0 = kgen.struct.extract %arg0[0] : <(memref<128xf32>, i32)>
  %1 = kgen.struct.extract %arg0[1] : <(memref<128xf32>, i32)>
  "use"(%0, %1) : (memref<128xf32>, i32) -> ()
  hlcf.return
}

// -----

// Non-struct arg passes through untouched; no `kgen.decomposed_arg_map` is
// set when nothing changed.
// CHECK-LABEL: kgen.func @passthrough
// CHECK-SAME:    (%[[A:[^:]+]]: memref<128xf32>)
// CHECK-NOT:     kgen.decomposed_arg_map
// CHECK:         "use"(%[[A]])
kgen.func @passthrough(%arg0: memref<128xf32>) {
  "use"(%arg0) : (memref<128xf32>) -> ()
  hlcf.return
}

// -----

// A surviving `kgen.call` to a decomposed callee is rewritten: the caller's
// own arg is a plain memref (so it isn't itself decomposed), and the struct
// it builds to call `call_target` gets flattened at the call site.
// CHECK-LABEL: kgen.func @call_target
// CHECK-SAME:    (%{{[^:]+}}: memref<128xf32>)
// CHECK-LABEL: kgen.func @call_source
// CHECK-SAME:    (%[[ARG:[^:]+]]: memref<128xf32>)
// CHECK:         %[[STRUCT:.+]] = kgen.struct.create(%[[ARG]])
// CHECK:         %[[LEAF:.+]] = kgen.struct.extract %[[STRUCT]][0]
// CHECK:         kgen.call @call_target(%[[LEAF]]) : (memref<128xf32>) -> ()
kgen.func @call_target(%arg0: !kgen.struct<(memref<128xf32>)>) {
  %0 = kgen.struct.extract %arg0[0] : <(memref<128xf32>)>
  "use"(%0) : (memref<128xf32>) -> ()
  hlcf.return
}
kgen.func @call_source(%arg0: memref<128xf32>) {
  %s = kgen.struct.create(%arg0) : !kgen.struct<(memref<128xf32>)>
  kgen.call @call_target(%s) : (!kgen.struct<(memref<128xf32>)>) -> ()
  hlcf.return
}

// -----

// A whole-aggregate use of a decomposed arg — no `struct.extract` narrows it
// to a single leaf — is rebuilt via `struct.create` rather than erroring.
// CHECK-LABEL: kgen.func @whole_aggregate_use
// CHECK-SAME:    (%[[A:[^:]+]]: memref<128xf32>)
// CHECK:         %[[REBUILT:.+]] = kgen.struct.create(%[[A]])
// CHECK:         "use"(%[[REBUILT]])
kgen.func @whole_aggregate_use(%arg0: !kgen.struct<(memref<128xf32>)>) {
  "use"(%arg0) : (!kgen.struct<(memref<128xf32>)>) -> ()
  hlcf.return
}

// -----

// A whole-aggregate arg passed straight to a call is rebuilt the same way —
// and since `callee` is also decomposed, `rewriteCalls` immediately
// extracts the same leaf back out of the rebuild at the call site.
// CHECK-LABEL: kgen.func @callee
// CHECK-SAME:    (%{{[^:]+}}: memref<128xf32>)
// CHECK-LABEL: kgen.func @caller_passes_whole_arg
// CHECK-SAME:    (%[[A:[^:]+]]: memref<128xf32>)
// CHECK:         %[[REBUILT:.+]] = kgen.struct.create(%[[A]])
// CHECK:         %[[LEAF:.+]] = kgen.struct.extract %[[REBUILT]][0]
// CHECK:         kgen.call @callee(%[[LEAF]]) : (memref<128xf32>) -> ()
kgen.func @callee(%arg0: !kgen.struct<(memref<128xf32>)>) {
  %0 = kgen.struct.extract %arg0[0] : <(memref<128xf32>)>
  "use"(%0) : (memref<128xf32>) -> ()
  hlcf.return
}
kgen.func @caller_passes_whole_arg(%arg0: !kgen.struct<(memref<128xf32>)>) {
  kgen.call @callee(%arg0) : (!kgen.struct<(memref<128xf32>)>) -> ()
  hlcf.return
}

// -----

// Conventions stay with their arguments: each leaf inherits the decomposed
// argument's, and the argument after it keeps its own.
// CHECK-LABEL: kgen.func @conventions
// CHECK-SAME:    (%[[A:[^:]+]]: memref<128xf32> owned, %[[B:[^:]+]]: i32 owned, %[[C:[^:]+]]: index)
// CHECK:         "use"(%[[A]], %[[B]], %[[C]])
kgen.func @conventions(%arg0: !kgen.struct<(memref<128xf32>, i32)> owned, %arg1: index) {
  %0 = kgen.struct.extract %arg0[0] : <(memref<128xf32>, i32)>
  %1 = kgen.struct.extract %arg0[1] : <(memref<128xf32>, i32)>
  "use"(%0, %1, %arg1) : (memref<128xf32>, i32, index) -> ()
  hlcf.return
}

// -----

// `fnArgAttrs` is re-indexed too: the pass-through argument's entry follows
// it, and the default policy gives leaves empty entries.
// CHECK-LABEL: kgen.func @arg_attrs
// CHECK-SAME:    fnArgAttrs = [{}, {}, {test.marker}]
kgen.func @arg_attrs(%arg0: !kgen.struct<(memref<128xf32>, i32)>, %arg1: index) attributes {
  fnArgAttrs = [{}, {test.marker}]
} {
  %0 = kgen.struct.extract %arg0[0] : <(memref<128xf32>, i32)>
  "use"(%0, %arg1) : (memref<128xf32>, index) -> ()
  hlcf.return
}

// -----

// A `struct.gep` through a pointer leaf traces like an extract; the use is
// re-materialized from the leaf by crossing the pointer and extracting.
// CHECK-LABEL: kgen.func @gep_through_pointer_leaf
// CHECK-SAME:    (%[[P:[^:]+]]: !kgen.pointer<struct<(i32, i64)>>)
// CHECK:         %[[S:.+]] = pop.load %[[P]]
// CHECK:         %[[V:.+]] = kgen.struct.extract %[[S]][1]
// CHECK:         "use"(%[[V]])
kgen.func @gep_through_pointer_leaf(%arg0: !kgen.struct<(pointer<struct<(i32, i64)>>)>) {
  %p = kgen.struct.extract %arg0[0] : <(pointer<struct<(i32, i64)>>)>
  %f = kgen.struct.gep %p[1] : <struct<(i32, i64)>>
  %v = pop.load %f : !kgen.pointer<i64>
  "use"(%v) : (i64) -> ()
  hlcf.return
}

// -----

// A pointer-typed leaf survives reconstruction the same as any other: no
// backing storage needs fabricating, since the pointer itself is the leaf.
// CHECK-LABEL: kgen.func @whole_pointer_use
// CHECK-SAME:    (%[[A:[^:]+]]: !kgen.pointer<struct<(memref<128xf32>)>>)
// CHECK:         %[[REBUILT:.+]] = kgen.struct.create(%[[A]])
// CHECK:         "use"(%[[REBUILT]])
kgen.func @whole_pointer_use(%arg0: !kgen.struct<(pointer<struct<(memref<128xf32>)>>)>) {
  "use"(%arg0) : (!kgen.struct<(pointer<struct<(memref<128xf32>)>>)>) -> ()
  hlcf.return
}

// -----

// An intermediate sub-struct — narrowed partway, short of either leaf under
// it — is rebuilt from just the leaves it actually contains.
// CHECK-LABEL: kgen.func @nested_struct_use
// CHECK-SAME:    (%[[A:[^:]+]]: memref<128xf32>, %[[B:[^:]+]]: i32)
// CHECK:         %[[REBUILT:.+]] = kgen.struct.create(%[[A]], %[[B]])
// CHECK:         "use"(%[[REBUILT]])
kgen.func @nested_struct_use(%arg0: !kgen.struct<(struct<(memref<128xf32>, i32)>)>) {
  %inner = kgen.struct.extract %arg0[0] : <(struct<(memref<128xf32>, i32)>)>
  "use"(%inner) : (!kgen.struct<(memref<128xf32>, i32)>) -> ()
  hlcf.return
}

// -----

// The chain cleanup drops a `pop.load` only when it is genuinely removable.
// A volatile load carries a write effect, so it outlives the sweep even with
// no uses; a plain one next to it does not.
// CHECK-LABEL: kgen.func @dead_volatile_load_survives
// CHECK:         pop.load volatile
// CHECK-NOT:     pop.load %
kgen.func @dead_volatile_load_survives(%arg0: !kgen.struct<(memref<128xf32>)>) {
  %p = "make_ptr"() : () -> !kgen.pointer<scalar<f32>>
  %volatile = pop.load volatile<1> %p : !kgen.pointer<scalar<f32>>
  %plain = pop.load %p : !kgen.pointer<scalar<f32>>
  %0 = kgen.struct.extract %arg0[0] : <(memref<128xf32>)>
  "use"(%0) : (memref<128xf32>) -> ()
  hlcf.return
}
