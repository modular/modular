// RUN: mo-opt %s --split-input-file \
// RUN:     --gc-pipeline-pre-jit="kernel-library-paths=%kernel_api_test_lib dump-mojo-text=true" \
// RUN:     -o /dev/null 2>&1 | FileCheck %s

// `dump-mojo-text=true` runs the DumpMojo pass, which writes the emitted Mojo to
// stderr; FileCheck matches against it (no temp dir needed).

// End-to-end device-graph synthesis + EmitMojo. An `isDeviceGraph` mo.graph
// with a single custom op lowers (through the synthesis stages) to a
// `mgp.device_graph.collective.create` whose per-device build body records the
// kernel via `DeviceGraphBuilder.region`, invoking the kernel's node-free
// `build_graph` method. `imposter_build_graph_capable` (`BuildGraphCapable` in
// the test kernel library) exposes both `execute` and a node-free `build_graph`.
// A second graph covers the fallback for kernels exposing only `execute`.

// The graph cache comes from the state context. `DeviceGraphCache` is not
// copyable, so the reference the state context hands back is taken by pointer
// rather than bound to a `var` of its own.
// CHECK: var [[CACHE:var[_0-9]+]] = Pointer(to = {{var[_0-9]+}}.get_device_graph_cache())

// The build closure passed to `DeviceGraph.collective.create` is a runtime
// closure that owns its captures, so it carries the `var` capture convention.
// It is indexed by a comptime device index and dispatches into the per-device
// body with `comptime if`. The build closure and per-kernel work closures are
// pinned @inline(.never) so AutomaticInline cannot merge the whole model's enqueue
// code into one function as this has significant compile time overhead.
// CHECK: @inline(.never)
// CHECK-NEXT: def [[BUILD:collective_build_for[_0-9]+]]{{.*}}(mut builder: {{.*}}DeviceGraphBuilder) raises {imm}:
// CHECK-NEXT: comptime if {{[a-z_0-9]+}} == 0:

// A buffer constant stays in the closure: it reads the constant-resource table
// of the compiled region it lives in, which a MOGG-ABI body does not carry. It
// crosses into the outlined body as an argument.
// CHECK: builtin_primitives.primitives.mgp_buffer_constant(
// CHECK: var [[CALL:var[_0-9]+]] = [[BODY:__device_graph_collective_create[_0-9]*]]({{.*}}builder{{.*}})

// The single graph output is registered on the builder: the yielded output
// tensor is packed into a fresh `AsyncValue` via `async_pack` and handed to
// `add_output`.
// CHECK: builder.add_output([[CALL]].async_pack())

// The graph is built via the static `DeviceGraph.collective.create(...)`.
// The collective is keyed on the graph inputs and cached via the same cache.
// CHECK: @inline(.never)
// CHECK-NEXT: def [[KEY:collective_key_for[_0-9]+]]{{.*}}() raises {imm} -> String:
// CHECK-NEXT: comptime if {{[a-z_0-9]+}} == 0:
// CHECK: raise Error("unexpected device index in collective device graph key")
// CHECK: [[ARRAY:var[_0-9]+]] = {{.*}}DeviceGraph.create_collective({{.*}}, [[BUILD]], [[KEY]], cache = [[CACHE]])
// CHECK: var [[G0:var[_0-9]+]] = [[ARRAY]][0]

// The recording stub is node-free (no result) and calls the kernel's
// `build_graph` method with the builder as the launch argument. The builder is
// a `mut` parameter: recording bodies mutate it.
// CHECK: def [[STUB:__[A-Za-z0-9_]+_build_graph]]({{.*}}: {{.*}}OpaquePointer[{{.*}}], {{.*}}: {{.*}}OpaquePointer[{{.*}}], mut {{.*}}: {{.*}}DeviceGraphBuilder) raises:
// CHECK: BuildGraphCapable.build_graph(

// Canonicalization hoists the ops whose operands all come from outside the
// dependencies body; outlining sinks them back in first, so this unit's work —
// including the tensor-to-buffer extraction that used to sit outside the
// closure — runs inside its own function.
// CHECK: def [[DEP:__device_graph_dependencies[_0-9]*]]({{.*}}mut {{.*}}: {{.*}}DeviceGraphBuilder{{.*}}) raises:
// CHECK: builtin_primitives.primitives.mgp_tensor_extract_buffer(
// CHECK: [[STUB]]({{.*}}, {{.*}}, {{.*}})

// The outlined build body takes the values it reads from the region — here the
// hoisted constant, the builder, the input — and returns the graph output.
// Parameters follow first use, not the region's argument order.
// CHECK: def [[BODY]]({{.*}}mut [[B:[a-z_0-9]+]]: {{.*}}DeviceGraphBuilder{{.*}}) raises -> {{.*}}:

// Each device input is recorded against a graph-owned stable location rather
// than the caller's buffer, so a replay does not depend on the caller's address
// surviving. `mgp.device_graph.execute` copies the live input in before replay.
// CHECK: var [[STABLE:var[_0-9]+]] = {{.*}}.add_input({{.*}})

// The recording allocates the output buffer inside a `region` work closure. The
// allocated buffer is a body result of the `mgp.device_graph.dependencies` op;
// since `region` returns only the body's last node, the buffer is surfaced
// through a captured `Optional` (None-initialized) that the `work` closure
// writes (so it captures by `mut`), then unwrapped with `.take()` after
// `region`.
// CHECK: var [[OUT:var[_0-9]+]] = Optional[{{.*}}](None)
// CHECK: @inline(.never)
// CHECK-NEXT: def [[WORK0:work[_0-9]+]](mut builder: {{.*}}DeviceGraphBuilder) raises {mut}:
// The allocation is a builder op: `mgp.device_graph.alloc` lowers to the
// `mgp_device_graph_alloc` primitive taking the builder directly -- no
// `builder.context()` extraction and no `mgp_buffer_alloc` in the work closure.
// The `[False]` parameter picks the pool's device memory manager; a `cpu` result
// buffer type emits `[True]` instead.
// CHECK-NOT: builder.context()
// CHECK-NOT: mgp_buffer_alloc
// CHECK: {{.*}}mgp_device_graph_alloc[False]({{.*}}, builder)
// CHECK: [[OUT]] =
// CHECK: var [[NODE0:var[_0-9]+]] = [[B]].region([[WORK0]], dependencies = [])
// CHECK: {{var[_0-9]+}} = [[OUT]].take()

// The dispatch is recorded in its own `imm`-capturing `region`, ordered after
// the alloc's node via the `dependencies=` list, and calls the unit's outlined
// body.
// CHECK: @inline(.never)
// CHECK-NEXT: def [[WORK1:work[_0-9]+]](mut builder: {{.*}}DeviceGraphBuilder) raises {imm}:
// CHECK: [[DEP]]({{.*}}, {{.*}}, builder)
// CHECK: [[B]].region([[WORK1]], dependencies = {{\[}}[[NODE0]]{{\]}})

mo.graph @dg(%x: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0> isDeviceGraph {
  %0 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>]
      (%x) : (!mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// -----

// Fallback for kernels without a node-free `build_graph` method:
// `imposter_execute_only` (`ExecuteOnly` in the test kernel library) exposes
// only `execute`. The `mgp.device_graph.recording_context` op views the builder
// as a recording device context (`builder.recording_context()`), and the stub
// dispatches the ordinary `execute` against it instead of calling `build_graph`.
// CHECK: def {{__[A-Za-z0-9_]+_build_graph}}({{.*}}: {{.*}}OpaquePointer[{{.*}}], {{.*}}: {{.*}}OpaquePointer[{{.*}}], mut [[BUILDER:[a-z_0-9]+]]: {{.*}}DeviceGraphBuilder) raises:
// CHECK: [[CTX:var[_0-9]+]] = [[BUILDER]].recording_context()
// CHECK-NOT: ExecuteOnly.build_graph(
// CHECK: ExecuteOnly.execute({{.*}}, [[CTX]])

mo.graph @dg_execute_only(%x: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0> isDeviceGraph {
  %0 = mo.custom [symbol = "imposter_execute_only", device = #M.device_ref<"gpu", 0>]
      (%x) : (!mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// -----

// A `!mo.buffer` graph input written in place (the KV-cache shape). The buffer
// lowers to a mut-marked `!mgp.tensor`, so its unpack carries the `mutable` kind and
// the Mojo-side `OwnedTensor[.., kind=.mutable].allocate_stable` registers an
// in-place marker instead of allocating a stable twin. The graph records it
// through `mgp.device_graph.add_input` like any other, and it is passed to
// `DeviceGraph.create` as a cache-key input (the mutable one contributes its
// address to the key via `write_graph_key`).
// CHECK-DAG: unpack_tensor[{{[0-9]+}}, {{[0-9]+}}, DType.float32, kind = max.gpu.host.DeviceGraphInputKind.mutable]
// CHECK-DAG: {{.*}}.add_input({{.*}})
// CHECK-DAG: DeviceGraph.create_collective(

mo.graph @dg_mut_input(%buf: !mo.buffer<[2, 3], f32, gpu:0>, %x: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.chain isDeviceGraph {
  %ch0 = mo.chain.create()
  %0 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>]
      (%x) : (!mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %ch1 = mo.mutable.store(%buf, %0, %ch0) : (!mo.buffer<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>, !mo.chain) -> (!mo.chain)
  mo.output %ch1 : !mo.chain
}

// -----

// A host-placed tensor input (dispatch metadata: an omitted device defaults to
// cpu). Its unpack carries the `host` kind, so the Mojo-side
// `OwnedTensor[.., kind=.host].allocate_stable` gets a host-pool stable twin
// pre-filled with the caller's bytes, and `write_graph_key` bakes the contents
// into the graph cache key. The graph records it through
// `mgp.device_graph.add_input` and passes it to `DeviceGraph.create` as a
// cache-key input.
// CHECK-DAG: unpack_tensor[{{[0-9]+}}, {{[0-9]+}}, DType.int64, kind = max.gpu.host.DeviceGraphInputKind.host]
// CHECK-DAG: {{.*}}.add_input({{.*}})
// CHECK-DAG: DeviceGraph.create_collective(

mo.graph @dg_host_input(%h: !mo.tensor<[4], si64>) -> !mo.tensor<[4], si64, gpu:0> isDeviceGraph {
  %0 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>]
      (%h) : (!mo.tensor<[4], si64>) -> !mo.tensor<[4], si64, gpu:0>
  mo.output %0 : !mo.tensor<[4], si64, gpu:0>
}

