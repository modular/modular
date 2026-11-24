// RUN: mo-opt %s | FileCheck %s

// C2 scaffolding roundtrip: the new advanced-fusion MOGG codegen ops and types
// (!mogg.pointer / !mogg.tensor / !mogg.dim) parse and print. ODS-only (no
// lowering); this exercises each op's assembly variants.

// CHECK-LABEL: func.func @codegen_ops
// A `!mogg.dim` is optionally named after its source dynamic dim (`%d0` prints
// as `!mogg.dim<D0>`); a bare `!mogg.dim` (`%dim`) carries no name. Both
// round-trip (the named form is checked on the `shape.create` below).
func.func @codegen_ops(%ptr: !mogg.pointer, %dim: !mogg.dim,
                       %d0: !mogg.dim<D0>,
                       %ctx: !mogg.context, %ab: !mogg.async<!mgp.buffer>,
                       %ta: !mo.tensor<[2, 2], f32>) {
  // shape.create: static-only and with a dynamic dim.
  // CHECK: mogg.shape.create() {{.*}}
  %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
  // CHECK: mogg.shape.create(%{{.*}}) {{.*}} : !mogg.dim<D0>
  %s1 = mogg.shape.create(%d0) <shape = #mosh<ape[D0, 2]> : !mosh.ape> : !mogg.dim<D0>

  // tensor.create: immutable input vs mutable output. The `device` records the
  // device kind (cpu vs gpu) locally.
  // CHECK: mogg.tensor.create(%{{.*}}, %{{.*}}) {{.*}} : !mogg.tensor
  %t0 = mogg.tensor.create(%ptr, %s0) <device = "cpu", elementType = f32, io = Output, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
  %t1 = mogg.tensor.create(%ptr, %s0) <device = "gpu", elementType = f32, io = Input, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor

  // rebind: a no-op handle conversion (async buffer -> pointer).
  // CHECK: mogg.rebind %{{.*}} : !mogg.async<!mgp.buffer> to !mogg.pointer
  %cp = mogg.rebind %ab : !mogg.async<!mgp.buffer> to !mogg.pointer

  // epilogue tensor: a `!mogg.shape` operand + captured tensors + device kind,
  // holding its own body (block args + a store), and the call that writes into
  // it (with an explicit device context).
  // CHECK: mogg.tensor.create.epilogue %{{.*}} ({{[^)]*}}) capturing {%{{[^ ]+}} = %{{.*}} : !mogg.tensor} ptr %{{.*}}
  %epi = mogg.tensor.create.epilogue %s0 (%val : !mogg.simd<f32>, %i : !mogg.dim) capturing {%cap = %t1 : !mogg.tensor} ptr %cp
      device = "gpu", elementType = f32, io = Input, mut = true {
    // CHECK: mogg.tensor.store %{{.*}}[%{{.*}}] = %{{.*}} : !mogg.tensor, !mogg.dim, !mogg.simd<f32>
    mogg.tensor.store %cap[%i] = %val : !mogg.tensor, !mogg.dim, !mogg.simd<f32>
  }
  // CHECK: mogg.call.opaque @kernel inputs(%{{.*}} : !mogg.tensor) outputs(%{{.*}} : !mogg.tensor) context(%{{.*}} : !mogg.context)
  mogg.call.opaque @kernel inputs(%t0 : !mogg.tensor) outputs(%epi : !mogg.tensor) context(%ctx : !mogg.context)
  // a bare call variant with no tensor operands and no context.
  // CHECK: mogg.call.opaque @bare
  mogg.call.opaque @bare

  // kernel region: !mo.tensor input (operand) + output (result) outside; inside,
  // one !mogg.pointer per input then output, a context, then the dynamic dims.
  // The body block args print as labeled groups (context before dims), but the
  // in-memory order stays [inputs, outputs, dims, context]. No terminator: the
  // output pointer block arg is the output. Like `mogg.func`, the op
  // carries a `device` attribute (the `MOExplicitDevice` trait) recording its
  // execution device.
  // CHECK: mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  // CHECK: inputs (%{{.*}})
  // CHECK: outputs (%{{.*}})
  // CHECK: context (%{{.*}})
  // CHECK: dims (%{{.*}}: !mogg.dim)
  // CHECK-SAME: attributes {device = #M.device_ref<"gpu", 0>}
  %kr = mogg.routine (%ta : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs(%in_ptr : !mogg.pointer)
    outputs(%out_ptr : !mogg.pointer)
    context(%kctx : !mogg.context)
    dims(%kd0 : !mogg.dim) attributes {device = #M.device_ref<"gpu", 0>} {
    // CHECK: mogg.call.opaque @inner
    mogg.call.opaque @inner
  }

  return
}
