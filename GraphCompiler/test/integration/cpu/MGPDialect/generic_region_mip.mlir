// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --mogg-sequence-fusion --canonicalize --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck %s
// RUN: mo-opt --mogg-import-packages --mgp-lower-primitives --canonicalize --mogg-outline-async-routines --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-strip-kernel-decls --jit --mogg-lower-compiled-routines --mgp-outline-models-to-callables %s -o %t_no_fused.mlir
// RUN: mt --execute %t_no_fused.mlir | FileCheck %s

// CHECK-LABEL:--- Running 'mip_add':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_add : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -20
  %res1 = index.add %lhs, %rhs
  %fact = index.constant 22
  %cond = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_div':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_div : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = index.divs %lhs, %rhs
  %fact = index.constant -21
  %cond = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_div_ceil':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_div_ceil : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -4
  %res1 = index.ceildivs %lhs, %rhs
  %fact = index.constant -10
  %cond = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_max':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_max : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = index.maxs %lhs, %rhs
  %fact = index.constant 42
  %cond = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_min':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_min : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -2
  %res1 = index.mins %lhs, %rhs
  %fact = index.constant -2
  %cond = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_mod':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_mod : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant 4
  %res1 = index.rems %lhs, %rhs
  %fact = index.constant 2
  %cond = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_mul':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_mul : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant 4
  %res1 = index.mul %lhs, %rhs
  %fact = index.constant 168
  %cond = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_nary_mul':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_nary_mul : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %a = index.constant 3
  %b = index.constant -5
  %c = index.constant 7
  %res1 = index.mul %a, %b
  %res = index.mul %res1, %c
  %fact = index.constant -105
  %cond = index.cmp eq(%res, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_nary_mul_same_input':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_nary_mul_same_input : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %a = index.constant 3
  %res1 = index.mul %a, %a
  %res = index.mul %res1, %a
  %fact = index.constant 27
  %cond = index.cmp eq(%res, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL:--- Running 'mip_and':
// CHECK-NEXT: 'mip_and' returned 1
mgp.model @mip_and : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %lhs = index.bool.constant true
  %rhs = index.bool.constant true
  %res1 = pop.and %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL:--- Running 'mip_or':
// CHECK-NEXT: 'mip_or' returned 1
mgp.model @mip_or : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %lhs = index.bool.constant true
  %rhs = index.bool.constant false
  %res1 = pop.or %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL:--- Running 'mip_xor':
// CHECK-NEXT: 'mip_xor' returned 1
mgp.model @mip_xor : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %lhs = index.bool.constant true
  %rhs = index.bool.constant false
  %res1 = pop.xor %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL:--- Running 'mip_select':
// CHECK-NOT:  <error: ASSERT>
mgp.model @mip_select : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %cond = kgen.param.constant: scalar<bool> = <true>
  %res1 = pop.select %cond, %one, %zero : index
  %fact = index.constant 1
  %cond1 = index.cmp eq(%res1, %fact)
  %ch1 = mgp.assert %cond1, "ASSERT"
  mef.output %ch1: !mef.chain
}
