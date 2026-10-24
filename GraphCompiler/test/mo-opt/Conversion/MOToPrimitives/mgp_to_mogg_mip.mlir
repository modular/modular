// RUN: mo-opt %s -allow-unregistered-dialect --mgp-lower-primitives | FileCheck %s

//--- Test lowering of Index Primitives

//--- index.constant will be tested throughout the tests ---/

// CHECK-LABEL: mgp.model @mip_add
// CHECK: index.add
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div
// CHECK: index.divs
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_div_ceil
// CHECK: index.ceildivs
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_eq
// CHECK: index.cmp eq({{.*}})
mgp.model @mip_cmp_eq : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 4
  %rhs = index.constant 4
  %res1 = index.cmp eq(%lhs, %rhs)
  %lhs1 = index.constant 2
  %res2 = index.cmp eq(%lhs1, %rhs)
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_cmp_lt
// CHECK: index.cmp slt({{.*}})
mgp.model @mip_cmp_lt : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 4
  %rhs = index.constant 4
  %res1 = index.cmp slt(%lhs, %rhs)
  %lhs1 = index.constant 2
  %res2 = index.cmp slt(%lhs1, %rhs)
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_cmp_le
// CHECK: index.cmp sle({{.*}})
mgp.model @mip_cmp_le : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %lhs = index.constant 5
  %rhs = index.constant 4
  %res1 = index.cmp sle(%lhs, %rhs)
  %lhs1 = index.constant 4
  %res2 = index.cmp sle(%lhs1, %rhs)
  mef.output %ch0 : !mef.chain
}


// CHECK-LABEL: mgp.model @mip_max
// CHECK: index.maxs
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_min
// CHECK: index.mins
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mod
// CHECK: index.rems
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_mul
// CHECK: index.mul
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @mip_and
// CHECK: pop.and
mgp.model @mip_and : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %lhs = index.bool.constant true
  %rhs = index.bool.constant true
  %res1 = pop.and %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_or
// CHECK: pop.or
mgp.model @mip_or : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %lhs = index.bool.constant true
  %rhs = index.bool.constant false
  %res1 = pop.or %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_xor
// CHECK: pop.xor
mgp.model @mip_xor : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %zero = index.constant 0
  %one = index.constant 1
  %lhs = index.bool.constant true
  %rhs = index.bool.constant false
  %res1 = pop.xor %lhs, %rhs
  mef.output %ch0, %res1: !mef.chain, i1
}

// CHECK-LABEL: mgp.model @mip_select
// CHECK: pop.select
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
  %ch1 = "test.sink"(%res1) : (index) -> !mef.chain
  mef.output %ch1: !mef.chain
}
