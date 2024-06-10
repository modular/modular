// RUN: kgen-opt -split-input-file -mem-2-reg -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @simple_add
kgen.generator @simple_add(%arg0: index, %arg1: index) -> index {
  // CHECK-NEXT: %0 = index.add %arg0, %arg1
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !kgen.pointer<index>

  %1 = pop.stack_allocation 1 x index
  pop.store %arg1, %1 : !kgen.pointer<index>

  %2 = pop.load %0 : !kgen.pointer<index>
  %3 = pop.load %1 : !kgen.pointer<index>
  %4 = index.add %2, %3
  pop.store %4, %1 : !kgen.pointer<index>

  %5 = pop.load %1 : !kgen.pointer<index>
  // CHECK-NEXT: return %0
  kgen.return %5 : index
}

// CHECK-LABEL: @use_in_region
kgen.generator @use_in_region(%arg0: index, %arg1: i1) -> index {
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !kgen.pointer<index>

  // CHECK-NEXT: hlcf.if
  hlcf.if %arg1 {
    // CHECK-NEXT: kgen.return %arg0
    %1 = pop.load %0 : !kgen.pointer<index>
    kgen.return %1 : index
  } else {
    hlcf.yield
  }

  // CHECK-NOT: pop.load
  %1 = pop.load %0 : !kgen.pointer<index>
  // CHECK: return %arg0 : index
  kgen.return %1 : index
}

// CHECK-LABEL: @store_in_region
kgen.generator @store_in_region(%arg0: index, %arg1: index, %arg2: i1) -> index {
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !kgen.pointer<index>

  // CHECK-NEXT: %0 = hlcf.if %arg2 -> index
  hlcf.if %arg2 {
    %1 = pop.load %0 : !kgen.pointer<index>
    // CHECK-NEXT: return %arg0
    kgen.return %1 : index
  } else {
    // CHECK: hlcf.yield %arg1 : index
    pop.store %arg1, %0 : !kgen.pointer<index>
    hlcf.yield
  }

  %1 = pop.load %0 : !kgen.pointer<index>
  // CHECK: return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @unknown_use
kgen.generator @unknown_use(%arg0: index) -> index {
  // CHECK-NEXT: stack_allocation
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !kgen.pointer<index>
  "unknown.use"(%0) : (!kgen.pointer<index>) -> ()
  %1 = pop.load %0 : !kgen.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: @nested_alloc
kgen.generator @nested_alloc(%arg0: index) -> index {
  // CHECK-NEXT: %0 = hlcf.loop
  %0 = hlcf.loop () -> index {
    %1 = pop.stack_allocation 1 x index
    pop.store %arg0, %1 : !kgen.pointer<index>
    // CHECK-NEXT: %1 = hlcf.loop
    %2 = hlcf.loop () -> index {
      %3 = pop.load %1 : !kgen.pointer<index>
      // CHECK-NEXT: hlcf.break %arg0
      hlcf.break %3 : index
    }
    // CHECK: break %1
    hlcf.break %2 : index
  }
  // CHECK: return %0
  kgen.return %0 : index
}

// CHECK-LABEL: @read_uninitialized
kgen.generator @read_uninitialized() -> index {
  // CHECK-NEXT: %0 = kgen.undef : index
  %0 = pop.stack_allocation 1 x index
  %1 = pop.load %0 : !kgen.pointer<index>
  // CHECK-NEXT: kgen.return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @if_empty_block
kgen.generator @if_empty_block(%arg0: i1, %arg1: index) -> index{
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  pop.store %arg1, %0 : !kgen.pointer<index>
  // CHECK-NEXT: %0 = hlcf.if %arg0 -> index
  hlcf.if %arg0 {
    %2 = pop.load %0 : !kgen.pointer<index>
    pop.store %2, %1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %arg1
    hlcf.yield
  } else {
    // CHECK: %1 = kgen.undef : index
    // CHECK-NEXT: yield %1
    hlcf.yield
  }
  %2 = pop.load %1 : !kgen.pointer<index>
  // CHECK: return %0
  kgen.return %2 : index
}

// CHECK-LABEL: @store_alloca
kgen.func @store_alloca() -> i32 {
  // CHECK-NEXT: pop.stack_allocation 1 x i32
  // CHECK-NEXT: pop.load
  %0 = pop.stack_allocation 1 x !kgen.pointer<i32>
  %1 = pop.stack_allocation 1 x i32
  pop.store %1, %0 : !kgen.pointer<pointer<i32>>
  %2 = pop.load %0 : !kgen.pointer<pointer<i32>>
  %3 = pop.load %2 : !kgen.pointer<i32>
  kgen.return %3 : i32
}

// CHECK-LABEL: @loop_variant
kgen.func @loop_variant(%arg0: index, %arg1: index, %lb: index, %ub: index, %step: index) -> (index, index) {
  %var0 = pop.stack_allocation 1 x index
  %var1 = pop.stack_allocation 1 x index
  // COM: var var0 = arg0
  // COM: var var1 = arg1
  pop.store %arg0, %var0 : !kgen.pointer<index>
  pop.store %arg1, %var1 : !kgen.pointer<index>

  %varIndex = pop.stack_allocation 1 x index
  pop.store %lb, %varIndex : !kgen.pointer<index>

  // COM: for i in range(lb, ub, step)
  // CHECK-NEXT: %0:3 = hlcf.loop (%arg5 = %arg0 : index, %arg6 = %arg1 : index, %arg7 = %arg2 : index)
  // CHECK-SAME: -> (index, index, index)
  hlcf.loop {
    %curIndex = pop.load %varIndex : !kgen.pointer<index>
    // CHECK-NEXT: %[[COND:.*]] = index.cmp slt(%arg7, %arg3)
    %cond = index.cmp slt(%curIndex, %ub)
    // CHECK-NEXT: hlcf.if %[[COND]]
    hlcf.if %cond {
      hlcf.yield
    } else {
      // CHECK: break %arg5, %arg6, %arg7
      hlcf.break
    }

    // COM: var0 += var1 + i
    %v00 = pop.load %var0 : !kgen.pointer<index>
    %v01 = pop.load %var1 : !kgen.pointer<index>
    %v02 = pop.load %varIndex : !kgen.pointer<index>
    // CHECK: %[[V0:.*]] = index.add %arg6, %arg7
    %v03 = index.add %v01, %v02
    // CHECK-NEXT: %[[V1:.*]] = index.add %[[V0]], %arg5
    %v04 = index.add %v03, %v00
    pop.store %v04, %var0 : !kgen.pointer<index>

    // COM: var1 *= var0
    %v10 = pop.load %var0 : !kgen.pointer<index>
    %v11 = pop.load %var1 : !kgen.pointer<index>
    // CHECK-NEXT: %[[V2:.*]] = index.mul %[[V1]], %arg6
    %v12 = index.mul %v10, %v11
    pop.store %v12, %var1 : !kgen.pointer<index>

    %i0 = pop.load %varIndex : !kgen.pointer<index>
    // CHECK-NEXT: %[[V3:.*]] = index.add %arg7, %arg4
    %i1 = index.add %i0, %step
    pop.store %i1, %varIndex : !kgen.pointer<index>
    // CHECK-NEXT: continue %[[V1]], %[[V2]], %[[V3]]
    hlcf.continue
  }

  // COM: return var0, var1
  %r0 = pop.load %var0 : !kgen.pointer<index>
  %r1 = pop.load %var1 : !kgen.pointer<index>
  kgen.return %r0, %r1 : index, index
}

// CHECK-LABEL: @try_region
kgen.func @try_region() {
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  %idx2 = index.constant 2
  %idx3 = index.constant 3
  pop.store %idx2, %0 : !kgen.pointer<index>
  pop.store %idx3, %1 : !kgen.pointer<index>
  // CHECK: %0 = lit.try -> index
  lit.try {
    // CHECK-NEXT: "use"(%idx2)
    %2 = pop.load %0 : !kgen.pointer<index>
    "use"(%2) : (index) -> ()
    pop.store %idx2, %1 : !kgen.pointer<index>
    // CHECK-NEXT: yield %idx2
    lit.try.yield
  // CHECK: except
  } except (%e: !kgen.struct<()>) {
    %2 = pop.load %1 : !kgen.pointer<index>
    // COM: This is dead code.
    // CHECK: "use"(%idx3)
    "use"(%2) : (index) -> ()
    lit.try.yield
  // CHECK: else (%arg0: index) {
  } else {
    // CHECK-NEXT: "use"(%idx2, %arg0)
    %2 = pop.load %0 : !kgen.pointer<index>
    %3 = pop.load %1 : !kgen.pointer<index>
    "use"(%2, %3) : (index, index) -> ()
    lit.try.yield
  // CHECK: }
  }
  // CHECK-NEXT: "use"(%idx3)
  pop.store %idx3, %0 : !kgen.pointer<index>
  %2 = pop.load %0 : !kgen.pointer<index>
  "use"(%2) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: @try_raise
kgen.func @try_raise(%err: index) -> index {
  %0 = pop.stack_allocation 1 x index
  // CHECK: %[[R:.*]] = lit.try -> index
  lit.try {
    %idx0 = index.constant 0
    pop.store %idx0, %0 : !kgen.pointer<index>
    // CHECK: lit.try.raise %arg0, %idx0 : index, index
    lit.try.raise %err : index
  // CHECK: except (%arg1: index, %arg2: index)
  } except (%e: index) {
    // CHECK: yield %arg2
    lit.try.yield
  } else {
    lit.try.yield
  }
  %1 = pop.load %0 : !kgen.pointer<index>
  // CHECK: return %[[R]]
  kgen.return %1 : index
}

// CHECK-LABEL: @pass_new_result
kgen.func @pass_new_result(%arg0: index) {
  %alloc = pop.stack_allocation 1 x index
  // CHECK-NEXT: %0:2 = hlcf.loop () -> (index, index)
  %0 = hlcf.loop () -> index {
    %idx0 = index.constant 0
    pop.store %idx0, %alloc : !kgen.pointer<index>
    // CHECK: break %arg0, %idx0
    hlcf.break %arg0 : index
  }
  kgen.return
}

// CHECK-LABEL: @unknown_region_op
kgen.generator @unknown_region_op() {
  // CHECK-NEXT: pop.stack_allocation
  %alloc0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: pop.stack_allocation
  %alloc1 = pop.stack_allocation 1 x index

  %idx0 = index.constant 0
  %idx1 = index.constant 1
  pop.store %idx0, %alloc0 : !kgen.pointer<index>
  pop.store %idx1, %alloc1 : !kgen.pointer<index>

  // CHECK: region Fn
  kgen.param.declare.region Fn = (%arg0: index) -> (index, index) {
    pop.store %arg0, %alloc1 : !kgen.pointer<index>
    %0 = pop.load %alloc1 : !kgen.pointer<index>
    %1 = pop.load %alloc0 : !kgen.pointer<index>
    kgen.return %0, %1 : index, index
  }

  kgen.return
}

// -----

// CHECK-LABEL: @for_variant
kgen.func @for_variant(%arg0: index, %arg1: index, %arg2: index, %arg4: !pop.scalar<f32>, %arg5: !pop.scalar<f32>) -> (index, index, index, !pop.scalar<f32>) {
  %var0 = pop.stack_allocation 1 x index
  %var1 = pop.stack_allocation 1 x index
  %var2 = pop.stack_allocation 1 x index
  %var3 = pop.stack_allocation 1 x !pop.scalar<f32>

  // COM: var var0 = arg0
  // COM: var var1 = arg1
  // COM: var var2 = arg2
  pop.store %arg0, %var0 : !kgen.pointer<index>
  pop.store %arg1, %var1 : !kgen.pointer<index>
  pop.store %arg2, %var2: !kgen.pointer<index>
  pop.store %arg4, %var3: !kgen.pointer<!pop.scalar<f32>>

  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  //CHECK-NEXT: %[[IDX0:.*]] = index.constant 0
  //CHECK-NEXT: %[[IDX1:.*]] = index.constant 1
  //CHECK-NEXT: %[[IDX2:.*]] = index.constant 2

  // COM: for i in range(2)
  // CHECK: %[[V0:.*]]:4 = hlcf.for [%idx2 to %idx0 step %idx1 sgt sub]
  // CHECK-SAME: (%arg5 = %idx2 : index,
  // CHECK-SAME   %arg6 = %arg0 : index, %arg7 = %arg1 : index, %arg8 = %arg2 : index, %arg9 = %arg3 : !pop.scalar<f32>,
  // CHECK-SAME   %arg10 = %arg4 : !pop.scalar<f32>, %arg11 = %arg0 : index, %arg12 = %arg1 : index, %arg13 = %arg2 : index, %arg14 = %arg3 : !pop.scalar<f32>)
  // CHECK-SAME: -> (index, index, index, !pop.scalar<f32>)
  hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg3 = %idx2 : index, %arg6 = %arg5: !pop.scalar<f32>) {
    // CHECK-NEXT: %[[V1:.*]] = index.sub %arg5, %[[IDX1]]
    %0 = index.sub %arg3, %idx1
    %v00 = pop.load %var0 : !kgen.pointer<index>
    %v01 = pop.load %var1 : !kgen.pointer<index>
    %v02 = pop.load %var2: !kgen.pointer<index>
    // COM: var1 + var2
    // CHECK-NEXT: %[[V2:.*]] = index.add %arg12, %arg13
    %v03 = index.add %v01, %v02
    // COM: var0 + var1 + var2
    // CHECK-NEXT: %[[V3:.*]] = index.add %[[V2]], %arg11
    %v04 = index.add %v03, %v00
    %v05 = pop.load %var3 : !kgen.pointer<!pop.scalar<f32>>

    // COM: var3 + %arg6
    // CHECK-NEXT: %[[V6:.*]] = pop.add %arg14, %arg10
    %v07 = pop.add %v05, %arg6: !pop.scalar<f32>

    pop.store %v07, %var3: !kgen.pointer<!pop.scalar<f32>>

    // COM: var0 = var0 + var1 + var2
    pop.store %v04, %var0 : !kgen.pointer<index>

    %v10 = pop.load %var0 : !kgen.pointer<index>
    %v11 = pop.load %var1 : !kgen.pointer<index>
    // COM: var0 * var1
    // CHECK-NEXT: %[[V4:.*]] = index.mul %[[V3]], %arg12
    %v12 = index.mul %v10, %v11
    // COM: var1 = var0 * var1
    pop.store %v12, %var1 : !kgen.pointer<index>

    %i0 = pop.load %var2: !kgen.pointer<index>
    // COM: var2 + iter
    // CHECK-NEXT: %[[V5:.*]] = index.add %arg13, %[[V1]]
    %i1 = index.add %i0, %0
    // COM: var2 = var2 + iter
    pop.store %i1, %var2: !kgen.pointer<index>

    // CHECK-DAG: hlcf.for.yield
    // CHECK-SAME: [induction_var (%[[V1]] : index)]
    // CHECK-SAME: [retvals (%[[V3]], %[[V4]], %[[V5]], %[[V6]] : index, index, index, !pop.scalar<f32>)]
    // CHECK-SAME: [iterargs (%[[V6]], %[[V3]], %[[V4]], %[[V5]], %[[V6]] : !pop.scalar<f32>, index, index, index, !pop.scalar<f32>)]
    hlcf.for.yield [induction_var (%0 : index)] [retvals ()] [iterargs (%v07: !pop.scalar<f32>)]
  } {unrollLevel = #hlcf<unroll_level full>}

  %r0 = pop.load %var0 : !kgen.pointer<index>
  %r1 = pop.load %var1 : !kgen.pointer<index>
  %r2 = pop.load %var2 : !kgen.pointer<index>
  %r3 = pop.load %var3 : !kgen.pointer<!pop.scalar<f32>>

  // CHECK-DAG: kgen.return %[[V0]]#0, %[[V0]]#1, %[[V0]]#2, %[[V0]]#3 : index, index, index, !pop.scalar<f32>
  kgen.return %r0, %r1, %r2, %r3 : index, index, index, !pop.scalar<f32>
}


// CHECK-LABEL: @nested_alloc_in_for
kgen.generator @nested_alloc_in_for(%arg0: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  //CHECK:      %[[IDX0:.*]] = index.constant 0
  //CHECK-NEXT: %[[IDX1:.*]] = index.constant 1
  //CHECK-NEXT: %[[IDX2:.*]] = index.constant 2

  // COM: var0 = %arg0
  %v0 = pop.stack_allocation 1 x index
  pop.store %arg0, %v0 : !kgen.pointer<index>

  // CHECK:      %[[V0:.*]] = hlcf.for [%idx2 to %idx0 step %idx1 sgt sub]
  // CHECK-SAME: (%arg1 = %idx2 : index, %arg2 = %arg0 : index, %arg3 = %idx0 : index, %arg4 = %arg0 : index)
  // CHECK-SAME: -> index
  hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg1 = %idx2 : index, %arg2 = %idx0 : index) {
    // COM: iter0
    // CHECK-NEXT: %[[V1:.*]] = index.sub %arg1, %[[IDX1]]
    %v00 = index.sub %arg1, %idx1

    %v02 = pop.load %v0 : !kgen.pointer<index>
    // COM: iter0 * var0
    // CHECK-NEXT: %[[V2:.*]] = index.mul %[[V1]], %arg4
    %v03 = index.mul %v00, %v02

    // COM: var2
    %v01 = pop.stack_allocation 1 x index
    pop.store %v03, %v01 : !kgen.pointer<index>

    // CHECK-NEXT: %[[V3:.*]] = hlcf.for [%idx2 to %idx0 step %idx1 sgt sub]
    // CHECK-SAME: (%arg5 = %idx2 : index, %arg6 = %2 : index, %arg7 = %idx0 : index, %arg8 = %2 : index)
    // CHECK-SAME: -> index
    hlcf.for [%idx2 to %idx0 step %idx1 sgt sub] (%arg3 = %idx2 : index, %arg4 = %idx0 : index) {
      // COM: iter1
      // CHECK-NEXT: %[[V4:.*]] = index.sub %arg5, %[[IDX1]]
      %v10 = index.sub %arg3, %idx1
      %3 = pop.load %v01 : !kgen.pointer<index>

      // COM: var2 + %arg4
      // CHECK-NEXT: %[[V5:.*]] = index.add %arg8, %arg7
      %4 = index.add %3, %arg4

      pop.store %4, %v01 : !kgen.pointer<index>
      // CHECK-DAG:  hlcf.for.yield
      // CHECK-SAME: [induction_var (%[[V4]] : index)]
      // CHECK-SAME: [retvals (%[[V5]] : index)]
      // CHECK-SAME: [iterargs (%[[V5]], %[[V5]] : index, index)]
      hlcf.for.yield [induction_var (%v10 : index)] [retvals ()] [iterargs (%4: index)]
    } {unrollLevel = #hlcf<unroll_level full>}
    %v04 = pop.load %v01 : !kgen.pointer<index>
    pop.store %v04, %v0 : !kgen.pointer<index>
    // CHECK-DAG:  hlcf.for.yield
    // CHECK-SAME: [induction_var (%[[V1]] : index)]
    // CHECK-SAME: [retvals (%[[V3]] : index)]
    // CHECK-SAME: [iterargs (%[[V3]], %[[V3]] : index, index)]
    hlcf.for.yield [induction_var (%v00 : index)] [retvals ()] [iterargs (%v04: index)]
  } {unrollLevel = #hlcf<unroll_level full>}

  %v1 = pop.load %v0 : !kgen.pointer<index>
  // CHECK-DAG: kgen.return %[[V0]]
  kgen.return %v1 : index
}

// CHECK-LABEL: kgen.func @elif
kgen.func @elif(%arg0 : index, %arg1: index, %arg2: index) -> index {
  %varCondition = pop.stack_allocation 1 x index
  pop.store %arg0, %varCondition : !kgen.pointer<index>

  %varThen = pop.stack_allocation 1 x index
  pop.store %arg1, %varThen : !kgen.pointer<index>

  %varElse = pop.stack_allocation 1 x index
  pop.store %arg2, %varElse : !kgen.pointer<index>

  // CHECK-NEXT: %idx0
  // CHECK:      %idx1
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  // CHECK-NEXT: %0:4 = hlcf.elif -> index, index, index, index {
  // CHECK-NEXT:   [[V3:%*.]] = index.add %arg0, %idx1
  // CHECK-NEXT:   [[V4:%*.]] = index.cmp eq([[V3]], %idx0)
  // CHECK-NEXT:   hlcf.elif.yield [[V4]], [[V3]], %arg1, %arg2 : i1, index, index, index
  // CHECK-NEXT: } then (%arg3: index, %arg4: index, %arg5: index){
  // CHECK-NEXT:   [[W3:%*.]] = index.add %arg4, %idx1
  // CHECK-NEXT:   hlcf.yield [[W3]], %arg3, [[W3]], %arg5 : index, index, index, index
  // CHECK-NEXT: } else (%arg3: index, %arg4: index, %arg5: index){
  // CHECK-NEXT:   [[U3:%*.]] = index.add %arg5, %idx1
  // CHECK-NEXT:   hlcf.yield [[U3]], %arg3, %arg4, [[U3]] : index, index, index, index
  // CHECK-NEXT: }
  %0 = hlcf.elif -> index {
    %1 = pop.load %varCondition : !kgen.pointer<index>
    %var2 = index.add %1, %idx1
    pop.store %var2, %varCondition : !kgen.pointer<index>
    %c = index.cmp eq(%var2, %idx0)
    hlcf.elif.yield %c : i1
  } then {
    %4 = pop.load %varThen : !kgen.pointer<index>
    %var5 = index.add %4, %idx1
    pop.store %var5, %varThen : !kgen.pointer<index>
    hlcf.yield %var5 : index
  } else {
    %5 = pop.load %varElse : !kgen.pointer<index>
    %var6 = index.add %5, %idx1
    pop.store %var6, %varElse : !kgen.pointer<index>
    hlcf.yield %var6 : index
  }

  %3 = pop.load %varCondition : !kgen.pointer<index>
  %6 = pop.load %varThen : !kgen.pointer<index>
  %7 = pop.load %varElse : !kgen.pointer<index>

  // CHECK-NEXT: %1 = index.add %0#1, %0#2
  // CHECK-NEXT: %2 = index.add %1, %0#3
  // CHECK-NEXT: kgen.return %2 : index
  %res1 = index.add %3, %6
  %res2 = index.add %res1, %7
  kgen.return %res2 : index
}

// CHECK-LABEL: kgen.func @lifetime_markers
kgen.func @lifetime_markers(%arg0: index) -> index {
  %0 = pop.stack_allocation 1 x index
  // CHECK-NEXT: lifetime.start()
  pop.stack_alloc.lifetime.start(%0) : !kgen.pointer<index>
  pop.store %arg0, %0 : !kgen.pointer<index>
  %1 = pop.load %0 : !kgen.pointer<index>
  // CHECK-NEXT: lifetime.end()
  pop.stack_alloc.lifetime.end(%0) : !kgen.pointer<index>
  // CHECK-NEXT: return %arg0
  kgen.return %1 : index
}

// CHECK-LABEL: kgen.func @lifetime_multiple_args
kgen.func @lifetime_multiple_args() {
  // CHECK-NEXT: %0 = pop.stack_allocation 1 x index
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x pointer<index>
  // CHECK-NEXT: lifetime.start(%0)
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<index>, !kgen.pointer<pointer<index>>
  pop.store %0, %1 : !kgen.pointer<pointer<index>>
  // CHECK-NEXT: lifetime.end(%0)
  pop.stack_alloc.lifetime.end(%1, %0) : !kgen.pointer<pointer<index>>, !kgen.pointer<index>
  kgen.return
}

// CHECK-LABEL: kgen.generator @param_for
kgen.generator @param_for() -> index {
  %mem = pop.stack_allocation 1 x index
  // CHECK-NEXT: %idx0 = index.constant 0
  %idx0 = index.constant 0
  pop.store %idx0, %mem : !kgen.pointer<index>
  // CHECK-NEXT: %0 = kgen.param.for
  // CHECK-NEXT: (%arg0 = %idx0 : index) -> index
  kgen.param.for decl: index in :index 2 iter :(index, !kgen.pointer<struct<(index, index, i1)>> byref_result) -> !kgen.none @wrapper {
    // CHECK-NEXT: %index = kgen.param.constant
    %0 = kgen.param.constant = <decl>
    %1 = pop.load %mem : !kgen.pointer<index>
    // CHECK-NEXT: %1 = index.add %arg0, %index
    %2 = index.add %1, %0
    pop.store %2, %mem : !kgen.pointer<index>
    // CHECK-NEXT: kgen.param.for.continue %1
    kgen.param.for.continue
  // CHECK-NEXT: } else (%arg0: index) {
  } else {
    %0 = pop.load %mem : !kgen.pointer<index>
    // CHECK-NEXT: %1 = index.add %arg0, %idx0
    %1 = index.add %0, %idx0
    pop.store %1, %mem : !kgen.pointer<index>
    // CHECK-NEXT: kgen.param.yield %1 : index
    kgen.param.yield
  // CHECK-NEXT: }
  }
  %1 = pop.load %mem : !kgen.pointer<index>
  // CHECK-NEXT: return %0
  kgen.return %1 : index
}
