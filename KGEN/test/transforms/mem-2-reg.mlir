// RUN: kgen-opt -split-input-file -mem-2-reg -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @simple_add
kgen.generator @simple_add(%arg0: index, %arg1: index) -> index {
  // CHECK-NEXT: %0 = index.add %arg0, %arg1
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>

  %1 = pop.stack_allocation 1 x index
  pop.store %arg1, %1 : !pop.pointer<index>

  %2 = pop.load %0 : !pop.pointer<index>
  %3 = pop.load %1 : !pop.pointer<index>
  %4 = index.add %2, %3
  pop.store %4, %1 : !pop.pointer<index>

  %5 = pop.load %1 : !pop.pointer<index>
  // CHECK-NEXT: return %0
  kgen.return %5 : index
}

// CHECK-LABEL: @use_in_region
kgen.generator @use_in_region(%arg0: index, %arg1: i1) -> index {
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>

  // CHECK-NEXT: hlcf.if
  hlcf.if %arg1 {
    // CHECK-NEXT: kgen.return %arg0
    %1 = pop.load %0 : !pop.pointer<index>
    kgen.return %1 : index
  } else {
    hlcf.yield
  }

  // CHECK-NOT: pop.load
  %1 = pop.load %0 : !pop.pointer<index>
  // CHECK: return %arg0 : index
  kgen.return %1 : index
}

// CHECK-LABEL: @store_in_region
kgen.generator @store_in_region(%arg0: index, %arg1: index, %arg2: i1) -> index {
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>

  // CHECK-NEXT: %0 = hlcf.if %arg2 -> index
  hlcf.if %arg2 {
    %1 = pop.load %0 : !pop.pointer<index>
    // CHECK-NEXT: return %arg0
    kgen.return %1 : index
  } else {
    // CHECK: hlcf.yield %arg1 : index
    pop.store %arg1, %0 : !pop.pointer<index>
    hlcf.yield
  }

  %1 = pop.load %0 : !pop.pointer<index>
  // CHECK: return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @unknown_use
kgen.generator @unknown_use(%arg0: index) -> index {
  // CHECK-NEXT: stack_allocation
  %0 = pop.stack_allocation 1 x index
  pop.store %arg0, %0 : !pop.pointer<index>
  "unknown.use"(%0) : (!pop.pointer<index>) -> ()
  %1 = pop.load %0 : !pop.pointer<index>
  kgen.return %1 : index
}

// CHECK-LABEL: @nested_alloc
kgen.generator @nested_alloc(%arg0: index) -> index {
  // CHECK-NEXT: %0 = hlcf.loop
  %0 = hlcf.loop () -> index {
    %1 = pop.stack_allocation 1 x index
    pop.store %arg0, %1 : !pop.pointer<index>
    // CHECK-NEXT: %1 = hlcf.loop
    %2 = hlcf.loop () -> index {
      %3 = pop.load %1 : !pop.pointer<index>
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
  %1 = pop.load %0 : !pop.pointer<index>
  // CHECK-NEXT: kgen.return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @if_empty_block
kgen.generator @if_empty_block(%arg0: i1, %arg1: index) -> index{
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  pop.store %arg1, %0 : !pop.pointer<index>
  // CHECK-NEXT: %0 = hlcf.if %arg0 -> index
  hlcf.if %arg0 {
    %2 = pop.load %0 : !pop.pointer<index>
    pop.store %2, %1 : !pop.pointer<index>
    // CHECK-NEXT: yield %arg1
    hlcf.yield
  } else {
    // CHECK: %1 = kgen.undef : index
    // CHECK-NEXT: yield %1
    hlcf.yield
  }
  %2 = pop.load %1 : !pop.pointer<index>
  // CHECK: return %0
  kgen.return %2 : index
}

// CHECK-LABEL: @store_alloca
kgen.func @store_alloca() -> i32 {
  // CHECK-NEXT: pop.stack_allocation 1 x i32
  // CHECK-NEXT: pop.load
  %0 = pop.stack_allocation 1 x !pop.pointer<i32>
  %1 = pop.stack_allocation 1 x i32
  pop.store %1, %0 : !pop.pointer<pointer<i32>>
  %2 = pop.load %0 : !pop.pointer<pointer<i32>>
  %3 = pop.load %2 : !pop.pointer<i32>
  kgen.return %3 : i32
}

// CHECK-LABEL: @loop_variant
kgen.func @loop_variant(%arg0: index, %arg1: index, %lb: index, %ub: index, %step: index) -> (index, index) {
  %var0 = pop.stack_allocation 1 x index
  %var1 = pop.stack_allocation 1 x index
  // COM: var var0 = arg0
  // COM: var var1 = arg1
  pop.store %arg0, %var0 : !pop.pointer<index>
  pop.store %arg1, %var1 : !pop.pointer<index>

  %varIndex = pop.stack_allocation 1 x index
  pop.store %lb, %varIndex : !pop.pointer<index>

  // COM: for i in range(lb, ub, step)
  // CHECK-NEXT: %0:3 = hlcf.loop (%arg5 = %arg0 : index, %arg6 = %arg1 : index, %arg7 = %arg2 : index)
  // CHECK-SAME: -> (index, index, index)
  hlcf.loop {
    %curIndex = pop.load %varIndex : !pop.pointer<index>
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
    %v00 = pop.load %var0 : !pop.pointer<index>
    %v01 = pop.load %var1 : !pop.pointer<index>
    %v02 = pop.load %varIndex : !pop.pointer<index>
    // CHECK: %[[V0:.*]] = index.add %arg6, %arg7
    %v03 = index.add %v01, %v02
    // CHECK-NEXT: %[[V1:.*]] = index.add %[[V0]], %arg5
    %v04 = index.add %v03, %v00
    pop.store %v04, %var0 : !pop.pointer<index>

    // COM: var1 *= var0
    %v10 = pop.load %var0 : !pop.pointer<index>
    %v11 = pop.load %var1 : !pop.pointer<index>
    // CHECK-NEXT: %[[V2:.*]] = index.mul %[[V1]], %arg6
    %v12 = index.mul %v10, %v11
    pop.store %v12, %var1 : !pop.pointer<index>

    %i0 = pop.load %varIndex : !pop.pointer<index>
    // CHECK-NEXT: %[[V3:.*]] = index.add %arg7, %arg4
    %i1 = index.add %i0, %step
    pop.store %i1, %varIndex : !pop.pointer<index>
    // CHECK-NEXT: continue %[[V1]], %[[V2]], %[[V3]]
    hlcf.continue
  }

  // COM: return var0, var1
  %r0 = pop.load %var0 : !pop.pointer<index>
  %r1 = pop.load %var1 : !pop.pointer<index>
  kgen.return %r0, %r1 : index, index
}

// CHECK-LABEL: @try_region
kgen.func @try_region() {
  %0 = pop.stack_allocation 1 x index
  %1 = pop.stack_allocation 1 x index
  %idx2 = index.constant 2
  %idx3 = index.constant 3
  pop.store %idx2, %0 : !pop.pointer<index>
  pop.store %idx3, %1 : !pop.pointer<index>
  // CHECK: %0 = lit.try -> index
  lit.try {
    // CHECK-NEXT: "use"(%idx2)
    %2 = pop.load %0 : !pop.pointer<index>
    "use"(%2) : (index) -> ()
    pop.store %idx2, %1 : !pop.pointer<index>
    // CHECK-NEXT: yield %idx2
    lit.try.yield
  // CHECK: except
  } except (%e: !pop.struct<>) {
    %2 = pop.load %1 : !pop.pointer<index>
    // COM: This is dead code.
    // CHECK: "use"(%idx3)
    "use"(%2) : (index) -> ()
    lit.try.yield
  // CHECK: else
  } else {
  // CHECK-NEXT: ^bb0(%arg0: index):
    // CHECK-NEXT: "use"(%idx2, %arg0)
    %2 = pop.load %0 : !pop.pointer<index>
    %3 = pop.load %1 : !pop.pointer<index>
    "use"(%2, %3) : (index, index) -> ()
    lit.try.yield
  // CHECK: }
  }
  // CHECK-NEXT: "use"(%idx3)
  pop.store %idx3, %0 : !pop.pointer<index>
  %2 = pop.load %0 : !pop.pointer<index>
  "use"(%2) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: @try_raise
kgen.func @try_raise(%err: index) -> index {
  %0 = pop.stack_allocation 1 x index
  // CHECK: %[[R:.*]] = lit.try -> index
  lit.try {
    %idx0 = index.constant 0
    pop.store %idx0, %0 : !pop.pointer<index>
    // CHECK: lit.try.raise %arg0, %idx0 : index, index
    lit.try.raise %err : index
  // CHECK: except (%arg1: index, %arg2: index)
  } except (%e: index) {
    // CHECK: yield %arg2
    lit.try.yield
  } else {
    lit.try.yield
  }
  %1 = pop.load %0 : !pop.pointer<index>
  // CHECK: return %[[R]]
  kgen.return %1 : index
}

// CHECK-LABEL: @pass_new_result
kgen.func @pass_new_result(%arg0: index) {
  %alloc = pop.stack_allocation 1 x index
  // CHECK-NEXT: %0:2 = hlcf.loop () -> (index, index)
  %0 = hlcf.loop () -> index {
    %idx0 = index.constant 0
    pop.store %idx0, %alloc : !pop.pointer<index>
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
  pop.store %idx0, %alloc0 : !pop.pointer<index>
  pop.store %idx1, %alloc1 : !pop.pointer<index>

  // CHECK: region Fn
  kgen.param.declare.region Fn = (%arg0: index) -> (index, index) {
    pop.store %arg0, %alloc1 : !pop.pointer<index>
    %0 = pop.load %alloc1 : !pop.pointer<index>
    %1 = pop.load %alloc0 : !pop.pointer<index>
    kgen.return %0, %1 : index, index
  }

  kgen.return
}

// -----

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "MLIR", isOptimized = true, emissionKind = Full>

#callerSp = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  name = "mem2reg_valueop",
  linkageName = "mem2reg_valueop",
  file = #file,
  line = 0,
  scopeLine = 0,
  subprogramFlags = "Definition"
> : !debuginfo.subroutine<(index) -> (): DW_CC_normal>

#local_variable = #debuginfo.local_variable<scope = #callerSp, name = "0", file = #file, line = 0, arg = 0, alignInBits = 0> : !debuginfo.unresolved<index>

#fileLoc = loc("foo.mlir":0:0)
#loc = loc(fused<#callerSp>[#fileLoc])

// CHECK-LABEL: @mem2reg_valueop
kgen.func @mem2reg_valueop(%arg0: index) {
  // CHECK-NEXT: debuginfo.value #local_variable = %arg0 : index
  %0 = pop.stack_allocation 1 x index loc(#loc)
  pop.store %arg0, %0 : !pop.pointer<index> loc(#loc)
  debuginfo.value #local_variable = %0 : !pop.pointer<index> loc(#loc)
  kgen.return loc(#loc)
} loc(#loc)
