// RUN: kgen-opt -check-lifetimes -split-input-file -mlir-print-debuginfo %s | FileCheck %s

// CHECK: ![[DI_PTR_TYPE:.*]] = !debuginfo.ti.ptr<!lit.declref<@S>>
// CHECK: ![[DI_S_TYPE:.*]] = !debuginfo.unresolved<!lit.declref<@S>>
// CHECK: #[[DIEXPR_IRVALUE:.*]] = #debuginfo.expr.irvalue : ![[DI_PTR_TYPE]]
// CHECK: #[[DIEXPR_DEREF:.*]] = #debuginfo.expr.deref<#[[DIEXPR_IRVALUE]]> : ![[DI_S_TYPE]]
// CHECK: #[[DISP:.*]] = #debuginfo.subprogram<compileUnit = #{{.*}}, scope = #{{.*}}, name = <"test">, linkageName = "test", file = #{{.*}}, line = 1, scopeLine = 1, subprogramFlags = Definition>
// CHECK: #[[DIVAR_X:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "x", file = #{{.*}}, line = 10> : ![[DI_S_TYPE]]
// CHECK: #[[DIVAR_Y:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "y", file = #{{.*}}, line = 13> : ![[DI_S_TYPE]]
// CHECK: #[[DIARG_X:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "x", file = #{{.*}}, line = 10, arg = 1> : ![[DI_S_TYPE]]
// CHECK: #[[DIARG_YS:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "ys", file = #{{.*}}, line = 13, arg = 2>
// CHECK: #[[DIARG_Z:.*]] = #debuginfo.local_variable<scope = #[[DISP]], name = "z", file = #{{.*}}, line = 15, arg = 3>

#file = #debuginfo.file<"test.mlir" in "">
#compile_unit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "LIT", isOptimized = true, emissionKind = Full>
#sp = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = <"test">, linkageName = "test", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition"> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#locX = loc(fused<#sp>["test.mlir":10:10])
#locUse = loc(fused<#sp>["test.mlir":12:10])
#locY = loc(fused<#sp>["test.mlir":13:10])
#locRet = loc(fused<#sp>["test.mlir":14:10])
#locZ = loc(fused<#sp>["test.mlir":15:10])

lit.struct.decl @S attributes {
  destructor =
    #kgen.symbol.constant<@S::@__del__> : !lit.signature<[1](!lit.ref<@S, mut *[0,0]> owned_in_mem) -> !kgen.none>} {

  lit.struct.field a : index

  lit.func @__init__(%self: !lit.ref<@S, mut selflife> init_self, %num: index) -> !kgen.none {
    %0 = lit.ref.struct.ger %self[a] : <index, mut selflife> from @S
    lit.ref.store %num, %0 : !lit.ref<index, mut selflife>

    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
  }
}

// CHECK-LABEL: lit.func @test_var
lit.func @test_var() -> index {
  // Create `x`.
  // CHECK-NEXT: %x = lit.var.decl "x" var : ![[VAR_X_TYPE:.*]] loc
  %x = lit.var.decl "x"  var : !lit.ref<@S, mut *"x`0"> loc(#locX)
  %0 = kgen.param.constant: index = <42> loc(#locX)
  // CHECK: debuginfo.value #[[DIVAR_X]] #[[DIEXPR_DEREF]] = %x : ![[VAR_X_TYPE]]
  // CHECK-NEXT: lifetime.start %x
  // CHECK-NEXT: lit.call @S::@__init__{{.*}}(%x,
  lit.call @S::@__init__[mut *"x`0"](%x, %0) : !lit.signature<[1]("self": !lit.ref<@S, mut *[0,0]> init_self, |, "num": index) -> !kgen.none> loc(#locX)

  // Use `x.a`.
  // CHECK: lit.ref.struct.ger {{.*}} loc(#[[LOC_USE:.*]])
  // CHECK-NEXT: lit.ref.load
  %x_a = lit.ref.struct.ger %x[a] : <index, mut *"x`0"> from @S loc(#locUse)
  %x_a_val = lit.ref.load %x_a : <index, mut *"x`0"> loc(#locUse)

  // `x` can be destroyed here.
  // CHECK-NEXT: debuginfo.kill #[[DIVAR_X]] loc(#[[LOC_USE]])
  // CHECK-NEXT: lit.call @S::@__del__{{.*}}(%x) : {{.*}} loc(#[[LOC_USE]])
  // CHECK-NEXT: lifetime.end %x

  // `y` is a synthetic variable. Should not have any debuginfo generated.
  %y = lit.var.decl "y"  synth : !lit.ref<@S, mut *"y`0"> loc(#locY)
  // CHECK-NOT: debuginfo.{{(value)|(kill)}}

  kgen.return %x_a_val : index loc(#locRet)
} loc(fused<#sp>["test.mlir":10:10])

// CHECK-LABEL: lit.func @test_uninit_var
lit.func @test_uninit_var() {
  // CHECK-NOT: debuginfo.{{(value)|(kill)}}
  %x = lit.var.decl "x"  var : !lit.ref<@S, mut *"x`0"> loc(#locX)
  kgen.return loc(#locRet)
} loc(fused<#sp>["test.mlir":10:10])

// CHECK-LABEL: lit.func @test_def_in_loop
lit.func @test_def_in_loop() {
  // Create `x`.
  // CHECK-NEXT: %x = lit.var.decl "x" var : ![[VAR_X_TYPE:.*]] loc
  %x = lit.var.decl "x"  var : !lit.ref<@S, mut *"x`0"> loc(#locX)
  // CHECK: hlcf.loop "loop0"
  hlcf.loop "loop0" {
    %0 = kgen.param.constant: index = <42> loc(#locX)
    // CHECK: debuginfo.value #[[DIVAR_X]] #[[DIEXPR_DEREF]] = %x : ![[VAR_X_TYPE]]
    // CHECK-NEXT: lifetime.start %x
    // CHECK-NEXT: lit.call @S::@__init__{{.*}}(%x,
    lit.call @S::@__init__[mut *"x`0"](%x, %0) : !lit.signature<[1]("self": !lit.ref<@S, mut *[0,0]> init_self, |, "num": index) -> !kgen.none> loc(#locX)
    // CHECK-NEXT: debuginfo.kill #[[DIVAR_X]]
    // CHECK-NEXT: call @S::@__del__
    // CHECK-NEXT: lifetime.end %x
    hlcf.continue loc(#locX)
  } loc(#locX)
  kgen.return loc(#locRet)
} loc(fused<#sp>["test.mlir":10:10])

// CHECK-LABEL: lit.func @test_def_twice
lit.func @test_def_twice() -> index {
  // Create `x`.
  // CHECK-NEXT: %x = lit.var.decl "x" var : ![[VAR_X_TYPE:.*]] loc
  %x = lit.var.decl "x"  var : !lit.ref<@S, mut *"x`0"> loc(#locX)
  %0 = kgen.param.constant: index = <42> loc(#locX)

  // CHECK: debuginfo.value #[[DIVAR_X]] #[[DIEXPR_DEREF]] = %x : ![[VAR_X_TYPE]]
  // CHECK-NEXT: lifetime.start %x
  // CHECK-NEXT: lit.call @S::@__init__
  lit.call @S::@__init__[mut *"x`0"](%x, %0) : !lit.signature<[1]("self": !lit.ref<@S, mut *[0,0]> init_self, |, "num": index) -> !kgen.none> loc(#locX)
  // CHECK: debuginfo.kill #[[DIVAR_X]]
  // CHECK-NEXT: call @S::@__del__
  // CHECK-NEXT: lifetime.end %x

  // CHECK-NEXT: debuginfo.value #[[DIVAR_X]]
  // CHECK-NEXT: lifetime.start %x
  // CHECK-NEXT: lit.call @S::@__init__
  lit.call @S::@__init__[mut *"x`0"](%x, %0) : !lit.signature<[1]("self": !lit.ref<@S, mut *[0,0]> init_self, |, "num": index) -> !kgen.none> loc(#locX)
  // CHECK-NEXT: debuginfo.kill #[[DIVAR_X]]
  // CHECK-NEXT: call @S::@__del__
  // CHECK-NEXT: lifetime.end %x

  kgen.return %0 : index loc(#locRet)
} loc(fused<#sp>["test.mlir":10:10])

// CHECK-LABEL: lit.func @test_consumed
lit.func @test_consumed() -> index {
  // Create `x`.
  // CHECK-NEXT: %x = lit.var.decl "x" var
  %x = lit.var.decl "x"  var : !lit.ref<@S, mut xlife> loc(#locX)
  %0 = kgen.param.constant: index = <42> loc(#locX)
  // CHECK: debuginfo.value #[[DIVAR_X]] #[[DIEXPR_DEREF]] = %x
  // CHECK-NEXT: lifetime.start %x
  // CHECK-NEXT: lit.call @S::@__init__
  lit.call @S::@__init__(%x, %0) : !lit.signature<("self": !lit.ref<@S, mut xlife> init_self, |, "num": index) -> !kgen.none> loc(#locX)

  // Create `y`.
  // CHECK: %[[VAR_Y:.*]] = lit.var.decl "y" var
  %y = lit.var.decl "y"  var : !lit.ref<@S, mut ylife> loc(#locY)

  // Move `x` into `y`.
  // CHECK: debuginfo.kill #[[DIVAR_X]] loc(#[[LOC_USE:.*]])
  // CHECK-NEXT: [[X_TRANSFER:%.*]] = lit.transfer_mem_ownership %x {{.*}} loc(#[[LOC_USE]])
  // CHECK-NEXT: debuginfo.value #[[DIVAR_Y]] #[[DIEXPR_DEREF]] = %[[VAR_Y]]
  // CHECK-NEXT: lifetime.start %y
  // CHECK-NEXT: lit.call @S::@__moveinit__{{.*}}(%y, [[X_TRANSFER]]) {{.*}} loc(#[[LOC_USE]])
  // CHECK-NEXT: lifetime.end %x
  %x_moved = lit.transfer_mem_ownership %x : !lit.ref<@S, mut xlife> -> !lit.ref<@S, mut xlifetrans> {paramDecl = #kgen<param.decl xlifetrans : !lit.lifetime<1>>} loc(#locUse)
  lit.call @S::@__moveinit__(%y, %x_moved) : !lit.signature<(!lit.ref<@S, mut ylife> init_self, !lit.ref<@S, mut xlifetrans> owned_in_mem) -> !kgen.none> loc(#locUse)

  // Last use of `y`.
  // CHECK: [[Y_A:%.*]] = lit.ref.struct.ger {{.*}} loc(#[[LOC_RET:.*]])
  // CHECK-NEXT: lit.ref.load [[Y_A]]
  // CHECK-NEXT: debuginfo.kill #[[DIVAR_Y]] loc(#[[LOC_RET]])
  // CHECK-NEXT: call @S::@__del__{{.*}}(%y)
  // CHECK-NEXT: lifetime.end %y
  %y_a = lit.ref.struct.ger %y[a] : <index, mut ylife> from @S loc(#locRet)
  %y_a_val = lit.ref.load %y_a : <index, mut ylife> loc(#locRet)
  kgen.return %y_a_val : index loc(#locRet)
} loc(fused<#sp>["test.mlir":10:10])

// CHECK-LABEL: lit.func @test_arg(%x: {{.*}}, %ys: {{.*}}, %z:
lit.func @test_arg(%x: !lit.ref<@S, mut *"x"> loc(#locX) owned_in_mem, %ys: !kgen.variadic<!lit.ref<@S, imm *"ys">, borrow_in_mem> loc(#locY) borrow|var, %z: index loc(#locZ) owned) -> index {
  // CHECK: debuginfo.value #[[DIARG_X]] #[[DIEXPR_DEREF]] = %x
  // CHECK-NOT: debuginfo.value {{.*}} = %ys
  // CHECK-NOT: debuginfo.value {{.*}} = %z
  %x_a = lit.ref.struct.ger %x[a] : <index, mut *"x"> from @S loc(#locUse)
  %x_a_val = lit.ref.load %x_a : <index, mut *"x"> loc(#locUse)

  // `x` can be destroyed here.
  // CHECK: debuginfo.kill #[[DIARG_X]] loc(#[[LOC_USE]])
  // CHECK-NEXT: lit.call @S::@__del__{{.*}}(%x) : {{.*}} loc(#[[LOC_USE]])

  // `ys` is shadowed (using a fake shadowing type for simplicity).
  %yshadow = lit.var.decl "ys" arg(1) : !lit.ref<!kgen.variadic<!lit.ref<@S, imm *"ys">, borrow_in_mem>, mut *"ys"> loc(#locY)
  lit.ref.store %ys, %yshadow : !lit.ref<!kgen.variadic<!lit.ref<@S, imm *"ys">, borrow_in_mem>, mut *"ys"> loc(#locY)
  // CHECK: %[[YSHADOW:.*]] = lit.var.decl "ys"
  // CHECK: debuginfo.value #[[DIARG_YS]] {{.*}} = %[[YSHADOW]]

  // `z` is shadowed.
  %zshadow = lit.var.decl "z" arg(2) : !lit.ref<index, mut *"z"> loc(#locZ)
  lit.ref.store %z, %zshadow : <index, mut *"z"> loc(#locZ)
  // CHECK: %[[ZSHADOW:.*]] = lit.var.decl "z"
  // CHECK: debuginfo.value #[[DIARG_Z]] {{.*}} = %[[ZSHADOW]]

  kgen.return %x_a_val : index loc(#locRet)
} loc(fused<#sp>["test.mlir":10:10])
