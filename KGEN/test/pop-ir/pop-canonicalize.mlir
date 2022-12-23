// RUN: kgen-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @struct_construct
kgen.func @struct_construct() -> !pop.struct<si4, ui4> {
  // CHECK-NEXT: constant: !pop.struct<si4, ui4> = <#pop.struct<-3, 7>>
  %0 = kgen.param.constant: si4 = <-3>
  %1 = kgen.param.constant: ui4 = <7>
  %2 = pop.struct.construct(%0, %1) : !pop.struct<si4, ui4>
  kgen.return %2 : !pop.struct<si4, ui4>
}

// CHECK-LABEL: @struct_get
kgen.func @struct_get() -> si4 {
  // CHECK-NEXT: constant: si4 = <-3>
  %0 = kgen.param.constant: !pop.struct<si4, ui4> = <#pop.struct<-3, 7>>
  %1 = pop.struct.get %0[0] : !pop.struct<si4, ui4>
  kgen.return %1 : si4
}

// CHECK-LABEL: @struct_replace
kgen.func @struct_replace() -> !pop.struct<si4, ui4> {
  // CHECK-NEXT: constant: !pop.struct<si4, ui4> = <#pop.struct<-5, 7>>
  %0 = kgen.param.constant: si4 = <-5>
  %1 = kgen.param.constant: !pop.struct<si4, ui4> = <#pop.struct<-3, 7>>
  %2 = pop.struct.replace %0, %1[0] : !pop.struct<si4, ui4>
  kgen.return %2 : !pop.struct<si4, ui4>
}

// CHECK-LABEL: @array_create
kgen.func @array_create() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: !pop.array<2, index> = <#pop.array<0, 0>>
  %idx0 = index.constant 0
  %0 = pop.array.create [%idx0, %idx0] : !pop.array<2, index>
  kgen.return %0 : !pop.array<2, index>
}

// CHECK-LABEL: @array_repeat
kgen.func @array_repeat() -> !pop.array<3, index> {
  // CHECK-NEXT: constant: !pop.array<3, index> = <#pop.array<0, 1, 0>>
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %0 = pop.array.repeat [%idx0, %idx1] : !pop.array<3, index>
  kgen.return %0 : !pop.array<3, index>
}

// CHECK-LABEL: @array_get
kgen.func @array_get() -> index {
  // CHECK-NEXT: constant = <1>
  %0 = kgen.param.constant: !pop.array<2, index> = <#pop.array<0, 1>>
  %1 = pop.array.get %0[1] : !pop.array<2, index>
  kgen.return %1 : index
}

// CHECK-LABEL: @array_replace
kgen.func @array_replace() -> !pop.array<2, index> {
  // CHECK-NEXT: constant: !pop.array<2, index> = <#pop.array<0, 1>>
  %0 = kgen.param.constant: !pop.array<2, index> = <#pop.array<0, 0>>
  %1 = index.constant 1
  %2 = pop.array.replace %1, %0[1] : !pop.array<2, index>
  kgen.return %2 : !pop.array<2, index>
}

// CHECK-LABEL: @variant_create
kgen.func @variant_create() -> !pop.variant<si4, ui4> {
  // CHECK-NEXT: constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %0 = kgen.param.constant: ui4 = <7>
  %1 = pop.variant.create %0 : ui4 -> !pop.variant<si4, ui4>
  kgen.return %1 : !pop.variant<si4, ui4>
}

// CHECK-LABEL: @variant_is
kgen.func @variant_is() -> i1 {
  // CHECK-NEXT: constant: i1 = <1>
  %0 = kgen.param.constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %1 = pop.variant.is ui4, %0 : !pop.variant<si4, ui4>
  kgen.return %1 : i1
}

// CHECK-LABEL: @variant_get
kgen.func @variant_get() -> ui4 {
  // CHECK-NEXT: constant: ui4 = <7>
  %0 = kgen.param.constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %1 = pop.variant.get %0 : !pop.variant<si4, ui4> as ui4
  kgen.return %1 : ui4
}

// CHECK-LABEL: @variant_get_ub
kgen.func @variant_get_ub() -> si4 {
  // CHECK: pop.variant.get
  %0 = kgen.param.constant: !pop.variant<si4, ui4> = <#pop.variant<:ui4 7>>
  %1 = pop.variant.get %0 : !pop.variant<si4, ui4> as si4
  kgen.return %1 : si4
}

// CHECK-LABEL: @variant_create_get
kgen.func @variant_create_get(%a: i32) -> i32 {
  %0 = pop.variant.create %a : i32 -> !pop.variant<i32, f32>
  %1 = pop.variant.get %0 : !pop.variant<i32, f32> as i32
  // CHECK: return %arg0
  kgen.return %1 : i32
}


// CHECK-LABEL: @call_indirect_partial_apply
kgen.generator @call_indirect_partial_apply(%fn: (index, i32) -> index, %arg0: index, %arg1: i32) -> index {
  // CHECK-NEXT: %0 = pop.call_indirect %arg0(%arg1, %arg2) : (index, i32) -> index
  %0 = pop.partial_apply %fn(?, %arg1) : (index, i32) -> index
  %1 = pop.call_indirect %0(%arg0) : !pop.closure<(index) -> index>
  // CHECK-NEXT: return %0
  kgen.return %1 : index
}

// CHECK-LABEL: @partial_apply_of_partial_apply
kgen.generator @partial_apply_of_partial_apply(%fn: (index, i32) -> index, %arg0: index, %arg1: i32) -> !pop.closure<() -> index> {
  // CHECK-NEXT: %0 = pop.partial_apply %arg0(%arg1, %arg2) : (index, i32) -> index
  %0 = pop.partial_apply %fn(?, %arg1) : (index, i32) -> index
  %1 = pop.partial_apply %0(%arg0) : !pop.closure<(index) -> index>
  // CHECK-NEXT: return %0
  kgen.return %1 : !pop.closure<() -> index>
}
