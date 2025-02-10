// RUN: kgen-opt %s -lower-lit -allow-unregistered-dialect -split-input-file -verify-parameters -kgen-print-inline-type-values | FileCheck %s

//===----------------------------------------------------------------------===//
// Self-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @ListNode = struct_inst<"ListNode"(next: pointer<typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>>) memoryOnly>
lit.struct.decl @ListNode {
  lit.struct.field next : !kgen.pointer<:type !lit.struct<@ListNode>>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare listnode: type = <[typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>, struct<(pointer<none>) memoryOnly>]>
  kgen.param.declare listnode: meta<!lit.struct<@ListNode>> = <[@ListNode]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Mutual-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @Bar = struct_inst<"Bar"(foo: typevalue<inst_struct_ref(#kgen.typeref<@Foo>)>)>
lit.struct.decl @Bar register_passable {
  lit.struct.field foo: !lit.struct<@Foo>
}

// CHECK: kgen.struct.generator @Foo = struct_inst<"Foo"(bar_ptr: pointer<typevalue<inst_struct_ref(#kgen.typeref<@Bar>)>>)>
lit.struct.decl @Foo register_passable {
  lit.struct.field bar_ptr: !kgen.pointer<@Bar>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare bar: type = <[typevalue<inst_struct_ref(#kgen.typeref<@Bar>)>, pointer<none>]>
  kgen.param.declare bar: meta<!lit.struct<@Bar>> = <[@Bar]>
  // CHECK: kgen.param.declare foo: type = <[typevalue<inst_struct_ref(#kgen.typeref<@Foo>)>, pointer<none>]>
  kgen.param.declare foo: meta<!lit.struct<@Foo>> = <[@Foo]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Parametric Self-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @ListNode = struct_inst<"ListNode"(next: typevalue<inst_struct_ref(#kgen.typeref<@Pointer<:type [typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>, struct<(pointer<none>) memoryOnly>]>>)>) memoryOnly>
lit.struct.decl @ListNode {
  lit.struct.field next : !lit.struct<@Pointer<:type !lit.struct<@ListNode>>>
}

// CHECK: kgen.struct.generator @Pointer<ty: type> = struct_inst<"Pointer"[ty]<:type ty>(address: pointer<typevalue<ty>>)>
lit.struct.decl @Pointer<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare listnode: type = <[typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>, struct<(pointer<none>) memoryOnly>]>
  kgen.param.declare listnode: meta<!lit.struct<@ListNode>> = <[@ListNode]>
  kgen.return
}
