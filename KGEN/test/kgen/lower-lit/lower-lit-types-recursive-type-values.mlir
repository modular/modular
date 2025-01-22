// RUN: kgen-opt %s -lower-lit -allow-unregistered-dialect -split-input-file -verify-parameters -kgen-print-inline-type-values | FileCheck %s

//===----------------------------------------------------------------------===//
// Self-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @ListNode : type
// CHECK-NEXT: kgen.struct.info :type [struct_inst<"ListNode"(next: pointer<typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>>) memoryOnly>, struct<(pointer<none>) memoryOnly>]
lit.struct.decl @ListNode {
  lit.struct.field next : !kgen.pointer<:type !lit.struct<@ListNode>>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare listnode: type = <[typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>, struct<(pointer<none>) memoryOnly>]>
  kgen.param.declare listnode: !lit.anystruct<@ListNode> = <[@ListNode]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Mutual-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @Bar : type
// CHECK-NEXT: kgen.struct.info :type [struct_inst<"Bar"(foo: typevalue<inst_struct_ref(#kgen.typeref<@Foo>)>)>, pointer<none>]
lit.struct.decl @Bar register_passable {
  lit.struct.field foo: !lit.struct<@Foo>
}

// CHECK: kgen.struct.generator @Foo : type
// CHECK-NEXT: kgen.struct.info :type [struct_inst<"Foo"(bar_ptr: pointer<typevalue<inst_struct_ref(#kgen.typeref<@Bar>)>>)>, pointer<none>]
lit.struct.decl @Foo register_passable {
  lit.struct.field bar_ptr: !kgen.pointer<@Bar>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare bar: type = <[typevalue<inst_struct_ref(#kgen.typeref<@Bar>)>, pointer<none>]>
  kgen.param.declare bar: !lit.anystruct<@Bar> = <[@Bar]>
  // CHECK: kgen.param.declare foo: type = <[typevalue<inst_struct_ref(#kgen.typeref<@Foo>)>, pointer<none>]>
  kgen.param.declare foo: !lit.anystruct<@Foo> = <[@Foo]>
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Parametric Self-Recursive Structs
//===----------------------------------------------------------------------===//

// CHECK: kgen.struct.generator @ListNode : type
// CHECK-NEXT: kgen.struct.info :type [struct_inst<"ListNode"(next: typevalue<inst_struct_ref(#kgen.typeref<@Pointer<:type [typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>, struct<(pointer<none>) memoryOnly>]>>)>) memoryOnly>, struct<(pointer<none>) memoryOnly>]
lit.struct.decl @ListNode {
  lit.struct.field next : !lit.struct<@Pointer<:type !lit.struct<@ListNode>>>
}

// CHECK: kgen.struct.generator @Pointer<ty: type> : type
// CHECK-NEXT: kgen.struct.info :type [struct_inst<"Pointer"[ty]<:type ty>(address: pointer<typevalue<ty>>)>, pointer<none>]
lit.struct.decl @Pointer<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

kgen.generator @type_values() {
  // CHECK: kgen.param.declare listnode: type = <[typevalue<inst_struct_ref(#kgen.typeref<@ListNode>)>, struct<(pointer<none>) memoryOnly>]>
  kgen.param.declare listnode: !lit.anystruct<@ListNode> = <[@ListNode]>
  kgen.return
}
