// RUN: kgen-opt -lower-lit -verify-parameters --kgen-print-inline-type-values %s | FileCheck %s

// CHECK: kgen.struct.generator @[[STRUCT_CONTAINER:.+]]<T: type> = struct_inst<"Container"[T]<:type T>(x: typevalue<T>) memoryOnly>
lit.struct.decl @Container<T: trait<@Trait>> {
  lit.struct.field x: !kgen.param<:trait<@Trait> T>
}

// CHECK: kgen.struct.generator @[[STRUCT_ELEMENT:.+]] = struct_inst<"Element" memoryOnly>
lit.struct.decl @Element {
}

// CHECK: kgen.generator @func<T: type>
// CHECK-SAME: (%arg0: !kgen.struct<(T) memoryOnly>
kgen.generator @func<T: trait<@Trait>>(%arg0: !lit.struct<@Container<:trait<@Trait> T>>) {
  kgen.return
}

kgen.generator @f() {
  kgen.return
}

// CHECK: kgen.generator @top
kgen.generator @top(%arg0: !lit.struct<@Container<:trait<@Trait> [@Element, {"f": () -> () = @f}]>>) {
  // CHECK-NEXT: call @func<:type [typevalue<inst_struct_ref(#kgen.typeref<@[[STRUCT_ELEMENT]]>)>, struct<() memoryOnly>, {{{.*}}}]>(%arg0) : (!kgen.struct<(struct<() memoryOnly>) memoryOnly>) -> ()
  kgen.call @func<:trait<@Trait> [@Element, {"f": () -> () = @f}]>(%arg0) : (!lit.struct<@Container<:trait<@Trait> [@Element, {"f": () -> () = @f}]>>) -> ()
  kgen.return
}
