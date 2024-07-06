// RUN: kgen-opt -lower-lit-types -verify-parameters --kgen-print-inline-type-values %s | FileCheck %s

lit.struct.decl @Container<T: trait<@Trait>> {
  lit.struct.field x: !kgen.paramref<:trait<@Trait> T>
}

lit.struct.decl @Element {
}

// CHECK-LABEL: kgen.generator @func<T: type>
// CHECK-SAME: (%arg0: !kgen.struct<(T) memoryOnly>
kgen.generator @func<T: trait<@Trait>>(%arg0: !lit.struct<@Container<:trait<@Trait> T>>) {
  kgen.return
}

kgen.generator @f() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @top
kgen.generator @top(%arg0: !lit.struct<@Container<:trait<@Trait> [@Element, {"f": () -> () = @f}]>>) {
  // CHECK-NEXT: call @func<:type [struct<() memoryOnly>, {{{.*}}}]>(%arg0) : (!kgen.struct<(struct<() memoryOnly>) memoryOnly>) -> ()
  kgen.call @func<:trait<@Trait> [@Element, {"f": () -> () = @f}]>(%arg0) : (!lit.struct<@Container<:trait<@Trait> [@Element, {"f": () -> () = @f}]>>) -> ()
  kgen.return
}
