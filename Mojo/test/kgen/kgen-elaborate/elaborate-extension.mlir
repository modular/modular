// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// Test that `#kgen.get_witness` resolves a trait conformance supplied by a
// `#kgen.extension` when the anchor type does not itself carry that trait.

kgen.generator @call(%arg0: !kgen.struct<(index)>) -> index {
  %c = kgen.param.constant : index = <1>
  hlcf.return %c : index
}

// The source closure struct: conforms to `def() -> T` with `T = index`.
kgen.struct.generator @make = struct_inst<"make"(data: index)> {
  kgen.conformance @"def() -> T" {
    kgen.witness "T" : type = index
    kgen.witness "__call__" : (!kgen.struct<(index)>) -> index = @call
  }
}

// The bridging extension struct: stateless, single type parameter `A`, conforms
// to `def() -> V` by forwarding each witness to `A`'s own `def() -> T` witnesses.
kgen.struct.generator @extension1<A: type> = struct_inst<"extension1"[A]<:type A>> {
  kgen.conformance @"def() -> V" {
    kgen.witness "V" : type = #kgen.get_witness<A, @"def() -> T", "T">
    kgen.witness "__call__" : (!kgen.param<A>) -> index = #kgen.get_witness<A, @"def() -> T", "__call__">
  }
}

// `sink` looks up the `def() -> V` __call__ witness on `F` and invokes it.
// CHECK-LABEL: kgen.func @"sink
// CHECK: kgen.call @call
kgen.generator @sink<V: type, F: type>(%arg: !kgen.param<F>) -> !kgen.param<V> {
  kgen.param.declare callFn : (!kgen.param<F>) -> !kgen.param<V> = <#kgen.get_witness<F, @"def() -> V", "__call__">>
  %result = kgen.call_param[(!kgen.param<F>) -> !kgen.param<V> : callFn](%arg)
  hlcf.return %result : !kgen.param<V>
}

// `forward` augments `G`'s trait view with an extension supplying `def() -> V`
// CHECK-LABEL: kgen.func @"forward
// CHECK-NEXT: kgen.call @"sink
// CHECK-NEXT: hlcf.return
kgen.generator @forward<T: type, G: type>(%arg: !kgen.param<G>) -> !kgen.param<T> {
  kgen.param.declare P1 : type = <#kgen.extension<G, [#kgen.type<typevalue<:!kgen.type #kgen.genref<@extension1<:type G>>>, struct<()>> : !kgen.type]> : !kgen.type>
  %arg1 = kgen.rebind %arg : !kgen.param<G> to !kgen.param<P1>
  %result = kgen.call @sink<:type T, :type P1>(%arg1) : (!kgen.param<P1>) -> !kgen.param<T>
  hlcf.return %result : !kgen.param<T>
}

#make = #kgen.type<typevalue<:!kgen.type #kgen.genref<@make>>, struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.func @foo
kgen.generator @foo(%arg0: !kgen.struct<(index)>) -> index {
  %0 = kgen.call @forward<:type index, :type #make>(%arg0) : (!kgen.struct<(index)>) -> index
  hlcf.return %0 : index
}

// -----

// An extension is physically transparent, so a layout struct built from one
// reduces to the anchor's layout.

kgen.generator @call2(%arg0: !kgen.struct<(index)>) -> index {
  %c = kgen.param.constant : index = <1>
  hlcf.return %c : index
}

kgen.struct.generator @make2 = struct_inst<"make2"(data: index)> {
  kgen.conformance @"def() -> T" {
    kgen.witness "T" : type = index
    kgen.witness "__call__" : (!kgen.struct<(index)>) -> index = @call2
  }
}

kgen.struct.generator @extension2<A: type> = struct_inst<"extension2"[A]<:type A>> {
  kgen.conformance @"def() -> V" {
    kgen.witness "V" : type = #kgen.get_witness<A, @"def() -> T", "T">
    kgen.witness "__call__" : (!kgen.param<A>) -> index = #kgen.get_witness<A, @"def() -> T", "__call__">
  }
}

#make2 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@make2>>, struct<(index)>> : !kgen.type
#extension2 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@extension2<:type #make2>>>, struct<()>> : !kgen.type

// A layout holding an extension reduces to the anchor's layout in that slot,
// so the rebind target is `struct<(struct<(index)>)>`: the same shape @make2
// has as a struct instance, but without the extension in it.
// CHECK-LABEL: kgen.func @extension_in_layout
// CHECK-NOT: kgen.rebind
kgen.generator @extension_in_layout(%arg0: !kgen.struct<(struct<(index)>)>) -> index {
  %arg1 = kgen.rebind %arg0 : !kgen.struct<(struct<(index)>)> to !kgen.struct<[#kgen.extension<#make2, [#extension2]>] layout>
  %elt = kgen.struct.extract %arg1[0] : <(struct<(index)>)>
  %0 = kgen.call @call2(%elt) : (!kgen.struct<(index)>) -> index
  hlcf.return %0 : index
}

// -----

// An extension anchored on another extension reduces through both to the
// innermost anchor's layout.

kgen.generator @call3(%arg0: !kgen.struct<(index)>) -> index {
  %c = kgen.param.constant : index = <1>
  hlcf.return %c : index
}

kgen.struct.generator @make3 = struct_inst<"make3"(data: index)> {
  kgen.conformance @"def() -> T" {
    kgen.witness "T" : type = index
    kgen.witness "__call__" : (!kgen.struct<(index)>) -> index = @call3
  }
}

kgen.struct.generator @extension3<A: type> = struct_inst<"extension3"[A]<:type A>> {
  kgen.conformance @"def() -> V" {
    kgen.witness "V" : type = #kgen.get_witness<A, @"def() -> T", "T">
    kgen.witness "__call__" : (!kgen.param<A>) -> index = #kgen.get_witness<A, @"def() -> T", "__call__">
  }
}

kgen.struct.generator @extension4<A: type> = struct_inst<"extension4"[A]<:type A>> {
  kgen.conformance @"def() -> W" {
    kgen.witness "W" : type = #kgen.get_witness<A, @"def() -> T", "T">
    kgen.witness "__call__" : (!kgen.param<A>) -> index = #kgen.get_witness<A, @"def() -> T", "__call__">
  }
}

#make3 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@make3>>, struct<(index)>> : !kgen.type
#extension3 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@extension3<:type #make3>>>, struct<()>> : !kgen.type
#extension4 = #kgen.type<typevalue<:!kgen.type #kgen.genref<@extension4<:type #make3>>>, struct<()>> : !kgen.type
#chained = #kgen.extension<#kgen.extension<#make3, [#extension3]>, [#extension4]> : !kgen.type

// CHECK-LABEL: kgen.func @nested_extension_in_layout
// CHECK-NOT: kgen.rebind
kgen.generator @nested_extension_in_layout(%arg0: !kgen.struct<(struct<(index)>)>) -> index {
  %arg1 = kgen.rebind %arg0 : !kgen.struct<(struct<(index)>)> to !kgen.struct<[#chained] layout>
  %elt = kgen.struct.extract %arg1[0] : <(struct<(index)>)>
  %0 = kgen.call @call3(%elt) : (!kgen.struct<(index)>) -> index
  hlcf.return %0 : index
}
