// RUN: kgen-opt %s -lower-lit -kgen-print-inline-type-values | FileCheck %s

// Struct generators are lowered last, one after another, so a layout first
// computed while lowering a later generator checks a conformance of a struct
// whose generator is already lowered. The parameter values in that check are
// still LIT, so they have to be lowered as well before they are bound to the
// generator's lowered parameters. A LIT `!lit.trait<@AnyType>` element type
// bound into the lowered `!kgen.param_list<type>` of the `param_list.reduce`
// below rebuilt a reduce whose generator no longer took its base, which failed
// the attribute verifier.
//
// `Opt` selects its storage the way `Optional` does through the `Tuple` of its
// payload, and `Holder` is the first to need the layouts of `Opt`.

lit.trait.decl @AnyType {}
lit.trait.decl @Niche {}

lit.struct.decl @Int register_passable {
  lit.struct.field v : !kgen.scalar<index>
  kgen.conformance @AnyType {
  }
}

lit.struct.decl @Ptr register_passable {
  lit.struct.field p : !kgen.pointer<none>
  kgen.conformance @AnyType {
  }
  kgen.conformance @Niche {
  }
}

// Conforms to @Niche when any element does.
lit.struct.decl @Tup<Ts: !kgen.param_list<!lit.trait<@AnyType>>> register_passable {
  lit.struct.field v : !kgen.scalar<index>
  kgen.conformance @AnyType {
  }
  kgen.conformance @Niche {
  } where #kgen.constraint<#kgen.param_list.reduce<#kgen<simd false> : !kgen.scalar<bool>, #kgen.param.decl.ref<"Ts"> : !kgen.param_list<!lit.trait<@AnyType>>, #kgen.gen<cond(*(0,0), *(0,0), conforms_to(:!lit.trait<@AnyType> #kgen.param_list.get<:param_list<!lit.trait<@AnyType>> *(0,1), *(0,2)>, :meta<!lit.trait<@AnyType, @Niche>> !lit.trait<@AnyType, @Niche>))> : !kgen.generator<<scalar<bool>, param_list<!lit.trait<@AnyType>>, index> scalar<bool>>>, loc(unknown)>
}

lit.struct.decl @Opt<T: !lit.trait<@AnyType>> register_passable {
  lit.struct.field s : !kgen.param<cond(conforms_to(:!lit.trait<@AnyType> !lit.struct<@Tup<:param_list<!lit.trait<@AnyType>> [T]>>, :meta<!lit.trait<@Niche>> !lit.trait<@Niche>), !lit.struct<@Int>, !kgen.scalar<bool>)>
}

// CHECK-LABEL: kgen.struct.generator @Holder
// CHECK-SAME:    (a: [typevalue<#kgen.genref<@Opt<:type [typevalue<#kgen.genref<@Int>>, scalar<index>]>>>, scalar<bool>],
// CHECK-SAME:     b: [typevalue<#kgen.genref<@Opt<:type [typevalue<#kgen.genref<@Ptr>>, pointer<none>]>>>, scalar<index>])
lit.struct.decl @Holder {
  lit.struct.field a : !lit.struct<@Opt<:!lit.trait<@AnyType> !lit.struct<@Int>>>
  lit.struct.field b : !lit.struct<@Opt<:!lit.trait<@AnyType> !lit.struct<@Ptr>>>
}
