// RUN: kgen-opt %s -lower-lit -verify-parameters=simplify=true \
// RUN:   -kgen-print-inline-type-values | FileCheck %s

// A storage type chosen by a conformance check on a type parameter, in the
// shape `Optional` and `Variant` use: the parameter is downcast to the trait
// the check established, and a nested wrapper selects its own storage by a
// further check on the upcast of that. Lowering has to fold the selection to
// one concrete layout wherever the instantiation is referenced.
//
// The struct is referenced from a field, from an SSA type and from a callee
// parameter, because those reach the layout lowering by different routes. If
// any route hands the parameter evaluator a constant whose halves are lowered
// to different depths, the downcast does not fold, the check behind it cannot
// resolve the struct, and the selection survives as a `cond` in that position
// only. The verifier then reports the callee's type against its declaration.

lit.trait.decl @AnyType {}
lit.trait.decl @Niche {}
lit.trait.decl @Custom {}

lit.struct.decl @Int register_passable {
  lit.struct.field v : !kgen.scalar<index>
}

// Conforms to @Niche and not to @Custom, so the inner selection takes the
// default branch.
lit.struct.decl @Ptr<T: !lit.trait<@AnyType>>(trait<@AnyType, @Niche>) register_passable {
  lit.struct.field p : !kgen.pointer<T>
  kgen.conformance @AnyType {
  }
  kgen.conformance @Niche {
  }
}

lit.struct.decl @CustomStorage<T: !lit.trait<@AnyType, @Custom>> register_passable {
  lit.struct.field v : !kgen.param<T>
}

lit.struct.decl @DefaultStorage<T: !lit.trait<@AnyType>> register_passable {
  lit.struct.field v : !kgen.param<T>
  lit.struct.field tag : !kgen.scalar<bool>
}

lit.struct.decl @NichedStorage<T: !lit.trait<@AnyType, @Niche>> register_passable {
  lit.struct.field m : !kgen.param<cond(conforms_to(:!lit.trait<@AnyType> upcast(:!lit.trait<@AnyType, @Niche> T), :meta<!lit.trait<@AnyType, @Custom>> !lit.trait<@AnyType, @Custom>), !lit.struct<@CustomStorage<:!lit.trait<@AnyType, @Custom> upcast(:!lit.trait<@AnyType, @Custom, @Niche> downcast(:!lit.trait<@AnyType, @Niche> T))>>, !lit.struct<@DefaultStorage<:!lit.trait<@AnyType> upcast(:!lit.trait<@AnyType, @Niche> T)>>)>
}

lit.struct.decl @Var<T: !lit.trait<@AnyType>> register_passable {
  lit.struct.field s : !kgen.param<cond(conforms_to(:!lit.trait<@AnyType> T, :meta<!lit.trait<@Niche>> !lit.trait<@Niche>), !lit.struct<@NichedStorage<:!lit.trait<@AnyType, @Niche> downcast(:!lit.trait<@AnyType> T)>>, !lit.struct<@DefaultStorage<:!lit.trait<@AnyType> T>>)>
}

!VarPtrInt = !lit.struct<@Var<:!lit.trait<@AnyType> !lit.struct<@Ptr<:!lit.trait<@AnyType> !lit.struct<@Int>>>>>

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    struct<(pointer<none>, scalar<bool>)>]) memoryOnly>
lit.struct.decl @Node {
  lit.struct.field f : !VarPtrInt
}

// CHECK-LABEL: kgen.generator @use
// CHECK-SAME:    (%arg0: !kgen.struct<(pointer<none>, scalar<bool>)>)
lit.fn @use(%a: !VarPtrInt) {
  hlcf.return
}

// CHECK-LABEL: kgen.generator @thunk
// CHECK-SAME:    callee: (!kgen.struct<(pointer<none>, scalar<bool>)>) -> !kgen.struct<(pointer<none>, scalar<bool>)>
lit.fn @thunk<callee: !lit.generator<(!VarPtrInt) -> !VarPtrInt>>(%a: !VarPtrInt) -> !VarPtrInt {
  %r = lit.call tail[!lit.generator<(!VarPtrInt) -> !VarPtrInt>: callee](%a)
  hlcf.return %r : !VarPtrInt
}
