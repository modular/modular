// RUN: kgen-opt %s -lower-lit -split-input-file -verify-parameters \
// RUN:   -verify-diagnostics

// Struct layouts that are legal but that this pass cannot build: in every case
// below the recursion is broken by an indirection, so the layout is finite and
// the struct should lower. Each `expected-error` records the limitation, which
// the diagnostic reports as such rather than blaming the code.
//
// Every case here has one cause: the recursion reaches the layout through the
// parameter list of a parametric type, where there is no indirection to erase
// at all. A layout is built from element type values, and a type value carries
// the layout of each of its type arguments.

//===----------------------------------------------------------------------===//
// A parametric pointer wrapper inside an aggregate. A `!kgen.struct` holds
// its elements as type values, so building @Node's layout demands the value
// form of `@Ptr<:type @Node>`, which carries @Node's layout.
//===----------------------------------------------------------------------===//

// FIXME: should lower. The error is anchored on the wrapper the cycle was
// re-entered on, so a note names @Node, the struct the recursive field
// actually belongs to.
// expected-error @below {{'Ptr' requires a recursive layout, which is not supported}}
lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// expected-note @below {{'Node' recurses through 'Ptr'}}
lit.struct.decl @Node {
  lit.struct.field next : !kgen.struct<(!lit.struct<@Ptr<:type !lit.struct<@Node>>>)>
}

// -----

//===----------------------------------------------------------------------===//
// One wrapper deeper, reached through a parametric struct's field.
//===----------------------------------------------------------------------===//

// FIXME: should lower, same cause as above.
// expected-error @below {{'Ptr' requires a recursive layout, which is not supported}}
lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Wrap<ty: type> register_passable {
  lit.struct.field x : !kgen.struct<(ty)>
}

// expected-note @below {{'Node' recurses through 'Ptr'}}
lit.struct.decl @Node {
  lit.struct.field next :
      !lit.struct<@Wrap<:type !lit.struct<@Ptr<:type !lit.struct<@Node>>>>>
}

// -----

//===----------------------------------------------------------------------===//
// Tuple stores `!kgen.struct<: <type values> isParamPack>`, and param
// packs deliberately opt out of `normalizeVariadicForUniquing` (see #84445),
// which is what keeps ordinary structs from reaching this. Mojo equivalent:
// struct Node: var next: Tuple[Pointer[Node, MutUntrackedOrigin]]
//===----------------------------------------------------------------------===//

lit.trait.decl @AnyType {}

// FIXME: should lower, same cause as above.
// expected-error @below {{'Ptr' requires a recursive layout, which is not supported}}
lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Tup<Ts: !kgen.param_list<!lit.trait<@AnyType>>> register_passable {
  lit.struct.field storage : !kgen.struct<:!kgen.param_list<!lit.trait<@AnyType>> Ts isParamPack>
}

// expected-note @below {{'Node' recurses through 'Ptr'}}
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@Tup<:param_list<trait<@AnyType>>
      [!lit.struct<@Ptr<:type !lit.struct<@Node>>>]>>
}

// -----

//===----------------------------------------------------------------------===//
// Through a variant with a concrete element list. Mojo reaches this via
// Variant's non-niche storage, so Optional[Span[Node, ...]] fails while
// Optional[Pointer[Node, ...]] compiles; the latter takes the niche path and
// never builds a variant.
// ===----------------------------------------------------------------------===//

// FIXME: should lower, same cause as above.
// expected-error @below {{'Ptr' requires a recursive layout, which is not supported}}
lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// expected-note @below {{'Node' recurses through 'Ptr'}}
lit.struct.decl @Node {
  lit.struct.field next :
      !kgen.variant<!lit.struct<@Ptr<:type !lit.struct<@Node>>>, !kgen.scalar<index>>
}

// -----

//===----------------------------------------------------------------------===//
// Through a variant whose element list is a rebind over a variadic.
//===----------------------------------------------------------------------===//

// FIXME: should lower, same cause as above.
// expected-error @below {{'Ptr' requires a recursive layout, which is not supported}}
lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @VarStorage<Ts: !kgen.param_list<!kgen.type>> register_passable {
  lit.struct.field impl : !kgen.variant<[rebind(:!kgen.param_list<!kgen.type> Ts)]>
}

// expected-note @below {{'Node' recurses through 'Ptr'}}
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@VarStorage<:param_list<type>
      [!lit.struct<@Ptr<:type !lit.struct<@Node>>>]>>
}

// -----

//===----------------------------------------------------------------------===//
// No concrete instantiation needed: a generic struct recursing at its own
// parameter hits the same defect, and every generic struct's generator body is
// built at its own parameters.
//===----------------------------------------------------------------------===//

// FIXME: should lower, same cause as above.
// expected-error @below {{'Ptr' requires a recursive layout, which is not supported}}
lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// expected-note @below {{'Rec' recurses through 'Ptr'}}
lit.struct.decl @Rec<ty: type> {
  lit.struct.field next :
      !kgen.struct<(!lit.struct<@Ptr<:type !lit.struct<@Rec<:type ty>>>>)>
}
