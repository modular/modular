// RUN: kgen-opt %s -lower-lit -split-input-file -verify-parameters \
// RUN:   -kgen-print-inline-type-values | FileCheck %s

// Struct layouts whose recursion is broken by an indirection. The layout is
// finite, so it lowers: the recursive occurrence becomes an opaque pointer and
// the indirection around it is kept.
//
// The shapes that reach the recursion through a parametric wrapper used to be
// rejected. What makes them work is that the value lowering never calls the
// layout lowering: computing the value representation of `@Ptr<@Node>` asks
// for no layout, so the only re-entry left is a layout one, on `@Node` itself,
// and the opaque pointer breaks it.

//===----------------------------------------------------------------------===//
// The wrapper is the immediate field type.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// Checked in full rather than just for the pointer, because this is also
// the smallest case that pins where each half comes from. `@Ptr`'s argument is
// a type constant carrying both halves - the instantiation's value half reads
// the argument's value half and its layout half reads the argument's layout -
// so `@Node`'s built layout appears nested inside `@Node`'s own value half.
//
// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    (next: [typevalue<#kgen.genref<@Ptr<:type
// CHECK-SAME:      [typevalue<#kgen.genref<@Node>>, struct<(pointer<none>) memoryOnly>]
// CHECK-SAME:      pointer<none>]) memoryOnly>
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@Ptr<:type !lit.struct<@Node>>>
}

// -----

//===----------------------------------------------------------------------===//
// A non-parametric wrapper behind an aggregate.
//===----------------------------------------------------------------------===//

lit.struct.decl @PtrToNode register_passable {
  lit.struct.field address : !kgen.pointer<:type !lit.struct<@Node>>
}

lit.struct.decl @Wrap<ty: type> register_passable {
  lit.struct.field x : !kgen.struct<(ty)>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@Wrap<:type !lit.struct<@PtrToNode>>>
}

// -----

//===----------------------------------------------------------------------===//
// A by-value wrapper around the pointer wrapper. The parameter is substituted
// as a type, so the layout completes without needing a value form.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Box<ty: type> register_passable {
  lit.struct.field val : !kgen.param<ty>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next :
      !lit.struct<@Box<:type !lit.struct<@Ptr<:type !lit.struct<@Node>>>>>
}

// -----

//===----------------------------------------------------------------------===//
// An array of the pointer wrapper. `!kgen.array` holds its element as a type,
// not as a type value, so nothing demands a value form here.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !kgen.array<2, !lit.struct<@Ptr<:type !lit.struct<@Node>>>>
}

// -----

//===----------------------------------------------------------------------===//
// A longer chain of memory-only, multi-field wrappers
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Box<ty: type> {
  lit.struct.field val : !kgen.param<ty>
  lit.struct.field tag : !kgen.scalar<index>
}

lit.struct.decl @Outer<ty: type> {
  lit.struct.field inner : !lit.struct<@Box<:type ty>>
  lit.struct.field tag : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next :
      !lit.struct<@Outer<:type !lit.struct<@Ptr<:type !lit.struct<@Node>>>>>
}

// -----

//===----------------------------------------------------------------------===//
// A parametric expression referring to the pointer wrapper. `get_alignof`
// stays unfolded through this pass - the target is still symbolic - which is
// why an unresolved size or alignment expression is not a problem here.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Aligned register_passable attributes {
  minAlignment = #kgen.param.expr<get_alignof,
    #kgen.type<!lit.struct<@Ptr<:type !kgen.scalar<index>>>> : !kgen.type,
    #kgen.param.expr<current_target> : !kgen.target> : index
} {
  lit.struct.field a : !kgen.scalar<index>
}

// CHECK-LABEL: kgen.struct.generator @User
// CHECK-SAME:    get_alignof
lit.struct.decl @User {
  lit.struct.field f : !lit.struct<@Aligned>
}

// -----

//===----------------------------------------------------------------------===//
// A raw pointer nested inside an aggregate rather than being the whole field
// type. The pointee is erased at any depth, so these need no wrapper struct to
// break the recursion.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !kgen.struct<(!kgen.pointer<:type !lit.struct<@Node>>)>
}

// -----

//===----------------------------------------------------------------------===//
// The same, nested inside an array.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    pointer<none>
lit.struct.decl @Node {
  lit.struct.field next : !kgen.array<2, !kgen.pointer<:type !lit.struct<@Node>>>
}

// -----

//===----------------------------------------------------------------------===//
// One wrapper deeper, reached through a parametric struct's field.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Wrap<ty: type> register_passable {
  lit.struct.field x : !kgen.struct<(ty)>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    , struct<(pointer<none>)>]) memoryOnly>
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

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @Tup<Ts: !kgen.param_list<!lit.trait<@AnyType>>> register_passable {
  lit.struct.field storage : !kgen.struct<:!kgen.param_list<!lit.trait<@AnyType>> Ts isParamPack>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    , struct<(pointer<none>) isParamPack>]) memoryOnly>
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@Tup<:param_list<trait<@AnyType>>
      [!lit.struct<@Ptr<:type !lit.struct<@Node>>>]>>
}

// -----

//===----------------------------------------------------------------------===//
// Through a variant with a concrete element list. Mojo reaches this via
// Variant's non-niche storage, so Optional[Span[Node, ...]] takes this path
// while Optional[Pointer[Node, ...]] takes the niche path and never builds a
// variant.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    (next: variant<pointer<none>, scalar<index>>) memoryOnly>
lit.struct.decl @Node {
  lit.struct.field next :
      !kgen.variant<!lit.struct<@Ptr<:type !lit.struct<@Node>>>, !kgen.scalar<index>>
}

// -----

//===----------------------------------------------------------------------===//
// Through a variant whose element list is a rebind over a variadic.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

lit.struct.decl @VarStorage<Ts: !kgen.param_list<!kgen.type>> register_passable {
  lit.struct.field impl : !kgen.variant<[rebind(:!kgen.param_list<!kgen.type> Ts)]>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    , variant<pointer<none>>]) memoryOnly>
lit.struct.decl @Node {
  lit.struct.field next : !lit.struct<@VarStorage<:param_list<type>
      [!lit.struct<@Ptr<:type !lit.struct<@Node>>>]>>
}


// -----

//===----------------------------------------------------------------------===//
// The wrapper reached inside a `!kgen.struct` element list.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// CHECK-LABEL: kgen.struct.generator @Node
// CHECK-SAME:    (next: struct<(pointer<none>)>) memoryOnly>
lit.struct.decl @Node {
  lit.struct.field next : !kgen.struct<(!lit.struct<@Ptr<:type !lit.struct<@Node>>>)>
}

// -----

//===----------------------------------------------------------------------===//
// The same shape with no concrete instantiation: a generic struct recursing at
// its own parameter, whose generator body is built at that parameter.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// CHECK-LABEL: kgen.struct.generator @Rec
// CHECK-SAME:    (next: struct<(pointer<none>)>) memoryOnly>
lit.struct.decl @Rec<ty: type> {
  lit.struct.field next :
      !kgen.struct<(!lit.struct<@Ptr<:type !lit.struct<@Rec<:type ty>>>>)>
}

// -----

//===----------------------------------------------------------------------===//
// An unbounded family of distinct instantiations behind the pointer wrapper:
// `@Rec<ty>` holds `@Rec<@Ptr<ty>>`, which holds `@Rec<@Ptr<@Ptr<ty>>>`, and
// so on. Both halves stay finite: each layout is a pointer, and the value
// half of a reference never expands a body. The by-value version of this
// shape is rejected before lowering - see
// lower-lit-types-recursion-by-value.mlir.
//===----------------------------------------------------------------------===//

lit.struct.decl @Ptr<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

// CHECK-LABEL: kgen.struct.generator @Rec
// CHECK-SAME:    (next: [typevalue<#kgen.genref<@Ptr<:type
// CHECK-SAME:      [typevalue<#kgen.genref<@Rec<:type [typevalue<#kgen.genref<@Ptr<:type ty>>>, pointer<none>]>>>, struct<(pointer<none>) memoryOnly>]
// CHECK-SAME:      pointer<none>]) memoryOnly>
lit.struct.decl @Rec<ty: type> {
  lit.struct.field next :
      !lit.struct<@Ptr<:type !lit.struct<@Rec<:type !lit.struct<@Ptr<:type ty>>>>>>
}

// CHECK-LABEL: kgen.struct.generator @User
// CHECK-SAME:    struct<(pointer<none>) memoryOnly>]) memoryOnly>
lit.struct.decl @User {
  lit.struct.field r : !lit.struct<@Rec<:type !kgen.scalar<index>>>
}

