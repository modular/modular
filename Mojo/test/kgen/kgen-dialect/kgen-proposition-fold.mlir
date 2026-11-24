// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values %s | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s

// COM: A scalar-bool `and` folds its operands into one set of clauses, inserted
// COM: in the order written, so several cases below state the same facts in
// COM: more than one order. The first RUN line checks a single fold, since a
// COM: refold of the printed form can hide an order dependence.

#A = #kgen.type<typevalue<#kgen.trait_ref<[@A]>>, type> : !kgen.type
#B = #kgen.type<typevalue<#kgen.trait_ref<[@B]>>, type> : !kgen.type
#AB = #kgen.type<typevalue<#kgen.trait_ref<[@A, @B]>>, type> : !kgen.type
#C = #kgen.type<typevalue<#kgen.trait_ref<[@C]>>, type> : !kgen.type

!BaseTrait = !lit.trait<@BaseTrait>
!DerivedTrait = !lit.trait<@DerivedTrait>

// CHECK-LABEL: @fold_leaves
kgen.generator @fold_leaves<a: scalar<bool>, b: scalar<bool>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <a>
  kgen.param.constant: scalar<bool> = <and(a, true)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(a, false)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <a>
  kgen.param.constant: scalar<bool> = <and(a, a)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b)>
  kgen.param.constant: scalar<bool> = <and(a, b)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b)>
  kgen.param.constant: scalar<bool> = <and(and(a, b), and(a, b))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(a, not(a))>
  hlcf.return
}

// COM: Identities conjoined over a shared member merge into one n-ary class.
// CHECK-LABEL: @fold_identity
kgen.generator @fold_identity<t1: type, t2: type, t3: type, t4: type>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <identical(:type t1, t2, t3)>
  kgen.param.constant: scalar<bool> = <and(identical(:type t1, t3), identical(:type t2, t3))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <identical(:type t1, t2, t3)>
  kgen.param.constant: scalar<bool> = <and(identical(:type t2, t3), identical(:type t1, t3))>

  // COM: The merged class proves the transitive pair.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <identical(:type t1, t2, t3)>
  kgen.param.constant: scalar<bool> = <and(identical(:type t1, t3), identical(:type t2, t3), identical(:type t1, t2))>

  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <identical(:type t1, t2, t3, t4)>
  kgen.param.constant: scalar<bool> = <and(and(identical(:type t1, t3), identical(:type t2, t3)), identical(:type t4, t3))>

  // COM: One class cannot hold two distinct type values.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(identical(:type t1, i32), identical(:type t1, i1))>
  hlcf.return
}

// COM: Members of different metatypes rebind to one representative type,
// COM: whichever side the rebinds are written on. The representative follows
// COM: parameter order, so both a `type` and a `non_struct_type` one are
// COM: covered. The `and` routes each identity through the fold.
// CHECK-LABEL: @fold_identity_rebind
kgen.generator @fold_identity_rebind<a: type, z: non_struct_type, t: type, n: non_struct_type, c: scalar<bool>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(c, identical(:type a, rebind(:non_struct_type z)))>
  kgen.param.constant: scalar<bool> = <and(identical(:type rebind(:type a), rebind(:non_struct_type z)), c)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(c, identical(:type a, rebind(:non_struct_type z)))>
  kgen.param.constant: scalar<bool> = <and(identical(:type rebind(:non_struct_type z), rebind(:type a)), c)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(c, identical(:non_struct_type n, rebind(:type t)))>
  kgen.param.constant: scalar<bool> = <and(identical(:type rebind(:type t), rebind(:non_struct_type n)), c)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(c, identical(:non_struct_type n, rebind(:type t)))>
  kgen.param.constant: scalar<bool> = <and(identical(:type rebind(:non_struct_type n), rebind(:type t)), c)>
  hlcf.return
}

// COM: `and` on integers is bitwise, and scalar `eq` is already an identity at
// COM: construction, so restating one leaves a single clause.
// CHECK-LABEL: @fold_integer
kgen.generator @fold_integer<x: scalar<index>, y: scalar<index>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<index> = <and(x, y)>
  kgen.param.constant: scalar<index> = <and(:scalar<index> x, y)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <identical(:scalar<index> x, y)>
  kgen.param.constant: scalar<bool> = <and(eq(:scalar<index> x, y), eq(:scalar<index> x, y))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <not(identical(:scalar<index> x, y))>
  kgen.param.constant: scalar<bool> = <and(not(eq(:scalar<index> x, y)), not(eq(:scalar<index> x, y)))>
  hlcf.return
}

// CHECK-LABEL: @fold_conformance
kgen.generator @fold_conformance<t: type>() {
  // COM: Separate bounds stay separate but jointly prove the packed bound, and
  // COM: each one alone.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(conforms_to(:type t, :type [typevalue<#kgen.trait_ref<[@A]>>, type]), conforms_to(:type t, :type [typevalue<#kgen.trait_ref<[@B]>>, type]))>
  kgen.param.constant: scalar<bool> = <and(conforms_to(:type t, :type #A), conforms_to(:type t, :type #B), conforms_to(:type t, :type #AB))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(conforms_to(:type t, :type [typevalue<#kgen.trait_ref<[@A]>>, type]), conforms_to(:type t, :type [typevalue<#kgen.trait_ref<[@B]>>, type]))>
  kgen.param.constant: scalar<bool> = <and(conforms_to(:type t, :type #A), conforms_to(:type t, :type #B), conforms_to(:type t, :type #A))>

  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <conforms_to(:type t, :type [typevalue<#kgen.trait_ref<[@A, @B]>>, type])>
  kgen.param.constant: scalar<bool> = <and(conforms_to(:type t, :type #AB), conforms_to(:type t, :type #A))>
  hlcf.return
}

// COM: An upcast only widens the metatype, so the stored conformance drops it.
// CHECK-LABEL: @fold_upcast_conformance
kgen.generator @fold_upcast_conformance<T: !DerivedTrait, c: scalar<bool>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(conforms_to(:trait<@DerivedTrait> T, :type [typevalue<#kgen.trait_ref<[@C]>>, type])), c)>
  kgen.param.constant: scalar<bool> = <and(not(conforms_to(:!BaseTrait upcast(:!DerivedTrait T), :type #C)), c)>
  hlcf.return
}

// CHECK-LABEL: @fold_or
kgen.generator @fold_or<a: scalar<bool>, b: scalar<bool>, c: scalar<bool>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <a>
  kgen.param.constant: scalar<bool> = <and(a, or(a, b))>

  // COM: A stored `or` is dropped once one of its disjuncts arrives.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <a>
  kgen.param.constant: scalar<bool> = <and(or(a, b), a)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b)>
  kgen.param.constant: scalar<bool> = <and(or(a, b), b, a)>

  // COM: A conjunctive disjunct can need the stored and the incoming fact.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b)>
  kgen.param.constant: scalar<bool> = <and(or(and(a, b), c), a, b)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b)>
  kgen.param.constant: scalar<bool> = <and(a, b, or(and(a, b), c))>
  hlcf.return
}

// CHECK-LABEL: @fold_not
kgen.generator @fold_not<a: scalar<bool>, b: scalar<bool>>() {
  // COM: A known conjunct is discharged from a negated conjunction, leaving the
  // COM: rest negated. Which side arrives first does not matter.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(b), a)>
  kgen.param.constant: scalar<bool> = <and(a, not(and(a, b)))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(a, not(and(a, b)), b)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(not(and(a, b)), a, b)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(b, not(and(a, b)), a)>

  // COM: A conjunction goal against its own negation.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(not(and(a, b)), and(a, b))>

  // COM: `not(a)` implies `not(and(a, b))`, so the wider one adds nothing.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <not(a)>
  kgen.param.constant: scalar<bool> = <and(not(a), not(and(a, b)))>
  hlcf.return
}

// COM: A later fact that shrinks a stored `not` makes it as strong as an
// COM: incoming `not`, so it drops the stored `not`s and `or`s that it implies.
// COM: Each fold is stated in both orders.
// CHECK-LABEL: @fold_not_shrink
kgen.generator @fold_not_shrink<a: scalar<bool>, b: scalar<bool>, c: scalar<bool>, d: scalar<bool>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(a), b)>
  kgen.param.constant: scalar<bool> = <and(not(and(a, b)), not(and(a, c)), b)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(a), b)>
  kgen.param.constant: scalar<bool> = <and(b, not(and(a, c)), not(and(a, b)))>

  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(a), b)>
  kgen.param.constant: scalar<bool> = <and(or(not(a), d), not(and(a, b)), b)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(a), b)>
  kgen.param.constant: scalar<bool> = <and(b, not(and(a, b)), or(not(a), d))>
  hlcf.return
}

// COM: Two of four negated pairs are shrunk by the positive pair, and the
// COM: shrunk ones imply a third.
// CHECK-LABEL: @fold_not_shrink_pairs
kgen.generator @fold_not_shrink_pairs<a4: scalar<bool>, a5: scalar<bool>, b4: scalar<bool>, b5: scalar<bool>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(a4), not(b5), a5, b4)>
  kgen.param.constant: scalar<bool> = <and(not(and(a4, b4)), not(and(a5, b5)), not(and(a4, b5)), a5, b4)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(a4), not(b5), a5, b4)>
  kgen.param.constant: scalar<bool> = <and(b4, a5, not(and(a4, b5)), not(and(a5, b5)), not(and(a4, b4)))>
  hlcf.return
}

// COM: A leading conjunction seeds the set without being refolded, so each
// COM: later clause must still interact with the seeded ones as it would in a
// COM: refold. A conjunction in any other position is refolded.
// CHECK-LABEL: @fold_onto_conjunction
kgen.generator @fold_onto_conjunction<a: scalar<bool>, b: scalar<bool>, c: scalar<bool>, d: scalar<bool>, e: scalar<bool>, t: type, t1: type, t2: type, t3: type>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b)>
  kgen.param.constant: scalar<bool> = <and(and(a, b), a)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(and(a, b), not(a))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(c, identical(:type t1, t2, t3))>
  kgen.param.constant: scalar<bool> = <and(and(c, identical(:type t1, t2)), identical(:type t3, t2))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(and(c, identical(:type t1, i32)), identical(:type t1, i1))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(b), a, c)>
  kgen.param.constant: scalar<bool> = <and(and(c, not(and(a, b))), a)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, c)>
  kgen.param.constant: scalar<bool> = <and(and(c, or(a, b)), a)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(c, conforms_to(:type t, :type [typevalue<#kgen.trait_ref<[@A, @B]>>, type]))>
  kgen.param.constant: scalar<bool> = <and(and(c, conforms_to(:type t, :type #A)), conforms_to(:type t, :type #AB))>

  // COM: A leaf ahead of the conjunction it joins.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b, c)>
  kgen.param.constant: scalar<bool> = <and(c, and(a, b))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(not(a), and(a, b))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(b), a, c)>
  kgen.param.constant: scalar<bool> = <and(a, and(c, not(and(a, b))))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(c, identical(:type t1, t2, t3))>
  kgen.param.constant: scalar<bool> = <and(identical(:type t3, t2), and(c, identical(:type t1, t2)))>

  // COM: Two multi-clause conjunctions, in either order.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(d), a, b, c)>
  kgen.param.constant: scalar<bool> = <and(and(a, b), and(c, not(and(a, d))))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(d), a, b, c)>
  kgen.param.constant: scalar<bool> = <and(and(c, not(and(a, d))), and(a, b))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(and(a, b), and(c, not(and(a, b))))>

  // COM: A shorter conjunction ahead of a longer one.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(not(e), a, b, c, d)>
  kgen.param.constant: scalar<bool> = <and(and(a, d), and(b, c, not(and(a, e))))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <and(and(a, d), and(b, c, not(and(a, b))))>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b, c, identical(:type t1, t2, t3))>
  kgen.param.constant: scalar<bool> = <and(and(a, identical(:type t3, t2)), and(b, c, identical(:type t1, t2)))>
  hlcf.return
}
