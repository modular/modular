// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s

// COM: A scalar-bool `and` folds its operands into one set of clauses, inserted
// COM: in the order written, so several cases below state the same facts in
// COM: more than one order.

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
kgen.generator @fold_or<a: scalar<bool>, b: scalar<bool>>() {
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <a>
  kgen.param.constant: scalar<bool> = <and(a, or(a, b))>

  // COM: A stored `or` is dropped once one of its disjuncts arrives.
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <a>
  kgen.param.constant: scalar<bool> = <and(or(a, b), a)>
  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <and(a, b)>
  kgen.param.constant: scalar<bool> = <and(or(a, b), b, a)>
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
