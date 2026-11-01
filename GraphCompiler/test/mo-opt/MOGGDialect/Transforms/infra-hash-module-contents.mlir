// RUN: mo-opt %s -mogg-hash-module-contents -allow-unregistered-dialect -split-input-file | FileCheck %s

// COM: Identical functions should hash to the same value, as should their
// COM: transitive dependencies.

// CHECK-LABEL @foo1
// CHECK: semantic_hash = "[[FOO_HASH:.*]]"
kgen.generator @foo_1(%arg0: si32, %arg1: si32) -> si32 {
  hlcf.return %arg0 : si32
}

// CHECK-LABEL @foo2
// CHECK: semantic_hash = "[[FOO_HASH]]"
kgen.generator @foo_2(%arg0: si32, %arg1: si32) -> si32 {
  hlcf.return %arg0 : si32
}

// CHECK-LABEL @bar1
// CHECK: semantic_hash = "[[BAR_HASH:.*]]"
kgen.generator @bar_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_1(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL @bar2
// CHECK: semantic_hash = "[[BAR_HASH]]"
kgen.generator @bar_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_2(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// -----

// COM: Swapping the callsite of functions should change the hash
// COM: This ensures we're properly factoring in the callee's hash on a
// COM: per-callsite basis.

// CHECK-LABEL @foo
kgen.generator @foo(%arg0: si32, %arg1: si32) -> si32 {
  hlcf.return %arg0 : si32
}

// CHECK-LABEL @bar
kgen.generator @bar(%arg0: si32, %arg1: si32) -> si32 {
  hlcf.return %arg1 : si32
}

// CHECK-LABEL: @test_1
// CHECK: semantic_hash = "[[HASH:.*]]"
kgen.generator @test_1(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @foo(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @bar(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x, %y : si32, si32
}

// CHECK-LABEL: @test_2
// CHECK-NOT: semantic_hash = "[[HASH]]"
kgen.generator @test_2(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @bar(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @foo(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x, %y : si32, si32
}

// -----

// COM: Equivalent recursive functions could hash to the same value, as should
// COM: equivalent callers.
// COM: Improving this is future work.

// CHECK-LABEL: @recursion_1
// CHECK-SAME: semantic_hash = "[[REC_HASH:.*]]"
kgen.generator @recursion_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @recursion_2(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL: @recursion_2
// CHECK-NOT: semantic_hash = "[[REC_HASH]]"
kgen.generator @recursion_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @recursion_1(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL: @test_1
// CHECK-SAME: semantic_hash = "[[CALLER_HASH:.*]]"
kgen.generator @test_1(%arg0: si32) -> si32 {
  %x = kgen.call @recursion_1(%arg0, %arg0) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL @test_2
// CHECK-NOT: semantic_hash = "[[CALLER_HASH]]"
kgen.generator @test_2(%arg0: si32) -> si32 {
  %x = kgen.call @recursion_2(%arg0, %arg0) : (si32, si32) -> si32
  hlcf.return %x : si32
}


// -----

// COM: Equivalent recursive functions should hash to the same value, as should
// COM: equivalent callers.

kgen.generator @scc_input() {
  "some_op" () : () -> ()
  hlcf.return
}

// CHECK-LABEL: @recursion_1
// CHECK: semantic_hash = "[[REC_HASH_1:.*]]"
kgen.generator @recursion_1(%arg0: si32, %arg1: si32) -> (si32, si32) {
  kgen.call @scc_input() : () -> ()

  %x = kgen.call @recursion_2(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @recursion_3(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x, %y : si32, si32
}

// CHECK-LABEL: @recursion_2
// CHECK: semantic_hash = "[[REC_HASH_2:.*]]"
kgen.generator @recursion_2(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @recursion_3(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @recursion_1(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x, %y : si32, si32
}

// CHECK-LABEL: @recursion_3
// CHECK-NOT: semantic_hash = "[[REC_HASH_1]]"
// CHECK-NOT: semantic_hash = "[[REC_HASH_2]]"
kgen.generator @recursion_3(%arg0: si32, %arg1: si32) -> (si32, si32) {
  %x = kgen.call @recursion_2(%arg0, %arg1) : (si32, si32) -> si32
  %y = kgen.call @recursion_1(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x, %y : si32, si32
}

// -----

// CHECK-LABEL: @foo_recursive_1
// CHECK: semantic_hash = "[[HASH1:.*]]"
kgen.generator @foo_recursive_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_recursive_2(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL: @foo_recursive_2
// CHECK: semantic_hash = "[[HASH2:.*]]"
kgen.generator @foo_recursive_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @foo_recursive_1(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL: @bar_recursive_1
// CHECK: semantic_hash = "[[HASH1]]"
kgen.generator @bar_recursive_1(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @bar_recursive_2(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// CHECK-LABEL: @bar_recursive_2
// CHECK: semantic_hash = "[[HASH2]]"
kgen.generator @bar_recursive_2(%arg0: si32, %arg1: si32) -> si32 {
  %x = kgen.call @bar_recursive_1(%arg0, %arg1) : (si32, si32) -> si32
  hlcf.return %x : si32
}

// -----

// Ensure that functions referred to in "parameter" positions are factored
// into the hash. This occurs commonly in Mojo code with functions like
// `ctx.enqueue_function[my_func]()`. There is no direct call to `my_func` in
// this instance.

// CHECK-LABEL: @parameter_function_1
// CHECK: semantic_hash = "[[FUNC_HASH:.*]]"
kgen.generator @parameter_function_1(%arg0: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %x = pop.add %arg0, %arg0 : !kgen.scalar<si32>
  hlcf.return %x : !kgen.scalar<si32>
}

// CHECK-LABEL: @parameter_function_2
// CHECK-NOT: semantic_hash = "[[FUNC_HASH]]"
kgen.generator @parameter_function_2(%arg0: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %x = pop.sub %arg0, %arg0 : !kgen.scalar<si32>
  hlcf.return %x : !kgen.scalar<si32>
}

kgen.generator @enqueue_function<callable: (!kgen.scalar<si32>) -> (!kgen.scalar<si32>)>() -> !kgen.scalar<si32> {
  %c0 = kgen.param.constant : !kgen.scalar<si32> = <1>
  %callee = kgen.param.constant : (!kgen.scalar<si32>) -> !kgen.scalar<si32> = <callable>
  %call = kgen.call_indirect %callee(%c0) : (!kgen.scalar<si32>) -> !kgen.scalar<si32>
  hlcf.return %call : !kgen.scalar<si32>
}

// CHECK-LABEL: @main_1
// CHECK: semantic_hash = "[[MAIN_HASH:.*]]"
kgen.generator @main_1() {
  kgen.call @enqueue_function<:(!kgen.scalar<si32>) -> !kgen.scalar<si32> @parameter_function_1>() : () -> ()
  hlcf.return
}

// CHECK-LABEL: @main_2
// CHECK-NOT: semantic_hash = "[[MAIN_HASH]]"
kgen.generator @main_2() {
  kgen.call @enqueue_function<:(!kgen.scalar<si32>) -> !kgen.scalar<si32> @parameter_function_2>() : () -> ()
  hlcf.return
}

// -----

// COM: Generators that differ only in trait conformance symbols should hash
// COM: to different values. TypeConformsToTraitAttr carries SymbolRefAttrs
// COM: that must be preserved in the bytecode hash (not normalized away like
// COM: GeneratorOp references).

// CHECK-LABEL: @trait_check_a
// CHECK: semantic_hash = "[[TRAIT_HASH_A:.*]]"
kgen.generator @trait_check_a<T: type>(%arg0: si32) -> si32 {
  "test_op"() {check = #kgen.type_conforms_to_trait<#kgen.param.decl.ref<"T"> : !kgen.type, #kgen.type<typevalue<#kgen.trait_ref<[@TraitA]>>, type> : !kgen.type>} : () -> ()
  hlcf.return %arg0 : si32
}

// CHECK-LABEL: @trait_check_b
// CHECK-NOT: semantic_hash = "[[TRAIT_HASH_A]]"
kgen.generator @trait_check_b<T: type>(%arg0: si32) -> si32 {
  "test_op"() {check = #kgen.type_conforms_to_trait<#kgen.param.decl.ref<"T"> : !kgen.type, #kgen.type<typevalue<#kgen.trait_ref<[@TraitB]>>, type> : !kgen.type>} : () -> ()
  hlcf.return %arg0 : si32
}

// -----

// COM: @region reaches @store only through @FusionStruct's conformance witness
// COM: (a genref in a type attr); the struct generator is not a callgraph node,
// COM: so @region never names @store directly. These two split modules differ
// COM: only in @store's baked-in constant, so their @region hashes must differ;
// COM: otherwise one model is served the other's cached kernel (the rope_theta
// COM: cross-model cache-poisoning bug).

// CHECK-LABEL: @store
// CHECK: semantic_hash = "[[STORE_FIRST:.*]]"
kgen.generator @store() -> !kgen.simd<1, f64> {
  %c = kgen.param.constant: simd<1, f64> = <#pop.float_literal_convert<#pop.float_literal<250000|1>>>
  hlcf.return %c : !kgen.simd<1, f64>
}

kgen.struct.generator @FusionStruct<n: i32> =
    struct_inst<"FusionStruct"[n]<:i32 n>(field0: index)>
{
  kgen.conformance @OutputFusion {
    kgen.witness "store" : () -> !kgen.simd<1, f64> = @store
  }
}

// CHECK-LABEL: @region
// CHECK: semantic_hash = "[[REGION_FIRST:.*]]"
kgen.generator @region() {
  "use"() {fused_output = #kgen.type<typevalue<#kgen.genref<@FusionStruct<:i32 0>>>, struct<(scalar<index>)>> : !kgen.type} : () -> ()
  hlcf.return
}

// -----

// CHECK-LABEL: @store
// CHECK-NOT: semantic_hash = "[[STORE_FIRST]]"
kgen.generator @store() -> !kgen.simd<1, f64> {
  %c = kgen.param.constant: simd<1, f64> = <#pop.float_literal_convert<#pop.float_literal<500000|1>>>
  hlcf.return %c : !kgen.simd<1, f64>
}

kgen.struct.generator @FusionStruct<n: i32> =
    struct_inst<"FusionStruct"[n]<:i32 n>(field0: index)>
{
  kgen.conformance @OutputFusion {
    kgen.witness "store" : () -> !kgen.simd<1, f64> = @store
  }
}

// CHECK-LABEL: @region
// CHECK-NOT: semantic_hash = "[[REGION_FIRST]]"
kgen.generator @region() {
  "use"() {fused_output = #kgen.type<typevalue<#kgen.genref<@FusionStruct<:i32 0>>>, struct<(scalar<index>)>> : !kgen.type} : () -> ()
  hlcf.return
}

// -----

// COM: Positive case: a third module byte-identical to the first (same @store
// COM: constant) must reproduce the first module's @store and @region hashes,
// COM: confirming equivalent fused kernels still dedup through the
// COM: struct-generator path rather than over-separating.

// CHECK-LABEL: @store
// CHECK: semantic_hash = "[[STORE_FIRST]]"
kgen.generator @store() -> !kgen.simd<1, f64> {
  %c = kgen.param.constant: simd<1, f64> = <#pop.float_literal_convert<#pop.float_literal<250000|1>>>
  hlcf.return %c : !kgen.simd<1, f64>
}

kgen.struct.generator @FusionStruct<n: i32> =
    struct_inst<"FusionStruct"[n]<:i32 n>(field0: index)>
{
  kgen.conformance @OutputFusion {
    kgen.witness "store" : () -> !kgen.simd<1, f64> = @store
  }
}

// CHECK-LABEL: @region
// CHECK: semantic_hash = "[[REGION_FIRST]]"
kgen.generator @region() {
  "use"() {fused_output = #kgen.type<typevalue<#kgen.genref<@FusionStruct<:i32 0>>>, struct<(scalar<index>)>> : !kgen.type} : () -> ()
  hlcf.return
}
