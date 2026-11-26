// RUN: mo-opt -split-input-file -verify-diagnostics %s


// mogg.call.build_graph must reference a kernel.decl op.
func.func @build_graph_missing_decl(
    %in: !mogg.tensor,
    %b: !mgp.device_graph.builder) {
  // expected-error@+1 {{'missing.kernel' does not refer to a kernel.decl op}}
  mogg.call.build_graph["missing.kernel"]
      inputs(%in : !mogg.tensor)
      builder(%b) moOpName = "mo.matmul"
  return
}

// -----

// A mogg.graph_blob is the one launch of the calls it holds, so it can't hold
// another.
func.func @nested_graph_blob(%ctx: !mogg.context) {
  // expected-error@+1 {{calls kernels only through 'mogg.call.opaque' and 'mogg.call.foreach', but holds 'mogg.graph_blob'}}
  mogg.graph_blob {
    mogg.graph_blob {
      mogg.call.opaque @missing.kernel context(%ctx : !mogg.context)
    }
  }
  return
}

// -----

func.func @empty_graph_blob() {
  // expected-error@+1 {{calls no kernel}}
  mogg.graph_blob {
  }
  return
}

// -----

// A mogg.graph_blob is one launch, on one device context.
func.func @graph_blob_call_on_two_contexts(%ctx0: !mogg.context, %ctx1: !mogg.context) {
  // expected-error@+1 {{calls each kernel on one device context}}
  mogg.graph_blob {
    mogg.call.opaque @missing.kernel context(%ctx0, %ctx1 : !mogg.context, !mogg.context)
  }
  return
}

// -----

func.func @graph_blob_two_contexts(%ctx0: !mogg.context, %ctx1: !mogg.context) {
  // expected-error@+1 {{calls kernels on more than one device context}}
  mogg.graph_blob {
    mogg.call.opaque @missing.kernel context(%ctx0 : !mogg.context)
    mogg.call.opaque @missing.kernel context(%ctx1 : !mogg.context)
  }
  return
}

// -----

func.func @execute_device_outside_graph_blob_call() {
  // expected-error@+1 {{expects parent op 'mogg.graph_blob.call'}}
  mogg.call.execute_device @missing.kernel
  return
}

// -----

func.func @execute_device_missing_decl(%ctx: !mogg.context, %cfg: !mogg.launch_config, %act: i32) {
  mogg.graph_blob.launch context(%ctx) config(%cfg) args {%a = %act : i32} {
    mogg.graph_blob.call active(%a) {
      // expected-error@+1 {{'missing.kernel' does not refer to a kernel.decl op}}
      mogg.call.execute_device @missing.kernel
    }
  }
  return
}

// -----

// mogg.call.execute_device calls a declarative kernel's `execute_device`.
!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

func.func @execute_device_not_declarative(%ctx: !mogg.context, %cfg: !mogg.launch_config, %act: i32) {
  mogg.graph_blob.launch context(%ctx) config(%cfg) args {%a = %act : i32} {
    mogg.graph_blob.call active(%a) {
      // expected-error@+1 {{kernel 'builtin_kernels.monoids.ReduceSum' has no 'execute_device' function}}
      mogg.call.execute_device @builtin_kernels.monoids.ReduceSum
    }
  }
  return
}

// -----

// A static permutation reads no permutations operand.
func.func @transpose_static_with_permutations(%input: !mogg.tensor, %perm: !mogg.tensor) {
  // expected-error@+1 {{takes a permutations operand only when some static permutation is dynamic}}
  %0 = mogg.tensor.create.transpose(%input, %perm) <elementType = f32, mut = false, outShape = #mosh<ape[3, 2]> : !mosh.ape, staticPermutations = array<i64: 1, 0>> : !mogg.tensor
  return
}

// -----

// A run-time permutation is read from the permutations operand.
func.func @transpose_dynamic_without_permutations(%input: !mogg.tensor) {
  // expected-error@+1 {{takes a permutations operand only when some static permutation is dynamic}}
  %0 = mogg.tensor.create.transpose(%input) <elementType = f32, mut = false, outShape = #mosh<ape[3, 2]> : !mosh.ape, staticPermutations = array<i64: -9223372036854775808, 0>> : !mogg.tensor
  return
}
