// RUN: mo-opt -split-input-file -verify-diagnostics %s

// The launch's region has one argument per operand.
func.func @launch_too_few_args(%ctx: !mogg.context, %cfg: !mogg.launch_config, %p: !mogg.pointer) {
  // expected-error@+1 {{expects one region argument per launch operand, of the operand's type}}
  "mogg.graph_blob.launch"(%ctx, %cfg, %p) ({
  ^bb0:
  }) : (!mogg.context, !mogg.launch_config, !mogg.pointer) -> ()
  return
}

// -----

func.func @launch_arg_type_mismatch(%ctx: !mogg.context, %cfg: !mogg.launch_config, %p: !mogg.pointer) {
  // expected-error@+1 {{expects one region argument per launch operand, of the operand's type}}
  "mogg.graph_blob.launch"(%ctx, %cfg, %p) ({
  ^bb0(%x: i32):
  }) : (!mogg.context, !mogg.launch_config, !mogg.pointer) -> ()
  return
}

// -----

// The launch's top level holds only call regions and barriers.
func.func @launch_holds_other_op(%ctx: !mogg.context, %cfg: !mogg.launch_config) {
  // expected-error@+1 {{holds only 'mogg.graph_blob.call' and 'mogg.graph_blob.barrier', but holds 'mogg.graph_blob.launch_config'}}
  mogg.graph_blob.launch context(%ctx) config(%cfg) args {%c = %ctx : !mogg.context} {
    %l = mogg.graph_blob.launch_config context(%c)
  }
  return
}

// -----

// Everything the kernel reads crosses as an operand.
func.func @launch_uses_value_from_above(%ctx: !mogg.context, %cfg: !mogg.launch_config, %a: i32) {
  // expected-note@+1 {{required by region isolation constraints}}
  mogg.graph_blob.launch context(%ctx) config(%cfg) {
    // expected-error@+1 {{using value defined outside the region}}
    mogg.graph_blob.call active(%a) {
      mogg.call.execute_device @missing.kernel
    }
  }
  return
}

// -----

func.func @barrier_outside_launch() {
  // expected-error@+1 {{expects parent op 'mogg.graph_blob.launch'}}
  mogg.graph_blob.barrier
  return
}

// -----

func.func @call_outside_launch(%a: i32) {
  // expected-error@+1 {{expects parent op 'mogg.graph_blob.launch'}}
  mogg.graph_blob.call active(%a) {
  }
  return
}

// -----

// Every thread reaches a barrier, so a barrier can't sit in a call region.
func.func @barrier_in_call(%ctx: !mogg.context, %cfg: !mogg.launch_config, %act: i32) {
  mogg.graph_blob.launch context(%ctx) config(%cfg) args {%a = %act : i32} {
    mogg.graph_blob.call active(%a) {
      // expected-error@+1 {{expects parent op 'mogg.graph_blob.launch'}}
      mogg.graph_blob.barrier
    }
  }
  return
}

// -----

func.func @call_with_no_call(%ctx: !mogg.context, %cfg: !mogg.launch_config, %act: i32) {
  mogg.graph_blob.launch context(%ctx) config(%cfg) args {%a = %act : i32} {
    // expected-error@+1 {{expected exactly one call in the body, got 0}}
    mogg.graph_blob.call active(%a) {
    }
  }
  return
}

// -----

func.func @call_with_two_calls(%ctx: !mogg.context, %cfg: !mogg.launch_config, %act: i32) {
  mogg.graph_blob.launch context(%ctx) config(%cfg) args {%a = %act : i32} {
    // expected-error@+1 {{expected exactly one call in the body, got 2}}
    mogg.graph_blob.call active(%a) {
      mogg.call.execute_device @missing.kernel
      mogg.call.execute_device @missing.kernel
    }
  }
  return
}

// -----

func.func @execute_configuration_missing_decl() {
  // expected-error@+1 {{'missing.kernel' does not refer to a kernel.decl op}}
  %cfg = mogg.call.execute_configuration @missing.kernel
  return
}

// -----

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

func.func @execute_configuration_not_declarative() {
  // expected-error@+1 {{kernel 'builtin_kernels.monoids.ReduceSum' has no 'execute_configuration' function}}
  %cfg = mogg.call.execute_configuration @builtin_kernels.monoids.ReduceSum
  return
}
