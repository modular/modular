// RUN: kgen-opt %s -split-input-file -verify-parameters \
// RUN:   -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file \
// RUN:   -elaborate-generators="use-parametric-interpret=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Tests `kgen.transparent_thunk_callee_expr` resolution via
// `IREvaluatorContext::resolveTransparentThunkCallee`. The thunk uses a
// `get_witness` expression that requires both parameter substitution and a
// witness-table lookup — see `kTransparentThunkCalleeExprAttr` in KGENOps.h
// for a description of the full elaboration sequence.

// The user's parametric kernel impl. `n` parameterises its emitted
// `nvvm.maxntid`, so a successful redirect carries `n=128` through to the
// emitted PTX function name and its attribute.
kgen.generator @user_kernel_impl<n: i32>() attributes {
  fnAttrs = [
    "nvvm.maxntid", #pop.array<n, 1, 1> : !pop.array<3, i32>
  ]
} {
  hlcf.return
}

// A parametric "closure wrapper" struct that conforms to `Callable`. Its
// witness for `__call__` is the user kernel instantiated with the wrapper's
// `n`, mirroring how a closure wrapper's witness points to the user impl
// applied to the closure's captures.
kgen.struct.generator @ClosureWrapper<n: i32> =
    struct_inst<"ClosureWrapper"[n]<:i32 n>(field0: index)>
{
  kgen.conformance @Callable {
    kgen.witness "__call__" : () -> () = @user_kernel_impl<:i32 n>
  }
}

// The transparent thunk. Parameterised over `W: type` (the callable type),
// it carries a `kgen.transparent_thunk_callee_expr` whose get_witness
// expression needs `W` substituted before it can resolve.
kgen.generator @callable_thunk<W: type>() attributes {
  "kgen.transparent_thunk_callee_expr" =
    #kgen.get_witness<W, @Callable, "__call__"> : !kgen.generator<() -> ()>
} {
  kgen.param.declare callee: () -> () =
      <#kgen.get_witness<W, @Callable, "__call__">>
  kgen.call_param[() -> () : callee]()
  hlcf.return
}

#wrapper128 = #kgen.type<
    typevalue<:!kgen.type #kgen.genref<@ClosureWrapper<:i32 128>>>,
    struct<(index)>
  > : !kgen.type

// CHECK-LABEL: kgen.func export @entry
kgen.generator export @entry() {
  kgen.param.declare nvptx: target =
      <#kgen.target<triple = "nvptx64-nvidia-cuda",
                    arch = "sm_80",
                    simd_bit_width = 128,
                    index_bit_width = 64,
                    tune_cpu = "sm_80",
                    data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">>

  // The offload targets the *thunk* instantiated with `W = ClosureWrapper<:i32 128>`.
  // `elaborate-generators` looks through the thunk, evaluates the
  // get_witness against the wrapper's conformance, and compiles the
  // resolved `@user_kernel_impl,n=128` instead. The emitted LLVM module
  // names that wrapped function (NOT the thunk) and carries its
  // `nvvm.maxntid` metadata as a function attribute.
  //
  // The compiled module is embedded as a single-line MLIR string so all checks
  // run on the same `kgen.param.constant` line via `CHECK-SAME`.
  //
  // CHECK: kgen.param.constant: string = <"
  // CHECK-SAME: define {{.*}}void @user_kernel_impl_n_128
  // CHECK-NOT:  callable_thunk
  // CHECK-SAME: nvvm.maxntid{{.*}}128,1,1
  %0 = kgen.compile_offload<nvptx, 2, "", "",
                            :() -> () @callable_thunk<:type #wrapper128>>
                            : !kgen.struct<(string, index)>
  hlcf.return
}
