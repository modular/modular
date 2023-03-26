//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// VerifyParameters
//===----------------------------------------------------------------------===//

def VerifyParameters : Pass<"verify-parameters", "mlir::ModuleOp"> {
  let summary = "Run the parameter verifier.";
  let description = [{
    The `verify-parameters` pass runs the KGEN parameter verifier. The pass
    ensures that all parameter uses are of the right type and reference
    parameters with declarations, it detects conflicting declarations, and it
    detects cycles in parameter definitions and uses.
  }];
}

//===----------------------------------------------------------------------===//
// LowerLITTerminators
//===----------------------------------------------------------------------===//

def LowerLITTerminators : Pass<"lower-lit-terminators", "mlir::ModuleOp"> {
  let summary = "Lower lexical terminators and prune unreachable code.";
  let description = [{
    The `lower-lit-terminators` pass will replace lexical Lit terminators with
    real terminators. It will prune any unreachable code (and warn about it),
    and verify that return parameters are the same at all return sites.

    Example:

    ```mlir
    lit.func @foo<() -> index>(%arg0: i32, %arg1: i32) -> i32 {
      lit.param_return<2>
      lit.return %arg0 : i32
      lit.return %arg1 : i32
      lit.end_func
    }
    ```

    Becomes:

    ```mlir
    lit.func @foo<() -> index>(%arg0: i32, %arg1: i32) -> i32 {
      kgen.param.result_bind<2>
      kgen.return %arg0 : i32
    }
    ```
  }];
}

//===----------------------------------------------------------------------===//
// LowerLIT
//===----------------------------------------------------------------------===//

def LowerLIT : Pass<"lower-lit", "mlir::ModuleOp"> {
  let summary = "Lower the `lit` dialects to the `kgen` dialect.";
  let description = [{
    This pass performs lowers high level abstractions from "lit" to "kgen"
    and diagnoses any errors that are found along the way.
  }];

  let dependentDialects = ["KGENDialect"];
}

//===----------------------------------------------------------------------===//
// LowerAsyncFunctions
//===----------------------------------------------------------------------===//

def LowerAsyncFunctions : Pass<"lower-async-functions", "mlir::ModuleOp"> {
  let summary = "Lower `async` functions by generating coroutine machinery.";
  let description = [{
    The `lower-async-functions` pass transforms `async` functions into normal
    `kgen.func` by adding the required POP-level coroutine machinery.
  }];

  let dependentDialects = ["POPDialect"];
}

//===----------------------------------------------------------------------===//
// LowerStructs
//===----------------------------------------------------------------------===//

def LowerStructs : Pass<"lower-structs", "mlir::ModuleOp"> {
  let summary = "Lower struct declarations to POP structs.";
  let description = [{
    This pass lowerings struct decls and decl reference types to in-inline POP
    struct types.
  }];

  let dependentDialects = ["POP::POPDialect"];
}

//===----------------------------------------------------------------------===//
// OutlineClosures
//===----------------------------------------------------------------------===//

def OutlineClosures : Pass<"outline-closures", "mlir::ModuleOp"> {
  let summary = "Outlines region parameters into explicit closures.";
  let description = [{
    This pass outlines region parameters into functions that are called with
    a struct that contains all captures.
  }];

  let dependentDialects = ["KGENDialect", "POP::POPDialect"];
}

//===----------------------------------------------------------------------===//
// LowerKGENToPOP
//===----------------------------------------------------------------------===//

def LowerKGENToPOP : Pass<"lower-kgen-to-pop", "FuncOp"> {
  let summary = "Perform staged lowering of KGEN structs and lists.";
  let description = [{
    The `lower-kgen-to-pop` pass is a staged lowering of "special" KGEN types:
    structs and lists. Due to restrictions on how these compose with LLVM, we
    handle these lowerings as a separate pass to POP, a smaller set of
    operations to worry about than all of LLVM IR.
  }];

  let dependentDialects = ["POP::POPDialect"];
}

//===----------------------------------------------------------------------===//
// LowerKGENToLLVM
//===----------------------------------------------------------------------===//

def LowerKGENToLLVM : Pass<"lower-kgen-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the `kgen` dialect to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion from the fully-elaborated and specified
    `kgen.func` ops and kgen types to the corresponding LLVM-dialect
    structures/types/ops.

    Functions marked as exported by a `kgen.export` operation are marked as
    external. All other functions are marked as internal linkage. For exported
    functions, wrapper functions are generated with C-compatible calling
    conventions. This breaks up argument and results structs in-place and
    converts arrays to be pass-by-reference.

    For example, consider the following function that accepts and returns a
    buffer:

    ```mlir
    llvm.func @slice(%a: !llvm.struct<(i64, ptr<f32>)>, %i: !llvm.ptr<i64>)
        -> !llvm.struct<(i64, ptr<f32>)>
    ```

    The resulting wrapper's signature will be:

    ```mlir
    llvm.func @slice_c(%a_0: i64, %a_1: !llvm.ptr<f32>, %i: !llvm.ptr<i64>,
                     %res_0: !llvm.ptr<i64>, %res_1: !llvm.ptr<ptr<f32>>)
    ```
  }];

  // We generate the LLVM dialect.
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// TweakSpilledAllocas
//===----------------------------------------------------------------------===//

def TweakSpilledAllocas : Pass<"tweak-spilled-allocas",
                               "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Trim and optimize lifetimes of spilled allocas.";
  let description = [{
    When LLVM lowers coroutines, spilled allocas are hoisted into the coroutine
    frame, invalidating their lifetime markers. LLVM's `coro-split` pass pukes
    when this happens, so prevent it by stripping lifetimes of spilled allocas.
    This pass must run prior to `lower-control-flow`, since the algorithm is
    trivial on region control-flow form.
  }];
}

//===----------------------------------------------------------------------===//
// LowerKGENCoroutinesAsync
//===----------------------------------------------------------------------===//

def LowerKGENCoroutinesAsync : Pass<"lower-coroutines-async",
                                    "mlir::ModuleOp"> {
  let summary = "Lower async functions to async LLVM coroutines.";
  let description = [{
    The `lower-coroutines-async` pass lowers KGEN coroutines implemented with
    POP coroutine operations to LLVM async coroutine intrinsics. LLVM async
    coroutine intrinsics require the front-end to handle control-flow and ABI.
    We generate the code to marshall function arguments and results through the
    coroutine async context and we handle control-flow using by storing the
    current resume function in the context.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// LowerPOPToLLVM
//===----------------------------------------------------------------------===//

def LowerPOPToLLVM : Pass<"lower-pop-to-llvm"> {
  let summary = "Lower the `pop` dialect to the `llvm` dialect.";
  let description = [{
    This pass performs a conversion of fully-elaborated and specified `pop`
    dialect operations to LLVM dialect operations.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// LowerGlobalPOPToLLVM
//===----------------------------------------------------------------------===//

def LowerGlobalPOPToLLVM : Pass<"lower-global-pop-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower global `pop` dialect operations to the `llvm` dialect.";
  let description = [{
    This module pass lowers `pop` operations that have global implications,
    and thus their lowering pass needs to run on the whole module. These are
    external calls and glboal constant operations.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// LowerPOPClosuresToLLVM
//===----------------------------------------------------------------------===//

def LowerPOPClosuresToLLVM : Pass<"lower-pop-closures-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower `pop` dialect closures to the `llvm` dialect.";
  let description = [{
    Lowers ops that contain `!pop.closure` type into LLVM dialect. For now,
    those ops are `!pop.partial_apply` and `!pop.call_indirect`.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// LowerControlFlow
//===----------------------------------------------------------------------===//

def LowerControlFlow : Pass<"lower-control-flow"> {
  let summary = "Lower structured control flow to LLVM and CFGs.";
  let description = [{
    This pass converts structured control-flow operations to LLVM operations and
    control-flow graph regions. Because this pass generates CFG regions, it must
    run after `lower-kgen-to-llvm`, which lowers `kgen.func` to `llvm.func`.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ElaborateGenerators
//===----------------------------------------------------------------------===//

def ElaborateGenerators : Pass<"elaborate-generators", "mlir::ModuleOp"> {
  let summary = "elaborate generators into fully concrete funcs";
  let description = [{
    This pass performs the elaboration step for the kgen pipeline,
    converting generators and generator interfaces into concrete
    funcs.
  }];

  let dependentDialects = ["::M::KGEN::KGENDialect"];

  let options = [
    Option<"shouldDoSearch", "enable-search", "bool", "false",
           "If true, benchmark implementations to find the fastest one.">
  ];
}

//===----------------------------------------------------------------------===//
// EmitLLVM
//===----------------------------------------------------------------------===//

def EmitLLVM : Pass<"emit-llvm", "mlir::ModuleOp"> {
  let summary = "emit LLVM IR";
  let description = [{
    This pass emits the LLVM IR for the concrete KGEN. If no file is provided
    then the IR will be printed to stdout.
  }];

  let dependentDialects = ["::M::KGEN::KGENDialect"];

  let options = [
    Option<"output", "output", "std::string", "", "Output file path.">,
  ];
}

//===----------------------------------------------------------------------===//
// PruneImpossibleVariants
//===----------------------------------------------------------------------===//

def PruneImpossibleVariants: Pass<"prune-impossible-variants",
                                  "mlir::ModuleOp"> {
  let summary = "remove variant cases that are known to be impossible";
  let description = [{
    The `prune-impossible-variants` pass runs a simple forward data-flow
    analysis to determine variant cases that are impossible.

    The pass folds `pop.variant.is` operations to constants when the operand is
    known to be or not to be the queried type and rewrites function signatures
    that contain variants of known types to those types.

    This pass must run post-elaboration.
  }];
}

//===----------------------------------------------------------------------===//
// AlwaysInlineParametric
//===----------------------------------------------------------------------===//

def AlwaysInlineParametric : Pass<"always-inline-param", "mlir::ModuleOp"> {
  let summary = "Process concrete 'always_inline' operations";
  let description = [{
    The `always-inline-param` pass will process all concrete calls to
    'always_inline' generators and inline them. If a cycle is encountered during
    inlining, the pass emits an error and fails to inline the cycle. This pass
    will strip debug info as appropriate for functions marked
    'always_inline(nodebug)'. This pass can only run on parametric KGEN IR.
  }];
}

//===----------------------------------------------------------------------===//
// ConstraintReduction
//===----------------------------------------------------------------------===//

def ConstraintReduction : Pass<"constraint-reduction", "KGEN::GeneratorOp"> {
  let summary = "Use a constraint set to reduce generator constraints.";
  let description = [{
    This pass analyses equivalence and equality constraints on a generator to
    determine whether there are redundant constraints that can be removed or
    whether the generator has a contradiction in its constraint set.
  }];
}

//===----------------------------------------------------------------------===//
// ForceInline
//===----------------------------------------------------------------------===//

def ForceInline : Pass<"force-inline", "mlir::ModuleOp"> {
  let summary = "Process all 'always_inline' functions";
  let description = [{
    The `force-inline` pass will process all calls to 'always_inline` functions
    and inline them. If a cycle is encountered during inlining, the pass emits
    an error and fails to inline the cycle. This pass will strip debug info as
    appropriate for functions marked 'always_inline(nodebug)'. This pass must
    run on concrete KGEN IR.
  }];
}

//===----------------------------------------------------------------------===//
// CleanupCompilerGlobals
//===----------------------------------------------------------------------===//

def CleanupCompilerGlobals: Pass<"cleanup-compiler-globals",
                                 "KGEN::FuncOp"> {
  let summary = "remove uses of global variables";
  let description = [{
    We use global variables to pass captures into lifted region parameters.
    These lifted regions are marked always_inline, so after inlining we
    can clean up usages of global variables. This pass is responsible for
    doing that.

    This pass must run post-elaboration, and after inlining.
  }];

  let dependentDialects = ["KGENDialect", "POP::POPDialect"];
}

//===----------------------------------------------------------------------===//
// EliminateDeadSymbols
//===----------------------------------------------------------------------===//

def EliminateDeadSymbols: Pass<"eliminate-dead-symbols", "ModuleOp"> {
  let summary = "remove dead symbols.";
  let description = [{
    This pass walks the symbols defined in the module and removes them
    if they aren't exported and have no uses.
  }];

  let dependentDialects = ["KGENDialect"];
}

//===----------------------------------------------------------------------===//
// Mem2Reg
//===----------------------------------------------------------------------===//

def Mem2Reg : Pass<"mem-2-reg"> {
  let summary = "Optimization pass that promotes memory to registers";
  let description = [{
    The `mem-2-reg` pass does memory elision over POP dialect operations by
    promoting values that live in stack memory, denoted by
    `pop.stack_allocation`, to register values, when all uses of the value by
    loads and stores are visible.

    This pass can run both before and after elaboration, but it can only analyze
    stack allocations with concrete sizes, and it can only run before
    control-flow lowering, since it does not work on CFGs.

    This pass is very basic and only elides stack allocations if all stores are
    in the same scope. It can replace loads spanning multiple regions though.

    Example:

    ```mlir
    %0 = pop.stack_allocation 1 x index
    pop.store %arg0, %0 : !pop.pointer<index>
    %1 = pop.load %0 : !pop.pointer<index>
    kgen.return %1 : index
    ```

    Will become:

    ```mlir
    kgen.return %arg0 : index
    ```

    However, stores within a nested region will not be elided (yet).

    ```mlir
    %0 = pop.stack_allocation 1 x index
    hlcf.if %arg2 {
      pop.store %arg0, %0 : !pop.pointer<index>
    } else {
      pop.store %arg1, %0 : !pop.pointer<index>
    }
    %1 = pop.load %0 : !pop.pointer<index>
    kgen.return %1 : index
    ```
  }];

  let statistics = [
    Statistic<"numAllocsElided", "num-allocs-elided",
              "the number of variables moved from memory to registers">,
    Statistic<"numLoadsElided", "num-loads-elided",
              "the number of loads elided">,
    Statistic<"numStoresElided", "num-stores-elided",
              "the number of stores elided">,
  ];
}
