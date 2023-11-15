//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_SLICEMOGGFUNCS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {

// The decorators we will look for on the generator to identify it as a MO
// kernel.
constexpr StringLiteral registerDecorator =
    "$utils::$_annotations::mogg_register";
constexpr StringLiteral registerOverrideDecorator =
    "$utils::$_annotations::mogg_register_override";
constexpr StringLiteral experimentalDecorator =
    "$utils::$_annotations::mogg_kgen_experiment_kernel";

constexpr StringLiteral tensorAllocDecorator =
    "$utils::$_annotations::mogg_tensor_allocator";
constexpr StringLiteral tensorMoveDecorator =
    "$utils::$_annotations::mogg_tensor_move_constructor";

template <typename LambdaToApply>
SmallVector<TypedAttr> forEachDecorator(GeneratorOp userKernel,
                                        LambdaToApply lambda) {
  SmallVector<TypedAttr> decoratorsToCopy;
  for (TypedAttr decorator : userKernel.getDecorators()) {
    // Keep track of the non mogg decorators to preserve them on the user
    // kernel.
    decoratorsToCopy.push_back(decorator);

    // Decorators are expected to the the apply of a symbol.
    auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator);
    if (!apply)
      continue;

    // The first operand is expected to be the symbol we are applying.
    auto sym = dyn_cast<KGEN::SymbolConstantAttr>(apply.getOperand(0));
    if (!sym)
      continue;

    StringRef decoratorName = sym.getSymbol().getLeafReference().strref();
    lambda(decorator, decoratorName, decoratorsToCopy);
  }
  return decoratorsToCopy;
}

class SliceMOGGFuncsPass
    : public M::KGEN::MOGGPreElab::impl::SliceMOGGFuncsBase<
          SliceMOGGFuncsPass> {
private:
  struct AnnotatedKernel {
    /// Every mogg kernel should have a registration hook mapping it onto an op.
    TypedAttr moggRegister;

    /// Currently we only run on MOGG kernels with the experimental attribute
    /// attached.
    TypedAttr experimentalAttr;

    /// When cloning the kernel we want to preserve the decorators unrelated to
    /// mogg.
    SmallVector<TypedAttr> nonMOGGDecorators;
  };

  std::optional<AnnotatedKernel> checkForMOGGAttrs(GeneratorOp userFunc) {
    AnnotatedKernel metadata;

    // Look for the mogg attributes on the kernels.
    auto lambda = [&](TypedAttr decorator, StringRef decoratorName,
                      SmallVector<TypedAttr> &attrsToCopy) {
      if (decoratorName.startswith(experimentalDecorator)) {
        metadata.experimentalAttr = decorator;
        // Drop the mogg decorator.
        attrsToCopy.pop_back();
      } else if (decoratorName.startswith(registerDecorator) ||
                 decoratorName.startswith(registerOverrideDecorator)) {
        metadata.moggRegister = decorator;

        // Drop the mogg decorator
        attrsToCopy.pop_back();
      }
    };

    // Capture the decorators unrelated to mogg so they can be preserved.
    metadata.nonMOGGDecorators = forEachDecorator(userFunc, lambda);

    // This is not a mogg kernel if it doesn't have a register and (for now) the
    // experimental attribute.
    if (!metadata.experimentalAttr || !metadata.moggRegister)
      return std::nullopt;
    return metadata;
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    SymbolTable symTab{mod};

    for (GeneratorOp userKernel :
         llvm::make_early_inc_range(mod.getOps<GeneratorOp>())) {

      std::optional<AnnotatedKernel> kernelMetadata =
          checkForMOGGAttrs(userKernel);
      if (!kernelMetadata.has_value())
        continue;

      // Slice out a new compute kernel. This replaces the old kernel as the
      // entry point for the thing we are going to execute.
      KGEN::GeneratorOp slicedComputeFunction = userKernel.clone();

      // Search for any function which allocates a new tensor and a move from
      // that into one of the input operands (meaning it is actually an output).
      KGEN::CallOp allocationFunc, moveConstructor;

      // Scan the kernel and identify the callsites of annotated functions that
      // we can understand.
      for (KGEN::CallOp call : slicedComputeFunction.getOps<KGEN::CallOp>()) {
        auto func = dyn_cast_or_null<KGEN::GeneratorOp>(symTab.lookup(
            cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
        if (!func)
          continue;

        auto identifyCalls = [&](TypedAttr decorator, StringRef decoratorName,
                                 SmallVector<TypedAttr> &attrsToCopy) {
          if (decoratorName.startswith(tensorAllocDecorator))
            allocationFunc = call;
          else if (decoratorName.startswith(tensorMoveDecorator))
            moveConstructor = call;
        };
        forEachDecorator(func, identifyCalls);
      }

      // Exit and clean up if the kernel is not what we expect.
      if (!moveConstructor || !allocationFunc) {
        slicedComputeFunction.erase();
        continue;
      }

      Value outputTensor = moveConstructor.getOperand(0);
      Value tmpTensor = moveConstructor.getOperand(1);
      tmpTensor.replaceAllUsesWith(outputTensor);

      // Remove the allocation and assignment from the sliced compute function.
      moveConstructor.erase();
      allocationFunc.erase();

      // Add the sliced functions to the KGEN as well.
      symTab.insert(slicedComputeFunction);

      // Remove the attributes from the user kernel as that should remain
      // untouched for the user to use directly in their code.
      userKernel.setDecorators(
          KGEN::DecoratorsAttr::get(ctx, kernelMetadata->nonMOGGDecorators));
    }
  }
};

} // namespace
