//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/MOGGUtils.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_MOGGIDENTIFYKERNELS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

static constexpr std::array<llvm::StringLiteral, 4> kMaxRegistrationDecorator =
    {"max", "compiler", "__init__", "register"};
static constexpr llvm::StringLiteral kExecuteFuncName = "execute";
static constexpr llvm::StringLiteral kShapeFuncName = "shape";

namespace {
class MOGGIdentifyKernelsPass
    : public M::KGEN::MOGGPreElab::impl::MOGGIdentifyKernelsBase<
          MOGGIdentifyKernelsPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder{moduleOp.getContext()};

    const auto identifyKernelsLambda =
        [&](LIT::StructDeclOp structDeclOp) -> WalkResult {
      auto loc = structDeclOp->getLoc();
      std::optional<StringAttr> registrationName;
      auto decorators = structDeclOp.getDecorators();
      if (decorators.empty())
        return WalkResult::advance();

      // Iterate over the decorators and to find max.compiler.register.
      for (auto decorator : decorators) {
        auto apply = dyn_cast<ParamOperatorAttr>(decorator);
        if (!apply || apply.getNumOperands() != 2)
          continue;

        auto sym = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
        if (!sym)
          continue;

        if (!symbolMatches(sym.getSymbol(), kMaxRegistrationDecorator))
          continue;

        auto [_, nameAttr] =
            cast<LIT::LITStructAttr>(apply.getOperand(1)).getValues().front();
        auto name = dyn_cast<StringAttr>(nameAttr);
        if (!name)
          continue;

        if (registrationName) {
          emitError(loc, "Only one op can be registered per kernel struct");
          return WalkResult::interrupt();
        }
        registrationName = name;
      }

      if (!registrationName)
        return WalkResult::advance();

      // Search for the `execute` and the optional `shape` functions. Until
      // traits can enforce @staticmethod, this has to be done in this pass.
      LIT::FuncOp executeFunc, shapeFunc;
      for (auto &op : structDeclOp.getFields().front()) {
        auto func = dyn_cast<LIT::FuncOp>(op);
        if (!func)
          continue;

        if (func.getSourceName() == kExecuteFuncName)
          executeFunc = func;
        else if (func.getSourceName() == kShapeFuncName)
          shapeFunc = func;
      }

      if (!executeFunc) {
        emitError(loc, "Kernels must have an execution entry point named " +
                           kExecuteFuncName);
        return WalkResult::interrupt();
      }

      if (!executeFunc.getIsStatic()) {
        emitError(loc, "Kernel entry point must be a static function");
        return WalkResult::interrupt();
      }

      if (shapeFunc && !shapeFunc.getIsStatic()) {
        emitError(loc, "Kernel shape function must be a static function");
        return WalkResult::interrupt();
      }

      executeFunc->setAttr(builder.getStringAttr(kMOGGExecuteFunctionLabel),
                           *registrationName);
      if (shapeFunc) {
        shapeFunc->setAttr(builder.getStringAttr(kMOGGShapeFunctionLabel),
                           *registrationName);
      }

      return WalkResult::advance();
    };

    if (moduleOp.walk(identifyKernelsLambda).wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
