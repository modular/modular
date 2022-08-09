//===- LowerHLKGEN.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/HLKGENDialect/HLKGENOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// FIXME: This shouldn't be needed here, this is because
// KGENPasses.h.inc/GEN_PASS_CLASSES is too monolithic.
namespace mlir::func {
class FuncDialect;
} // namespace mlir::func
namespace mlir::LLVM {
class LLVMDialect;
} // namespace mlir::LLVM

using namespace M;
using namespace KGEN;

/// Lower an hlkgen.generator to kgen.generator.
static LogicalResult lowerHLGenerator(KGEN::HLGeneratorOp gen) {
  OpBuilder b(gen);

  // Directly lower since these operations are exactly identical right now.
  auto result = b.create<GeneratorOp>(
      gen.getLoc(), gen.getFunctionTypeAttr(), gen.getParamDeclsAttr(),
      gen.getNumInputParametersAttr(), gen.getConstraintsAttr(),
      gen.getConstraintMessages(), gen.getImplementsAttr());

  // Move over the body unmodified.
  auto *bodyBlock = gen.getBodyBlock();
  gen.getBody().getBlocks().remove(bodyBlock);
  result.getBody().push_back(bodyBlock);

  // Move over the symbol.
  SymbolTable::setSymbolName(result, SymbolTable::getSymbolName(gen));

  // Remove the old operation.
  gen->erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass boilerplate.
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_CLASSES
#include "KGEN/KGENPasses.h.inc"

class LowerHLKGENPass : public LowerHLKGENBase<LowerHLKGENPass> {
public:
  void runOnOperation() override {
    // TODO: This has to be a module pass because this mutates the body of the
    // module, but we could trivially parallelize this within the pass.
    ModuleOp module = getOperation();
    for (auto hlGenerator :
         llvm::make_early_inc_range(module.getOps<KGEN::HLGeneratorOp>())) {
      if (failed(lowerHLGenerator(hlGenerator)))
        signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> M::KGEN::createLowerHLKGENPass() {
  return std::make_unique<LowerHLKGENPass>();
}
