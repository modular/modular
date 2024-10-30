//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/CustomDialect/CustomUtils.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/TransformUtils/SlicingUtils.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace Custom;

namespace M::KGEN {
#define GEN_PASS_DEF_REGISTERCUSTOMOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct RegisterCustomOpsPass
    : public impl::RegisterCustomOpsBase<RegisterCustomOpsPass> {
  using RegisterCustomOpsBase::RegisterCustomOpsBase;

  void runOnOperation() override;
};

} // namespace

void RegisterCustomOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ExportMap exportedSymbols;
  SmallVector<std::pair<GeneratorOp, ArrayAttr>> updates;
  for (auto gen : getOperation().getOps<GeneratorOp>()) {
    ArrayAttr patterns = gen.getPatternsAttr();
    if (!patterns)
      continue;
    exportedSymbols.try_emplace(gen.getSymNameAttr(), ExportKind::Exported);
    SmallVector<Attribute> patternTemplates;
    for (auto sym : patterns.getAsRange<SymbolConstantAttr>()) {
      StringAttr name = sym.getSymbol().getLeafReference();
      auto params = ParameterExprArrayAttr::get(ctx, sym.getParamValues());
      patternTemplates.push_back(
          ArrayAttr::get(ctx, ArrayRef<Attribute>{name, params}));
      exportedSymbols.try_emplace(name, ExportKind::Exported);
    }
    updates.emplace_back(gen, ArrayAttr::get(ctx, patternTemplates));
  }
  if (exportedSymbols.empty())
    return;

  // Slice a module containing the custom op definitions.
  OwningOpRef<ModuleOp> slice = getOperation().clone();
  for (auto [gen, patterns] : updates)
    gen.setPatternsAttr(patterns);

  // Serialize and store these on the module so they can be accessed later.
  DenseResourceElementsAttr customOpResource =
      writeModuleToBytecodeAttr(*slice);
  getOperation()->setAttr(kCustomOpImplModuleAttr, customOpResource);
}
