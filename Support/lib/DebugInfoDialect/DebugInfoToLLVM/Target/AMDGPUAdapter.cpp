//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUAdapter.h"

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace M::DebugInfo;

namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// getAMDGPUAdapter
//===----------------------------------------------------------------------===//

TargetAdapter DebugInfo::getAMDGPUAdapter(bool tradeoffPerfForVariableDI) {
  return TargetAdapter{
      [](DIAttrTypeReplacer &replacer, RewritePatternSet &patterns) {},
      [](ModuleOp module) {}, [](ModuleOp module) {}};
}
