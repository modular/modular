//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "kgen-infer-function-attrs"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// InferFunctionAttrsPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_INFERFUNCTIONATTRS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct FunctionAttrs {
  bool isConvergent = false;
};

struct CallGraphNode
    : public SCCNode<CallGraphNode, FuncOp, KGENCallOpInterface> {
  using SCCNode::SCCNode;

  FunctionAttrs attrs;
};

struct CallGraph : public SCCGraph<CallGraph, CallGraphNode> {
  CallGraph(const SymbolTable &symtab) : symtab(symtab) {}

  /// Analyze function body to determine which attributes need to be added or
  /// removed.
  bool doAnalysis(CallGraphNode *node);

  /// Propagate all attributes that need to be added or removed to a function.
  void doRewrite(const CallGraphNode *node);

  /// Symbol table for function lookup.
  const SymbolTable &symtab;
};

/// Propagate attributes from function body to a function.
/// One of the crucial attribute to be propagated is `convergent` that tells
/// that function cannot be made control-dependent on any other value (see
/// https://llvm.org/docs/ConvergentOperations.html)
bool CallGraph::doAnalysis(CallGraphNode *node) {
  bool changed = false;
  FuncOp func = node->func;
  if (node->attrs.isConvergent || func.isConvergent()) {
    node->attrs.isConvergent = true;
    return false;
  }

  llvm::LLVMContext llvmCtx;
  func.walk([&](Operation *op) -> WalkResult {
    // TODO: Use trait when it's available in upstream
    if (auto barrier = dyn_cast<mlir::NVVM::Barrier0Op>(op)) {
      node->attrs.isConvergent = true;
      changed = true;
      return WalkResult::interrupt();
    }

    // Propagate `convergent` attribute from intrinsic.
    if (auto intrinsic = dyn_cast<POP::CallLLVMIntrinsicOp>(op)) {
      auto intrinsicName = cast<StringAttr>(intrinsic.getIntrin());
      llvm::Intrinsic::ID intrinsicID =
          llvm::Intrinsic::lookupIntrinsicID(intrinsicName.getValue());
      llvm::AttributeSet attrSet =
          llvm::Intrinsic::getFnAttributes(llvmCtx, intrinsicID);
      // Check if intrinsic is convergent
      if (attrSet.hasAttribute(llvm::Attribute::Convergent)) {
        node->attrs.isConvergent = true;
        changed = true;
        return WalkResult::interrupt();
      }
    }

    // Propagate `convergent` attribute from callee.
    if (auto call = dyn_cast<KGEN::CallOp>(op)) {
      auto callee = symtab.lookup<FuncOp>(call.getCalleeSymbol().getAttr());
      const CallGraphNode &calleeNode = nodes.find(callee)->second;
      if (calleeNode.attrs.isConvergent) {
        node->attrs.isConvergent = true;
        changed = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return changed;
}

/// Propagate all attributes that need to be added or removed to a function.
void CallGraph::doRewrite(const CallGraphNode *node) {
  FuncOp func = node->func;
  if (node->attrs.isConvergent)
    func.setConvergent(true);
}

struct InferFunctionAttrsPass
    : impl::InferFunctionAttrsBase<InferFunctionAttrsPass> {
  void runOnOperation() override;
};

void InferFunctionAttrsPass::runOnOperation() {
  AsyncRT::Runtime &runtime =
      *loadContext(&getContext())->get<AsyncRT::Runtime>();
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  CallGraph cg(symtab);
  cg.build(getOperation(), symtab);
  cg.run(runtime);
}

} // namespace
