//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_ARGPROMOTION
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class ArgPromotionPass : public impl::ArgPromotionBase<ArgPromotionPass> {
public:
  using ArgPromotionBase::ArgPromotionBase;
  void runOnOperation() override;
};

struct Node : public SCCNode<Node, FuncOp, CallOp> {
  using SCCNode::SCCNode;
};

struct Graph : public SCCGraph<Graph, Node> {
  using SCCGraph::SCCGraph;

  /// Perform analysis on promotable arguments within a single node.
  bool doAnalysis(Node *node);
  /// Promote arguments on a function.
  void doRewrite(const Node *node);

  /// Check non-direct-call operations for references to functions. These
  /// functions cannot be modified.
  void checkNonCallOp(Operation *op) {
    mlir::AttrTypeWalker walker;
    walker.addWalk(
        [&](FlatSymbolRefAttr ref) { cantPromote.insert(ref.getAttr()); });
    for (const NamedAttribute &attr : op->getAttrs())
      walker.walk(attr.getValue());
    // Note: At this stage in the pipeline, there should be no function
    // references inside types or locations.
  }

  /// These are functions that have references outside of direct calls. We can
  /// only modify the ABI of internal functions that are directly called.
  DenseSet<StringAttr> cantPromote;
};
} // namespace

bool Graph::doAnalysis(Node *node) { return false; }

void Graph::doRewrite(const Node *node) {}

void ArgPromotionPass::runOnOperation() {
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  AsyncRT::Runtime &runtime =
      *loadContext(&getContext())->get<AsyncRT::Runtime>();

  Graph cg;
  cg.build(getOperation(), symtab);

  cg.dump();
  for (StringAttr cant : cg.cantPromote)
    cant.dump();

  cg.run(runtime);
}
