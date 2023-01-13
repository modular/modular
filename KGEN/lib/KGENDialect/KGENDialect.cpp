//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// KGENDialectFoldInterface
//===----------------------------------------------------------------------===//

namespace {
struct KGENDialectFoldInterface : public mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Never hoist a constant out of a declaration scope. We could scan the
  /// parameters declarations to find the highest scope a constant could be
  /// hoisted into, but that is expensive to do.
  bool shouldMaterializeInto(Region *region) const override {
    return isa<DeclInterface>(region->getParentOp());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// KGENDialectInlinerInterface
//===----------------------------------------------------------------------===//

namespace {
struct KGENDialectInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All individual operations are legal to inline.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const override {
    return true;
  }

  /// FuncOp are legal to inline if they have the force_inline FnEffect. Other
  /// callables we don't want inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    if (auto func = dyn_cast<FuncOp>(callable)) {
      return bitEnumContainsAny(func.getFullSignature().getFnEffects(),
                                FnEffects::ForceInline);
    }

    return !isa<FuncInterface>(callable);
  }

  /// Region bodies are always able to be inlined assuming the callable check
  /// passed.
  bool isLegalToInline(Region *, Region *, bool,
                       BlockAndValueMapping &) const override {
    return true;
  }

  /// For now, we're only inlining kgen.func ops - so we don't have to deal with
  /// return parameters or anything.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const override {
    auto ret = cast<ReturnOp>(op);
    assert(ret.getNumOperands() == valuesToRepl.size());
    for (auto [operand, val] : llvm::zip(ret.getOperands(), valuesToRepl))
      val.replaceAllUsesWith(operand);
  }

  /// If the top level thing can be inlined, assume everything in it can too.
  bool shouldAnalyzeRecursively(Operation *op) const override { return false; }
};
} // namespace

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void KGENDialect::initialize() {
  registerAttributes();
  registerTypes();
  addInterfaces<KGENDialectFoldInterface, KGENDialectInlinerInterface>();
  injectAttrInterfaces();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/KGENDialect/KGEN.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/KGENDialect/KGENDialect.cpp.inc"
