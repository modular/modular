//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/DebugInfoToLLVM/DIExpressionSimplifier.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include <deque>

using namespace M;

namespace LLVM = mlir::LLVM;

#define DEBUG_TYPE "support-di-expression-simplifier"

namespace {
//===----------------------------------------------------------------------===//
// LLVMDIExpressionSimplifier
//===----------------------------------------------------------------------===//

/// Simplifier for LLVM::DIExpressionAttr.
/// Useful for optimizations that cannot be applied on DebugInfo dialect's
/// DIExpression representation due to manipulations that break types.
///
/// Users of this simplifier register their own rewrite patterns. Each pattern
/// matches on a contiguous range of LLVM DIExpressionElemAttrs, and can be
/// used to rewrite it into a new range of DIExpressionElemAttrs of any length.
///
/// For example, adjacent DW_OP_LLVM_fragment operators can be merged into one.
/// This is not allowed in DebugInfo dialect's DI Expressions since this erases
/// the intermediate types, but can be performed in LLVM dialect since its DI
/// Expression is untyped.
class LLVMDIExpressionSimplifier {
public:
  using OperatorT = LLVM::DIExpressionElemAttr;

  class ExprRewritePattern {
  public:
    using OperatorT = LLVMDIExpressionSimplifier::OperatorT;
    using OpIterT = std::deque<OperatorT>::const_iterator;
    using OpIterRange = llvm::iterator_range<OpIterT>;

    virtual ~ExprRewritePattern() = default;
    /// Check whether a particular prefix of operators matches this pattern.
    /// The provided argument is guaranteed non-empty.
    /// Return the iterator after the last matched element.
    virtual OpIterT match(OpIterRange) const = 0;
    /// Replace the operators with a new list of operators.
    /// The provided argument is guaranteed to be the same length as returned
    /// by the `match` function.
    virtual SmallVector<OperatorT> replace(OpIterRange) const = 0;
  };

  /// Register a rewrite pattern with the simplifier.
  /// Rewriter patterns are attempted in the order of registration.
  void addPattern(std::unique_ptr<ExprRewritePattern> pattern);

  /// Simplify a DIExpression according to all the patterns registered.
  /// A non-negative `maxNumRewrites` will limit the number of rewrites this
  /// simplifier applies.
  LLVM::DIExpressionAttr simplify(LLVM::DIExpressionAttr expr,
                                  int64_t maxNumRewrites = -1) const;

private:
  /// The registered patterns.
  SmallVector<std::unique_ptr<ExprRewritePattern>> patterns;
};

void LLVMDIExpressionSimplifier::addPattern(
    std::unique_ptr<ExprRewritePattern> pattern) {
  patterns.emplace_back(std::move(pattern));
}

LLVM::DIExpressionAttr
LLVMDIExpressionSimplifier::simplify(LLVM::DIExpressionAttr expr,
                                     int64_t maxNumRewrites) const {
  ArrayRef<OperatorT> operators = expr.getOperations();

  // `inputs` contains the unprocessed postfix of operators.
  // `result` contains the already finalized prefix of operators.
  // Invariant: concat(result, inputs) is equivalent to `operators` after some
  // application of the rewrite patterns.
  // Using a deque for inputs so that we have efficient front insertion and
  // removal. Random access is not necessary for patterns.
  std::deque<OperatorT> inputs(operators.begin(), operators.end());
  SmallVector<OperatorT> result;

  int64_t numRewrites = 0;
  while (!inputs.empty() &&
         (maxNumRewrites < 0 || numRewrites < maxNumRewrites)) {
    bool foundMatch = false;
    for (const std::unique_ptr<ExprRewritePattern> &pattern : patterns) {
      ExprRewritePattern::OpIterT matchEnd = pattern->match(inputs);
      if (matchEnd == inputs.begin())
        continue;

      foundMatch = true;
      SmallVector<OperatorT> replacement =
          pattern->replace(llvm::make_range(inputs.cbegin(), matchEnd));
      inputs.erase(inputs.begin(), matchEnd);
      inputs.insert(inputs.begin(), replacement.begin(), replacement.end());
      ++numRewrites;
      break;
    }

    if (!foundMatch) {
      // If no match, pass along the current operator.
      result.push_back(inputs.front());
      inputs.pop_front();
    }
  }

  if (maxNumRewrites >= 0 && numRewrites >= maxNumRewrites) {
    LLVM_DEBUG(llvm::dbgs()
               << "LLVMDIExpressionSimplifier exceeded max num rewrites ("
               << maxNumRewrites << ")\n");
    // Skip rewriting the rest.
    result.append(inputs.begin(), inputs.end());
  }

  return LLVM::DIExpressionAttr::get(expr.getContext(), result);
}

//===----------------------------------------------------------------------===//
// Known Patterns
//===----------------------------------------------------------------------===//

/// Adjacent DW_OP_LLVM_fragment ops can be merged into one.
class MergeFragments : public LLVMDIExpressionSimplifier::ExprRewritePattern {
public:
  OpIterT match(OpIterRange operators) const override {
    OpIterT it = operators.begin();
    if (it == operators.end() ||
        it->getOpcode() != llvm::dwarf::DW_OP_LLVM_fragment)
      return operators.begin();

    ++it;
    if (it == operators.end() ||
        it->getOpcode() != llvm::dwarf::DW_OP_LLVM_fragment)
      return operators.begin();

    return ++it;
  }

  SmallVector<OperatorT> replace(OpIterRange operators) const override {
    OpIterT it = operators.begin();
    OperatorT first = *(it++);
    OperatorT second = *it;
    // Add offsets & select the size of the earlier operator (the one closer to
    // the IR value).
    uint64_t offset = first.getArguments()[0] + second.getArguments()[0];
    uint64_t size = first.getArguments()[1];
    OperatorT newOp = OperatorT::get(
        first.getContext(), llvm::dwarf::DW_OP_LLVM_fragment, {offset, size});
    return SmallVector<OperatorT>{newOp};
  }
};
} // namespace

void DebugInfo::simplifyLLVMDIExpressionRecursively(Operation *op) {
  LLVMDIExpressionSimplifier simplifier;
  simplifier.addPattern(std::make_unique<MergeFragments>());

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&simplifier](LLVM::DIExpressionAttr expr) {
    return simplifier.simplify(expr);
  });
  replacer.recursivelyReplaceElementsIn(op);
}
