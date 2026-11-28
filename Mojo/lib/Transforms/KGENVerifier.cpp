//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "Target/TargetLowering.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_KGENVERIFIERPASS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

/// True for first-party + target specific dialects; others may be mid-lowering.
static bool shouldVerifyOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect)
    return false;
  // Curated subset of registerAllKGENDialects(); keep in sync.
  return llvm::StringSwitch<bool>(dialect->getNamespace())
      .Case("kgen", true)
      .Case("pop", true)
      .Case("lit", true)
      .Case("hlcf", true)
      .Case("co", true)
      .Case("interp", true)
      .Case("M", true)
      .Case("debuginfo", true)
      .Case("nvvm", true)
      .Case("rocdl", true)
      .Default(false);
}

/// Reject SIMD types whose resolved length cannot be lowered: non-positive
/// lengths, lengths which aren't a power of two, and lengths beyond LLVM
/// SelectionDAG's 2^15 operand limit (MOCO-1388).
/// FIXME: These should be correct on construction from the Mojo code itself,
/// but as per MOCO-2839 we're (often) silently dropping assertions during the
/// folding of @always_inline("builtin") functions. This is a clumsy fallback
/// but a good backstop to ensure we don't generate invalid code.
static LogicalResult verifySIMDLengths(Operation *op,
                                       DenseSet<Type> &checkedTypes) {
  constexpr int64_t maxSIMDLength = 1LL << 15;
  SIMDType invalid;
  auto checkType = [&](Type type) {
    if (!type) {
      llvm::reportFatalInternalError(
          Twine("'") + op->getName().getStringRef() +
          "' has a null operand, result or block argument type");
    }
    if (!checkedTypes.insert(type).second)
      return;
    type.walk([&](SIMDType simd) {
      std::optional<int64_t> size = simd.getResolvedSize();
      if (size &&
          (*size <= 0 || *size > maxSIMDLength || !llvm::isPowerOf2_64(*size)))
        invalid = simd;
    });
  };

  for (Type type : op->getOperandTypes())
    checkType(type);
  for (Type type : op->getResultTypes())
    checkType(type);
  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())
      for (BlockArgument arg : block.getArguments())
        checkType(arg.getType());

  if (!invalid)
    return success();
  return op->emitError("SIMD vector length must be a power of two between 1 "
                       "and 2^15, found ")
         << invalid;
}

/// True when a fully resolved type's layout overflows signed 64-bit object
/// size. Unknown / parametric layouts return false so they can be checked
/// after elaboration.
static bool hasOverflowingLayout(Type type, TargetInfoAttr target) {
  if (auto arr = dyn_cast<POP::ArrayType>(type)) {
    std::optional<int64_t> n = arr.getResolvedSize();
    if (!n)
      return false;
    if (*n < 0)
      return true;
    std::optional<int64_t> elem =
        DataLayoutInterface::getTypeAllocSize(target, arr.getElementType());
    if (!elem)
      return false;
    return !DataLayoutInterface::checkedSizeMul(*n, *elem);
  }
  if (auto simd = dyn_cast<SIMDType>(type)) {
    std::optional<int64_t> n = simd.getResolvedSize();
    std::optional<KGENDType> dtype = simd.getResolvedDType();
    if (!n || !dtype)
      return false;
    if (*n < 0)
      return true;
    if (simd.getTypeSize(target))
      return false;
    return dtype->isArithmetic() || dtype->isBool() || dtype->isAddress();
  }
  if (auto strct = dyn_cast<StructType>(type)) {
    std::optional<SmallVector<Type>> elems = strct.getElementTypes();
    if (!elems)
      return false;
    for (Type elem : *elems) {
      if (!DataLayoutInterface::getTypeAllocSize(target, elem) ||
          !DataLayoutInterface::getTypeABIAlign(target, elem))
        return false;
    }
    return !strct.getTypeSize(target);
  }
  return false;
}

/// Reject types whose size wraps in layout math. Wrapped sizes used to
/// lower to zero-byte stack slots that aliased neighbouring locals.
static LogicalResult verifyOversizedTypes(Operation *op, TargetInfoAttr target,
                                          DenseSet<Type> &checkedTypes) {
  Type invalid;
  auto checkType = [&](Type type) {
    if (!type) {
      llvm::reportFatalInternalError(
          Twine("'") + op->getName().getStringRef() +
          "' has a null operand, result or block argument type");
    }
    if (!checkedTypes.insert(type).second)
      return;
    type.walk([&](Type t) {
      if (hasOverflowingLayout(t, target))
        invalid = t;
    });
  };

  for (Type type : op->getOperandTypes())
    checkType(type);
  for (Type type : op->getResultTypes())
    checkType(type);
  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())
      for (BlockArgument arg : block.getArguments())
        checkType(arg.getType());

  if (!invalid)
    return success();
  return op->emitError("type is too large to fit in addressable memory, found ")
         << invalid;
}

namespace {
struct KGENVerifierPass : public impl::KGENVerifierPassBase<KGENVerifierPass> {
  using KGENVerifierPassBase::KGENVerifierPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    std::atomic<size_t> numErrors = 0;
    const size_t maxErrors = *KGENPassCLOptions::kgenVerifierMaxErrors();

    // Per-op (non-recursive): skips off-allowlist ops; structural/dominance
    // checks come from the PM verifier (off in MODULAR_PRODUCTION).
    // SIMD lengths and oversized layouts are checked on every op regardless
    // of dialect, since mid-lowering ops can still carry KGEN types.
    TargetInfoAttr target = lookupTargetInfo(op);
    auto verifyOp = [&](Operation *operation, DenseSet<Type> &checkedSIMD,
                        DenseSet<Type> &checkedSizes) {
      if (failed(verifySIMDLengths(operation, checkedSIMD)))
        ++numErrors;
      if (target &&
          failed(verifyOversizedTypes(operation, target, checkedSizes)))
        ++numErrors;
      if (shouldVerifyOp(operation) &&
          failed(mlir::verify(operation, /*verifyRecursively=*/false)))
        ++numErrors;
    };
    auto verifyTree = [&](Operation *root) {
      DenseSet<Type> checkedSIMD;
      DenseSet<Type> checkedSizes;
      return root->walk([&](Operation *operation) {
        verifyOp(operation, checkedSIMD, checkedSizes);
        return numErrors >= maxErrors ? WalkResult::interrupt()
                                      : WalkResult::advance();
      });
    };

    if (auto module = dyn_cast<mlir::ModuleOp>(op)) {
      // Dominance checks renumber a block's ops on first use, so workers must
      // not share a block. The module's block is safe because its ops take no
      // operands.
      SmallVector<Operation *> children;
      for (Operation &child : *module.getBody())
        children.push_back(&child);
      mlir::parallelForEach(&getContext(), children,
                            [&](Operation *child) { (void)verifyTree(child); });
      if (numErrors < maxErrors) {
        DenseSet<Type> checkedSIMD;
        DenseSet<Type> checkedSizes;
        verifyOp(op, checkedSIMD, checkedSizes);
      }
      if (numErrors >= maxErrors)
        signalPassFailure();
    } else if (verifyTree(op).wasInterrupted()) {
      signalPassFailure();
    }
    const TargetLowering *lowering = nullptr;
    if (target) {
      ErrorOr<const TargetLowering *> loweringOr =
          TargetLoweringRegistry::get().lookup(target.getTriple());
      lowering = loweringOr.isError() ? nullptr : *loweringOr;
    }
    if (useMLIRVerifierOnly || !lowering || !lowering->needsVerification()) {
      if (numErrors > 0)
        signalPassFailure();
      return;
    }

    // Target-specific verification.
    if (op->walk([&](Operation *operation) {
            if (failed(lowering->verifyOp(operation)))
              ++numErrors;
            if (numErrors >= maxErrors)
              return WalkResult::interrupt();
            return WalkResult::advance();
          }).wasInterrupted()) {
      signalPassFailure();
    }
    if (numErrors > 0)
      signalPassFailure();
  }
};
} // namespace
