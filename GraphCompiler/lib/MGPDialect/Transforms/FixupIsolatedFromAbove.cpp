//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Transforms/FixupIsolatedFromAbove.h"

#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "mgp-fixup-isolated-from-above"

using namespace M;
using namespace M::MGP;
using namespace mlir;

[[maybe_unused]] static SetVector<Value> getValues(Operation *op) {
  SetVector<Value> result;
  for (Value value : op->getResults())
    result.insert(value);
  for (Value value : op->getOperands())
    result.insert(value);
  return result;
}

/// Returns false if a value is shared between inOp and outOp.
[[maybe_unused]] static bool checkRewrite(Operation *inOp, Operation *outOp) {
#ifdef MODULAR_DEBUG
  SetVector<Value> inValues = getValues(inOp);
  SetVector<Value> outValues = getValues(outOp);
  SetVector<Value> intersection = llvm::set_intersection(inValues, outValues);
  if (!intersection.empty()) {
    llvm::errs() << "a rewritten operation is continuing to use values shared "
                    "with the original operation.\n";
    llvm::errs() << "input operation:\n" << *inOp << "\n";
    llvm::errs() << "output operation:\n" << *outOp << "\n";
    for (Value v : intersection)
      llvm::errs() << "shared value:\n" << v << "\n";
    return false;
  }
#endif
  return true;
}

/// Rewrites inBlock into outBlock, where freeValues will instead be bound
/// by new arguments in outBlock. If mustAlsoReturn is true, freeValues will
/// also become new block outputs (as needed for a grt.while). If, in addition,
/// beforeLastOutput is true, any new block outputs will come just before the
/// existing last block output (as needed for the output of a grt.while 'cond'
/// block).
///
/// Could be reused at higher dialects provided they implement
/// GRTControlFlowInterface for the control flow ops, and we don't depend
/// on the concrete MEF::OutputOp type for checking for output terminators.
static void fixup(OpBuilder &builder, Block *inBlock,
                  const SetVector<Value> &freeValues, Block *outBlock,
                  bool mustAlsoReturn, bool beforeLastOutput) {
  IRMapping mapping;
  builder.setInsertionPoint(outBlock, outBlock->begin());

  LLVM_DEBUG(llvm::dbgs() << "fixup begin\n");

  for (BlockArgument inArg : inBlock->getArguments()) {
    BlockArgument outArg =
        outBlock->addArgument(inArg.getType(), inArg.getLoc());
    LLVM_DEBUG(llvm::dbgs() << "mapping " << inArg << " to " << outArg << "\n");
    mapping.map(inArg, outArg);
  }
  for (Value inFreeValue : freeValues) {
    BlockArgument outArg =
        outBlock->addArgument(inFreeValue.getType(), inFreeValue.getLoc());
    LLVM_DEBUG(llvm::dbgs()
               << "mapping " << inFreeValue << " to " << outArg << "\n");
    mapping.map(inFreeValue, outArg);
  }

  for (Operation *inOp : getDataflowOrder(inBlock)) {
    LLVM_DEBUG(llvm::dbgs() << "looking at op " << *inOp << "\n");
    if (auto inOutputOp = dyn_cast<MEF::OutputOp>(inOp)) {
      if (mustAlsoReturn) {
        SmallVector<Value> outOperands;
        // Rewrite existing operands (possibly excluding last).
        size_t n = inOutputOp->getNumOperands();
        if (beforeLastOutput) {
          assert(n > 0);
          --n;
        }
        for (size_t i = 0; i < n; ++i) {
          Value outOperand = mapping.lookupOrNull(inOutputOp->getOperand(i));
          ASSERT_STREAM(outOperand, << "unmapped operand");
          outOperands.emplace_back(outOperand);
        }
        // Include the additional arguments.
        for (Value inFreeValue : freeValues) {
          Value outFreeValue = mapping.lookupOrNull(inFreeValue);
          ASSERT_STREAM(outFreeValue, << "unmapped free value");
          outOperands.emplace_back(outFreeValue);
        }
        if (beforeLastOutput) {
          // Rewrite last operand.
          Value outOperand = mapping.lookupOrNull(inOutputOp->getOperand(n));
          ASSERT_STREAM(outOperand, << "unmapped operand");
          outOperands.emplace_back(outOperand);
        }
        [[maybe_unused]] Operation *outOutputOp =
            MEF::OutputOp::create(builder, inOp->getLoc(), outOperands);
        ASSERT_STREAM(checkRewrite(inOutputOp, outOutputOp),
                      << "invalid rewrite for output op");
        continue;
      }
    }

    if (auto inCfOp = dyn_cast<GRT::GRTControlFlowInterface>(inOp)) {
      bool subMustAlsoReturn = inCfOp.getFlavor() == GRT::kLoopLike;

      SetVector<Value> nestedFreeValues;
      for (Region &region : inCfOp->getRegions())
        accumFreeValues(nestedFreeValues, &region.front(), &region.front());
      assert(llvm::all_of(nestedFreeValues, [&](Value nestedFreeValue) {
        return nestedFreeValue.getParentBlock() == inBlock ||
               freeValues.contains(nestedFreeValue);
      }));

      SmallVector<Value> outOperands;
      for (Value operand : inCfOp->getOperands()) {
        Value outOperand = mapping.lookupOrNull(operand);
        ASSERT_STREAM(outOperand, << "unmapped operand");
        outOperands.emplace_back(outOperand);
      }
      // Include the additional arguments.
      for (Value inNestedFreeValue : nestedFreeValues) {
        Value outNestedFreeValue = mapping.lookupOrNull(inNestedFreeValue);
        ASSERT_STREAM(outNestedFreeValue, << "unmapped free value");
        outOperands.emplace_back(outNestedFreeValue);
      }

      // New op continues to return existing results (possibly excluding last).
      size_t n = inCfOp->getNumResults();
      if (beforeLastOutput) {
        assert(n > 0);
        --n;
      }
      SmallVector<Type> outResultTypes;
      for (size_t i = 0; i < n; ++i)
        outResultTypes.emplace_back(inCfOp->getResult(i).getType());
      if (subMustAlsoReturn) {
        // But new op must also return a result for each new argument.
        for (Value inNestedFreeValue : nestedFreeValues)
          outResultTypes.emplace_back(inNestedFreeValue.getType());
      }
      if (beforeLastOutput)
        // Account for last result.
        outResultTypes.emplace_back(inCfOp->getResult(n).getType());

      GRT::GRTControlFlowInterface outCfOp =
          cloneWithArgs(builder, inCfOp, outResultTypes, outOperands);
      ASSERT_STREAM(checkRewrite(inCfOp, outCfOp),
                    << "invalid rewrite for control flow op");

      // Additional results for the free values need no mapping.
      for (const auto [inResult, outResult] :
           llvm::zip(inCfOp->getResults(), outCfOp->getResults())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "mapping " << inResult << " to " << outResult << "\n");
        mapping.map(inResult, outResult);
      }

      auto insertPoint = builder.saveInsertionPoint();
      for (size_t i = 0, m = inCfOp->getNumRegions(); i < m; ++i) {
        // A grt.while 'cond' block returns the condition last, so new values
        // go before it.
        bool subBeforeLastOutput = inCfOp.getFlavor() == GRT::kLoopLike &&
                                   inCfOp->getNumRegions() == 2 && i == 0;
        Block *inSubBlock = &inCfOp->getRegion(i).front();
        Block *outSubBlock = &outCfOp->getRegion(i).emplaceBlock();
        fixup(builder, inSubBlock, nestedFreeValues, outSubBlock,
              subMustAlsoReturn, subBeforeLastOutput);
      }
      builder.restoreInsertionPoint(insertPoint);
      continue;
    }

    [[maybe_unused]] Operation *outOp = builder.clone(*inOp, mapping);
    ASSERT_STREAM(checkRewrite(inOp, outOp), << "invalid rewrite of op");
  }

  LLVM_DEBUG(llvm::dbgs() << "fixup end\n");
}

ModelOp M::MGP::fixupIsolatedFromAbove(mlir::OpBuilder &builder,
                                       ModelOp inModelOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "before fixupIsolatedFromAbove:\n";
    inModelOp.print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n\n";
  });

  builder.setInsertionPointAfter(inModelOp);
  auto outModelOp =
      MGP::ModelOp::create(builder, inModelOp.getLoc(),
                           /*devices=*/inModelOp.getDevicesAttr(),
                           /*frameworkName=*/inModelOp.getFrameworkNameAttr(),
                           /*inputOutputTypes=*/inModelOp.getInputOutputTypes(),
                           /*sym_name=*/inModelOp.getName(),
                           /*sym_visibility=*/nullptr,
                           /*isSubgraph=*/inModelOp.getIsSubgraph(),
                           /*isDeviceGraph=*/inModelOp.getIsDeviceGraph(),
                           /*theExtraAttrs=*/inModelOp.getTheExtraAttrsAttr());
  LLVM_DEBUG(llvm::dbgs() << "init\n");
  fixup(builder, inModelOp.getInitBlock(),
        /*freeValues=*/{}, &outModelOp.getInit().emplaceBlock(),
        /*mustAlsoReturn=*/false, /*beforeLastOutput=*/false);
  LLVM_DEBUG(llvm::dbgs() << "execute\n");
  fixup(builder, inModelOp.getExecuteBlock(),
        /*freeValues=*/{}, &outModelOp.getExecute().emplaceBlock(),
        /*mustAlsoReturn=*/false, /*beforeLastOutput=*/false);
  return outModelOp;
}
