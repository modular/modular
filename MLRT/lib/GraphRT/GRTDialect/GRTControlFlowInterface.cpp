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

#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/Compiler/VerifyUtils.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

mlir::LogicalResult M::GRT::verifyControlFlowInterfaceOp(Operation *op) {
  auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op);

  switch (cfOp.getFlavor()) {
  case kLoopLike:
    if (cfOp->getNumRegions() < 1 || cfOp->getNumRegions() > 2)
      return op->emitOpError("expects one or two sub-regions");
    break;
  case kBranchLike:
    if (cfOp->getNumRegions() == 0)
      return op->emitOpError("expects at least one sub-region");
    break;
  }

  if (cfOp.getFlavor() == kLoopLike) {
    // The op's results must match the op's inputs.
    if (failed(checkMatchingTypes(op, "result", op->getResultTypes(),
                                  cfOp.getInputs().getTypes())))
      return failure();
  }

  // All the nested blocks must accept arguments matching the op's inputs.
  SmallVector<StringRef> regionNames = cfOp.getRegionNames();
  assert(regionNames.size() == cfOp->getNumRegions());
  for (size_t i = 0, n = cfOp->getNumRegions(); i < n; ++i) {
    Block *block = &cfOp->getRegion(i).front();
    if (failed(checkArgumentTypes(op, regionNames[i], block,
                                  cfOp.getInputs().getTypes())))
      return failure();
  }

  // All the nested blocks must return results matching the op's results.
  for (size_t i = 0, n = cfOp->getNumRegions(); i < n; ++i) {
    Block *block = &cfOp->getRegion(i).front();
    auto outputOp = cast<M::MEF::OutputOp>(block->getTerminator());
    SmallVector<Type> outputTypes =
        llvm::to_vector(outputOp.getOperands().getTypes());
    if (cfOp.getFlavor() == kLoopLike) {
      // Ok for nested blocks to return additional results, eg for
      // conditions.
      if (outputTypes.size() > op->getNumResults())
        outputTypes.erase(outputTypes.begin() + op->getNumResults(),
                          outputTypes.end());
    }
    if (failed(checkMatchingTypes(
            op,
            (Twine("mef.output operands in '") + regionNames[i] + "' block")
                .str(),
            outputTypes, op->getResultTypes())))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GRTControlFlowInterface
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.cpp.inc"
