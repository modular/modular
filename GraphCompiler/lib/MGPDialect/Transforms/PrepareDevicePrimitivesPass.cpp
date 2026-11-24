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

#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

#define DEBUG_TYPE "mgp-prepare-device-primitives"

namespace M::MGP {

#define GEN_PASS_DEF_PREPAREDEVICEPRIMITIVESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// DevicePrimitivesPass
//===----------------------------------------------------------------------===//

class PrepareDevicePrimitivesPass
    : public impl::PrepareDevicePrimitivesPassBase<
          PrepareDevicePrimitivesPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    OpBuilder builder(op);

    op->walk([&](Operation *origOp) {
      if (auto hasDeviceOp = dyn_cast<MGPHasDeviceOpInterface>(origOp)) {
        assert(!origOp->hasSuccessors());
        assert(origOp->getNumRegions() == 0);
        builder.setInsertionPoint(origOp);
        std::string newOpName = (Twine(origOp->getName().getStringRef()) + "." +
                                 hasDeviceOp.getDevice().getLabel())
                                    .str();
        LLVM_DEBUG(llvm::dbgs() << "renaming " << origOp->getName() << " to "
                                << newOpName << "\n");

        NamedAttrList newAttrs;
        for (auto namedAttr : origOp->getAttrDictionary()) {
          if (auto deviceRefAttr =
                  dyn_cast<DeviceRefAttr>(namedAttr.getValue())) {
            std::string newAttrName =
                (Twine(namedAttr.getName().getValue()) + "Id").str();
            LLVM_DEBUG(llvm::dbgs() << "renaming " << namedAttr.getName()
                                    << " to " << newAttrName << "\n");
            Type ui64 = builder.getType<IntegerType>(64, IntegerType::Unsigned);
            APInt apint(ui64.getIntOrFloatBitWidth(), deviceRefAttr.getId(),
                        ui64.isSignedInteger());
            newAttrs.set(newAttrName, IntegerAttr::get(ui64, apint));
          } else {
            newAttrs.set(namedAttr.getName(), namedAttr.getValue());
          }
        }

        Operation *newOp = Operation::create(
            /*location=*/origOp->getLoc(),
            /*name=*/OperationName(newOpName, origOp->getContext()),
            /*resultTypes=*/origOp->getResults().getTypes(),
            /*operands=*/origOp->getOperands(),
            /*attributes=*/std::move(newAttrs),
            /*properties=*/PropertyRef{},
            /*successors=*/{},
            /*numRegions=*/0);
        builder.setInsertionPoint(origOp);
        builder.insert(newOp);
        origOp->replaceAllUsesWith(newOp->getResults());
        origOp->erase();
      }
    });
  }
};

} // namespace
} // namespace M::MGP
