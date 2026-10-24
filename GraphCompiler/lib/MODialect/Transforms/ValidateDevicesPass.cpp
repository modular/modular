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

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"

#include "GraphCompiler/MODialect/Transforms/Utils.h"

#define DEBUG_TYPE "validate-devices"

namespace M::MO {

#define GEN_PASS_DEF_VALIDATEDEVICESPASS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

}; // namespace M::MO

using namespace M;
using namespace MO;

namespace {

//===----------------------------------------------------------------------===//
// ValidateDevicesPass
//===----------------------------------------------------------------------===//
//
// Verifies every input and result tensor dtype is valid for the device
// architecture. Reports only the first invalid combination.

struct ValidateDevicesPass
    : public impl::ValidateDevicesPassBase<ValidateDevicesPass> {
  using ValidateDevicesPassBase::ValidateDevicesPassBase;

  bool isSupportedArchDTypeCombination(llvm::Triple::ArchType arch,
                                       DType dtype) {
    // CPUs don't support sub-32-bit floats, but bf16 (x86_64, aarch64) and f16
    // (x86_64) are permitted for casting.
    if ((arch == llvm::Triple::ArchType::x86_64 ||
         arch == llvm::Triple::ArchType::aarch64) &&
        dtype == DType::bf16)
      return true;
    if (arch == llvm::Triple::ArchType::x86_64 && dtype == DType::f16)
      return true;
    if (llvm::is_contained(
            {llvm::Triple::ArchType::x86_64, llvm::Triple::ArchType::aarch64},
            arch) &&
        dtype.isFloat() && dtype.getWidthInBits() < 32) {
      return false;
    }

    return true;
  }

  static std::string formatDeviceForUser(DeviceRefAttr device) {
    DeviceRef deviceRef;
    deviceRef.id = device.getId();
    deviceRef.label = device.getLabel();
    return deviceRef.toString();
  }

  static void emitErrorOnOp(Operation *op, DeviceRefAttr device,
                            StringRef dtype) {
    op->emitError() << "The " << dtype
                    << " data type is not supported on device '"
                    << formatDeviceForUser(device) << "'.";
  }

  LogicalResult process(llvm::Triple::ArchType targetArch, Block *block) {
    for (Value arg : block->getArguments()) {
      if (isa<MO::TensorType>(arg.getType())) {
        TensorType tensorType = dynCastFromTensorLike(arg.getType());
        LLVM_DEBUG(llvm::dbgs() << "Checking input '" << arg << "'\n");
        if (!isSupportedArchDTypeCombination(targetArch,
                                             tensorType.getDtype())) {
          emitErrorOnOp(block->getParentOp(), tensorType.getDeviceRef(),
                        tensorType.getDtype().getAsString());
          return failure();
        }
      }
    }

    WalkResult result = graphOp->walk([&](Operation *op) {
      for (Value res : op->getResults()) {
        if (isa<MO::TensorType>(res.getType())) {
          TensorType tensorType = dynCastFromTensorLike(res.getType());
          LLVM_DEBUG(llvm::dbgs() << "Checking result '" << res << "'\n");
          if (!isSupportedArchDTypeCombination(targetArch,
                                               tensorType.getDtype())) {
            emitErrorOnOp(op, tensorType.getDeviceRef(),
                          tensorType.getDtype().getAsString());
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return failure();

    return success();
  }

  void runOnOperation() override {
    graphOp = getOperation();

    LLVM_DEBUG({
      llvm::dbgs() << "defaultDeviceTargetTriple: '"
                   << defaultDeviceTargetTriple << "'\n";
      llvm::dbgs() << "defaultDeviceTargetArch: '" << defaultDeviceTargetArch
                   << "'\n";
    });

    if (!defaultDeviceTargetTriple.empty() &&
        !defaultDeviceTargetArch.empty()) {
      TargetInfo targetInfo{llvm::Triple(defaultDeviceTargetTriple),
                            defaultDeviceTargetArch};
      graphOp.setDeviceTargetInfo(
          fromRuntimeTargetInfo(graphOp.getContext(), targetInfo));
    }

    TargetInfoAttr targetInfo = graphOp.getDeviceTargetInfo();
    if (!targetInfo) {
      LLVM_DEBUG({
        llvm::dbgs() << "The graph op has no target information, "
                        "ValidateDevicesPass will do nothing.\n";
      });
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Running ValidateDevicesPass for target '"
                   << targetInfo.getTriple().getTriple() << "'\n";
    });

    llvm::Triple::ArchType targetArch = targetInfo.getTriple().getArch();
    if (failed(process(targetArch, graphOp.getBodyBlock()))) {
      return signalPassFailure();
    }
  }

  GraphOp graphOp;
};

} // namespace
