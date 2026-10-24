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

#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/StringSet.h"

namespace M::MOGG {

FusionLeafAttr createFusionLeaf(Operation *op) {
  std::string name;
  if (auto shapeFuncOp = dyn_cast<MO::InvokeShapeFuncOp>(op)) {
    name = "shapeof(";
    name += shapeFuncOp.getShapeFuncName();
    name += ")";
  } else if (auto layoutTransformOp = dyn_cast<MO::LayoutTransformOp>(op)) {
    name = layoutTransformOp->getName().getIdentifier().getValue();
    name += "_";
    name += layoutTransformOp.getInput()
                .getType()
                .getLayoutAttr()
                .getFormat()
                .getValue();
    name += "_to_";
    name += layoutTransformOp.getResult()
                .getType()
                .getLayoutAttr()
                .getFormat()
                .getValue();
  } else if (auto customOp = dyn_cast<MO::CustomOp>(op)) {
    name = "custom__";
    name += customOp.getSymbol();
  } else {
    name = op->getName().getIdentifier().getValue();
  }

  auto ctx = op->getContext();
  LocationAttr loc = op->getLoc();

  TypeAttr signature;
  auto funcType =
      FunctionType::get(ctx, op->getOperandTypes(), op->getResultTypes());
  signature = TypeAttr::get(funcType);

  return FusionLeafAttr::get(ctx, StringAttr::get(ctx, name), loc, signature);
}

bool dependsOnNoAliasConstantLikesOnly(mlir::Operation *operation) {
  if (!operation)
    return false;
  // An op with a block-arg operand fails the walk: it depends on a runtime
  // value, including a mixed op that also takes a constant.
  return !walkDataflowGraph<WalkDirection::Producers>(
              operation,
              [](mlir::Operation *op) -> mlir::WalkResult {
                if (op->getNumOperands() == 0) {
                  if (!isa<MO::ConstantLike, MO::ChainCreateOp>(op))
                    return mlir::WalkResult::interrupt();
                  auto extConst = dyn_cast<MO::ConstantExternalOp>(op);
                  return (!extConst || !extConst.getHasAlias())
                             ? mlir::WalkResult::advance()
                             : mlir::WalkResult::interrupt();
                }
                for (mlir::Value operand : op->getOperands())
                  if (!operand.getDefiningOp())
                    return mlir::WalkResult::interrupt();
                return mlir::WalkResult::advance();
              })
              .wasInterrupted();
}

bool isTileCodegenForOp(mlir::Operation *op, mlir::ModuleOp moduleOp) {
  auto kernelStubOp = op->getParentOfType<MOGG::FuncOp>();
  StringRef deviceLabel = kernelStubOp
                              ? kernelStubOp.getDevice()
                              : MO::getDeviceOrDefaultAny(op).getLabel();
  auto deviceRef = M::DeviceRefAttr::get(op->getContext(), deviceLabel, 0);
  return MO::DeviceInfoMapping(moduleOp).get(deviceRef).getTileBasedFusion();
}

} // namespace M::MOGG
