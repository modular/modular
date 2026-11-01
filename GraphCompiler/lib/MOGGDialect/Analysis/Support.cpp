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

#include "GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Mojo/KGENDialect/KGENOps.h"

namespace M::MOGG {

/// Returns the name of the given `op` is registered under in mojo.
MojoOperationInfo getMojoOperationInfo(Operation *op) {
  // Custom operations call the symbol and function directly.
  if (auto customOp = dyn_cast<MO::CustomOp>(op))
    return {customOp.getSymbol()};

  // Layout transforms map to a hardcoded op name based on the inferred layouts
  if (auto layoutTransformOp = dyn_cast<MO::LayoutTransformOp>(op)) {
    auto startLayout = layoutTransformOp.getInput().getType().getLayoutAttr();
    auto endLayout = layoutTransformOp.getResult().getType().getLayoutAttr();

    StringAttr opName = StringAttr::get(
        op->getContext(), llvm::Twine("layout_transform_") +
                              startLayout.getFormat().strref() + "_to_" +
                              endLayout.getFormat().strref());
    return {opName};
  }

  if (auto shapeInvoke = dyn_cast<MO::InvokeShapeFuncOp>(op))
    return {shapeInvoke.getShapeFuncName()};

  // Else we register 1-1 with the official name of the operation.
  return {op->getName().getStringRef()};
}

ErrorOr<KernelDeclAdaptor> getMojoKernelAdaptor(const MojoRegistry &registry,
                                                Operation *op) {
  MojoOperationInfo info = getMojoOperationInfo(op);

  if (!registry.containsKernel(info.operationName)) {
    return Error(
        llvm::formatv("no kernel registered for '{}'", info.operationName)
            .str());
  }
  DeviceRefAttr device = MojoRegistry::getDeviceRefForOp(op);
  auto declOpOr = registry.getResolvedKernel(info.operationName, device);
  if (declOpOr.isError())
    return declOpOr.takeError();
  KernelDeclAdaptor adaptor(*declOpOr);

  if (!adaptor.executeFunction.has_value() &&
      !adaptor.elementwiseFunction.has_value() && !adaptor.isReduction()) {
    return Error(
        llvm::formatv(
            "Could not find an execute or elementwise function for '{}'",
            info.operationName)
            .str());
  }

  return adaptor;
}

} // namespace M::MOGG
