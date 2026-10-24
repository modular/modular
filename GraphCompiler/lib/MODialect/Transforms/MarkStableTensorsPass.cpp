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

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "mo-mark-stable-tensors"

namespace M::MO {

#define GEN_PASS_DEF_MARKSTABLETENSORSPASS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace MO;

namespace {

//===----------------------------------------------------------------------===//
// MarkStableTensorsPass
//===----------------------------------------------------------------------===//

struct MarkStableTensorsPass final
    : public impl::MarkStableTensorsPassBase<MarkStableTensorsPass> {
  using Base::Base;

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    if (!graphOp.getIsDeviceGraph())
      return;

    auto moduleOp = graphOp->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      graphOp.emitError()
          << "device-graphable mo.graph has no parent ModuleOp; cannot "
             "look up kernel registrations";
      return signalPassFailure();
    }

    MOGG::MojoRegistry registry(moduleOp);

    // For `mo.custom`, the kernel registration is named by the `symbol`
    // attribute; for built-in MO ops, the op name itself is the
    // registration string.
    auto getRegistrationName = [](Operation *op) {
      if (auto custom = dyn_cast<CustomOp>(op))
        return custom.getSymbol();
      return op->getName().getStringRef();
    };

    WalkResult result =
        graphOp.walk([&](Operation *op) {
          if (op->hasTrait<OpTrait::IsTerminator>())
            return WalkResult::advance();

          StringRef registration = getRegistrationName(op);

          if (!registry.containsKernel(registration))
            return WalkResult::advance();

          DeviceRefAttr device = MOGG::MojoRegistry::getDeviceRefForOp(op);
          auto kernelOr = registry.getResolvedKernel(registration, device);
          if (kernelOr.isError()) {
            op->emitError(kernelOr.getError());
            return WalkResult::interrupt();
          }

          MOGG::KernelDeclAdaptor adaptor(*kernelOr);
          if (!adaptor.buildGraphFunction)
            return WalkResult::advance();

          const MOGG::KernelDeclFunctionAdaptor &buildGraph =
              *adaptor.buildGraphFunction;
          if (llvm::none_of(buildGraph.inputArguments,
                            [](const MOGG::KernelDeclOperandAdaptor &arg) {
                              return arg.isStableTensorType();
                            }))
            return WalkResult::advance();

          auto mappingOr = MOGG::mapMOOpToKernelImplementation(op, buildGraph);
          if (mappingOr.isError()) {
            op->emitError()
                << "cannot map operands onto the `build_graph` "
                   "function of kernel '"
                << kernelOr->getSymName() << "': " << mappingOr.getError();
            return WalkResult::interrupt();
          }

          for (const auto &[index, arg] : mappingOr->operandMapping) {
            if (!arg.isStableTensorType())
              continue;

            OpOperand &operand = op->getOpOperand(index);

            auto tensor = dyn_cast<TensorValue>(operand.get());
            if (!tensor) {
              op->emitError()
                  << "operand at position " << index
                  << " feeds a stable tensor argument of the `build_graph` "
                     "function of kernel '"
                  << kernelOr->getSymName() << "' but is not a tensor";
              return WalkResult::interrupt();
            }

            if (tensor.getType().isStable())
              continue;

            OpBuilder builder(op);
            auto stable = StableCreateOp::create(builder, op->getLoc(), tensor);
            operand.assign(stable);
          }

          return WalkResult::advance();
        });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
