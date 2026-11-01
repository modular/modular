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
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

#define DEBUG_TYPE "mo-verify-device-graph"

namespace M::MO {

#define GEN_PASS_DEF_VERIFYDEVICEGRAPHPASS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace MO;

namespace {

//===----------------------------------------------------------------------===//
// VerifyDeviceGraphPass
//===----------------------------------------------------------------------===//
//
// In an `isDeviceGraph` `mo.graph`, every kernel-backed op must map to a
// `mogg.kernel.decl` with `build_graph` populated. Ops without a kernel
// registration (e.g. `mo.constant`) are skipped.

struct VerifyDeviceGraphPass final
    : public impl::VerifyDeviceGraphPassBase<VerifyDeviceGraphPass> {
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

    graphOp.getBodyBlock()->walk(
        [&](mlir::Operation *op) {
          // Terminators are tagged as control ops but need no validation.
          if (op->hasTrait<OpTrait::IsTerminator>())
            return;

          // Unsupported until CUDA conditional nodes (or equivalent) land.
          if (isa<MOControlOpInterface>(op)) {
            op->emitError()
                << "device-graphable mo.graph cannot contain control-flow op '"
                << op->getName().getStringRef()
                << "' (control flow is not yet supported)";
            signalPassFailure();
            return;
          }

          StringRef registration =
              TypeSwitch<Operation *, StringRef>(op)
                  .Case([](CustomOp custom) { return custom.getSymbol(); })
                  .Default(op->getName().getStringRef());

          if (!registry.containsKernel(registration))
            return;

          DeviceRefAttr device = MOGG::MojoRegistry::getDeviceRefForOp(op);
          auto kernelOr = registry.getResolvedKernel(registration, device);
          if (kernelOr.isError()) {
            op->emitError(kernelOr.getError());
            signalPassFailure();
            return;
          }
          MOGG::KernelDeclOp kernel = *kernelOr;
          if (!kernel.getBuildGraph()) {
            op->emitError()
                << "op '" << op->getName().getStringRef()
                << "' is in a device-graphable mo.graph but its kernel "
                   "registration ('"
                << registration << "', kernel symbol '" << kernel.getSymName()
                << "') does not expose a `build_graph` entry point";
            signalPassFailure();
          }
        });
  }
};

} // namespace
