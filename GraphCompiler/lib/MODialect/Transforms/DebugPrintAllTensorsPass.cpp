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

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Pass/Pass.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_DEBUGPRINTALLTENSORSPASS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

struct DebugPrintAllTensorsPass
    : public impl::DebugPrintAllTensorsPassBase<DebugPrintAllTensorsPass> {
  using DebugPrintAllTensorsPassBase::DebugPrintAllTensorsPassBase;

  std::string valueToLabel(Value value) {
    std::string str;
    llvm::raw_string_ostream os(str);

    if (auto op = value.getDefiningOp())
      os << " " << op->getName() << " ";

    value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());

    return str;
  }

  /// Insert mo.debug.tensor.prints into block.
  void process(OpBuilder &builder, Location loc, Block *block) {
    mlir::sortTopologically(block);

    builder.setInsertionPointToStart(block);

    Value currentChain = MO::ChainCreateOp::create(builder, loc, ValueRange{});

    for (Value arg : block->getArguments()) {
      if (isa<MO::TensorType>(arg.getType())) {
        currentChain = MO::DebugTensorPrintOp::create(
            builder, loc, /*inChain=*/currentChain, /*input=*/arg,
            /*label=*/valueToLabel(arg));
      }
    }

    for (Operation &op : block->getOperations()) {
      for (Region &subRegion : op.getRegions())
        process(builder, op.getLoc(), &subRegion.front());

      builder.setInsertionPointAfter(&op);
      for (Value res : op.getResults()) {
        if (isa<MO::TensorType>(res.getType())) {
          currentChain = MO::DebugTensorPrintOp::create(
              builder, op.getLoc(), /*inChain=*/currentChain, /*input=*/res,
              /*label=*/valueToLabel(res));
        }
      }
    }
  }

  bool hasDebugPrints(GraphOp graphOp) {
    bool hasDebugPrint = false;

    graphOp.walk([&hasDebugPrint](DebugTensorPrintOp printOp) {
      hasDebugPrint = true;
      return WalkResult::interrupt();
    });

    return hasDebugPrint;
  }

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    if (hasDebugPrints(graphOp)) {
      graphOp->emitError(
          "debug-print-all-tensors should only be invoked on graphs that don't "
          "already have mo.debug.tensor.print ops");
      return signalPassFailure();
    }
    OpBuilder builder(graphOp->getContext());
    process(builder, graphOp.getLoc(), graphOp.getBodyBlock());
  }
};

} // namespace
