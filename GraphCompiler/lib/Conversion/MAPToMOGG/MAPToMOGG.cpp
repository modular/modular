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
//
// Lowers `map` dialect kernels into `mogg` codegen ops. Each `map` op that can
// appear in a kernel implements `LowerableToMOGGInterface` (see
// `MAPOpsLowerableToMOGG.cpp`); `MAPToMOGGState` walks each `map.iter.opaque`
// and dispatches on whatever op it finds.
//
// A `map.iter.opaque` becomes a `mogg.routine` whose body rebuilds each tensor
// from a `!mogg.pointer` block argument via `mogg.shape.create` /
// `mogg.tensor.create`. A nested `map.call.opaque` becomes a
// `mogg.call.opaque`, a `map.iter.elementwise.body` a `mogg.call.foreach`, and
// a `map.iter.epilogue` a `mogg.tensor.create.epilogue` whose body is lowered
// by the same dispatcher.
//
// The kernel gains one `!mogg.dim` block argument per unique dynamic dimension
// (deduped across inputs, outputs, and nested shape attributes), placed
// between the pointers and the context.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h"
#include "GraphCompiler/Interfaces/MAPToMOGGState.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_MAPTOMOGG
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h.inc"

namespace {

struct MAPToMOGGPass : public MAP::impl::MAPToMOGGBase<MAPToMOGGPass> {
  using Base::Base;

  void runOnOperation() override {
    MOGG::MojoRegistry registry(getOperation());

    SmallVector<MAP::IterOpaqueOp> worklist;
    getOperation()->walk([&](MAP::IterOpaqueOp op) {
      worklist.push_back(op);
      return WalkResult::skip();
    });

    for (MAP::IterOpaqueOp op : worklist) {
      MAPToMOGGState state(registry, op);
      state.enableComputeLambdas = enableComputeLambdas;
      if (failed(state.lowerOp(op.getOperation())))
        return signalPassFailure();
    }

    // Post-order: the lowering erases the op.
    WalkResult result = getOperation()->walk([&](MO::InvokeShapeFuncOp op) {
      MAPToMOGGState state(registry, op);
      state.enableComputeLambdas = enableComputeLambdas;
      return failed(lowerInvokeShapeFuncToMOGG(op, state))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

} // namespace M::MAP
