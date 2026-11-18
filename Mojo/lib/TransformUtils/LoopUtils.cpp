//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/TransformUtils/LoopUtils.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace M;
using namespace HLCF;
using namespace KGEN;

bool KGEN::isDroppableOnExitIteration(Operation *op) {
  // A debug record describes the iteration the raised loop no longer runs, and
  // a `-g` range iterator body is full of them.
  if (isa_and_nonnull<DebugInfo::DebugInfoDialect>(op->getDialect()))
    return true;

  // `hlcf.if` declares no memory effects either way, so judge it by what it
  // runs; a region that leaves the loop is not droppable.
  if (auto ifOp = dyn_cast<IfOp>(op)) {
    for (Region &region : ifOp->getRegions()) {
      for (Block &block : region) {
        if (!isa<YieldOp>(block.getTerminator()))
          return false;
        for (Operation &nested : block.without_terminator()) {
          if (!KGEN::isDroppableOnExitIteration(&nested))
            return false;
        }
      }
    }
    return true;
  }

  return mlir::isMemoryEffectFree(op);
}
