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

#include "GraphCompiler/Interfaces/MAPToMOGGState.h"
#include "GraphCompiler/Interfaces/LowerableToMOGGInterface.h"

using namespace mlir;

namespace M::MAP {

LogicalResult MAPToMOGGState::lowerOp(Operation *op) {
  auto lowerable = dyn_cast<LowerableToMOGGInterface>(op);
  if (!lowerable)
    return op->emitOpError("MAPToMOGG: unsupported op");
  return lowerable.lowerToMOGG(*this);
}

LogicalResult MAPToMOGGState::lowerRegionBody(
    Block &block, llvm::function_ref<bool(Operation *)> shouldLower) {
  for (Operation &op : block) {
    if (!shouldLower(&op))
      continue;
    if (failed(lowerOp(&op)))
      return failure();
  }
  return success();
}

} // namespace M::MAP
