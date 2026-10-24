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

#include "GraphCompiler/MODialect/Transforms/ShapeInferencePassesCommon.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace MO;

namespace M::MO::ShapeInferencePassCommon {

//===----------------------------------------------------------------------===//
// Common Patterns
//===----------------------------------------------------------------------===//

LogicalResult FoldConsecutiveRebinds::deviceAwareMatchAndRewrite(
    RebindOp op, DeviceAwareRewriter &rewriter) const {
  Location loc = op->getLoc();
  auto producerOp = op.getInput().getDefiningOp<RebindOp>();
  if (!producerOp) {
    return rewriter.notifyMatchFailure(loc, "not preceded by other mo.rebind");
  }

  rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), producerOp.getInput(),
                                        op.getOutputParamDecls());
  return success();
}

} // namespace M::MO::ShapeInferencePassCommon
