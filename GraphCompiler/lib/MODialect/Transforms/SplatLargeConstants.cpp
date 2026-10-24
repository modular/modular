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
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"

namespace M::MO {

#define GEN_PASS_DEF_SPLATLARGECONSTANTS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

struct SplatLargeConstants
    : public impl::SplatLargeConstantsBase<SplatLargeConstants> {

  SplatLargeConstants() : SplatLargeConstantsBase() {}

  SplatLargeConstants(const SplatLargeConstantsOptions &opts)
      : SplatLargeConstantsBase(opts) {}

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    MLIRContext *ctx = graphOp->getContext();
    OpBuilder builder{graphOp.getContext()};
    auto shapeType = ShapeType::get(ctx);

    graphOp.walk([&](ConstantOp constantOp) {
      auto constantValue = constantOp.getValue();
      if (constantValue.size() < numElements)
        return;

      // Scalar constants have nothing to broadcast.
      if (constantValue.size() == 1)
        return;

      builder.setInsertionPoint(constantOp);
      Location loc = constantOp->getLoc();
      auto elementType = constantValue.getElementType();
      auto constantOpType = constantOp.getType();

      int bitwidth = elementType.getIntOrFloatBitWidth();
      int bytewidth = bitwidth / 8;
      SmallVector<uint8_t> zeroValue(bytewidth, 0);
      auto newArrayType = ArrayType::get(1, elementType);
      auto splatValue = M::ArrayElementsAttr::get(zeroValue, newArrayType);
      auto newConstantType =
          constantOpType.cloneWith(ShapeAttr::get({1}, shapeType));
      auto newConstantOp =
          ConstantOp::create(builder, loc, newConstantType, splatValue);
      auto broadcastOp = Static::BroadcastToOp::create(
          builder, loc, constantOpType, newConstantOp);
      constantOp->replaceAllUsesWith(broadcastOp);
      constantOp->erase();
    });
  }
};

} // namespace M::MO
