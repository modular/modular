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

#include "GraphCompiler/Conversion/RMOSupport/Broadcast.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"
#include <numeric>

using namespace M;
using namespace M::MO;

class BroadcastTest : public ::testing::Test {
public:
  BroadcastTest() : Test(), builder(&ctx), loc(UnknownLoc::get(&ctx)) {}

protected:
  void SetUp() override {
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
    loc.getContext()->allowUnregisteredDialects(true);
    loc.getContext()->loadDialect<MDialect>();
    loc.getContext()->loadDialect<MODialect>();
    loc.getContext()->loadDialect<RMO::RMODialect>();
  }

  void TearDown() override { module->erase(); }

  MO::TensorValue getTestValue(ShapeAttr shape, DeviceRefAttr deviceRef = {}) {
    auto type = MO::TensorType::get(shape, Float32Type::get(&ctx), deviceRef);
    SmallVector<APFloat> values(1, APFloat(0.f));
    MO::TensorValue out = ConstantOp::create(
        builder, loc, ArrayRef<APFloat>(values), Float32Type::get(&ctx));
    // Overwrite the output type
    out.setType(type);
    return out;
  }

  MO::TensorValue getTestValue(ArrayRef<int64_t> shape,
                               DeviceRefAttr deviceRef = {}) {
    auto shapeTy = builder.getType<ShapeType>();
    auto shapeAttr = ShapeAttr::get(shape, shapeTy);
    return getTestValue(shapeAttr, deviceRef);
  }

  MO::TensorValue getTestValue(ArrayRef<TypedAttr> shape,
                               DeviceRefAttr deviceRef = {}) {
    auto shapeTy = builder.getType<ShapeType>();
    auto shapeAttr = builder.getAttr<ShapeAttr>(shape, shapeTy);
    return getTestValue(shapeAttr, deviceRef);
  }

  TypedAttr getParametricDim(const std::string &name) {
    auto idxTy = ShapeAttr::getDimType(builder.getContext());
    TypedAttr typeAttr = KGEN::ParamDeclRefAttr::get(name, idxTy);
    return typeAttr;
  }

  TypedAttr getIntegerDim(int64_t v) {
    return MOSH::ShapeAttr::getDim(builder.getContext(), v);
  }

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ModuleOp module;
  OpBuilder builder;
  Location loc;
};

TEST_F(BroadcastTest, Failure) {
  // TODO(GEX-715): Audit asserts on production builds.
#ifndef NDEBUG
  MO::TensorValue x = getTestValue({1, 3});
  MO::TensorValue y = getTestValue({1, 1});
  EXPECT_DEATH(
      RMO::broadcastTo(builder, loc, x, y),
      "don't use this helper unless you can guarantee x broadcasts to y");
#endif
}

TEST_F(BroadcastTest, Basic) {
  MO::TensorValue x = getTestValue({1, 1});
  MO::TensorValue y = getTestValue({1, 3});
  MO::TensorValue bcastX = RMO::broadcastTo(builder, loc, x, y);
  SmallVector<int64_t> refDims{1ll, 3ll};
  ASSERT_EQ(bcastX.getType().getIntDims(), refDims);
}

TEST_F(BroadcastTest, Multi) {
  MO::TensorValue x = getTestValue({5, 1, 2});
  MO::TensorValue y = getTestValue({1, 1, 3, 1});
  auto [bcastX, bcastY] = RMO::reifyImplicitBroadcasting(builder, loc, x, y);
  SmallVector<int64_t> refDims{1ll, 5ll, 3ll, 2ll};
  ASSERT_EQ(bcastX.getType().getIntDims(), refDims);
  ASSERT_EQ(bcastY.getType().getIntDims(), refDims);
}

TEST_F(BroadcastTest, ReshapeAndBroadcast) {
  auto deviceRef = DeviceRefAttr::getDefault(&ctx);
  MO::TensorValue x = getTestValue({5}, deviceRef);
  MO::TensorValue y = getTestValue({3, 5, 1}, deviceRef);

  auto [bcastX1, bcastY1] = RMO::reifyImplicitBroadcasting(builder, loc, x, y);

  SmallVector<int64_t> refDims1{3, 5, 5};
  SmallVector<int64_t> refDims2{3, 5, 1};

  TensorType x1Type = bcastX1.getType();
  TensorType y1Type = bcastY1.getType();
  ASSERT_EQ(x1Type.getIntDims(), refDims1);
  ASSERT_EQ(y1Type.getIntDims(), refDims1);
  ASSERT_EQ(x1Type.getDeviceRef(), deviceRef);
  ASSERT_EQ(y1Type.getDeviceRef(), deviceRef);

  auto bcastX2 = RMO::reshapeAndBroadcastTo(builder, loc, x,
                                            SmallVector<int64_t>{1, 5, 1}, y);

  ASSERT_EQ(bcastX2.getType().getIntDims(), refDims2);
  ASSERT_EQ(bcastX2.getType().getDeviceRef(), deviceRef);

  // Also test that the devices are correctly set within the pattern.
  for (Type opType : bcastX2.getDefiningOp()->getOperandTypes()) {
    llvm::errs() << bcastX2.getDefiningOp();
    ASSERT_EQ(cast<TensorType>(opType).getDeviceRef(), deviceRef);
  }
}
