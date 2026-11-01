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

#include "GraphCompiler/Matchers.h"

#include "GraphCompiler/Conversion/RMOSupport/Broadcast.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"
#include <numeric>

using namespace M;
using namespace M::MO;
using namespace mlir;
using namespace GraphCompiler::Matchers;

class MatchersTest : public ::testing::Test {
public:
  MatchersTest() : Test(), builder(&ctx), loc(UnknownLoc::get(&ctx)) {}

protected:
  void SetUp() override {
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
    loc.getContext()->allowUnregisteredDialects(true);
    loc.getContext()->loadDialect<MDialect>();
    loc.getContext()->loadDialect<MODialect>();
    loc.getContext()->loadDialect<MOGG::MOGGDialect>();

    SmallVector<Type> inputTypes{
        MO::TensorType::get({1}, Float32Type::get(&ctx)),
        MO::TensorType::get({1, 2}, Float32Type::get(&ctx)),
        MO::TensorType::get({1, 2, 3}, Float32Type::get(&ctx))};
    graph = MO::GraphOp::create(builder, module.getLoc(), "test", inputTypes,
                                SmallVector<Type>{});
    builder.setInsertionPointToEnd(graph.getBodyBlock());
    Value in0 = graph.getBody().front().getArgument(0);
    Value in1 = graph.getBody().front().getArgument(1);
    Value in2 = graph.getBody().front().getArgument(2);
    Value shape1 = MO::ShapeOfOp::create(builder, graph.getLoc(),
                                         cast<MO::TensorValue>(in1));
    Value shape2 = MO::ShapeOfOp::create(builder, graph.getLoc(),
                                         cast<MO::TensorValue>(in2));
    A = MO::BroadcastToOp::create(builder, graph.getLoc(), in1.getType(), in0,
                                  shape1);
    B = MO::AddOp::create(builder, graph.getLoc(), A, in1);
    C = MO::BroadcastToOp::create(builder, graph.getLoc(), B.getType(), B,
                                  shape2);
    D = MO::SubOp::create(builder, graph.getLoc(), C, in2);
  }

  void TearDown() override { module->erase(); }

  // A non-type template parameter cannot have type 'float' before
  // C++20clang(template_nontype_parm_bad_structural_type)
  template <int value = 0>
  MO::TensorValue getFloatConst(ArrayRef<int64_t> shape) {
    auto type = MO::TensorType::get(shape, Float32Type::get(&ctx));
    SmallVector<APFloat> values(1, APFloat(float(value)));
    MO::TensorValue out = ConstantOp::create(
        builder, loc, ArrayRef<APFloat>(values), Float32Type::get(&ctx));
    // Overwrite the output type
    out.setType(type);
    return out;
  }

  MO::TensorValue getIntConst(ArrayRef<int64_t> shape, int64_t bitwidth = 64) {
    auto iType =
        IntegerType::get(loc.getContext(), bitwidth, IntegerType::Signed);
    auto type = MO::TensorType::get(shape, iType);
    SmallVector<int64_t> values{1};
    MO::TensorValue out =
        ConstantOp::create(builder, loc, ArrayRef<int64_t>(values));
    // Overwrite the output type
    out.setType(type);
    return out;
  }

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ModuleOp module;
  GraphOp graph;
  OpBuilder builder;
  Location loc;
  TensorValue A, B, C, D;
};

TEST_F(MatchersTest, Constants) {
  MO::TensorValue x = getIntConst({});
  MO::TensorValue x32 = getIntConst({}, 32);
  MO::TensorValue y = getFloatConst({});
  MO::TensorValue i = getIntConst({1});
  MO::TensorValue j = getFloatConst({1});
  Value c;

  auto match = m_ScalarConstant(1);
  auto floatMatch = m_ScalarConstant(0.0);
  auto intZero = m_ScalarConstant(0);
  auto capture = m_ScalarConstant(&c, 1);

  ASSERT_FALSE(match.match(y));
  ASSERT_FALSE(match.match(i));
  ASSERT_FALSE(match.match(j));
  ASSERT_TRUE(match.match(x));
  ASSERT_TRUE(match.match(x32));
  ASSERT_FALSE(floatMatch.match(x));
  ASSERT_TRUE(floatMatch.match(y));
  ASSERT_FALSE(intZero.match(x));
  ASSERT_TRUE(intZero.match(y));

  ASSERT_FALSE(capture.match(y));
  ASSERT_FALSE(capture.match(i));
  ASSERT_FALSE(capture.match(j));
  ASSERT_NE(c, x);
  ASSERT_TRUE(capture.match(x));
  ASSERT_EQ(c, x);
  ASSERT_NE(c, x32);
  ASSERT_TRUE(capture.match(x32));
  ASSERT_EQ(c, x32);
}

TEST_F(MatchersTest, Ranked) {
  Value ranked;
  auto pattern = m_Ranked(2, m_Any(&ranked));

  ASSERT_FALSE(pattern.match(graph.getBody().front().getArgument(0)));
  ASSERT_NE(graph.getBody().front().getArgument(1), ranked);
  ASSERT_TRUE(pattern.match(graph.getBody().front().getArgument(1)));
  ASSERT_EQ(graph.getBody().front().getArgument(1), ranked);
  ASSERT_FALSE(pattern.match(graph.getBody().front().getArgument(2)));
}

TEST_F(MatchersTest, MOVector_Matrix) {
  Value v, m;
  auto vector = m_MOVector(m_Any(&v));
  auto matrix = m_MOMatrix(m_Any(&m));

  ASSERT_FALSE(matrix.match(graph.getBody().front().getArgument(0)));
  ASSERT_NE(graph.getBody().front().getArgument(1), m);
  ASSERT_TRUE(matrix.match(graph.getBody().front().getArgument(1)));
  ASSERT_EQ(graph.getBody().front().getArgument(1), m);
  ASSERT_FALSE(matrix.match(graph.getBody().front().getArgument(2)));

  ASSERT_NE(graph.getBody().front().getArgument(0), v);
  ASSERT_TRUE(vector.match(graph.getBody().front().getArgument(0)));
  ASSERT_FALSE(vector.match(graph.getBody().front().getArgument(1)));
  ASSERT_FALSE(vector.match(graph.getBody().front().getArgument(2)));
}

TEST_F(MatchersTest, Optional) {
  Value ranked;
  auto any = m_Any(&ranked);
  auto pattern = m_Optional(m_Ranked(2, any), m_Ranked(1, any));

  ASSERT_NE(graph.getBody().front().getArgument(0), ranked);
  ASSERT_TRUE(pattern.match(graph.getBody().front().getArgument(0)));
  ASSERT_EQ(graph.getBody().front().getArgument(0), ranked);
  ASSERT_TRUE(pattern.match(graph.getBody().front().getArgument(1)));
  ASSERT_EQ(graph.getBody().front().getArgument(1), ranked);
  ASSERT_FALSE(pattern.match(graph.getBody().front().getArgument(2)));
}

TEST_F(MatchersTest, Typed) {
  Value typed;
  auto pattern1 = m_Typed<MO::TensorType>(m_Any(&typed));
  auto pattern2 = m_Typed<RankedTensorType>(m_Any(&typed));

  ASSERT_FALSE(pattern2.match(graph.getBody().front().getArgument(0)));
  ASSERT_NE(graph.getBody().front().getArgument(0), typed);
  ASSERT_TRUE(pattern1.match(graph.getBody().front().getArgument(0)));
  ASSERT_EQ(graph.getBody().front().getArgument(0), typed);
}

TEST_F(MatchersTest, Logging) {
  auto pattern = m_Op<SubOp>(m_Op<SubOp>(m_Any(), m_Any()), m_Any());
  GraphCompiler::Logger log;

  pattern.match(D.getDefiningOp(), &log);

  std::string logMsg;
  llvm::raw_string_ostream ss(logMsg);
  log.print(ss);
  ASSERT_EQ(
      logMsg,
      "Hit a Dead End at OpPattern mo.sub didn't match MLIR OpType! Rewinding "
      "Stack.\n %4 = \"mo.broadcast_to\"(%3, %1) <{outputParamDecls = "
      "#kgen<param.decls[]>}> : (!mo.tensor<[1, 2], f32>, !mo.tensor<[3], "
      "si32>) -> !mo.tensor<[1, 2], f32>\n  %5 = \"mo.sub\"(%4, %arg2) "
      ": "
      "(!mo.tensor<[1, 2], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[1, "
      "2], f32>\n");
}

TEST_F(MatchersTest, SplatOrBroadcastedConstantOfAnyType) {
  MO::TensorValue cst123 = getFloatConst<123>({1, 2, 3});
  auto match123f =
      m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(123.f);
  auto match123i =
      m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(123);
  auto match1234 =
      m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(123.4);

  ASSERT_TRUE(match123f.match(cst123));
  ASSERT_TRUE(match123i.match(cst123));
  ASSERT_FALSE(match1234.match(cst123));
}
