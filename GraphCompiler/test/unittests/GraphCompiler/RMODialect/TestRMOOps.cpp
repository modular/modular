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
// File for testing helper methods and implicit type builders in `RMOOps.cpp`
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/RMODialect/RMOOps.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "gtest/gtest.h"
#include <limits.h>

using namespace M;
using M::MO::TensorType;
using M::MO::TensorValue;

namespace {

template <typename T>
std::string toString(const T &attr) {
  std::string s;
  llvm::raw_string_ostream stream(s);
  stream << attr;
  return s;
}

TEST(testRMOOpHelpers, testRefineShape) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<RMO::RMODialect, MO::MODialect>();

  auto dimType = ShapeAttr::getDimType(&ctx);

  // Declare dimensions
  auto paramDimAttr =
      KGEN::ParamDeclRefAttr::get(KGEN::ParamDeclAttr::get("D", dimType));
  auto paramExprDimAttr = KGEN::ParamOperatorAttr::get(
      KGEN::POC::Mul, SmallVector<TypedAttr>{paramDimAttr, paramDimAttr});
  auto paramIntegerAttr = ShapeAttr::getDim(&ctx, 42);

  // Take the cross product of `dimTypes` and split into two streams of dims
  SmallVector<TypedAttr> dimTypes = {paramDimAttr, paramExprDimAttr,
                                     paramIntegerAttr};
  SmallVector<TypedAttr> shape1Dims;
  SmallVector<TypedAttr> shape2Dims;
  for (TypedAttr outer : dimTypes) {
    for (TypedAttr inner : dimTypes) {
      shape1Dims.push_back(outer);
      shape2Dims.push_back(inner);
    }
  }

  TensorType shape1Ty = TensorType::get(&ctx, shape1Dims, DType::f32);
  TensorType shape2Ty = TensorType::get(&ctx, shape2Dims, DType::f32);

  // result of refining shape1 with shape 2 and vice versa
  SmallVector<TypedAttr> shape1Shape2Refined = {
      paramDimAttr,     paramDimAttr,     paramIntegerAttr,
      paramExprDimAttr, paramExprDimAttr, paramIntegerAttr,
      paramIntegerAttr, paramIntegerAttr, paramIntegerAttr};
  SmallVector<TypedAttr> shape2Shape1Refined = {
      paramDimAttr,     paramExprDimAttr, paramIntegerAttr,
      paramDimAttr,     paramExprDimAttr, paramIntegerAttr,
      paramIntegerAttr, paramIntegerAttr, paramIntegerAttr};
  TensorType refined1Ty =
      TensorType::get(&ctx, shape1Shape2Refined, DType::f32);
  TensorType refined2Ty =
      TensorType::get(&ctx, shape2Shape1Refined, DType::f32);

  // What the built tensor types look like:
  const char *shape1Str =
      "!mo.tensor<[D, D, D, mul(D, D), mul(D, D), mul(D, D), 42, 42, 42], f32>";
  const char *shape2Str =
      "!mo.tensor<[D, mul(D, D), 42, D, mul(D, D), 42, D, mul(D, D), 42], f32>";
  const char *refined1Str =
      "!mo.tensor<[D, D, 42, mul(D, D), mul(D, D), 42, 42, 42, 42], f32>";
  const char *refined2Str =
      "!mo.tensor<[D, mul(D, D), 42, D, mul(D, D), 42, 42, 42, 42], f32>";
  EXPECT_EQ(toString(shape1Ty), shape1Str);
  EXPECT_EQ(toString(shape2Ty), shape2Str);
  EXPECT_EQ(toString(refined1Ty), refined1Str);
  EXPECT_EQ(toString(refined2Ty), refined2Str);

  // Test refinement protocol on crossproduct of results. Dimension by
  // dimension, the first operand's dimension tends to win unless we have an
  // integral value.
  EXPECT_EQ(toString(M::RMO::refineShape(shape1Ty, shape1Ty)), shape1Str);
  EXPECT_EQ(toString(M::RMO::refineShape(shape1Ty, shape2Ty)), refined1Str);

  EXPECT_EQ(toString(M::RMO::refineShape(shape2Ty, shape1Ty)), refined2Str);
  EXPECT_EQ(toString(M::RMO::refineShape(shape2Ty, shape2Ty)), shape2Str);
}

TEST(testRMOOpHelpers, testRefineDifferentIntegralDims) {
  // TODO(GEX-715): Audit asserts on production builds.
#ifndef NDEBUG
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<RMO::RMODialect, MO::MODialect>();

  // Declare dimensions / shape attr
  auto paramInteger1Attr = ShapeAttr::getDim(&ctx, 1);
  auto paramInteger2Attr = ShapeAttr::getDim(&ctx, 2);

  TensorType shape1Ty = TensorType::get(
      &ctx, SmallVector<TypedAttr>{paramInteger1Attr}, DType::f32);
  TensorType shape2Ty = TensorType::get(
      &ctx, SmallVector<TypedAttr>{paramInteger2Attr}, DType::f32);

  EXPECT_DEATH(
      M::RMO::refineShape(shape1Ty, shape2Ty),
      "trying to refine two tensors whose integer dimensions don't match.");
#endif
}

} // namespace
