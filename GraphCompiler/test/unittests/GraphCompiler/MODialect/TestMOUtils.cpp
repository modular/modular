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

#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "gtest/gtest.h"
#include <limits.h>

using namespace M;

namespace {

inline double
constructAndGetFloatArrayAttrData(double num, Type &eleType,
                                  const llvm::fltSemantics &semantic,
                                  bool *losesInfo) {
  APFloat convertedFloat = APFloat(num);
  convertedFloat.convert(semantic, llvm::RoundingMode::TowardZero, losesInfo);

  auto attr = FloatArrayElementsAttr::get({convertedFloat}, eleType);
  return (*attr.getValues().begin()).convertToDouble();
}

inline int64_t constructAndGetIntArrayAttrData(int64_t num, Type &eleType,
                                               DType dtype) {
  APInt convertedInt = APInt(dtype.getWidthInBits(), num, true);

  auto attr =
      IntArrayElementsAttr::get(ArrayType::get(1, eleType), {convertedInt});

  return (*attr.getValues().begin()).getSExtValue();
}

template <typename inT, typename outT, typename elementType>
std::optional<SmallVector<outT>>
constructAndGetDenseAttrData(SmallVector<inT> originalData,
                             elementType &eleType, mlir::MLIRContext *ctx) {
  ArrayRef<char> data((char *)originalData.data(),
                      originalData.size_in_bytes());
  ArrayType type = ArrayType::get(originalData.size(), eleType);
  auto attr = DenseResourceElementsAttr::get(
      type,
      DenseResourceElementsHandle::getManagerInterface(ctx).insert(
          "test", mlir::UnmanagedAsmResourceBlob::allocateWithAlign(data, 64)));
  return MO::tryGetConstantBlob<outT>(attr, getEquivalentDType(eleType));
}

TEST(testGetConstantBlobFloatArrayAttr, allDtypes) {
  SmallVector<double> testCases = {-453, -32, -0.02, 0, 0.58, 3.14, 121.24};

  SmallVector<std::pair<const llvm::fltSemantics &(*)(), DType>>
      semanticGenAndDtype = {{&APFloat::BFloat, DType::bf16},
                             {&APFloat::IEEEhalf, DType::f16},
                             {&APFloat::IEEEsingle, DType::f32},
                             {&APFloat::IEEEdouble, DType::f64}};

  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();
  for (auto &[semanticGen, dtype] : semanticGenAndDtype) {
    mlir::FloatType fltType = getEquivalentFloatType(&ctx, dtype);
    bool losesInfo; // will always be true for DType < f64
    for (double expected : testCases) {
      const llvm::fltSemantics &semantic = semanticGen();
      double actual = constructAndGetFloatArrayAttrData(expected, fltType,
                                                        semantic, &losesInfo);
      EXPECT_NEAR(expected, actual, abs(expected * 0.01));
    }
  }
}

TEST(testGetConstantBlobIntArrayAttr, allDtypes) {
  SmallVector<int64_t> testCases = {-50, -8, 1, 0, 5, 80};
  SmallVector<DType> dtypes = {DType::si8, DType::si16, DType::si32};

  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();
  for (const DType &dtype : dtypes) {
    mlir::IntegerType intType = getEquivalentIntegerType(&ctx, dtype);
    for (int64_t expected : testCases) {
      int64_t actual =
          constructAndGetIntArrayAttrData(expected, intType, dtype);
      EXPECT_EQ(expected, actual);
    }
  }
}

TEST(testGetConstantBlobDenseResourceAttr, testSI64) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  SmallVector<int64_t> data = {33, 4, -100};
  mlir::IntegerType fltType = getEquivalentIntegerType(&ctx, DType::si64);
  std::optional<SmallVector<int64_t>> actual =
      constructAndGetDenseAttrData<int64_t, int64_t, mlir::IntegerType>(
          data, fltType, &ctx);
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual.value(), data);
}

TEST(testGetConstantBlobDenseResourceAttr, testSI32) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  SmallVector<int32_t> data = {33, 4, -100};
  mlir::IntegerType fltType = getEquivalentIntegerType(&ctx, DType::si32);
  std::optional<SmallVector<int32_t>> actual =
      constructAndGetDenseAttrData<int32_t, int32_t, mlir::IntegerType>(
          data, fltType, &ctx);
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual.value(), data);
}

TEST(testGetConstantBlobDenseResourceAttr, testF64) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  SmallVector<double> data = {3.0, 4.0, -100};
  mlir::FloatType fltType = getEquivalentFloatType(&ctx, DType::f64);
  std::optional<SmallVector<double>> actual =
      constructAndGetDenseAttrData<double, double, mlir::FloatType>(
          data, fltType, &ctx);
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual.value(), data);
}

TEST(testGetConstantBlobDenseResourceAttr, testF32) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  SmallVector<float> data = {3.0, 4.0, -100};
  mlir::FloatType fltType = getEquivalentFloatType(&ctx, DType::f32);
  std::optional<SmallVector<float>> actual =
      constructAndGetDenseAttrData<float, float, mlir::FloatType>(data, fltType,
                                                                  &ctx);
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual.value(), data);
}

TEST(testGetConstantBlobDenseResourceAttr, testF16) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  // calculated by hand
  SmallVector<uint16_t> data = {0x4248, 0xDD10, 0x4400};
  SmallVector<double> expected = {3.140625, -324, 4.0};
  mlir::FloatType fltType = getEquivalentFloatType(&ctx, DType::f16);
  std::optional<SmallVector<double>> actual =
      constructAndGetDenseAttrData<uint16_t, double, mlir::FloatType>(
          data, fltType, &ctx);
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual.value(), expected);
}

TEST(testGetConstantBlobDenseResourceAttr, testBF16) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  // calculated by hand
  SmallVector<uint16_t> data = {0x4049, 0xC467, 0x4080};
  SmallVector<double> expected = {3.140625, -924, 4.0};
  mlir::FloatType fltType = getEquivalentFloatType(&ctx, DType::bf16);
  std::optional<SmallVector<double>> actual =
      constructAndGetDenseAttrData<uint16_t, double, mlir::FloatType>(
          data, fltType, &ctx);
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual.value(), expected);
}

} // namespace
