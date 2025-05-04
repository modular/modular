//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace KGEN::LIT;
using namespace mlir;
using namespace testing;

//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//

namespace {
class FuncTypeGeneratorTypeTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  FuncTypeGeneratorTypeTest() { ctx.loadDialect<KGENDialect, LITDialect>(); }
};
} // namespace

TEST_F(FuncTypeGeneratorTypeTest, TestSpecialization) {
  auto indexType = IndexType::get(&ctx);
  auto typeType = TypeType::get(&ctx);
  auto indexTypeAttr = TypeParamAttr::get(indexType, typeType);
  auto ref0Type = ParamType::get(ParamIndexRefAttr::get(0, typeType));
  FunctionType funcType = FunctionType::get(&ctx, {ref0Type}, {ref0Type});
  SmallVector<Type> inputParamTypes = {typeType};

  // Test bare KGEN Signature
  {
    FuncTypeGeneratorType sigGen =
        FuncTypeGeneratorType::get(inputParamTypes, funcType);
    FuncTypeGeneratorType concreteSigGen =
        sigGen.getSpecializedGenerator({indexTypeAttr});

    EXPECT_EQ(concreteSigGen,
              FuncTypeGeneratorType::get(
                  /*inputParamTypes=*/{},
                  FunctionType::get(&ctx, {indexType}, {indexType})));
  }

  // Test Signature with metadata
  {
    auto posOnly = PogMetadataAttr::get(
        StringAttr::get(&ctx), PassingKind::PosOnly, VariadicKind::None);
    PogListAttr pogs =
        PogListAttr::get(&ctx, SmallVector<PogMetadataAttr>{posOnly});
    FnMetadataAttr fnMetadata = FnMetadataAttr::get(
        pogs,
        /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
        /*isNestedOriginExclusivityCheckingDisabled=*/false);
    FnMetadataAttr fnMetadataNoParams = FnMetadataAttr::get(
        pogs,
        /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
        /*isNestedOriginExclusivityCheckingDisabled=*/false);
    FuncTypeGeneratorType sigGen =
        FuncTypeGeneratorType::get(inputParamTypes, funcType, /*argConvs=*/{},
                                   /*effects=*/{}, fnMetadata, pogs);
    FuncTypeGeneratorType concreteSigGen =
        sigGen.getSpecializedGenerator({indexTypeAttr});

    EXPECT_EQ(concreteSigGen,
              FuncTypeGeneratorType::get(
                  /*inputParamTypes=*/{},
                  FunctionType::get(&ctx, {indexType}, {indexType}),
                  /*argConvs=*/{},
                  /*effects=*/{}, fnMetadataNoParams, PogListAttr::get(&ctx)));
  }
}
