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
// SignatureGeneratorType
//===----------------------------------------------------------------------===//

namespace {
class SignatureGeneratorTypeTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  SignatureGeneratorTypeTest() { ctx.loadDialect<KGENDialect, LITDialect>(); }
};
} // namespace

TEST_F(SignatureGeneratorTypeTest, TestConversion) {
  auto indexType = IndexType::get(&ctx);
  auto typeType = TypeType::get(&ctx);
  auto originType = LIT::OriginType::get(&ctx, /*isMutable=*/false);
  FunctionType funcType = FunctionType::get(&ctx, {indexType}, {indexType});
  SmallVector<Type> inputParamTypes = {originType, indexType, typeType};
  SmallVector<ArgConvention> argConvs = {ArgConvention::ReadReg};
  FnEffects fnEffects = FnEffects().setCapturing();

  // Test bare KGEN Signature
  {
    SignatureType sig =
        SignatureType::get(funcType, inputParamTypes, /*resultParamTypes=*/{},
                           argConvs, fnEffects);
    SignatureGeneratorType sigGen = SignatureGeneratorType::get(
        inputParamTypes, funcType, argConvs, fnEffects);

    EXPECT_EQ(sig.asSignatureGenerator(), sigGen);
    EXPECT_EQ(sigGen.asOldSignature(), sig);
  }

  // Test Signature with metadata
  {
    auto inferred =
        PogMetadataAttr::get(StringAttr::get(&ctx), PassingKind::Inferred,
                             /*isVariadic=*/false);
    auto posOnly =
        PogMetadataAttr::get(StringAttr::get(&ctx), PassingKind::PosOnly,
                             /*isVariadic=*/false);
    PogListAttr paramPogs = PogListAttr::get(
        &ctx, SmallVector<PogMetadataAttr>{inferred, posOnly, posOnly});
    PogListAttr argPogs =
        PogListAttr::get(&ctx, SmallVector<PogMetadataAttr>{posOnly});
    FnMetadataAttr fnMetadata = FnMetadataAttr::get(
        argPogs, paramPogs,
        /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
        /*isNestedOriginExclusivityCheckingDisabled=*/false);
    LITSignatureType sig = LITSignatureType::get(
        funcType, inputParamTypes, argConvs, fnEffects, fnMetadata);
    SignatureGeneratorType sigGen = SignatureGeneratorType::get(
        inputParamTypes, funcType, argConvs, fnEffects, fnMetadata, paramPogs);
    EXPECT_EQ(sig.asSignatureGenerator(), sigGen);
    EXPECT_EQ(sigGen.asOldSignature(), sig);
  }
}

TEST_F(SignatureGeneratorTypeTest, TestSpecialization) {
  auto indexType = IndexType::get(&ctx);
  auto typeType = TypeType::get(&ctx);
  auto indexTypeAttr = TypeConstantAttr::get(indexType, typeType);
  auto ref0Type = ParamRefType::get(ParamIndexRefAttr::get(0, typeType));
  FunctionType funcType = FunctionType::get(&ctx, {ref0Type}, {ref0Type});
  SmallVector<Type> inputParamTypes = {typeType};

  // Test bare KGEN Signature
  {
    SignatureGeneratorType sigGen =
        SignatureGeneratorType::get(inputParamTypes, funcType);
    SignatureGeneratorType concreteSigGen =
        sigGen.getSpecializedGenerator({indexTypeAttr});

    EXPECT_EQ(concreteSigGen,
              SignatureGeneratorType::get(
                  /*inputParamTypes=*/{},
                  FunctionType::get(&ctx, {indexType}, {indexType})));
  }

  // Test Signature with metadata
  {
    auto posOnly =
        PogMetadataAttr::get(StringAttr::get(&ctx), PassingKind::PosOnly,
                             /*isVariadic=*/false);
    PogListAttr pogs =
        PogListAttr::get(&ctx, SmallVector<PogMetadataAttr>{posOnly});
    FnMetadataAttr fnMetadata = FnMetadataAttr::get(
        pogs, pogs,
        /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
        /*isNestedOriginExclusivityCheckingDisabled=*/false);
    FnMetadataAttr fnMetadataNoParams = FnMetadataAttr::get(
        pogs, PogListAttr::get(&ctx),
        /*numImplicitOriginDecls=*/0, /*captureOrigins=*/nullptr,
        /*isNestedOriginExclusivityCheckingDisabled=*/false);
    SignatureGeneratorType sigGen =
        SignatureGeneratorType::get(inputParamTypes, funcType, /*argConvs=*/{},
                                    /*effects=*/{}, fnMetadata, pogs);
    SignatureGeneratorType concreteSigGen =
        sigGen.getSpecializedGenerator({indexTypeAttr});

    EXPECT_EQ(concreteSigGen,
              SignatureGeneratorType::get(
                  /*inputParamTypes=*/{},
                  FunctionType::get(&ctx, {indexType}, {indexType}),
                  /*argConvs=*/{},
                  /*effects=*/{}, fnMetadataNoParams, PogListAttr::get(&ctx)));
  }
}
