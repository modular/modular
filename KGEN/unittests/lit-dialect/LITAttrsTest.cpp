//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENTypes.h"
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
// OriginAttrsTest
//===----------------------------------------------------------------------===//

namespace {
class OriginUnionAttrTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  OriginUnionAttrTest() { ctx.loadDialect<KGENDialect, LITDialect>(); }
};
} // namespace

TEST_F(OriginUnionAttrTest, TestDeduplication) {
  OriginType imm = OriginType::get(&ctx, false);
  TypedAttr paramX = ParamDeclRefAttr::get(StringAttr::get(&ctx, "x"), imm);
  TypedAttr paramY = ParamDeclRefAttr::get(StringAttr::get(&ctx, "y"), imm);
  TypedAttr a0 = OriginFieldAttr::get(paramY, StringAttr::get(&ctx, "f0"));

  // Two identical ones.
  EXPECT_EQ(OriginUnionAttr::get({a0, a0}, imm), a0);

  // Two identical ones mixed with another in the middle.
  EXPECT_EQ(cast<OriginUnionAttr>(OriginUnionAttr::get({a0, paramX, a0}, imm))
                .getNumOperands(),
            2);

  // More interleaved.
  EXPECT_EQ(cast<OriginUnionAttr>(
                OriginUnionAttr::get({a0, paramX, a0, paramY, a0, paramX}, imm))
                .getNumOperands(),
            3);
}
