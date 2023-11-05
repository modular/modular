//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using namespace DebugInfo;
using namespace mlir;
using namespace testing;

//===----------------------------------------------------------------------===//
// SourceNameAttrTest
//===----------------------------------------------------------------------===//

namespace {
class SourceNameAttrTest : public Test {
protected:
  MLIRContext ctx;

  SourceNameAttrTest() { ctx.loadDialect<DebugInfoDialect>(); }
};
} // namespace

TEST_F(SourceNameAttrTest, TestEncodeDecode) {
  StringRef testStr = R"mlir(
    #builtin_name = #debuginfo.source_name<"builtin">
    #test_name = #debuginfo.source_name<"test">
    #int_name = #debuginfo.source_name<"int" from #builtin_name>
    #simd_name = #debuginfo.source_name<"simd" from #builtin_name>
    #Int_name = #debuginfo.source_name<"Int" from #int_name>
    #SIMD_name = #debuginfo.source_name<"SIMD"[#Int_name] from #simd_name>
    #func_name = #debuginfo.source_name<"func"(#SIMD_name)<"1"> from #test_name>

    #strange = #debuginfo.source_name<"strange*">
    #weird = #debuginfo.source_name<"weird&name"<":struct<index> { 1 }", "^&*"> from #strange>

    module attributes {
      kgen.test0 = #func_name,
      kgen.test1 = #weird
    } {}
  )mlir";

  OwningOpRef<ModuleOp> module = mlir::parseSourceString<ModuleOp>(
      testStr, mlir::ParserConfig(&ctx), "TestEncode_testStr");
  ASSERT_TRUE(module);

  {
    auto sourceName = (*module)->getAttrOfType<SourceNameAttr>("kgen.test0");
    ASSERT_TRUE(sourceName);

    StringRef expected =
        "test::func(builtin::simd::SIMD[builtin::int::Int])<1>";
    EXPECT_EQ(sourceName.encode().getValue(), expected);

    ErrorOr<SourceNameAttr> decoded = SourceNameAttr::decode(&ctx, expected);
    ASSERT_FALSE(decoded.isError());
    EXPECT_EQ(decoded.takeValue(), sourceName);
  }

  {
    auto sourceName = (*module)->getAttrOfType<SourceNameAttr>("kgen.test1");
    ASSERT_TRUE(sourceName);

    StringRef expected =
        "`strange*`::`weird&name`<`:struct<index> { 1 }`,`^&*`>";
    EXPECT_EQ(sourceName.encode().getValue(), expected);

    ErrorOr<SourceNameAttr> decoded = SourceNameAttr::decode(&ctx, expected);
    ASSERT_FALSE(decoded.isError());
    EXPECT_EQ(decoded.takeValue(), sourceName);
  }
}
