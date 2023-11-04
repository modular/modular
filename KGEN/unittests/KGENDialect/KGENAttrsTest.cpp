//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace testing;

//===----------------------------------------------------------------------===//
// PackageArchiveArrayAttrTest
//===----------------------------------------------------------------------===//

namespace {
class PackageArchiveArrayAttrTest : public Test {
protected:
  MLIRContext context;

  PackageArchiveArrayAttrTest() {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();

    context.loadDialect<KGENDialect>();
  }
};
} // namespace

TEST_F(PackageArchiveArrayAttrTest, getTargetArchive_empty) {
  // Construct an empty array of package archives.
  auto archives = PackageArchiveArrayAttr::get(&context, {});
  EXPECT_THAT(archives.getTargetsAsString(), StrEq("none"));

  // An archive for this target doesn't exist in the array.
  TargetInfoAttr targetInfo =
      getTargetInfoFor(&context, llvm::sys::getDefaultTargetTriple(), "", "")
          .takeValue();
  EXPECT_EQ(archives.getTargetArchive(targetInfo), std::nullopt);
}

TEST_F(PackageArchiveArrayAttrTest, getTargetArchive_noMatch) {
  // Construct an array of package archives. We'll search for one of the
  // targets, but their contents are unimportant and so we use dummy data.
  DenseResourceElementsAttr dummy =
      createResourceAttr(&context, "dummy-data", "dummy-name");
  TargetInfoAttr targetInfo =
      getTargetInfoFor(&context, llvm::sys::getDefaultTargetTriple(), "", "")
          .takeValue();
  auto archive = PackageArchiveAttr::get(targetInfo, dummy, dummy);
  auto archives = PackageArchiveArrayAttr::get(&context, archive);
  // There's only one archive, so a string representing the archives' targets
  // should not contain a comma.
  EXPECT_THAT(archives.getTargetsAsString(), Not(Contains(',')));

  // Now search for a target that doesn't exist in the array.
  TargetInfoAttr needle =
      getTargetInfoFor(&context, llvm::sys::getDefaultTargetTriple(),
                       "different-arch", "")
          .takeValue();
  EXPECT_EQ(archives.getTargetArchive(needle), std::nullopt);
}

TEST_F(PackageArchiveArrayAttrTest, getTargetArchive_match) {
  // Construct an array of package archives. We'll search for one of the
  // targets, but their contents are unimportant and so we use dummy data.
  std::string triple = llvm::sys::getDefaultTargetTriple();
  DenseResourceElementsAttr dummy =
      createResourceAttr(&context, "dummy-data", "dummy-name");

  TargetInfoAttr targetInfoOne =
      getTargetInfoFor(&context, triple, "arch-one", "").takeValue();
  auto archiveOne = PackageArchiveAttr::get(targetInfoOne, dummy, dummy);

  TargetInfoAttr targetInfoTwo =
      getTargetInfoFor(&context, triple, "arch-two", "").takeValue();
  auto archiveTwo = PackageArchiveAttr::get(targetInfoTwo, dummy, dummy);

  auto archives =
      PackageArchiveArrayAttr::get(&context, {archiveOne, archiveTwo});
  // There are multiple archives, so a string representing the archives' targets
  // should contain a comma.
  EXPECT_THAT(archives.getTargetsAsString(), Contains(','));

  // Now search for the second target.
  EXPECT_EQ(archives.getTargetArchive(targetInfoTwo), archiveTwo);
}

//===----------------------------------------------------------------------===//
// SourceNameAttrTest
//===----------------------------------------------------------------------===//

namespace {
class SourceNameAttrTest : public Test {
protected:
  MLIRContext ctx;

  SourceNameAttrTest() { ctx.loadDialect<KGENDialect>(); }
};
} // namespace

TEST_F(SourceNameAttrTest, TestEncodeDecode) {
  StringRef testStr = R"mlir(
    #builtin_name = #kgen.source_name<"builtin">
    #test_name = #kgen.source_name<"test">
    #int_name = #kgen.source_name<"int" from #builtin_name>
    #simd_name = #kgen.source_name<"simd" from #builtin_name>
    #Int_name = #kgen.source_name<"Int" from #int_name>
    #SIMD_name = #kgen.source_name<"SIMD"[#Int_name] from #simd_name>
    #func_name = #kgen.source_name<"func"(#SIMD_name)<"1"> from #test_name>

    #strange = #kgen.source_name<"strange*">
    #weird = #kgen.source_name<"weird&name"<":struct<index> { 1 }", "^&*"> from #strange>

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
