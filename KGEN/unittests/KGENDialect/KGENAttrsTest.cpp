//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "Support/Compiler/MLIRDenseAttr.h"
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
