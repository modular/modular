//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MDialect.h"
#include "Support/PlatformUtils.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace M;

TEST(ArchTarget, GetFeatures) {
  // Initialize the LLVM targets so we can look up the current target machine.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();
  auto targetInfo =
      M::getMArchFeatures(&ctx, "x86_64-unknown-linux-gnu", "skylake-avx512",
                          "generic", "", "", llvm::Reloc::Static);
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getRelocationModel(), llvm::Reloc::Static);

  targetInfo = M::getMArchFeatures(&ctx, "x86_64-apple-macosx11.0", "x86-64",
                                   "apple", "", "", llvm::Reloc::PIC_);
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getArch(), "x86-64");
  EXPECT_EQ(targetInfo->getRelocationModel(), llvm::Reloc::PIC_);

  targetInfo = M::getMArchFeatures(&ctx, "arm64-apple-macosx11.0", "arm64",
                                   "apple-m1", "", "", llvm::Reloc::PIC_);
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getArch(), "apple-m1");
}

TEST(ArchTarget, getMArchTargetInfo) {
  llvm::InitializeAllTargets();

  ErrorOr<TargetInfo> info = M::getMArchTargetInfo(
      "aarch64-unknown-linux-gnu", "armv8.2-a", "neoverse-n1", "");
  ASSERT_FALSE(info.isError()) << info.getError();
  EXPECT_EQ(info->arch, "neoverse-n1");
  EXPECT_EQ(info->triple.str(), "aarch64-unknown-linux-gnu");
}

// getTargetInfoFor must expand the explicit --target-features delta against
// the CPU model defaults so that hasFeature() reflects what LLVM will actually
// compile for. znver4 enables avx512f by default; omitting -avx512f from the
// feature string should not make it invisible to Mojo's compile-time queries.
TEST(ArchTarget, GetTargetInfoForExpandsFeatures) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  MLIRContext ctx{MLIRContext::Threading::DISABLED};
  ctx.loadDialect<MDialect>();

  constexpr StringLiteral triple = "x86_64-unknown-linux-gnu";
  constexpr StringLiteral cpu = "znver4";
  // Feature string that enables/disables some other features but does not
  // mention avx512f
  constexpr StringLiteral featuresWithoutDisablingAvx512f =
      "+avx,+avx2,-avx512bw,-avx512cd,-avx512dq,-avx512vl";

  // avx512f is part of znver4's CPU model defaults. Without an explicit
  // -avx512f in the feature string, LLVM keeps it enabled — hasFeature must
  // agree.
  auto targetOn =
      M::getTargetInfoFor(&ctx, triple, cpu, featuresWithoutDisablingAvx512f,
                          "", "", llvm::Reloc::Static);
  ASSERT_FALSE(targetOn.isError()) << targetOn.getError();
  EXPECT_TRUE(targetOn->hasFeature("avx512f"));
  EXPECT_TRUE(targetOn->hasFeature("avx2"));

  // With an explicit -avx512f, hasFeature must return false.
  constexpr StringLiteral featuresWithAvx512fDisabled =
      "+avx,+avx2,-avx512bw,-avx512cd,-avx512dq,-avx512f,-avx512vl";
  auto targetOff =
      M::getTargetInfoFor(&ctx, triple, cpu, featuresWithAvx512fDisabled, "",
                          "", llvm::Reloc::Static);
  ASSERT_FALSE(targetOff.isError()) << targetOff.getError();
  EXPECT_FALSE(targetOff->hasFeature("avx512f"));
  EXPECT_TRUE(targetOff->hasFeature("avx2"));
}
