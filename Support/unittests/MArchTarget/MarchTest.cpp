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
