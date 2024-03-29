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
#if defined(__linux__) && defined(MODULAR_X86_64)
  auto targetInfo = M::getMArchFeatures(&ctx, "skylake-avx512", "generic", "");
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
#elif defined(__APPLE__) && defined(MODULAR_X86_64)
  auto targetInfo = M::getMArchFeatures(&ctx, "x86-64", "apple", "");
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getArch(), "x86-64");
#elif defined(__APPLE__) && defined(MODULAR_ARM_NEON)
  auto targetInfo = M::getMArchFeatures(&ctx, "arm64", "apple-m1", "");
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getArch(), "apple-m1");
#endif
}

TEST(ArchTarget, getMArchTargetInfo) {
  llvm::InitializeAllTargets();

  ErrorOr<TargetInfo> info =
      M::getMArchTargetInfo("armv8.2-a", "neoverse-n1", "");
  ASSERT_FALSE(info.isError()) << info.getError();
  EXPECT_EQ(info->arch, "neoverse-n1");
  // FIXME(#17421): The triple's OS name is set to the host machine's OS name,
  // which is incorrect for cross-compilation. So here we only test the first 2
  // components of the triple, so as not to include the host OS mame.
  EXPECT_THAT(info->triple.str(), testing::StartsWith("aarch64-unknown-"));
}
