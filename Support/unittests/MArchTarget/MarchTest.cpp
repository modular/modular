//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MDialect.h"
#include "Support/PlatformUtils.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace M;

TEST(ArchTarget, GetFeatures) {
  // Initialize the LLVM targets so we can look up the current target machine.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  MLIRContext ctx;
  ctx.loadDialect<MDialect>();
#if defined(__linux__) && defined(MODULAR_X86_64)
  auto targetInfo = M::getMArchFeatures(&ctx, "skylake-avx512", "generic", "");
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
#elif defined(__APPLE__)
  auto targetInfo = M::getMArchFeatures(&ctx, "arm64", "apple-m1", "");
  ASSERT_FALSE(targetInfo.isError()) << targetInfo.getError();
  EXPECT_EQ(targetInfo->getCpu(), "apple-m1");
#endif
}
