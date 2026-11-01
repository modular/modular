//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Support/MojoStubCompiler.h"

#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/MDialect/MDialect.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::MOGG;

namespace {

TEST(VirtualCpuTargetParse, DescriptorOverridesTripleCpuFeatures) {
  // Cross-arch: an x86_64 descriptor resolved against an arm64 host.
  auto specOr = parseVirtualCpuTarget(
      "triple=x86_64-unknown-linux-gnu;cpu=znver4;features=+avx2,+avx512f",
      "arm64-apple-macosx");
  ASSERT_FALSE(specOr.isError()) << specOr.takeError().get();
  auto spec = specOr.takeValue();
  EXPECT_EQ(spec.triple, "x86_64-unknown-linux-gnu");
  EXPECT_EQ(spec.cpu, "znver4");
  EXPECT_EQ(spec.features, "+avx2,+avx512f");
  EXPECT_TRUE(spec.hasExplicitFeatures);
}

TEST(VirtualCpuTargetParse, BareArchKeepsHostTripleNameOnly) {
  auto specOr = parseVirtualCpuTarget("x86-64-v3", "x86_64-unknown-linux-gnu");
  ASSERT_FALSE(specOr.isError()) << specOr.takeError().get();
  auto spec = specOr.takeValue();
  EXPECT_TRUE(spec.triple.empty());
  EXPECT_EQ(spec.cpu, "x86-64-v3");
  EXPECT_FALSE(spec.hasExplicitFeatures);
}

TEST(VirtualCpuTargetParse, GenericDesugarsAgainstDescriptorTriple) {
  // "generic" de-sugars against the descriptor's triple (x86-64), not the host.
  auto specOr = parseVirtualCpuTarget(
      "triple=x86_64-unknown-linux-gnu;cpu=generic", "arm64-apple-macosx");
  ASSERT_FALSE(specOr.isError()) << specOr.takeError().get();
  EXPECT_EQ(specOr.takeValue().cpu, "x86-64");
}

TEST(VirtualCpuTargetParse, NativeIsRejected) {
  // "native" re-leaks the host CPU; rejected in both forms.
  EXPECT_TRUE(
      parseVirtualCpuTarget("native", "x86_64-unknown-linux-gnu").isError());
  EXPECT_TRUE(parseVirtualCpuTarget("cpu=native", "x86_64-unknown-linux-gnu")
                  .isError());
}

TEST(VirtualCpuTargetParse, UnknownDescriptorFieldIsRejected) {
  EXPECT_TRUE(
      parseVirtualCpuTarget("cpu=x86-64;bogus=1", "x86_64-unknown-linux-gnu")
          .isError());
}

TEST(VirtualCpuTargetParse, DescriptorRequiresCpu) {
  EXPECT_TRUE(parseVirtualCpuTarget("triple=x86_64-unknown-linux-gnu",
                                    "x86_64-unknown-linux-gnu")
                  .isError());
}

TEST(VirtualCpuTargetResolve, DescriptorIsHonoredByGetTargetInfo) {
  // End-to-end: the descriptor must reach getTargetInfo's resolved arch/triple.
  // Host-independent: the resolved arch is the descriptor's, not the host's.
  initLLVM();
  mlir::MLIRContext ctx;
  ctx.loadDialect<MDialect>();

  llvm::scope_exit restore{[prev = M::Driver::getVirtualCpuTarget()] {
    M::Driver::setVirtualCpuTarget(prev);
  }};
  M::Driver::setVirtualCpuTarget(
      "triple=x86_64-unknown-linux-gnu;cpu=x86-64;features=+sse2");

  auto targetOr = getTargetInfo(&ctx);
  ASSERT_FALSE(targetOr.isError()) << targetOr.takeError().get();
  auto target = targetOr.takeValue();
  EXPECT_EQ(target.getArch(), "x86-64");
  EXPECT_TRUE(llvm::StringRef(target.getTripleStr()).starts_with("x86_64"))
      << target.getTripleStr();
}

TEST(VirtualCpuTargetResolve, ForHostExecutionIgnoresTheOverride) {
  // The constant-folding sub-compilation executes its artifact in this
  // process, so it must resolve the true host even while a virtual CPU
  // target retargets every other compile (a Windows-ABI fold object dies
  // materializing `__chkstk` on a Linux or macOS host).
  initLLVM();
  mlir::MLIRContext ctx;
  ctx.loadDialect<MDialect>();

  llvm::scope_exit restore{[prev = M::Driver::getVirtualCpuTarget()] {
    M::Driver::setVirtualCpuTarget(prev);
  }};

  // The host baseline, resolved with no override in effect.
  M::Driver::setVirtualCpuTarget("");
  auto hostOr = getTargetInfo(&ctx);
  ASSERT_FALSE(hostOr.isError()) << hostOr.takeError().get();
  auto host = hostOr.takeValue();

  M::Driver::setVirtualCpuTarget(
      "triple=aarch64-w64-windows-gnu-elf;cpu=generic");
  auto foldOr = getTargetInfo(&ctx, "", "", /*forHostExecution=*/true);
  ASSERT_FALSE(foldOr.isError()) << foldOr.takeError().get();
  auto fold = foldOr.takeValue();

  EXPECT_EQ(fold.getTripleStr(), host.getTripleStr());
  EXPECT_EQ(fold.getArch(), host.getArch());
  EXPECT_EQ(fold.getFeatures(), host.getFeatures());

  // And without the flag the same override still wins, so this cannot pass
  // by the override never engaging.
  auto overriddenOr = getTargetInfo(&ctx);
  ASSERT_FALSE(overriddenOr.isError()) << overriddenOr.takeError().get();
  EXPECT_NE(overriddenOr.takeValue().getTripleStr(), host.getTripleStr());
}

} // namespace
