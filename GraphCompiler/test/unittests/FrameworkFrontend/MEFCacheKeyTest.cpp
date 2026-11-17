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

#include "GraphCompiler/MEFCacheKey.h"

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/Passes.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Context.h"
#include "Support/FileSystemExtras.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include "gtest/gtest.h"

#include <fstream>

using namespace M;

namespace {

/// Test fixture that constructs the inputs the hasher needs: an MLIR
/// context with the dialects ``computeMEFCacheKey`` touches, default
/// `GCPipelineOptions`, a fresh `Context`.
class MEFCacheKeyTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<mlir::BuiltinDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<MOGG::MOGGDialect>();
    ctx.loadDialect<MO::MODialect>();
    maxContext = ContextRef::create();
  }

  /// Parse the given MLIR source as a `ModuleOp`. Returns an owned op so
  /// the caller can compute a key against it.
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef src) {
    auto mod = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
    EXPECT_TRUE(mod) << "failed to parse MLIR fixture:\n" << src.str();
    return mod;
  }

  /// Build a hasher with default-constructed options. Tests skip if
  /// `build()` fails — the hasher's prefix needs the built-in kernel
  /// packages on disk, which the bazel test sandbox may not surface.
  /// The optional `configureGC` callback runs before the build, so a
  /// caller can mutate `gcPipelineOptions` (e.g. populate
  /// `kernelLibraryPaths`).
  std::optional<MEFCacheKeyHasher>
  tryBuild(llvm::function_ref<void(MO::GCPipelineOptions &)> configureGC = {}) {
    MO::GCPipelineOptions gc;
    if (configureGC)
      configureGC(gc);
    auto hasherOr = MEFCacheKeyHasher::build(ctx, gc, maxContext);
    if (hasherOr.isError())
      return std::nullopt;
    return std::move(hasherOr).takeValue();
  }

  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ContextRef maxContext;
};

constexpr size_t kBLAKE3DigestBytes = 32;

constexpr llvm::StringLiteral kEmptyModule = "module {}";
// Symbols get canonicalized by `normalizeAllNames`, so the names alone
// don't differentiate modules. These fixtures differ in signature shape
// — argument count and type — which survives normalization.
constexpr llvm::StringLiteral kFuncFooModule = R"mlir(
  module {
    func.func private @foo(%a: i32) -> i32
  }
)mlir";
constexpr llvm::StringLiteral kFuncBarModule = R"mlir(
  module {
    func.func private @bar(%a: i64, %b: i64) -> i64
  }
)mlir";

/// RAII helper that appends `dir` to the Mojo module import path env var and
/// restores the previous value on destruction. The prior value is kept as a
/// prefix so the built-in kernel packages still resolve.
class AppendImportPathDir {
public:
  explicit AppendImportPathDir(llvm::StringRef dir) {
    if (const char *prev = ::getenv(kImportPathEnvVar))
      previous = prev;
    std::string next = previous ? *previous + "," + dir.str() : dir.str();
    ::setenv(kImportPathEnvVar, next.c_str(), /*overwrite=*/1);
  }
  ~AppendImportPathDir() {
    if (previous)
      ::setenv(kImportPathEnvVar, previous->c_str(), /*overwrite=*/1);
    else
      ::unsetenv(kImportPathEnvVar);
  }

private:
  static constexpr const char *kImportPathEnvVar =
      "MODULAR_MOJO_MAX_IMPORT_PATH";
  std::optional<std::string> previous;
};

/// Create a self-cleaning temp directory containing a fake Mojo binary
/// package `pkg.mojoc` with the given bytes.
std::optional<TempDir> makePackageDir(llvm::StringRef contents) {
  auto dirOr = TempDir::create("mef-import-path-test.%%%%%%");
  EXPECT_FALSE(dirOr.isError()) << "failed to create temp dir";
  if (dirOr.isError())
    return std::nullopt;
  TempDir dir = dirOr.takeValue();
  std::ofstream out(dir.getPath() / "pkg.mojoc", std::ios::binary);
  out << contents.str();
  return dir;
}

} // namespace

TEST_F(MEFCacheKeyTest, KeyHasExpectedLength) {
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto keyOr = hasher->keyForModule(*mod);
  ASSERT_FALSE(keyOr.isError()) << keyOr.takeError().get();
  EXPECT_EQ(keyOr.takeValue().size(), kBLAKE3DigestBytes);
}

TEST_F(MEFCacheKeyTest, GraphCounterDoesNotReachKey) {
  // A graph's `counter` numbers the symbolic dimension names it mints, so it
  // moves with work done to the graph rather than with what it computes. Two
  // modules differing only in it are the same program.
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto fresh = parse(R"mlir(
    module {
      mo.graph @g(%arg0: !mo.tensor<[4], f32, cpu:0>) -> !mo.tensor<[4], f32, cpu:0> {
        mo.output %arg0 : !mo.tensor<[4], f32, cpu:0>
      } {counter = 0 : i64}
    }
  )mlir");
  auto advanced = parse(R"mlir(
    module {
      mo.graph @g(%arg0: !mo.tensor<[4], f32, cpu:0>) -> !mo.tensor<[4], f32, cpu:0> {
        mo.output %arg0 : !mo.tensor<[4], f32, cpu:0>
      } {counter = 32 : i64}
    }
  )mlir");
  if (!fresh || !advanced)
    GTEST_SKIP() << "unable to build the graph fixture";

  auto freshKey = hasher->keyForModule(*fresh);
  auto advancedKey = hasher->keyForModule(*advanced);
  ASSERT_FALSE(freshKey.isError());
  ASSERT_FALSE(advancedKey.isError());
  EXPECT_EQ(freshKey.takeValue(), advancedKey.takeValue());
}

TEST_F(MEFCacheKeyTest, ContentDigestIgnoresWhichDevicesWerePresent) {
  // A split's two halves hold different devices -- the compile-only one holds
  // none -- and the device-info mapping records them. It must not reach the
  // digest, or an artifact exported on one half could never be imported on the
  // other; the tile-based fusion bit it carries is part of the program and
  // must.
  auto cpuOnly = parse(R"mlir(
    module attributes {mo.device_info_mapping = {cpu = #M.device_info<"cpu", "cpu", "", "", tile_based_fusion = false>}} {
      func.func private @foo(%a: i32) -> i32
    }
  )mlir");
  auto withVirtualGpu = parse(R"mlir(
    module attributes {mo.device_info_mapping = {cpu = #M.device_info<"cpu", "cpu", "", "", tile_based_fusion = false>, gpu = #M.device_info<"gpu", "cuda", "sm_90a", "", tile_based_fusion = false>}} {
      func.func private @foo(%a: i32) -> i32
    }
  )mlir");
  if (!cpuOnly || !withVirtualGpu)
    GTEST_SKIP() << "unable to build the graph fixture";

  auto cpuDigest =
      computeMefContentDigest(*cpuOnly, MO::GCPipelineOptions{}, maxContext);
  auto gpuDigest = computeMefContentDigest(*withVirtualGpu,
                                           MO::GCPipelineOptions{}, maxContext);
  if (cpuDigest.isError())
    GTEST_SKIP() << "kernel packages unavailable; skip "
                 << cpuDigest.takeError().get();
  ASSERT_FALSE(gpuDigest.isError()) << gpuDigest.takeError().get();
  EXPECT_EQ(cpuDigest.takeValue(), gpuDigest.takeValue());
}

TEST_F(MEFCacheKeyTest, ContentDigestIgnoresTheTreeTheGraphWasBuiltIn) {
  // The kernel-library paths record where the building tree put its kernels.
  // A split records in one tree and initializes in another, so they cannot be
  // part of the identity. What the paths hold is hashed from the libraries a
  // compile is handed, and `KernelLibraryContentInvalidatesKey` covers that.
  auto built = parse(R"mlir(
    module {
      mo.graph @g(%arg0: !mo.tensor<[4], f32, cpu:0>) -> !mo.tensor<[4], f32, cpu:0> attributes {_kernel_library_paths = ["/build/tree/kernels.mojoc"]} {
        mo.output %arg0 : !mo.tensor<[4], f32, cpu:0>
      } {counter = 0 : i64}
    }
  )mlir");
  auto replayed = parse(R"mlir(
    module {
      mo.graph @g(%arg0: !mo.tensor<[4], f32, cpu:0>) -> !mo.tensor<[4], f32, cpu:0> attributes {_kernel_library_paths = ["/runfiles/tree/kernels.mojoc"]} {
        mo.output %arg0 : !mo.tensor<[4], f32, cpu:0>
      } {counter = 0 : i64}
    }
  )mlir");
  if (!built || !replayed)
    GTEST_SKIP() << "unable to build the graph fixture";

  auto builtDigest =
      computeMefContentDigest(*built, MO::GCPipelineOptions{}, maxContext);
  auto replayedDigest =
      computeMefContentDigest(*replayed, MO::GCPipelineOptions{}, maxContext);
  if (builtDigest.isError())
    GTEST_SKIP() << "kernel packages unavailable; skip "
                 << builtDigest.takeError().get();
  ASSERT_FALSE(replayedDigest.isError()) << replayedDigest.takeError().get();
  EXPECT_EQ(builtDigest.takeValue(), replayedDigest.takeValue());
}

TEST_F(MEFCacheKeyTest, ContentDigestCoversTheBody) {
  auto foo = parse(kFuncFooModule);
  auto bar = parse(kFuncBarModule);
  if (!foo || !bar)
    GTEST_SKIP() << "unable to build the graph fixture";

  auto fooDigest =
      computeMefContentDigest(*foo, MO::GCPipelineOptions{}, maxContext);
  auto barDigest =
      computeMefContentDigest(*bar, MO::GCPipelineOptions{}, maxContext);
  if (fooDigest.isError())
    GTEST_SKIP() << "kernel packages unavailable; skip "
                 << fooDigest.takeError().get();
  ASSERT_FALSE(barDigest.isError()) << barDigest.takeError().get();
  EXPECT_NE(fooDigest.takeValue(), barDigest.takeValue());
}

TEST_F(MEFCacheKeyTest, KeyIsDeterministic) {
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kFuncFooModule);
  auto first = hasher->keyForModule(*mod);
  auto second = hasher->keyForModule(*mod);
  ASSERT_FALSE(first.isError());
  ASSERT_FALSE(second.isError());
  EXPECT_EQ(first.takeValue(), second.takeValue());
}

TEST_F(MEFCacheKeyTest, DifferentModulesProduceDifferentKeys) {
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto a = parse(kFuncFooModule);
  auto b = parse(kFuncBarModule);
  auto keyA = hasher->keyForModule(*a);
  auto keyB = hasher->keyForModule(*b);
  ASSERT_FALSE(keyA.isError());
  ASSERT_FALSE(keyB.isError());
  EXPECT_NE(keyA.takeValue(), keyB.takeValue());
}

TEST_F(MEFCacheKeyTest, HasherStateIsImmutable) {
  // Computing a different key in between mustn't disturb the BLAKE3 state.
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto a = parse(kFuncFooModule);
  auto b = parse(kFuncBarModule);
  auto first = hasher->keyForModule(*a);
  (void)hasher->keyForModule(*b);
  auto third = hasher->keyForModule(*a);
  ASSERT_FALSE(first.isError());
  ASSERT_FALSE(third.isError());
  EXPECT_EQ(first.takeValue(), third.takeValue());
}

TEST_F(MEFCacheKeyTest, KernelLibraryPathInvalidatesKey) {
  // Two hashers differing only in `kernelLibraryPaths` should produce
  // distinct keys for the same module. Use a temp file with known
  // non-empty contents as a stand-in "kernel library" — pointing at a
  // real, readable path the hasher can open.
  llvm::SmallString<128> tmpPath;
  int tmpFd = -1;
  auto ec = llvm::sys::fs::createTemporaryFile("mef-cache-key-test", "dat",
                                               tmpFd, tmpPath);
  ASSERT_FALSE(ec) << "failed to create temp file: " << ec.message();
  {
    std::ofstream out(tmpPath.c_str(), std::ios::binary);
    out << "synthetic kernel library bytes for MEFCacheKeyTest";
  }
  llvm::sys::fs::file_t tmpFile = llvm::sys::fs::convertFDToNativeFile(tmpFd);
  llvm::sys::fs::closeFile(tmpFile);
  llvm::scope_exit cleanup{[&] { llvm::sys::fs::remove(tmpPath); }};

  auto plain = tryBuild();
  if (!plain)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  std::string tmpPathStr(tmpPath.str());
  auto withExtraHasher = tryBuild(
      [&](MO::GCPipelineOptions &gc) { gc.kernelLibraryPaths = {tmpPathStr}; });
  if (!withExtraHasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto plainKey = plain->keyForModule(*mod);
  auto extraKey = withExtraHasher->keyForModule(*mod);
  ASSERT_FALSE(plainKey.isError());
  ASSERT_FALSE(extraKey.isError());
  EXPECT_NE(plainKey.takeValue(), extraKey.takeValue());
}

TEST_F(MEFCacheKeyTest, KernelLibraryContentInvalidatesKey) {
  // The key covers what a user kernel library holds, not only where it lives,
  // so editing a kernel misses the old artifact instead of reusing it.
  llvm::SmallString<128> tmpPath;
  int tmpFd = -1;
  auto ec = llvm::sys::fs::createTemporaryFile("mef-cache-key-content-test",
                                               "dat", tmpFd, tmpPath);
  ASSERT_FALSE(ec) << "failed to create temp file: " << ec.message();
  llvm::sys::fs::file_t tmpFile = llvm::sys::fs::convertFDToNativeFile(tmpFd);
  llvm::sys::fs::closeFile(tmpFile);
  llvm::scope_exit cleanup{[&] { llvm::sys::fs::remove(tmpPath); }};

  const std::string path(tmpPath.str());
  auto write = [&](llvm::StringRef contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << contents.str();
  };

  write("the first kernels");
  auto before = tryBuild(
      [&](MO::GCPipelineOptions &gc) { gc.kernelLibraryPaths = {path}; });
  if (!before)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  write("the edited kernels");
  auto after = tryBuild(
      [&](MO::GCPipelineOptions &gc) { gc.kernelLibraryPaths = {path}; });
  if (!after)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto beforeKey = before->keyForModule(*mod);
  auto afterKey = after->keyForModule(*mod);
  ASSERT_FALSE(beforeKey.isError());
  ASSERT_FALSE(afterKey.isError());
  EXPECT_NE(beforeKey.takeValue(), afterKey.takeValue());
}

TEST_F(MEFCacheKeyTest, KernelLibraryPathIsRelocatable) {
  // MOCO-4152: keying on contents not path lets a CI-warmed cache hit under a
  // different install location.
  auto writeLib = [](llvm::SmallVectorImpl<char> &outPath) {
    int fd = -1;
    auto ec = llvm::sys::fs::createTemporaryFile("mef-reloc-test", "dat", fd,
                                                 outPath);
    EXPECT_FALSE(ec) << "failed to create temp file: " << ec.message();
    {
      std::ofstream out(outPath.data(), std::ios::binary);
      // Identical bytes in both libraries — only the path differs.
      out << "identical synthetic kernel library bytes";
    }
    llvm::sys::fs::file_t file = llvm::sys::fs::convertFDToNativeFile(fd);
    llvm::sys::fs::closeFile(file);
  };

  llvm::SmallString<128> pathA, pathB;
  writeLib(pathA);
  writeLib(pathB);
  llvm::scope_exit cleanup{[&] {
    llvm::sys::fs::remove(pathA);
    llvm::sys::fs::remove(pathB);
  }};
  ASSERT_NE(pathA.str(), pathB.str()) << "temp paths must differ";

  std::string pathAStr(pathA.str()), pathBStr(pathB.str());
  auto hasherA = tryBuild(
      [&](MO::GCPipelineOptions &gc) { gc.kernelLibraryPaths = {pathAStr}; });
  auto hasherB = tryBuild(
      [&](MO::GCPipelineOptions &gc) { gc.kernelLibraryPaths = {pathBStr}; });
  if (!hasherA || !hasherB)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto keyA = hasherA->keyForModule(*mod);
  auto keyB = hasherB->keyForModule(*mod);
  ASSERT_FALSE(keyA.isError());
  ASSERT_FALSE(keyB.isError());
  EXPECT_EQ(keyA.takeValue(), keyB.takeValue())
      << "key must not depend on the kernel library's absolute path";
}

TEST_F(MEFCacheKeyTest, ImportPathPackageContentInvalidatesKey) {
  // The kernel implementations live in packages resolved via the module
  // import path (`linalg`, `nn`, ...), not in the two built-in re-export
  // packages. A content change in any import-path package must move the
  // key, or a serve run after a kernel-source rebuild silently reuses the
  // stale MEF.
  std::optional<TempDir> pkgDir =
      makePackageDir("import-path package bytes v1");
  ASSERT_TRUE(pkgDir);

  AppendImportPathDir env(pkgDir->getPath().string());
  auto before = tryBuild();
  if (!before)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  {
    std::ofstream out(pkgDir->getPath() / "pkg.mojoc", std::ios::binary);
    out << "import-path package bytes v2";
  }
  auto after = tryBuild();
  if (!after)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto beforeKey = before->keyForModule(*mod);
  auto afterKey = after->keyForModule(*mod);
  ASSERT_FALSE(beforeKey.isError());
  ASSERT_FALSE(afterKey.isError());
  EXPECT_NE(beforeKey.takeValue(), afterKey.takeValue())
      << "an import-path kernel package content change must change the key";
}

TEST_F(MEFCacheKeyTest, ImportPathPackageIsRelocatable) {
  // MOCO-4152 still holds for import-path packages: identical bytes under a
  // different directory must not perturb the key.
  std::optional<TempDir> dirA =
      makePackageDir("identical import-path package bytes");
  std::optional<TempDir> dirB =
      makePackageDir("identical import-path package bytes");
  ASSERT_TRUE(dirA && dirB);
  ASSERT_NE(dirA->getPath(), dirB->getPath()) << "temp dirs must differ";

  std::optional<MEFCacheKeyHasher> hasherA, hasherB;
  {
    AppendImportPathDir env(dirA->getPath().string());
    hasherA = tryBuild();
  }
  {
    AppendImportPathDir env(dirB->getPath().string());
    hasherB = tryBuild();
  }
  if (!hasherA || !hasherB)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto keyA = hasherA->keyForModule(*mod);
  auto keyB = hasherB->keyForModule(*mod);
  ASSERT_FALSE(keyA.isError());
  ASSERT_FALSE(keyB.isError());
  EXPECT_EQ(keyA.takeValue(), keyB.takeValue())
      << "key must not depend on an import-path package's absolute path";
}

TEST_F(MEFCacheKeyTest, CasEnabledInvalidatesKey) {
  // MOCO-4152: cache-dir's path is stripped, but set-vs-unset toggles CAS
  // outlining, which changes the artifact — so the folded-in `cas-enabled` bit
  // must still move the key.
  auto plain = tryBuild();
  if (!plain)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto withCas = tryBuild(
      [](MO::GCPipelineOptions &gc) { gc.cacheDir = "/some/cas/store/path"; });
  if (!withCas)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto plainKey = plain->keyForModule(*mod);
  auto casKey = withCas->keyForModule(*mod);
  ASSERT_FALSE(plainKey.isError());
  ASSERT_FALSE(casKey.isError());
  EXPECT_NE(plainKey.takeValue(), casKey.takeValue())
      << "enabling CAS must change the key even though the path is stripped";
}

TEST_F(MEFCacheKeyTest, CacheDirPathIsRelocatable) {
  // MOCO-4152: CAS-on is captured, but the store location is not — so two
  // CAS-enabled hashers differing only by cache-dir path must match.
  auto hasherA = tryBuild(
      [](MO::GCPipelineOptions &gc) { gc.cacheDir = "/install/a/cas"; });
  auto hasherB = tryBuild(
      [](MO::GCPipelineOptions &gc) { gc.cacheDir = "/install/b/cas"; });
  if (!hasherA || !hasherB)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto keyA = hasherA->keyForModule(*mod);
  auto keyB = hasherB->keyForModule(*mod);
  ASSERT_FALSE(keyA.isError());
  ASSERT_FALSE(keyB.isError());
  EXPECT_EQ(keyA.takeValue(), keyB.takeValue())
      << "key must not depend on the CAS store's absolute path";
}

TEST_F(MEFCacheKeyTest, MlirTimingOutDirIsRelocatable) {
  // MOCO-4152: mlir-timing-out-dir is a pure diagnostic output folder with no
  // effect on the artifact, so it must not perturb the key.
  auto hasherA = tryBuild([](MO::GCPipelineOptions &gc) {
    gc.mlirTimingOutDir = "/install/a/timing";
  });
  auto hasherB = tryBuild([](MO::GCPipelineOptions &gc) {
    gc.mlirTimingOutDir = "/install/b/timing";
  });
  if (!hasherA || !hasherB)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto keyA = hasherA->keyForModule(*mod);
  auto keyB = hasherB->keyForModule(*mod);
  ASSERT_FALSE(keyA.isError());
  ASSERT_FALSE(keyB.isError());
  EXPECT_EQ(keyA.takeValue(), keyB.takeValue())
      << "key must not depend on the MLIR timing output directory's path";
}

TEST_F(MEFCacheKeyTest, NanCheckInvalidatesKey) {
  // `nan-check` is a `GCPipelineOptions` field (it gates `NanCheckPass`), so it
  // is captured structurally via `gcPipelineOptions.print()` + the textual
  // pipeline — the hasher does *not* read `max-debug.nan-check` from `Config`
  // directly. Pin that: two hashers differing only in `nanCheck` must produce
  // distinct keys for the same module.
  auto plain = tryBuild();
  if (!plain)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto withNanCheck =
      tryBuild([](MO::GCPipelineOptions &gc) { gc.nanCheck = true; });
  if (!withNanCheck)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto plainKey = plain->keyForModule(*mod);
  auto nanKey = withNanCheck->keyForModule(*mod);
  ASSERT_FALSE(plainKey.isError());
  ASSERT_FALSE(nanKey.isError());
  EXPECT_NE(plainKey.takeValue(), nanKey.takeValue());
}

TEST_F(MEFCacheKeyTest, VirtualCpuTargetInvalidatesKey) {
  // The target reaches codegen, so it must reach the key too; else a
  // host-warmed cache falsely hits under a different CPU target (up to
  // executing instructions the running CPU lacks).
  llvm::scope_exit restore{[prev = M::Driver::getVirtualCpuTarget()] {
    M::Driver::setVirtualCpuTarget(prev);
  }};

  M::Driver::setVirtualCpuTarget(""); // host
  auto host = tryBuild();
  if (!host)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  M::Driver::setVirtualCpuTarget("generic"); // portable baseline, != host CPU
  auto generic = tryBuild();
  if (!generic)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto hostKey = host->keyForModule(*mod);
  auto genericKey = generic->keyForModule(*mod);
  ASSERT_FALSE(hostKey.isError());
  ASSERT_FALSE(genericKey.isError());
  EXPECT_NE(hostKey.takeValue(), genericKey.takeValue());
}

TEST_F(MEFCacheKeyTest, VirtualCpuTargetDescriptorReachesKey) {
  // Two distinct descriptors must produce distinct keys. Both pin an x86_64
  // triple so the test is host-independent.
  llvm::scope_exit restore{[prev = M::Driver::getVirtualCpuTarget()] {
    M::Driver::setVirtualCpuTarget(prev);
  }};

  M::Driver::setVirtualCpuTarget("triple=x86_64-unknown-linux-gnu;cpu=x86-64");
  auto baseline = tryBuild();
  if (!baseline)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  M::Driver::setVirtualCpuTarget(
      "triple=x86_64-unknown-linux-gnu;cpu=znver4;features=+avx512f");
  auto znver4 = tryBuild();
  if (!znver4)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kEmptyModule);
  auto baselineKey = baseline->keyForModule(*mod);
  auto znver4Key = znver4->keyForModule(*mod);
  ASSERT_FALSE(baselineKey.isError());
  ASSERT_FALSE(znver4Key.isError());
  EXPECT_NE(baselineKey.takeValue(), znver4Key.takeValue());
}

TEST_F(MEFCacheKeyTest, ConvenienceMatchesIncremental) {
  // `computeMEFCacheKey` is documented to equal
  // `MEFCacheKeyHasher::build(...).keyForModule(...)`. Pin the
  // equivalence so any drift in the convenience surface is caught.
  auto hasher = tryBuild();
  if (!hasher)
    GTEST_SKIP() << "kernel packages unavailable; skip";

  auto mod = parse(kFuncFooModule);
  auto incrementalKey = hasher->keyForModule(*mod);
  ASSERT_FALSE(incrementalKey.isError());

  MO::GCPipelineOptions gc;
  auto convenienceKey = computeMEFCacheKey(*mod, gc, maxContext);
  ASSERT_FALSE(convenienceKey.isError());

  EXPECT_EQ(incrementalKey.takeValue(), convenienceKey.takeValue());
}
