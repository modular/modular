//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Covers the offload-ownership hooks on TargetTraits/TargetBackend: trait
// defaults, registry dispatch for a mock external-toolchain target, the
// `overrideExported` fold of `forcesExportedSymbols`, and the
// `lowerAndEmitOffload` default/override contract.
//
// What this does not cover is the driver side: the per-kernel loop in
// `compileOffloads` that reads these hooks. No in-tree backend owns offload
// lowering yet, so there is no lit suite covering it either, and
// `ObjectCompiler::create` wants `modular.cfg`, a cache dir and `lld`, which
// a gtest cannot supply. Tracked in MOCO-4850.
//
//===----------------------------------------------------------------------===//

#include "Mojo/Compiler/Target/TargetBackend.h"
// Completes the forward-declared `Cache::TransformCache` so this TU can
// construct and destroy `OffloadEmitContext`.
#include "Cache/CachedTransform.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Target/TargetTraits.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::KGEN;

namespace {

// A fake accelerator whose device artifact comes from an external toolchain.
constexpr llvm::StringLiteral kMockTriplePrefix = "kgenmock-";

struct MockOffloadTraits final : TargetTraits {
  llvm::StringRef name() const override { return "kgenmock"; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.str().starts_with(kMockTriplePrefix);
  }
  bool forcesExportedSymbols() const override { return true; }
  bool emitsOffloadObjectFile() const override { return false; }
  llvm::StringRef getAsmExtension() const override { return ".mock.mlir"; }
  llvm::StringRef getLLVMExtension() const override { return ".mock.mlir"; }
  llvm::StringRef getObjectExtension() const override { return ".mockbin"; }
  llvm::StringRef getBitcodeExtension() const override { return ".mock.bc"; }
  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }
  bool isBaseTarget() const override { return true; }
};

struct MockOffloadBackend final : TargetBackend {
  const TargetTraits *traits() const override {
    static MockOffloadTraits traits;
    return &traits;
  }
  bool ownsOffloadLowering() const override { return true; }
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  lowerAndEmitOffload(mlir::Operation *module,
                      const OffloadEmitContext &ctx) const override {
    return std::unique_ptr<llvm::MemoryBuffer>(
        llvm::MemoryBuffer::getMemBufferCopy("mock-artifact:" +
                                             ctx.options.targetCpu));
  }
  SplitStrategy splitStrategy(const CompilationOptions &) const override {
    return SplitStrategy::None;
  }
  bool isBaseTarget() const override { return true; }
  ErrorOr<BufferRef> emitAssembly(llvm::Module &,
                                  EmitContext &) const override {
    return Error("mock target does not emit LLVM assembly");
  }
  ErrorOr<BufferRef> emitObject(llvm::Module &, EmitContext &) const override {
    return Error("mock target does not emit LLVM objects");
  }
  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef>,
                                   llvm::StringRef,
                                   EmitContext &) const override {
    return Error("mock target does not create archives");
  }
};

// Registries are process-global with no removal API; register once for every
// test in this binary.
struct RegisterMocks {
  RegisterMocks() {
    TargetTraitsRegistry::get().add(std::make_unique<MockOffloadTraits>());
    TargetBackendRegistry::get().add(std::make_unique<MockOffloadBackend>());
  }
};
RegisterMocks registerMocks;

TEST(TargetBackendTest, TraitsRegistryDispatch) {
  llvm::Triple mockTriple("kgenmock-none-unknown");
  ErrorOr<const TargetTraits *> traitsOr =
      TargetTraitsRegistry::get().lookup(mockTriple);
  ASSERT_FALSE(traitsOr.isError());
  const TargetTraits *traits = *traitsOr;
  EXPECT_EQ(traits->name(), "kgenmock");
  EXPECT_TRUE(traits->forcesExportedSymbols());
  EXPECT_FALSE(traits->emitsOffloadObjectFile());
}

TEST(TargetBackendTest, OffloadTraitDefaultsAreOff) {
  MockOffloadTraits mock;
  // Compare against the base-class defaults via a plain traits object.
  struct PlainTraits final : TargetTraits {
    llvm::StringRef name() const override { return "plain"; }
    bool matches(const llvm::Triple &) const override { return false; }
    llvm::StringRef getAsmExtension() const override { return ".s"; }
    llvm::StringRef getLLVMExtension() const override { return ".ll"; }
    llvm::StringRef getObjectExtension() const override { return ".o"; }
    llvm::StringRef getBitcodeExtension() const override { return ".bc"; }
    llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
      return commonEmissionKinds();
    }
    bool isBaseTarget() const override { return true; }
  } plain;
  EXPECT_FALSE(plain.forcesExportedSymbols());
  EXPECT_TRUE(mock.forcesExportedSymbols());
}

TEST(TargetBackendTest, OverrideExportedFoldsForcedSymbols) {
  EXPECT_TRUE(overrideExported(llvm::Triple("kgenmock-none-unknown")));
  EXPECT_FALSE(overrideExported(llvm::Triple("x86_64-unknown-linux-gnu")));
}

TEST(TargetBackendTest, BackendRegistryDispatchAndOwnership) {
  llvm::Triple mockTriple("kgenmock-none-unknown");
  ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(mockTriple);
  ASSERT_FALSE(backendOr.isError());
  const TargetBackend *backend = *backendOr;
  EXPECT_TRUE(backend->ownsOffloadLowering());

  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  CompilationOptions options;
  // The arch reaches the backend through the context, not a separate argument.
  options.targetCpu = "mockarch";
  OffloadEmitContext emitCtx{options, module->getLoc()};
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> artifactOr =
      backend->lowerAndEmitOffload(*module, emitCtx);
  ASSERT_FALSE(artifactOr.isError());
  EXPECT_EQ((*artifactOr)->getBuffer(), "mock-artifact:mockarch");
}

TEST(TargetBackendTest, LowerAndEmitOffloadDefaultErrors) {
  // A backend that does not opt into offload ownership keeps the erroring
  // default implementation.
  struct PlainBackend final : TargetBackend {
    SplitStrategy splitStrategy(const CompilationOptions &) const override {
      return SplitStrategy::None;
    }
    bool isBaseTarget() const override { return true; }
    ErrorOr<BufferRef> emitAssembly(llvm::Module &,
                                    EmitContext &) const override {
      return Error("unused");
    }
    ErrorOr<BufferRef> emitObject(llvm::Module &,
                                  EmitContext &) const override {
      return Error("unused");
    }
    ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef>,
                                     llvm::StringRef,
                                     EmitContext &) const override {
      return Error("unused");
    }
  } plain;
  EXPECT_FALSE(plain.ownsOffloadLowering());

  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  CompilationOptions options;
  OffloadEmitContext emitCtx{options, module->getLoc()};
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> artifactOr =
      plain.lowerAndEmitOffload(*module, emitCtx);
  EXPECT_TRUE(artifactOr.isError());
}

// Every offload compile derives its options from the group through one helper,
// so a per-kernel target cannot silently miss what the bundled path applies.
TEST(TargetBackendTest, OffloadEmissionOptionsCarryLinkAndFpMode) {
  CompilationOptions options;
  FpMode hostFpMode = options.fpMode;

  EXPECT_FALSE(applyOffloadEmissionOptions("target-abi=lp64d,some-opt",
                                           "-lfoo,-lbar", hostFpMode, options)
                   .has_value());
  EXPECT_EQ(options.emissionLinkOptions, "-lfoo,-lbar");
  // `target-abi` is intercepted but stays in the string: it identifies the
  // offload in debug output and cache keys.
  EXPECT_EQ(options.targetABI, "lp64d");
  EXPECT_EQ(options.emissionOptions, "target-abi=lp64d,some-opt");
}

TEST(TargetBackendTest, OffloadEmissionOptionsStripFpMode) {
  CompilationOptions options;
  FpMode hostFpMode;
  hostFpMode.contract = true;

  // fp-mode is not a registered cl option, so it must not survive into the
  // string that later reaches `parseEmissionOptions`.
  EXPECT_FALSE(
      applyOffloadEmissionOptions("contract=off", "", hostFpMode, options)
          .has_value());
  EXPECT_EQ(options.emissionOptions, "");
  EXPECT_FALSE(options.fpMode.contract);

  EXPECT_EQ(
      applyOffloadEmissionOptions("contract=bogus", "", hostFpMode, options),
      "contract=bogus");
}

} // namespace
