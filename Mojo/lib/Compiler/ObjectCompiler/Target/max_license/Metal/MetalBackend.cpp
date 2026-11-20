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
// TargetBackend for Apple Metal (air64): writes AIR bitcode and links a
// metallib via `xcrun metal`, or packages it in-tree (MetallibWriter) when the
// Metal toolchain is absent or MODULAR_METAL_INTREE_METALLIB is set.
//
//===----------------------------------------------------------------------===//

#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/Compiler/SaveAsmOutput.h"
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Mojo/ToolCommon/CompilationOptions.h"

#include "Target/max_license/Metal/LLVM/Transforms/InstructionRewrite.h"
#include "Target/max_license/Metal/LLVM/Transforms/MetalAIRPass.h"
#include "Target/max_license/Metal/LLVM/Transforms/MetalModuleRewritePass.h"
#include "Target/max_license/Metal/LLVM/Transforms/MetalRewriteDebugInfo.h"
#include "Target/max_license/Metal/LLVM/Transforms/MetalVerifier.h"
#include "Target/max_license/Metal/MetalAddressSpace.h"
#include "Target/max_license/Metal/MetalTraits.h"
#include "Target/max_license/Metal/MetallibWriter.h"
#include "LLVM/Transforms/PointerRewriter.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

namespace {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
llvm::cl::opt<std::string> objectCompilerUseCustomAIR(
    "kgen-object-compiler-use-custom-air",
    llvm::cl::desc(
        "Specify custom AIR file to compile a program. The custom AIR file is "
        "not removed after compilation. WARNING: to run the program AIR file "
        "must contain all kernels used by it."));
#pragma GCC diagnostic pop

constexpr const char *kXcrunPath = "/usr/bin/xcrun";

constexpr const char *kInTreeMetallibEnv = "MODULAR_METAL_INTREE_METALLIB";

/// Packages AIR in-tree when asked to, or when there is no Metal toolchain to
/// fall back on (e.g. cross-compiling from Linux). An empty value counts as
/// unset so templated CI configs still get the fallback.
ErrorOr<bool> useInTreeMetallib() {
  std::optional<std::string> env =
      llvm::sys::Process::GetEnv(kInTreeMetallibEnv);
  if (!env || env->empty())
    return !llvm::sys::fs::can_execute(kXcrunPath);
  std::string value = llvm::StringRef(*env).lower();
  if (value == "1" || value == "true" || value == "on" || value == "yes")
    return true;
  if (value == "0" || value == "false" || value == "off" || value == "no")
    return false;
  return Error(Twine(kInTreeMetallibEnv) + "='" + *env +
               "' is not a boolean; use 1/0, true/false, on/off or yes/no");
}

/// Rejects a MACOSX_DEPLOYMENT_TARGET the AIR cannot run on, as `xcrun metal`
/// does, so the failure happens at build time instead of at library load.
ErrorOrSuccess checkDeploymentTarget(const llvm::Module &module) {
  std::optional<std::string> env =
      llvm::sys::Process::GetEnv("MACOSX_DEPLOYMENT_TARGET");
  if (!env || env->empty())
    return success();
  llvm::VersionTuple target;
  if (target.tryParse(*env))
    return Error(Twine("invalid version number in 'MACOSX_DEPLOYMENT_TARGET=") +
                 *env + "'");
  llvm::VersionTuple required =
      llvm::Triple(module.getTargetTriple()).getOSVersion();
  if (target < required)
    return Error(Twine("MACOSX_DEPLOYMENT_TARGET=") + *env +
                 " is older than macOS " + required.getAsString() +
                 ", which this kernel's AIR version requires");
  return success();
}

/// Maps an `xcrun metal` error message to actionable guidance.
std::string getPrettyMetallibError(llvm::StringRef metalCompilerErrorMessage,
                                   llvm::Module &module) {
  if (metalCompilerErrorMessage.contains("is using language version ") ||
      metalCompilerErrorMessage.contains("air version set to")) {
    unsigned metalMajor = 3;
    if (llvm::NamedMDNode *md =
            module.getNamedMetadata("air.language_version")) {
      if (md->getNumOperands() > 0) {
        llvm::MDNode *node = md->getOperand(0);
        if (node->getNumOperands() >= 2) {
          if (auto *majorMD =
                  dyn_cast<llvm::ConstantAsMetadata>(node->getOperand(1)))
            metalMajor =
                cast<llvm::ConstantInt>(majorMD->getValue())->getZExtValue();
        }
      }
    }
    if (metalMajor >= 4) {
      return "Targeting Metal 4.0 requires Xcode with Metal 4.0 support and "
             "macOS 26.0 or later.\nIf after update you still see compilation "
             "error, please submit a bug report.";
    }
    return "Please make sure Xcode version is at least 16.0 and macOS is at "
           "least 15.0.\nIf after update you still see compilation error, "
           "please submit a bug report.";
  }
  if (metalCompilerErrorMessage.contains("unable to find utility \"metallib\""))
    return "Please make sure Xcode is installed and setup correctly\n" +
           metalCompilerErrorMessage.str();
  if (metalCompilerErrorMessage.contains("missing Metal Toolchain"))
    return "Please make sure Xcode metal toolchain is installed and setup "
           "correctly\n" +
           metalCompilerErrorMessage.str();
  if (metalCompilerErrorMessage.contains("ignoring file"))
    return "Metal Compiler dropped the AIR file from the metallib due to:\n" +
           metalCompilerErrorMessage.str();
  return "Metal Compiler failed to compile metallib. Please submit a bug "
         "report.\n" +
         metalCompilerErrorMessage.str();
}

class MetalBackend final : public TargetBackend {
public:
  const TargetTraits *traits() const override { return &MetalTraits::get(); }

  SplitStrategy splitStrategy(const CompilationOptions &) const override {
    return SplitStrategy::None;
  }
  bool isCodegenInterprocedural() const override { return true; }
  bool isOffload() const override { return true; }

  std::optional<unsigned> sharedMemoryAddressSpace() const override {
    return MetalAddressSpace::ThreadGroup;
  }

  // GPU branches are far costlier than on CPU, so raise the SimplifyCFG bonus.
  llvm::SimplifyCFGOptions
  adjustSimplifyCFGOptions(llvm::SimplifyCFGOptions options) const override {
    return options.bonusInstThreshold(2);
  }

  // AddressSanitizer is supported, but GPUs don't run the host ThreadSanitizer.
  void addSanitizers(llvm::ModulePassManager &mpm,
                     const CompilationOptions &options) const override {
    addAddressSanitizerPass(mpm, options);
  }

  // Recursion is not supported for GPU kernel entry functions (same as clang).
  void attachCodegenAttributes(llvm::Function *kernelEntry) const override {
    kernelEntry->addFnAttr(llvm::Attribute::NoRecurse);
  }

  // Metal consumes AIR bitcode (LLVM 17 format), not standard LLVM bitcode.
  void emitBitcode(llvm::Module &module,
                   llvm::raw_pwrite_stream &os) const override {
    M::KGEN::LLVM::WriteBitcode17ToFile(module, os,
                                        /*ShouldPreserveUseListOrder=*/false,
                                        /*ModuleSummaryIndex=*/nullptr,
                                        /*GenerateHash=*/false,
                                        /*ModuleHash=*/nullptr);
  }

  // Metal fully owns its pipeline: a fixed AIR legalization sequence that
  // replaces the standard optimization pipeline (opt-level independent).
  bool buildLLVMPipeline(llvm::ModulePassManager &mpm, llvm::PassBuilder &,
                         const CompilationOptions &) const override {
    // Initial LLVM IR must be correct for Metal.
    mpm.addPass(MetalVerifierPass());
    // Module-level lowering before per-function rewrites: inlines
    // static_string_* globals and replaces print sentinel calls.
    mpm.addPass(MetalModuleRewritePass());
    llvm::FunctionPassManager fpm;
    // Rewrite all Metal-unsupported LLVM IR intrinsics and instructions.
    fpm.addPass(InstructionRewritePass());
    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
    // Address space conversions.
    mpm.addPass(MetalAIRPass());
    // Add bitcasts for typed IR.
    mpm.addPass(PointerRewriter());
    // Rewrite debuginfo to what Instruments expects (for now just strip it).
    mpm.addPass(MetalRewriteDebugInfoPass());
    // Verify that IR is now correct to the best of our knowledge.
    mpm.addPass(MetalVerifierPass());
    return true;
  }

  // Registers this backend's `-passes` names for kgen-llvm-opt:
  //   kgen-metal-air           Transform IR to Apple AIR for Metal GPU
  //   kgen-pointer-rewriter    Rewrite opaque pointers to typed pointers
  //   kgen-metal-verifier      Reject IR with float types wider than f32
  //   kgen-metal-rewrite-di    Rewrite DebugInfo for Metal/Instruments
  //   kgen-instruction-rewrite Rewrite unsupported intrinsics/instructions
  void registerPipelinePasses(llvm::PassBuilder &passBuilder) const override {
    passBuilder.registerPipelineParsingCallback(
        [](llvm::StringRef name, llvm::ModulePassManager &mpm,
           llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
          if (name == "kgen-metal-air") {
            mpm.addPass(MetalAIRPass());
            return true;
          }
          if (name == "kgen-pointer-rewriter") {
            mpm.addPass(PointerRewriter());
            return true;
          }
          if (name == "kgen-metal-verifier") {
            mpm.addPass(MetalVerifierPass());
            return true;
          }
          if (name == "kgen-metal-rewrite-di") {
            mpm.addPass(MetalRewriteDebugInfoPass());
            return true;
          }
          return false;
        });
    passBuilder.registerPipelineParsingCallback(
        [](llvm::StringRef name, llvm::FunctionPassManager &fpm,
           llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
          if (name == "kgen-instruction-rewrite") {
            fpm.addPass(InstructionRewritePass());
            return true;
          }
          return false;
        });
  }

  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const override {
    CompilationOptions adjusted = options;
    // air64 needs arm64 for TargetMachine; drop Metal-specific features.
    adjusted.targetTriple = "arm64" + moduleTriple.substr(5).str();
    adjusted.targetFeatures = "";
    return adjusted;
  }

  void finalizeModuleForTarget(llvm::Module &module, llvm::TargetMachine &tm,
                               llvm::StringRef originalTriple) const override {
    module.setDataLayout(tm.createDataLayout());
    // Restore the original air64 triple for GPU.
    module.setTargetTriple(llvm::Triple(originalTriple));
  }

  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &) const override {
    // LLVM IR text is the closest readable equivalent to Metal assembly.
    WriteableBufferRef buf = WriteableBuffer::get();
    *buf << module;
    return buf;
  }

  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override {
    const CompilationOptions &options = ctx.options;

    // Write AIR bitcode using the Metal bitcode writer.
    llvm::SmallString<256> airTempFile;
    if (auto err =
            llvm::sys::fs::createTemporaryFile("kernel", ".air", airTempFile)) {
      return Error("Failed to create temporary AIR file: " + err.message());
    }
    llvm::FileRemover airRemover(airTempFile);

    std::error_code airEc;
    llvm::raw_fd_ostream airOS(airTempFile, airEc, llvm::sys::fs::OF_None);
    if (airEc)
      return Error("Failed to open AIR file for writing: " + airEc.message());

    M::KGEN::LLVM::WriteBitcode17ToFile(module, airOS,
                                        /*ShouldPreserveUseListOrder=*/false,
                                        /*ModuleSummaryIndex=*/nullptr,
                                        /*GenerateHash=*/false,
                                        /*ModuleHash=*/nullptr);
    airOS.flush();
    airOS.close();

    if (!options.saveTempsPrefix.empty()) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> airBuffer =
          llvm::MemoryBuffer::getFile(airTempFile);
      if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, ".air",
                                          *airBuffer.get())))
        return Error("failed to save air to saveTempsPrefix");
    }

    ErrorOr<bool> inTree = useInTreeMetallib();
    if (inTree.isError())
      return Error(Twine(inTree.getError()));
    if (*inTree) {
      if (!objectCompilerUseCustomAIR.getValue().empty())
        return Error("-kgen-object-compiler-use-custom-air needs the `xcrun "
                     "metal` packager; set " +
                     Twine(kInTreeMetallibEnv) + "=0 on a host with Xcode");
      if (ErrorOrSuccess err = checkDeploymentTarget(module))
        return Error(Twine(err.getError()));
      auto airBuffer = llvm::MemoryBuffer::getFile(airTempFile);
      if (!airBuffer)
        return Error("Failed to read AIR file: " +
                     airBuffer.getError().message());
      llvm::Expected<std::string> metallib =
          writeMetallib(module, (*airBuffer)->getBuffer());
      if (!metallib)
        return Error("Failed to package metallib: " +
                     llvm::toString(metallib.takeError()));
      if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, ".metallib",
                                          llvm::StringRef(*metallib))))
        return Error("failed to save metallib to saveTempsPrefix");
      WriteableBufferRef buf = WriteableBuffer::get();
      (*buf) << *metallib;
      return buf;
    }

    // Convert AIR to metallib using `xcrun metallib`.
    llvm::SmallString<256> metallibTempFile;
    if (auto err = llvm::sys::fs::createTemporaryFile("kernel", ".metallib",
                                                      metallibTempFile)) {
      return Error("Failed to create temporary metallib file: " +
                   err.message());
    }
    llvm::FileRemover metallibRemover(metallibTempFile);

    const char *xcrunPath = kXcrunPath;
    const std::string optLevel =
        std::string("-O") + std::to_string(options.optimizationLevel);

    std::vector<llvm::StringRef> metallibArgs = {
        xcrunPath, "-sdk",      "macosx", "metal",
        optLevel,  airTempFile, "-o",     metallibTempFile,
    };

    // The custom AIR option (a static) outlives metallibArgs' StringRefs.
    const std::string &customAIR = objectCompilerUseCustomAIR.getValue();
    if (!customAIR.empty()) {
      llvm::errs() << "WARNING: Using custom AIR file: " << customAIR << '\n';
      metallibArgs = {
          xcrunPath, "-sdk",    "macosx", "metal",
          optLevel,  customAIR, "-o",     metallibTempFile,
      };
    }

    std::string errorMsg;
    llvm::SmallString<256> metallibErrorFile;
    llvm::FileRemover metallibErrorRemover;
    if (llvm::sys::fs::createTemporaryFile("metallib_error", "log",
                                           metallibErrorFile))
      metallibErrorFile = "";
    else
      metallibErrorRemover.setFile(metallibErrorFile);
    int result = llvm::sys::ExecuteAndWait(
        metallibArgs[0], metallibArgs, /*env=*/std::nullopt,
        /*redirects=*/
        {/*stdin=*/std::nullopt, /*stdout=*/std::nullopt,
         /*stderr=*/metallibErrorFile.empty()
             ? std::nullopt
             : std::make_optional(metallibErrorFile)},
        /*secondsToWait=*/60, /*memoryLimit=*/0, &errorMsg);

    std::string metallibStderr;
    if (!metallibErrorFile.empty()) {
      if (auto metallibErrorBuffer =
              llvm::MemoryBuffer::getFile(metallibErrorFile))
        metallibStderr = metallibErrorBuffer.get()->getBuffer().str();
    }

    // `xcrun metal` can exit 0 after reporting a failure (e.g. a dropped AIR
    // input) on stderr, and a clean compile prints nothing, so stderr fails.
    if (!metallibStderr.empty())
      return Error(getPrettyMetallibError(metallibStderr, module));
    if (result != 0)
      return Error("xcrun metallib compilation failed with code " +
                   llvm::Twine(result).str() + ": " + errorMsg);

    auto metallibBuffer = llvm::MemoryBuffer::getFile(metallibTempFile);
    if (!metallibBuffer)
      return Error("Failed to read metallib file: " +
                   metallibBuffer.getError().message());
    if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, ".metallib",
                                        *metallibBuffer.get())))
      return Error("failed to save metallib to saveTempsPrefix");

    WriteableBufferRef buf = WriteableBuffer::get();
    (*buf) << (*metallibBuffer)->getBuffer();
    return buf;
  }

  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef moduleName,
                                   EmitContext &ctx) const override {
    // A metallib is a single self-contained artifact.
    if (objects.size() != 1)
      return Error("metal backend expects a single object to archive");
    return objects[0];
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<MetalBackend> registerMetalBackend;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
