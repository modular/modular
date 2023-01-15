//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LowerToObject.h"
#include "Support/ErrorOr.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"

#include <filesystem>
#include <mutex>

using namespace M;
using namespace KGEN;
using namespace Cache;

/// Setup the machine properties from the current architecture.
static ErrorOr<std::unique_ptr<llvm::TargetMachine>>
createHostTargetMachine(const CompilationOptions &options) {
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    return Error("no target exists for '" + targetTriple +
                 "': " + errorMessage);

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_, /*CM=*/std::nullopt,
      /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/true));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

//===----------------------------------------------------------------------===//
// ExecutionEngine implementation
//===----------------------------------------------------------------------===//

M::ErrorOr<ExecutionEngine>
ExecutionEngine::create(const CompilationOptions &options) {
  ExecutionEngine ee(nullptr, options);

  // Ensure the native target is initialized.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm

  // Create the target machine.
  RETURN_ERROR(createHostTargetMachine(options));

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
                                       const llvm::Triple &tt) {
    auto objectLayer =
        std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, []() {
          return std::make_unique<llvm::SectionMemoryManager>();
        });

    if (options.debugLevel == CompilationOptions::kNoDebug)
      return objectLayer;

    // Register JIT event listeners if they are enabled.
    if (ee.gdbListener)
      objectLayer->registerJITEventListener(*ee.gdbListener);
    if (ee.perfListener)
      objectLayer->registerJITEventListener(*ee.perfListener);

    // Make sure the debug info sections aren't stripped.
    objectLayer->setProcessAllSections(true);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    if (tt.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    return objectLayer;
  };

  // Create the JIT.
  auto jitOr = llvm::orc::LLJITBuilder()
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .create();
  if (!jitOr)
    return M::Error(llvm::toString(jitOr.takeError()));

  ee.jit = std::move(*jitOr);
  return ee;
}

ExecutionEngine::ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit,
                                 CompilationOptions options)
    : options(options), ctx(std::make_unique<llvm::LLVMContext>()),
      jit(std::move(jit)),
      gdbListener(llvm::JITEventListener::createGDBRegistrationListener()),
      perfListener(nullptr) {
  // Attach the perf listener.
  if (auto *listener = llvm::JITEventListener::createPerfJITEventListener())
    perfListener = listener;
  else if (auto *listener =
               llvm::JITEventListener::createIntelJITEventListener())
    perfListener = listener;
}

ExecutionEngine::~ExecutionEngine() = default;
ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

ErrorOrSuccess ExecutionEngine::add(StringRef name, BufferRef obj) {
  // Create a new dylib so that we don't have ODR violations.
  auto dylibOr = jit->createJITDylib(name.str());
  if (!dylibOr)
    return M::Error(toString(dylibOr.takeError()));

  // Resolve symbols that are statically linked in the current process.
  dylibOr->addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix())));

  // If the addObjectFile succeeds we store a ref to this buffer so the data
  // won't be deallocated until the JIT is destroyed. This version of
  // MemoryBuffer::getMemBuffer produces a non-owning buffer.
  std::unique_ptr<llvm::MemoryBuffer> objMemBuf =
      llvm::MemoryBuffer::getMemBuffer(obj->getBuffer(), /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  if (auto err = jit->addObjectFile(*dylibOr, std::move(objMemBuf)))
    return M::Error(toString(std::move(err)));

  // Store a ref to the buffer data.
  objBuffers.push_back(obj.copy());

  return success();
}

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef libName,
                                              StringAttr symbol) {
  auto *dylib = jit->getJITDylibByName(libName);
  if (!dylib)
    return Error("could not find JITDylib for " + libName);

  auto addr = jit->lookup(*dylib, symbol.getValue());
  if (!addr)
    return M::Error(toString(addr.takeError()));

  return CompiledFunc(addr->toPtr<void *>());
}
