//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/LowerToObject.h"
#include "Support/ErrorOr.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCEHFrameRegistrar.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

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
                                       const llvm::Triple &tt)
      -> std::unique_ptr<llvm::orc::ObjectLinkingLayer> {
    auto objectLayer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    if (tt.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    // If we don't want any debugging in this binary, then stop here.
    if (options.debugLevel == CompilationOptions::kNoDebug)
      return objectLayer;

    // Get the registrar for the GDB JIT loader interface.
    if (tt.isOSBinFormatMachO()) {
      llvm::orc::JITDylib &dylib =
          session.createBareJITDylib("$platform-stdlib");
      // We have to explicitly define these wrapper symbols on macOS because
      // they're hidden visibility.
      cantFail(dylib.define(llvm::orc::absoluteSymbols(
          {{session.intern("_llvm_orc_registerJITLoaderGDBWrapper"),
            {llvm::pointerToJITTargetAddress(
                 &llvm_orc_registerJITLoaderGDBWrapper),
             llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute}},
           {session.intern("_llvm_orc_registerJITLoaderGDBAllocAction"),
            {llvm::pointerToJITTargetAddress(
                 &llvm_orc_registerJITLoaderGDBAllocAction),
             llvm::JITSymbolFlags::Exported |
                 llvm::JITSymbolFlags::Absolute}}})));

      objectLayer->addPlugin(
          cantFail(llvm::orc::GDBJITDebugInfoRegistrationPlugin::Create(
              session, dylib, tt)));
    } else if (tt.isOSBinFormatELF()) {
      // Register the DebugObjectManagerPlugin.
      objectLayer->addPlugin(
          std::make_unique<llvm::orc::DebugObjectManagerPlugin>(
              session,
              cantFail(llvm::orc::createJITLoaderGDBRegistrar(session))));
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
    : options(options), jit(std::move(jit)) {}

ExecutionEngine::~ExecutionEngine() = default;
ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

ErrorOrSuccess ExecutionEngine::add(StringRef name, BufferRef obj) {
  llvm::orc::JITDylib *dylib = jit->getJITDylibByName(name);
  if (!dylib) {
    // Create a new dylib so that we don't have ODR violations.
    auto dylibOr = jit->createJITDylib(name.str());
    if (!dylibOr)
      return M::Error(toString(dylibOr.takeError()));
    dylib = &*dylibOr;

    // Resolve symbols that are statically linked in the current process.
    dylib->addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            jit->getDataLayout().getGlobalPrefix())));
  }

  // If the addObjectFile succeeds we store a ref to this buffer so the data
  // won't be deallocated until the JIT is destroyed. This version of
  // MemoryBuffer::getMemBuffer produces a non-owning buffer.
  std::unique_ptr<llvm::MemoryBuffer> objMemBuf =
      llvm::MemoryBuffer::getMemBuffer(obj->getBuffer(), /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  if (auto err = jit->addObjectFile(*dylib, std::move(objMemBuf)))
    return M::Error(toString(std::move(err)));

  // Store a ref to the buffer data.
  objBuffers.push_back(obj.copy());

  return success();
}

// TODO (8082): This should not be necessary.
ErrorOrSuccess ExecutionEngine::add(StringRef libName, StringRef functionName,
                                    void *fn) {
  llvm::orc::JITDylib *dylib = jit->getJITDylibByName(libName);
  if (!dylib) {
    // Create a new dylib so that we don't have ODR violations.
    auto dylibOr = jit->createJITDylib(libName.str());
    if (!dylibOr)
      return M::Error(toString(dylibOr.takeError()));
    dylib = &*dylibOr;

    // Resolve symbols that are statically linked in the current process.
    dylib->addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            jit->getDataLayout().getGlobalPrefix())));
  }

  if (auto err = dylib->define(llvm::orc::absoluteSymbols(
          {{jit->mangleAndIntern(functionName),
            {llvm::pointerToJITTargetAddress(fn),
             llvm::JITSymbolFlags::Exported |
                 llvm::JITSymbolFlags::Absolute}}}))) {
    return Error(toString(std::move(err)));
  }

  return success();
}

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef libName,
                                              StringRef symbol) {
  auto *dylib = jit->getJITDylibByName(libName);
  if (!dylib)
    return Error("could not find JITDylib for " + libName);

  auto addr = jit->lookup(*dylib, symbol);
  if (!addr)
    return M::Error(toString(addr.takeError()));

  return CompiledFunc(addr->toPtr<void *>());
}
