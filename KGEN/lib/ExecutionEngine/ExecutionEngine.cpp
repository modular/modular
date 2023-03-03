//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/LowerToObject.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCEHFrameRegistrar.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#include <Windows.h>
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#else
// Just need *a* definition here. It is fully unused for non-windows builds.
static void *__ImageBase = nullptr;
#endif // _MSC_VER

using namespace M;
using namespace KGEN;
using namespace Cache;

/// A standard name (that a user is unlikely to create) that we can use for a
/// JITDylib to define platform-specific symbols we want to be in the JIT'ed
/// address space.
static constexpr StringLiteral platformStdlibName = "$platform-stdlib";

//===----------------------------------------------------------------------===//
// ExecutionEngine implementation
//===----------------------------------------------------------------------===//

M::ErrorOr<ExecutionEngine>
ExecutionEngine::create(const CompilationOptions &options) {
  ExecutionEngine ee(nullptr, options);

  // Create the target machine.
  RETURN_ERROR(KGEN::createTargetMachine(options, /*isJIT=*/false));

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

      // COFF requires __ImageBase.
      llvm::orc::JITDylib &dylib =
          session.createBareJITDylib(platformStdlibName.str());
      llvm::cantFail(dylib.define(llvm::orc::absoluteSymbols(
          {{session.intern("__ImageBase"),
            {llvm::pointerToJITTargetAddress(&__ImageBase),
             llvm::JITSymbolFlags::Exported |
                 llvm::JITSymbolFlags::Absolute}}})));
    }

    // If we don't want any debugging in this binary, then stop here.
    if (options.debugLevel == CompilationOptions::kNoDebug)
      return objectLayer;

    // Get the registrar for the GDB JIT loader interface.
    if (tt.isOSBinFormatMachO()) {
      llvm::orc::JITDylib &dylib =
          session.createBareJITDylib(platformStdlibName.str());
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
  auto dylibOr = getOrCreateDylib(name);
  if (dylibOr.isError())
    return dylibOr.takeError();

  llvm::orc::JITDylib *dylib = *dylibOr;

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
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();

  llvm::orc::JITDylib *dylib = *dylibOr;

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

ErrorOr<llvm::orc::JITDylib *>
ExecutionEngine::getOrCreateDylib(StringRef libName) {
  llvm::orc::JITDylib *dylib = jit->getJITDylibByName(libName);
  if (!dylib) {
    auto dylibOr = jit->createJITDylib(libName.str());
    if (!dylibOr)
      return M::Error(toString(dylibOr.takeError()));
    dylib = &*dylibOr;

    // Resolve symbols that are statically linked in the current process.
    dylib->addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            jit->getDataLayout().getGlobalPrefix())));

    // Make sure to expose symbols from the platform stdlib.
    llvm::orc::JITDylib *stdlib = jit->getJITDylibByName(platformStdlibName);
    if (stdlib)
      dylib->addToLinkOrder(*stdlib);
  }
  return dylib;
}
