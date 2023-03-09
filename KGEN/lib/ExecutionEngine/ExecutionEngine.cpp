//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/ExecutionEngine/ORCCASID.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Algorithms.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/ExecutionEngine/Orc/COFFPlatform.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCEHFrameRegistrar.h"
#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

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

namespace {
/// Provide a key that doesn't do any hashing - we only want to read things from
/// keys provided to this.
struct ReadOnlyKey {
  using KeyTy = StringRef;
  static std::string hashKey(KeyTy key) { return key.str(); }
};
} // namespace

M::ErrorOr<ExecutionEngine>
ExecutionEngine::create(const CompilationOptions &options) {
  // Create a BlobCache ref.
  RuntimeAndCache<ReadOnlyKey> runtimeAndCache(".kgen_cache/orc");
  if (auto err = runtimeAndCache.setup())
    return err.takeError();

  BlobCache<ReadOnlyKey> &orcCache = runtimeAndCache.getCache();
  AsyncValueRef<std::optional<BufferRef>> orcRTBuf =
      orcCache.find(MODULAR_ORC_RT_CAS_ID);

  ExecutionEngine ee(nullptr, options);

  // Create the target machine.
  auto tmOr = KGEN::createTargetMachine(options, /*isJIT=*/false);
  if (tmOr.isError())
    return tmOr.takeError();
  std::unique_ptr<llvm::TargetMachine> tm = std::move(*tmOr);

  // Write the orc_rt file to a temporary path so we can pass that path. This is
  // a temporary work-around until COFF can be called with a memory buffer.
  std::error_code ec;
  std::filesystem::path path = std::filesystem::temp_directory_path(ec);
  if (ec)
    return Error(ec.message());

  path = path / "liborc_rt.a";

  // Write the object to the path.
  LLCL::await(orcRTBuf);
  if (orcRTBuf.isError())
    return std::move(orcRTBuf.takeDiagnostic().getMessage());
  std::optional<BufferRef> rtBuf = std::move(*orcRTBuf);
  if (rtBuf) {
    // Write the runtime to the temp file.
    llvm::raw_fd_ostream tmp(path.string().c_str(), ec);
    if (ec)
      return Error(ec.message());

    tmp << (*rtBuf)->getBuffer();
  }

  // Define an optional error we can set to something if we hit an error in a
  // nested closure.
  std::optional<Error> outError = std::nullopt;

  // Setup the platform.
  auto setupPlatform = [&](llvm::orc::ExecutionSession &session,
                           llvm::orc::ObjectLinkingLayer &objLinkingLayer) {
    llvm::orc::JITDylib &platformStdlib =
        session.createBareJITDylib(platformStdlibName.str());
    const llvm::Triple &tt = session.getTargetTriple();
    if (rtBuf && tt.isOSBinFormatMachO()) {
      if (auto platform = llvm::orc::MachOPlatform::Create(
              session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
              platformStdlib, path.string().c_str()))
        session.setPlatform(std::move(*platform));
      else
        outError = Error(toString(platform.takeError()));
    } else if (rtBuf && tt.isOSBinFormatELF()) {
      if (auto platform = llvm::orc::ELFNixPlatform::Create(
              session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
              platformStdlib, path.string().c_str()))
        session.setPlatform(std::move(*platform));
      else
        outError = Error(toString(platform.takeError()));
    } else if (rtBuf && tt.isOSBinFormatCOFF()) {
      // Windows needs some help to load dylibs, apparently.
      auto loadDynamicLibrary = [tt, &tm](llvm::orc::JITDylib &jd,
                                          StringRef dllName) -> llvm::Error {
        if (!dllName.endswith_insensitive(".dll"))
          return llvm::make_error<llvm::StringError>(
              "DLLName not ending with .dll", llvm::inconvertibleErrorCode());

        if (auto dylibGeneratorOr =
                llvm::orc::DynamicLibrarySearchGenerator::Load(
                    dllName.data(), tm->createDataLayout().getGlobalPrefix()))
          jd.addGenerator(std::move(*dylibGeneratorOr));
        else
          return dylibGeneratorOr.takeError();
        return llvm::Error::success();
      };

      if (auto platform = llvm::orc::COFFPlatform::Create(
              session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
              platformStdlib, path.string().c_str(), loadDynamicLibrary))
        session.setPlatform(std::move(*platform));
      else
        outError = Error(toString(platform.takeError()));
    }
  };

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
                                       const llvm::Triple &tt)
      -> std::unique_ptr<llvm::orc::ObjectLinkingLayer> {
    auto objectLayer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);

    // Set up the platform support now that we have an object layer.
    setupPlatform(session, *objectLayer);
    // Bail if we hit an error.
    if (outError)
      return nullptr;

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    if (tt.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    // If we don't want any debugging in this binary, then stop here.
    if (options.debugLevel == CompilationOptions::kNoDebug)
      return objectLayer;

    auto returnError = [&](llvm::Error &&err) {
      outError = Error(toString(std::move(err)));
      return nullptr;
    };

    // Get the registrar for the GDB JIT loader interface.
    if (tt.isOSBinFormatMachO()) {
      llvm::orc::JITDylib &dylib =
          *session.getJITDylibByName(platformStdlibName);
      // We have to explicitly define these wrapper symbols on macOS because
      // they're hidden visibility.
      auto err = dylib.define(llvm::orc::absoluteSymbols(
          {{session.intern("_llvm_orc_registerJITLoaderGDBWrapper"),
            {llvm::pointerToJITTargetAddress(
                 &llvm_orc_registerJITLoaderGDBWrapper),
             llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute}},
           {session.intern("_llvm_orc_registerJITLoaderGDBAllocAction"),
            {llvm::pointerToJITTargetAddress(
                 &llvm_orc_registerJITLoaderGDBAllocAction),
             llvm::JITSymbolFlags::Exported |
                 llvm::JITSymbolFlags::Absolute}}}));
      if (err)
        return returnError(std::move(err));

      if (auto plugin = llvm::orc::GDBJITDebugInfoRegistrationPlugin::Create(
              session, dylib, tt))
        objectLayer->addPlugin(std::move(*plugin));
      else
        return returnError(plugin.takeError());
    } else if (tt.isOSBinFormatELF()) {
      // Register the DebugObjectManagerPlugin.
      if (auto plugin = llvm::orc::createJITLoaderGDBRegistrar(session)) {
        objectLayer->addPlugin(
            std::make_unique<llvm::orc::DebugObjectManagerPlugin>(
                session, std::move(*plugin)));
      } else {
        return returnError(plugin.takeError());
      }
    }

    return objectLayer;
  };

  // Create the JIT.
  auto jitOr = llvm::orc::LLJITBuilder()
                   .setPlatformSetUp(llvm::orc::setUpOrcPlatform)
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .create();
  if (!jitOr)
    return M::Error(llvm::toString(jitOr.takeError()));

  // If we hit an error during object layer creation, return it.
  if (outError)
    return std::move(*outError);

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
  assert(jit && "must have the JIT already constructed");
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
