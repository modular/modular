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
#include "llvm/Support/Base64.h"
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

/// Write the orc_rt buffer to a temporary path so we can pass that path. This
/// is a temporary work-around until COFF can be called with a memory buffer.
/// See #10097.
static ErrorOrSuccess writeORCRTToFile(BufferRef &buf, std::string &outPath) {
  std::error_code ec;
  std::filesystem::path path = std::filesystem::temp_directory_path(ec);
  if (ec)
    return Error(ec.message());

  path = path / "liborc_rt.a";
  outPath = path.string();

  // Write the runtime to the temp file.
  llvm::raw_fd_ostream tmp(outPath.c_str(), ec);
  if (ec)
    return Error(ec.message());

  tmp << buf->getBuffer();
  return success();
}

/// Set up the ORC platform for the various different binary formats/platforms
/// we support. This requires that we have an ExecutionSession *and* an
/// ObjectLinkingLayer.
///
/// The main reason to use the platform like this is that it automatically sets
/// up the various symbols that complex code will need to execute on a target.
static ErrorOrSuccess
setupPlatform(StringRef orcRTPath, llvm::TargetMachine &tm,
              llvm::orc::ExecutionSession &session,
              llvm::orc::ObjectLinkingLayer &objLinkingLayer) {
  llvm::orc::JITDylib &platformStdlib =
      session.createBareJITDylib(platformStdlibName.str());

  // Add the current process symbols in.
  if (auto generator =
          llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
              tm.createDataLayout().getGlobalPrefix()))
    platformStdlib.addGenerator(std::move(*generator));
  else
    return Error(toString(generator.takeError()));

  const llvm::Triple &tt = session.getTargetTriple();
  // TODO: (#10184) Ensure we have no memory leaks on non-COFF platforms.
  if (!tt.isOSBinFormatCOFF())
    return success();

  if (tt.isOSBinFormatMachO()) {
    if (auto platform = llvm::orc::MachOPlatform::Create(
            session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
            platformStdlib, orcRTPath.data()))
      session.setPlatform(std::move(*platform));
    else
      return Error(toString(platform.takeError()));
  } else if (tt.isOSBinFormatELF()) {
    if (auto platform = llvm::orc::ELFNixPlatform::Create(
            session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
            platformStdlib, orcRTPath.data()))
      session.setPlatform(std::move(*platform));
    else
      return Error(toString(platform.takeError()));
  } else if (tt.isOSBinFormatCOFF()) {
    // Windows needs some help to load dylibs, apparently.
    auto loadDynamicLibrary = [tt, &tm](llvm::orc::JITDylib &jd,
                                        StringRef dllName) -> llvm::Error {
      if (!dllName.endswith_insensitive(".dll"))
        return llvm::make_error<llvm::StringError>(
            "DLLName not ending with .dll", llvm::inconvertibleErrorCode());

      if (auto dylibGeneratorOr =
              llvm::orc::DynamicLibrarySearchGenerator::Load(
                  dllName.data(), tm.createDataLayout().getGlobalPrefix()))
        jd.addGenerator(std::move(*dylibGeneratorOr));
      else
        return dylibGeneratorOr.takeError();
      return llvm::Error::success();
    };

    if (auto platform = llvm::orc::COFFPlatform::Create(
            session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
            platformStdlib, orcRTPath.data(), loadDynamicLibrary))
      session.setPlatform(std::move(*platform));
    else
      return Error(toString(platform.takeError()));
  }
  return success();
}

M::ErrorOr<ExecutionEngine>
ExecutionEngine::create(const CompilationOptions &options) {
  // Create a BlobCache ref.
  std::filesystem::path base = ".kgen_cache";
  base /= "orc";
  RuntimeAndCache<ReadOnlyKey> runtimeAndCache(base.string());
  if (auto err = runtimeAndCache.setup())
    return err.takeError();

  BlobCache<ReadOnlyKey> &orcCache = runtimeAndCache.getCache();

  // Decode the base64 CAS ID to do the lookup with the raw bytes.
  std::vector<char> bytes;
  bytes.reserve(32);
  llvm::cantFail(llvm::decodeBase64(M::CASID::kOrcRT, bytes));
  AsyncValueRef<std::optional<BufferRef>> orcRTBuf =
      orcCache.find(StringRef(bytes.data(), bytes.size()));
  // Await the orc runtime buffer.
  LLCL::await(orcRTBuf);

  ExecutionEngine ee(nullptr, options);

  // Create the target machine.
  auto tmOr = KGEN::createTargetMachine(options, /*isJIT=*/false);
  if (tmOr.isError())
    return tmOr.takeError();
  std::unique_ptr<llvm::TargetMachine> tm = std::move(*tmOr);

  if (orcRTBuf.isError())
    return std::move(orcRTBuf.takeDiagnostic().getMessage());

  std::optional<BufferRef> rtBuf = std::move(*orcRTBuf);
  std::string orcRTPath;
  // TODO: (#10184) Turn this back on for ELF/NIX and MachO
  if (!rtBuf.has_value() && tm->getTargetTriple().isOSBinFormatCOFF())
    return Error("could not find orc_rt in the cache");

  if (auto err = writeORCRTToFile(*rtBuf, orcRTPath))
    return err.takeError();

  // Define an optional error we can set to something if we hit an error in a
  // nested closure.
  std::optional<Error> outError = std::nullopt;

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
                                       const llvm::Triple &tt)
      -> std::unique_ptr<llvm::orc::ObjectLinkingLayer> {
    auto objectLayer = std::make_unique<llvm::orc::ObjectLinkingLayer>(session);

    // Set up the platform support now that we have an object layer.
    if (rtBuf) {
      if (auto err = setupPlatform(orcRTPath, *tm, session, *objectLayer)) {
        outError = err.takeError();
        return nullptr;
      }
    }

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
