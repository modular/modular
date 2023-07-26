//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "Cache/BlobCache.h"
#include "Cache/Support/Keys.h"
#include "KGEN/ExecutionEngine/COMPILERRTCASID.h"
#include "KGEN/ExecutionEngine/ORCCASID.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "LLCL/Runtime/Algorithms.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ExecutionEngine/Orc/COFFPlatform.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace KGEN;
using namespace Cache;

//===----------------------------------------------------------------------===//
// MaterializationLayer
//===----------------------------------------------------------------------===//

char MaterializationLayer::ID;

MaterializationLayer::MaterializationLayer(llvm::orc::ExecutionSession &sess,
                                           const llvm::DataLayout &dl,
                                           AddToSearchOrderFn add)
    : session(sess), dataLayout(dl), addToSearchOrder(std::move(add)) {}

ErrorOr<llvm::orc::JITDylib *>
MaterializationLayer::getOrCreateDylib(StringRef libName) {
  if (llvm::orc::JITDylib *dylib = session.getJITDylibByName(libName))
    return dylib;

  auto dylibOr = session.createJITDylib(libName.str());
  if (!dylibOr)
    return M::Error(toString(dylibOr.takeError()));
  llvm::orc::JITDylib &dylib = *dylibOr;

  // Add the dylib to the search order.
  if (auto err = addToSearchOrder(libName, &dylib))
    return err.takeError();

  return &dylib;
}

llvm::orc::SymbolStringPtr
MaterializationLayer::mangleAndIntern(StringRef name) {
  std::string mangledName;
  llvm::raw_string_ostream mangledNameStream(mangledName);
  llvm::Mangler::getNameWithPrefix(mangledNameStream, name, dataLayout);
  return session.intern(mangledName);
}

//===----------------------------------------------------------------------===//
// StaticSymbolLayer
//===----------------------------------------------------------------------===//

char StaticSymbolLayer::ID;

StaticSymbolLayer::StaticSymbolLayer(llvm::orc::ExecutionSession &sess,
                                     const llvm::DataLayout &dl,
                                     AddToSearchOrderFn add)
    : llvm::RTTIExtends<StaticSymbolLayer, MaterializationLayer>(
          sess, dl, std::move(add)) {}

ErrorOrSuccess StaticSymbolLayer::add(StringRef libName, StringRef funcName,
                                      void *fn) {
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();

  llvm::orc::JITDylib *dylib = *dylibOr;
  if (auto err = dylib->define(llvm::orc::absoluteSymbols(
          {{mangleAndIntern(funcName),
            {llvm::orc::ExecutorAddr::fromPtr(fn),
             llvm::JITSymbolFlags::Exported |
                 llvm::JITSymbolFlags::Absolute}}}))) {
    return Error(toString(std::move(err)));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StaticArchiveObjectMaterializationUnit
//===----------------------------------------------------------------------===//

namespace {
class StaticArchiveObjectMaterializationUnit
    : public llvm::orc::MaterializationUnit {
public:
  StaticArchiveObjectMaterializationUnit(llvm::orc::ObjectLayer &objLayer,
                                         llvm::MemoryBufferRef objectBuffer,
                                         Interface &interface)
      : MaterializationUnit(interface), objectBuffer(objectBuffer),
        genLayer(objLayer) {}

  /// Provide a name for this MU that will show up in ORC debug logs.
  StringRef getName() const override {
    return "KGEN::StaticArchiveObjectMaterializationUnit";
  }

  /// Given a MaterializationResponsibility, push the object file buffer onto
  /// the base layer.
  void materialize(
      std::unique_ptr<llvm::orc::MaterializationResponsibility> mr) override {
    genLayer.emit(std::move(mr),
                  llvm::MemoryBuffer::getMemBuffer(
                      objectBuffer, /*RequiresNullTerminator=*/false));
  }

  /// Notify that the symbol `name` has been overridden.
  void discard(const llvm::orc::JITDylib &jd,
               const llvm::orc::SymbolStringPtr &name) override {}

  llvm::MemoryBufferRef objectBuffer;
  llvm::orc::ObjectLayer &genLayer;
};
} // namespace

//===----------------------------------------------------------------------===//
// StaticArchiveMaterializationLayer
//===----------------------------------------------------------------------===//

char StaticArchiveLayer::ID;

StaticArchiveLayer::StaticArchiveLayer(llvm::orc::ObjectLayer &objLayer,
                                       llvm::orc::ExecutionSession &sess,
                                       const llvm::DataLayout &dl,
                                       AddToSearchOrderFn add)
    : llvm::RTTIExtends<StaticArchiveLayer, MaterializationLayer>(
          sess, dl, std::move(add)),
      objectLayer(objLayer) {}

ErrorOrSuccess StaticArchiveLayer::add(StringRef libName,
                                       Cache::BufferRef archive) {
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();
  llvm::orc::JITDylib *dylib = *dylibOr;

  // If the archive creation succeeds we store a ref to this buffer so the
  // data won't be deallocated until the JIT is destroyed. This version of
  // MemoryBuffer::getMemBuffer produces a non-owning buffer.
  std::unique_ptr<llvm::MemoryBuffer> archiveMemBuf =
      llvm::MemoryBuffer::getMemBuffer(archive->getBuffer(),
                                       /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  auto archiveBinary =
      llvm::object::Archive::create(archiveMemBuf->getMemBufferRef());
  if (!archiveBinary)
    return M::Error(toString(archiveBinary.takeError()));

  // Store a ref to the buffer data.
  archiveBuffers.push_back(archive.copy());

  // Generate a materialization unit for each of the children in this archive.
  // TODO: We really shouldn't have to do this, we should be able to use a
  // static library generator instead. This unfortunately doesn't work well with
  // the current generator model in orc, where some platforms (like MSVC) define
  // "terminal" generators as part of platform setup.
  llvm::orc::ResourceTrackerSP resourceTracker =
      dylib->getDefaultResourceTracker();
  llvm::Error err = llvm::Error::success();
  for (auto &child : (*archiveBinary)->children(err)) {
    if (err)
      return Error(toString(std::move(err)));
    auto childBufferOr = child.getMemoryBufferRef();
    if (!childBufferOr)
      return M::Error(toString(childBufferOr.takeError()));

    auto childInterface =
        llvm::orc::getObjectFileInterface(session, *childBufferOr);
    if (!childInterface)
      return M::Error(toString(childInterface.takeError()));
    auto defineErr =
        dylib->define(std::make_unique<StaticArchiveObjectMaterializationUnit>(
                          objectLayer, *childBufferOr, *childInterface),
                      resourceTracker);
    if (defineErr)
      return M::Error(toString(std::move(defineErr)));
  }
  if (err)
    return Error(toString(std::move(err)));

  return success();
}

/// A standard name (that a user is unlikely to create) that we can use for a
/// JITDylib to define platform-specific symbols we want to be in the JIT'ed
/// address space.
static constexpr StringLiteral platformStdlibName = "$platform-stdlib";
static constexpr StringLiteral compilerRTlibName = "$compilerrt-lib";

//===----------------------------------------------------------------------===//
// ExecutionEngine implementation
//===----------------------------------------------------------------------===//

using Keys::ReadOnlyKey;

/// Set up the ORC platform for the various different binary formats/platforms
/// we support. This requires that we have an ExecutionSession *and* an
/// ObjectLinkingLayer.
///
/// The main reason to use the platform like this is that it automatically sets
/// up the various symbols that complex code will need to execute on a target.
static ErrorOrSuccess
setupPlatform(StringRef orcRTPath, const llvm::DataLayout &dataLayout,
              llvm::orc::JITDylib &platformStdlib,
              llvm::orc::ExecutionSession &session,
              llvm::orc::ObjectLinkingLayer &objLinkingLayer) {
  const llvm::Triple &tt = session.getTargetTriple();

  // Add the current process symbols in.
  // NOTE: COFF JIT currently doesn't support in process symbols, as it can
  // currently hit conflicts with symbols in the current COFF ORC runtime.
  if (!tt.isOSBinFormatCOFF()) {
    if (auto generator =
            llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
                session))
      platformStdlib.addGenerator(std::move(*generator));
    else
      return Error(toString(generator.takeError()));
  }

  // No path to the orc runtime, exit early.
  if (orcRTPath.empty())
    return success();

  // The ELFNixPlatform has memory leaks, don't set it up.
  if (tt.isOSBinFormatELF())
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
    auto loadDynamicLibrary = [&session](llvm::orc::JITDylib &jd,
                                         StringRef dllName) -> llvm::Error {
      if (!dllName.ends_with_insensitive(".dll"))
        return llvm::make_error<llvm::StringError>(
            "DLLName not ending with .dll", llvm::inconvertibleErrorCode());

      // Get or create a dylib for this DLL.
      auto *libJD = session.getJITDylibByName(dllName);
      if (!libJD) {
        auto generatorOr = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
            session, dllName.data());
        if (!generatorOr)
          return generatorOr.takeError();
        libJD = &session.createBareJITDylib(dllName.str());
        libJD->addGenerator(std::move(*generatorOr));
      }
      jd.addToLinkOrder(*libJD);
      return llvm::Error::success();
    };

    if (auto platform = llvm::orc::COFFPlatform::Create(
            session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
            platformStdlib, orcRTPath.data(), std::move(loadDynamicLibrary)))
      session.setPlatform(std::move(*platform));
    else
      return Error(toString(platform.takeError()));
  }
  return success();
}

static ErrorOr<std::optional<BufferRef>> extractRTFromCache(StringRef casID) {
  // Create a BlobCache ref.
  std::filesystem::path base = ".kgen_cache";
  base /= "orc";
  RuntimeAndCache<ReadOnlyKey> runtimeAndCache(base.string());
  if (auto err = runtimeAndCache.setup(KGEN_VERSION_STRING))
    return err.takeError();

  BlobCache<ReadOnlyKey> &cache = runtimeAndCache.getCache();

  // Decode the base64 CAS ID to do the lookup with the raw bytes.
  std::vector<char> bytes;
  bytes.reserve(32);
  llvm::cantFail(llvm::decodeBase64(casID, bytes));
  AsyncValueRef<std::optional<BufferRef>> rtBuf =
      cache.find(StringRef(bytes.data(), bytes.size()));
  // Await the runtime buffer.
  LLCL::await(rtBuf);

  // Take the diagnostic.
  if (rtBuf.isError())
    return std::move(rtBuf.takeDiagnostic().getMessage());

  return std::move(*rtBuf);
}

/// Returns the path to a KGENCompilerRTShared dynamic library suitable for the
/// target triple, or an error if none exists.
static ErrorOr<std::filesystem::path>
getKGENCompilerRTSharedPath(llvm::orc::ExecutionSession &session, Config &cfg) {
  // First, attempt to get the direct path to the binary.
  // TODO: This is a variable the package installer should set when installing
  //       the SDK.
  StringRef libPath = cfg.getValue("mojo.compilerrt.path");
  std::filesystem::path path = libPath.str();
  if (libPath.empty()) {
    StringRef derivedPath = cfg.getValue("derived.path");
    // If we don't have the derived path, then use the home dir as the path.
    if (derivedPath.empty())
      path = Config::getModularHomeDirPath();
    else
      path = std::filesystem::path(derivedPath.str()) / "build";

    if (session.getTargetTriple().isOSBinFormatMachO()) {
      path /= "lib";
      path /= "libKGENCompilerRTShared.dylib";
    } else if (session.getTargetTriple().isOSBinFormatELF()) {
      path /= "lib";
      path /= "libKGENCompilerRTShared.so";
    } else if (session.getTargetTriple().isOSBinFormatCOFF()) {
      path /= "bin";
      path /= "KGENCompilerRTShared.dll";
    }
  }

  return path;
}

/// Initialize the CompilerRT dylib.
static ErrorOrSuccess initializeCompilerRT(llvm::orc::ExecutionSession &session,
                                           Config &cfg) {
  auto compilerRTBuf = extractRTFromCache(M::CASID::kCompilerRT);
  if (compilerRTBuf.isError())
    return compilerRTBuf.takeError();
  std::optional<BufferRef> rtBuf = std::move(*compilerRTBuf);
  std::string compilerRTPath;
  // If we have rtBuf we can write it to a file and use that. Otherwise, attempt
  // to read it from the build dir.
  if (rtBuf) {
    if (auto err = writeTempFile("compiler_rt-%%%%%%%.a", (*rtBuf)->getBuffer(),
                                 compilerRTPath))
      return err.takeError();
  } else {
    ErrorOr<std::filesystem::path> rtPath =
        getKGENCompilerRTSharedPath(session, cfg);
    if (failed(rtPath))
      return rtPath.takeError();
    compilerRTPath = rtPath->string();
  }

  auto generatorOr = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
      session, compilerRTPath.c_str());
  if (!generatorOr)
    return Error(toString(generatorOr.takeError()));
  auto *libJD = &session.createBareJITDylib(compilerRTlibName.str());
  libJD->addGenerator(std::move(*generatorOr));
  return success();
}

/// Search for the specified sanitizer library path. This currently only allows
/// clang and libclang_rt.* sanitizers.
static ErrorOr<std::string>
findSanitizerLibraryPath(Sanitizers which, llvm::orc::ExecutionSession &session,
                         Config &cfg) {
  // No sanitizer on the build, return an empty string.
  if (!which)
    return "";

#ifdef _WIN32
  return Error("cannot find sanitizer libraries on windows");
#endif

  std::string configName =
      llvm::formatv("mojo.sanitizer.{0}",
                    which.has(Sanitizers::kAddress) ? "address" : "thread");

  // We may have simply been provided the path to the sanitizer we want!
  StringRef sanitizerPath = cfg.getValue(configName);

  // Ensure it exists, but as long as it does, use it unmodified.
  std::error_code ec;
  if (!sanitizerPath.empty() &&
      std::filesystem::exists(sanitizerPath.str(), ec) && !ec)
    return sanitizerPath.str();

  if (ec)
    return Error(ec.message());

  // Get the sanitizer library name. We only have macOS and linux here because
  // we don't support windows in this function anyway.
  StringRef shortenedSanitizer =
      which.has(Sanitizers::kAddress) ? "asan" : "tsan";
  std::string sanitizerLibName;
  if (session.getTargetTriple().isOSBinFormatMachO()) {
    sanitizerLibName =
        llvm::formatv("libclang_rt.{0}_osx_dynamic.dylib", shortenedSanitizer);
  } else if (session.getTargetTriple().isOSBinFormatELF()) {
    // TODO: This only supports clang sanitizers - we could in theory check GCC
    //       sanitizers as well.
    sanitizerLibName =
        llvm::formatv("libclang_rt.{0}-{1}.so", shortenedSanitizer,
                      session.getTargetTriple().getArchName());
  }

  // Create tempfile for stdout.
  auto tmpOutOr = TempFile::create("sanitizer-lib-out-%%%%%.tmp");
  if (tmpOutOr.isError())
    return tmpOutOr.takeError();

  // Get the system clang, so we can attempt to use it to print the sanitizer
  // path.
  llvm::ErrorOr<std::string> clangOr = llvm::sys::findProgramByName("clang");
  if (!clangOr)
    return Error("unable to find the system compiler: " +
                 clangOr.getError().message() + " - please set " + configName);

  // Redirects here are stdin(0), stdout(1), and stderr(2).
  std::string err;
  int result = llvm::sys::ExecuteAndWait(
      *clangOr,
      /*Args=*/
      {*clangOr, "--print-file-name", sanitizerLibName},
      /*Env=*/std::nullopt,
      /*Redirects=*/
      {std::nullopt, tmpOutOr->getPath().string(), std::nullopt},
      /*SecondsToWait=*/1, /*MemoryLimit=*/0, /*ErrMsg=*/&err);
  if (result != 0)
    return Error("failed to execute system compiler: " + err);

  // Read the file that has the result of calling the system compiler and set it
  // in the config.
  auto fileOr = Cache::Buffer::getFile(tmpOutOr->getPath());
  if (fileOr.isError())
    return fileOr.takeError();

  // Set the value in the config - make sure to trim off trailing whitespace.
  cfg.setValue(configName, (*fileOr)->getBuffer().rtrim());

  // And return the value directly from the config.
  return cfg.getValue(configName).str();
}

M::ErrorOr<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(ExecutionEngineOptions options,
                        const llvm::TargetMachine &tm) {
  // Create the data layout from the target machine.
  const llvm::DataLayout &layout = tm.createDataLayout();
  const llvm::Triple &tt = tm.getTargetTriple();

  // Construct the ExecutionSession. The user may have passed in an
  // ExecutorProcessControl that we need to use.
  std::unique_ptr<llvm::orc::ExecutorProcessControl> epc =
      std::move(options.epc);
  if (!epc) {
    auto pageSize = llvm::sys::Process::getPageSize();
    if (!pageSize)
      return Error(toString(pageSize.takeError()));

    size_t slabSize = 1024 * 1024 * 1024;

    auto managerOr = llvm::orc::MapperJITLinkMemoryManager::CreateWithMapper<
        llvm::orc::InProcessMemoryMapper>(slabSize);
    if (!managerOr)
      return Error(toString(managerOr.takeError()));

    epc = std::make_unique<llvm::orc::SelfExecutorProcessControl>(
        std::make_shared<llvm::orc::SymbolStringPool>(),
        std::make_unique<llvm::orc::DynamicThreadPoolTaskDispatcher>(), tt,
        *pageSize, /*MemMgr=*/std::move(*managerOr));
  }
  auto sessionPtr =
      std::make_unique<llvm::orc::ExecutionSession>(std::move(epc));

  // Now we can actually create the ExecutionEngine.
  auto ee = std::unique_ptr<ExecutionEngine>(
      new ExecutionEngine(std::move(sessionPtr), layout));

  // Get the ORC runtime binary.
  auto orcRTBuf = extractRTFromCache(M::CASID::kOrcRT);
  if (orcRTBuf.isError())
    return orcRTBuf.takeError();

  std::optional<BufferRef> rtBuf = std::move(*orcRTBuf);
  bool haveOrcRT = rtBuf.has_value();
  std::string orcRTPath;

  // TODO(#10097): Orc now supports passing in an archive, remove the usage of
  // files for the orcrt.
  if (haveOrcRT)
    if (auto err = writeTempFile("liborc_rt-%%%%%%%.a", (*rtBuf)->getBuffer(),
                                 orcRTPath))
      return err.takeError();

  // Windows *requires* the orc runtime.
  if (!haveOrcRT && tt.isOSBinFormatCOFF())
    return Error("could not find orc_rt in the cache");

  // Construct the object linking layer.
  ee->objectLayer =
      std::make_unique<llvm::orc::ObjectLinkingLayer>(*ee->executionSession);

  // Construct the platform stdlib - this way we don't have to worry about
  // whether or not we have it later on.
  llvm::orc::JITDylib &platformStdlib =
      ee->executionSession->createBareJITDylib(platformStdlibName.str());

  // If we have the platform support library, use it.
  if (auto err = setupPlatform(orcRTPath, ee->dataLayout, platformStdlib,
                               *ee->executionSession, *ee->objectLayer))
    return err.takeError();

  // Open the config object so we can use it.
  auto cfgOr = Config::open();
  if (cfgOr.isError())
    return cfgOr.takeError();
  Config cfg = std::move(*cfgOr);

  // Find the path to the sanitizer library.
  auto pathOr =
      findSanitizerLibraryPath(options.sanitizers, *ee->executionSession, cfg);
  if (pathOr.isError())
    return pathOr.takeError();

  // Pull in ASAN/TSAN if we have them, and they're requested.
  if (options.sanitizers.has(Sanitizers::kAddress)) {
    // If the asan symbols already exist in the target process, DO NOT re-init
    // asan - we will get hard-to-debug failures that occur on initialization.
    llvm::Expected<llvm::orc::ExecutorSymbolDef> asanInit =
        ee->executionSession->lookup({&platformStdlib}, "__asan_init");
    if (!asanInit) {
      // Consume the error - we don't care what it was.
      llvm::consumeError(asanInit.takeError());
      // Now, try and find the thing.
      assert(!pathOr->empty() &&
             "we didn't specify which sanitizer to findSanitizerLibraryPath?");
      // Find and load the asan dylib.
      if (auto generator = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
              *ee->executionSession, pathOr->c_str()))
        platformStdlib.addGenerator(std::move(*generator));
      else
        return Error(llvm::toString(generator.takeError()));
    }
  } else if (options.sanitizers.has(Sanitizers::kThread)) {
    assert(!pathOr->empty() &&
           "we didn't specify which sanitizer to findSanitizerLibraryPath?");
    // Find and load the tsan dylib.
    if (auto generator = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
            *ee->executionSession, pathOr->c_str()))
      platformStdlib.addGenerator(std::move(*generator));
    else
      return Error(llvm::toString(generator.takeError()));
  }

  // COFF format binaries (Windows) need special handling to deal with
  // exported symbol visibility.
  if (tt.isOSBinFormatCOFF()) {
    ee->objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
    ee->objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
  }

  if (options.registerDebugPlugins) {
    llvm::orc::ExecutionSession &session = *ee->executionSession;

    // Get the registrar for the GDB JIT loader interface.
    if (tt.isOSBinFormatMachO()) {
      // We have to explicitly define these wrapper symbols on macOS because
      // they're hidden visibility.
      auto err = platformStdlib.define(llvm::orc::absoluteSymbols(
          {{session.intern("_llvm_orc_registerJITLoaderGDBWrapper"),
            {llvm::orc::ExecutorAddr::fromPtr(
                 &llvm_orc_registerJITLoaderGDBWrapper),
             llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute}},
           {session.intern("_llvm_orc_registerJITLoaderGDBAllocAction"),
            {llvm::orc::ExecutorAddr::fromPtr(
                 &llvm_orc_registerJITLoaderGDBAllocAction),
             llvm::JITSymbolFlags::Exported |
                 llvm::JITSymbolFlags::Absolute}}}));
      if (err)
        return Error(llvm::toString(std::move(err)));

      // Create and register the JIT DebugInfo plugin.
      auto plugin = llvm::orc::GDBJITDebugInfoRegistrationPlugin::Create(
          session, platformStdlib, tt);
      if (!plugin)
        return Error(llvm::toString(plugin.takeError()));

      ee->objectLayer->addPlugin(std::move(*plugin));
    } else if (tt.isOSBinFormatELF()) {
      // Register the DebugObjectManagerPlugin.
      auto registrar = llvm::orc::createJITLoaderGDBRegistrar(session);
      if (!registrar)
        return Error(llvm::toString(registrar.takeError()));

      ee->objectLayer->addPlugin(
          std::make_unique<llvm::orc::DebugObjectManagerPlugin>(
              session, std::move(*registrar)));
    }
  }

  // Add the platform dylib to the search order.
  if (auto err = ee->addToSearchOrder(platformStdlibName, &platformStdlib))
    return err.takeError();

  // Prepare the CompilerRT dylib.
  if (auto err = initializeCompilerRT(*ee->executionSession, cfg))
    return err.takeError();

  return std::move(ee);
}

ErrorOr<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::createWithStandardLayers(ExecutionEngineOptions options,
                                          const llvm::TargetMachine &tm) {
  auto engineOr = ExecutionEngine::create(std::move(options), tm);
  if (engineOr.isError())
    return engineOr.takeError();

  // Add the standard layers.
  (*engineOr)->addLayer<StaticSymbolLayer>();
  (*engineOr)->addLayer<StaticArchiveLayer>((*engineOr)->getLinkingLayer());

  return std::move(*engineOr);
}

ExecutionEngine::ExecutionEngine(
    std::unique_ptr<llvm::orc::ExecutionSession> session,
    const llvm::DataLayout &dl)
    : executionSession(std::move(session)),
      // Parse the layout so that we own the underlying memory. DataLayout is a
      // bit weird, it seems like it has some internal data structures that
      // every instance shares.
      dataLayout(dl.getStringRepresentation()) {}

ExecutionEngine::~ExecutionEngine() {
  if (!executionSession)
    return;

  // If the execution engine has initialized the ORC runtime, the ELFNix and
  // COFF platform implementations need manual shutdown. The MachOPlatform
  // implementation is more sophisticated and performs shutdown automatically
  // through the JITLink LinkGraph allocation actions.
  const llvm::Triple &triple = executionSession->getTargetTriple();
  if (executionSession->getPlatform() &&
      // FIXME: On Windows, this complains about symbol not found. The Windows
      // build seems happy even without the shutdown, so disable it for now.
      (triple.isOSBinFormatELF() /*|| triple.isOSBinFormatCOFF()*/)) {
    ErrorOr<CompiledFunc> shutdown =
        lookup(triple.isOSBinFormatELF() ? "__orc_rt_elfnix_platform_shutdown"
                                         : "__orc_rt_coff_platform_shutdown");
    if (shutdown.isError()) {
      llvm::report_fatal_error(
          Twine("failed to find ELF/COFF platform shutdown function: ") +
          shutdown.takeError().get());
    }
    struct OrcRTCWrapperFunctionResult {
      char *data;
      size_t size;
    };
    shutdown->invoke<OrcRTCWrapperFunctionResult, char *, size_t>(nullptr, 0);
  }

  if (auto Err = executionSession->endSession())
    executionSession->reportError(std::move(Err));
}

ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

bool ExecutionEngine::libraryExists(llvm::StringRef libName) {
  llvm::orc::JITDylib *dylib = executionSession->getJITDylibByName(libName);
  return dylib != nullptr;
}

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef symbol) {
  return lookupWithSearchOrder(searchOrder, symbol);
}

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef libName,
                                              StringRef symbol) {
  llvm::orc::JITDylib *dylib = executionSession->getJITDylibByName(libName);
  if (!dylib)
    return Error("could not find JITDylib with name: " + libName);

  return lookupWithSearchOrder(llvm::orc::makeJITDylibSearchOrder({dylib}),
                               symbol);
}

ErrorOrSuccess
ExecutionEngine::runProgram(StringRef libName, StringRef entryPoint,
                            function_ref<ErrorOrSuccess(void *)> runFn) {
  using namespace llvm::orc;

  // The platform will be non-null if the ORC RT was available.
  if (executionSession->getPlatform()) {
    // If the ORC runtime is available, it can be used to get platform support
    // for stuff like global initializers and destructors. Invoke the function
    // by using the JIT dylib functions directly.
    ErrorOr<CompiledFunc> dlopenFn = lookup("__orc_rt_jit_dlopen");
    if (dlopenFn.isError())
      return dlopenFn.takeError();
    ErrorOr<CompiledFunc> dlsymFn = lookup("__orc_rt_jit_dlsym");
    if (dlsymFn.isError())
      return dlsymFn.takeError();
    ErrorOr<CompiledFunc> dlerrorFn = lookup("__orc_rt_jit_dlerror");
    if (dlerrorFn.isError())
      return dlerrorFn.takeError();
    ErrorOr<CompiledFunc> dlcloseFn = lookup("__orc_rt_jit_dlclose");
    if (dlcloseFn.isError())
      return dlcloseFn.takeError();

    std::string libNameStr = libName.str();
    void *dylib = dlopenFn->invoke<void *, const char *, int>(
        libNameStr.c_str(), /*ORC_RT_RTLD_LAZY=*/0x1);
    if (!dylib)
      return Error(dlerrorFn->invoke<const char *>());

    std::string entryPointStr = entryPoint.str();
    void *fnPtr = dlsymFn->invoke<void *, void *, const char *>(
        dylib, entryPointStr.c_str());
    if (!fnPtr)
      return Error(dlerrorFn->invoke<const char *>());

    ErrorOrSuccess result = runFn(fnPtr);

    // A result of `-1` indicates that `dlclose` failed.
    if (dlcloseFn->invoke<int, void *>(dylib) == -1)
      return Error(dlerrorFn->invoke<const char *>());

    return result;
  }

  // Lookup the entry point symbol and directly invoke it rather than going
  // through the runtime.
  ErrorOr<CompiledFunc> mainFn = lookup(entryPoint);
  if (mainFn.isError())
    return mainFn.takeError();
  return runFn(mainFn->getFunctionPointer());
}

llvm::orc::SymbolStringPtr
KGEN::ExecutionEngine::mangleAndIntern(StringRef name) {
  std::string mangledName;
  llvm::raw_string_ostream mangledNameStream(mangledName);
  llvm::Mangler::getNameWithPrefix(mangledNameStream, name, dataLayout);
  return executionSession->intern(mangledName);
}

ErrorOrSuccess ExecutionEngine::addToSearchOrder(StringRef name,
                                                 llvm::orc::JITDylib *dylib) {
  auto [_, didInsert] = knownDylibs.insert(name);
  assert(didInsert && "must have uniquely-named dylibs");

  // If this isn't the platform stdlib, setup CompilerRT.
  if (name != platformStdlibName) {
    dylib->addToLinkOrder(
        *executionSession->getJITDylibByName(compilerRTlibName));
  }

  // Use higher preference for newer dylibs.
  searchOrder.insert(searchOrder.begin(),
                     {dylib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols});
  return success();
}

ErrorOr<CompiledFunc> ExecutionEngine::lookupWithSearchOrder(
    const llvm::orc::JITDylibSearchOrder &order, llvm::StringRef symbol) {
  // Look up this symbol with the search order provided.
  llvm::Expected<llvm::orc::ExecutorSymbolDef> sym =
      executionSession->lookup(order, mangleAndIntern(symbol));
  if (sym)
    return CompiledFunc(sym->getAddress().toPtr<void *>());

  // Check to see if any of the layers have errors.
  auto found = llvm::find_if(
      layers, [](const auto &layer) { return layer->hasError(); });
  // If not, return the error returned by the ORC.
  if (found == layers.end())
    return Error(llvm::toString(sym.takeError()));

  // Add the additional context from the layer's error.
  return Error(llvm::toString(sym.takeError()) +
               " (from the layer: " + (*found)->takeError().get() + ")");
}
