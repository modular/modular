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
#include "LLCL/Runtime/Algorithms.h"
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
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Process.h"
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

  auto archiveOr = llvm::orc::StaticLibraryDefinitionGenerator::Create(
      objectLayer, std::move(archiveMemBuf));
  if (auto err = archiveOr.takeError())
    return M::Error(toString(std::move(err)));
  dylib->addGenerator(std::move(*archiveOr));

  // Store a ref to the buffer data.
  archiveBuffers.push_back(archive.copy());

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

/// Write the rt buffer to a temporary path so we can pass that path.
static ErrorOrSuccess writeRTToFile(StringRef prefix, BufferRef &buf,
                                    std::string &outPath) {
  std::error_code ec;
  std::filesystem::path path = std::filesystem::temp_directory_path(ec);
  if (ec)
    return Error(ec.message());

  // Write to a temporary file, but make it unique so that parallel running
  // processes don't overwrite and corrupt the file.
  path = path / (prefix + "_rt-%%%%%%%.a").str();
  outPath = path.string();

  auto tmpfileOr = TempFile::create(path.string());
  if (tmpfileOr.isError())
    return tmpfileOr.takeError();

  // Write the runtime to the temp file.
  llvm::raw_fd_ostream tmp(tmpfileOr->getFD(), /*shouldClose=*/false);
  if (ec)
    return Error(ec.message());

  tmp << buf->getBuffer();
  outPath = tmpfileOr->getPath().string();
  tmpfileOr->keep();
  return success();
}

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
  // No path to the orc runtime, exit early.
  if (orcRTPath.empty())
    return success();
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

  // TODO: (#10184) Ensure we have no memory leaks on linux platforms.
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
  if (auto err = runtimeAndCache.setup())
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

/// Initialize the CompilerRT dylib.
static ErrorOrSuccess
initializeCompilerRT(llvm::orc::ExecutionSession &session) {
  auto compilerRTBuf = extractRTFromCache(M::CASID::kCompilerRT);
  if (compilerRTBuf.isError())
    return compilerRTBuf.takeError();
  std::optional<BufferRef> rtBuf = std::move(*compilerRTBuf);

  std::string compilerRTPath;
  if (auto err = writeRTToFile("compilerrt", *rtBuf, compilerRTPath))
    return err.takeError();

  auto generatorOr = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
      session, compilerRTPath.data());
  if (!generatorOr)
    return Error(toString(generatorOr.takeError()));
  auto *libJD = &session.createBareJITDylib(compilerRTlibName.str());
  libJD->addGenerator(std::move(*generatorOr));
  return success();
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
  std::string orcRTPath;

  // TODO(#10097): Orc now supports passing in an archive, remove the usage of
  // files for the orcrt.
  if (rtBuf.has_value())
    if (auto err = writeRTToFile("liborc", *rtBuf, orcRTPath))
      return err.takeError();

  // Windows *requires* the orc runtime.
  if (!rtBuf.has_value() && tt.isOSBinFormatCOFF())
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
  if (auto err = initializeCompilerRT(*ee->executionSession))
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
  if (executionSession)
    if (auto Err = executionSession->endSession())
      executionSession->reportError(std::move(Err));
}

ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef symbol) {
  llvm::Expected<llvm::orc::ExecutorSymbolDef> sym =
      executionSession->lookup(searchOrder, mangleAndIntern(symbol));
  if (!sym) {
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

  return CompiledFunc(sym->getAddress().toPtr<void *>());
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
