//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "Cache/BlobCache.h"
#include "Cache/Support/Keys.h"
#include "KGEN/ExecutionEngine/JIT/MaterializationLayer.h"
#include "KGEN/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "KGEN/Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ExecutionEngine/Orc/COFFPlatform.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderPerf.h"
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

/// A standard name (that a user is unlikely to create) that we can use for a
/// JITDylib to define platform-specific symbols we want to be in the JIT'ed
/// address space.
static constexpr StringLiteral platformStdlibName = "$platform-stdlib";
static constexpr StringLiteral compilerRTlibName = "$compilerrt-lib";
static constexpr StringLiteral mlirclibName = "$mlirc-lib";

//===----------------------------------------------------------------------===//
// ExecutionEngine implementation
//===----------------------------------------------------------------------===//

using Keys::ReadOnlyKey;

/// Create a unix-like system platform of the given type, and set that as the
/// platform of the given session.
template <typename T>
static ErrorOrSuccess
setUnixPlatform(llvm::orc::JITDylib &platformStdlib,
                llvm::orc::ExecutionSession &session,
                llvm::orc::ObjectLinkingLayer &objLinkingLayer,
                std::unique_ptr<llvm::MemoryBuffer> orcRTBuf) {
  // Create a generator for the ORC runtime archive.
  auto orcRuntimeArchiveGenerator =
      toModularErrorOr(llvm::orc::StaticLibraryDefinitionGenerator::Create(
          objLinkingLayer, std::move(orcRTBuf)));
  if (orcRuntimeArchiveGenerator.isError())
    return orcRuntimeArchiveGenerator.takeError();

  auto platformOr = toModularErrorOr(
      T::Create(session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
                platformStdlib, std::move(*orcRuntimeArchiveGenerator)));
  if (platformOr.isError())
    return platformOr.takeError();
  session.setPlatform(std::move(*platformOr));
  return success();
}

/// Set up the ORC platform for the various different binary formats/platforms
/// we support. This requires that we have an ExecutionSession *and* an
/// ObjectLinkingLayer.
///
/// The main reason to use the platform like this is that it automatically sets
/// up the various symbols that complex code will need to execute on a target.
static ErrorOrSuccess
setupPlatform(const std::optional<BufferRef> &orcRTBuf,
              const llvm::DataLayout &dataLayout,
              llvm::orc::JITDylib &platformStdlib,
              llvm::orc::ExecutionSession &session,
              llvm::orc::ObjectLinkingLayer &objLinkingLayer) {
  const llvm::Triple &tt = session.getTargetTriple();

  // Add the current process symbols in.
  // NOTE: COFF JIT currently doesn't support in process symbols, as it can
  // currently hit conflicts with symbols in the current COFF ORC runtime.
  if (!tt.isOSBinFormatCOFF()) {
    auto generator = toModularErrorOr(
        llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
            session));
    if (generator.isError())
      return generator.takeError();
    platformStdlib.addGenerator(std::move(*generator));
  }

  // No orc runtime, exit early.
  if (!orcRTBuf)
    return success();

  // Disable the runtime on Linux, since there are issues with multi-tenancy
  // and memory leaks. Disable the runtime on MacOS due to issues with
  // libunwind.
  if (tt.isOSBinFormatELF() || tt.isOSBinFormatMachO())
    return success();

  auto orcRTMemBuf =
      llvm::MemoryBuffer::getMemBufferCopy((*orcRTBuf)->getBuffer());
  if (tt.isOSBinFormatMachO()) {
    if (auto error = setUnixPlatform<llvm::orc::MachOPlatform>(
            platformStdlib, session, objLinkingLayer, std::move(orcRTMemBuf)))
      return error;
  } else if (tt.isOSBinFormatELF()) {
    if (auto error = setUnixPlatform<llvm::orc::ELFNixPlatform>(
            platformStdlib, session, objLinkingLayer, std::move(orcRTMemBuf)))
      return error;
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

    auto platform = toModularErrorOr(llvm::orc::COFFPlatform::Create(
        session, cast<llvm::orc::ObjectLinkingLayer>(objLinkingLayer),
        platformStdlib, std::move(orcRTMemBuf), std::move(loadDynamicLibrary)));
    if (platform.isError())
      return platform.takeError();
    session.setPlatform(std::move(*platform));
  }
  return success();
}

/// Initialize the mlirc and CompilerRT dylib.
static ErrorOrSuccess
initializeCompilerRT(llvm::orc::ExecutionSession &session, MojoConfig &cfg,
                     const llvm::DataLayout &layout,
                     const ExecutionEngineOptions &options) {
  std::error_code ec;

  // mlirc dylib. Grab the symbols from the current process.
  {
    auto *libJD = &session.createBareJITDylib(mlirclibName.str());
    libJD->addGenerator(llvm::cantFail(
        llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
            session, [=](const llvm::orc::SymbolStringPtr &symbolStringPtr) {
              return (*symbolStringPtr).starts_with("mlir");
            })));
  }

  // CompilerRT dylib.
  std::string compilerRTPath = cfg.getCompilerRTPath().str();
  if (!std::filesystem::exists(compilerRTPath, ec) || ec)
    return Error(std::string("unable to locate compiler_rt ") + compilerRTPath);

  auto generatorOr =
      toModularErrorOr(llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
          session, compilerRTPath.c_str()));
  if (generatorOr.isError()) {
    return Error(Twine("error '") + Twine(generatorOr.getError()) +
                 "' while loading compiler runtime library from '" +
                 compilerRTPath.c_str() + "'");
  }
  auto *libJD = &session.createBareJITDylib(compilerRTlibName.str());
  libJD->addGenerator(std::move(*generatorOr));

  // Allow pulling in sanitizer methods from the current process, as we
  // currently can't activate any of these runtimes otherwise (they must
  // generally be loaded first in the host process).
  libJD->addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          layout.getGlobalPrefix(),
          [=](const llvm::orc::SymbolStringPtr &symbolStringPtr) {
            return llvm::any_of(ArrayRef<StringRef>{"__asan", "__tsan"},
                                [&](StringRef prefix) {
                                  return (*symbolStringPtr).starts_with(prefix);
                                });
          })));

  return success();
}

/// Grab a memory buffer for the Orc runtime.
static ErrorOr<std::optional<BufferRef>> initializeOrcRT(MojoConfig &cfg) {
  std::error_code ec;
  std::string orcRTPath = cfg.getOrcRTPath().str();
  if (!std::filesystem::exists(orcRTPath, ec) || ec) {
    ErrorOr<std::filesystem::path> cfgPath = cfg.getConfigFilePath();
    if (cfgPath.isError())
      return cfgPath.takeError();
    return Error("unable to locate orc_rt at " + Twine(orcRTPath) + ". " +
                 "Tried reading the config from: " + cfgPath.get().c_str() +
                 ".");
  }
  return Buffer::getFile(orcRTPath);
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
    auto pageSize = toModularErrorOr(llvm::sys::Process::getPageSize());
    if (pageSize.isError())
      return pageSize.takeError();

    size_t slabSize = 1024 * 1024 * 1024;

    auto managerOr = toModularErrorOr(
        llvm::orc::MapperJITLinkMemoryManager::CreateWithMapper<
            llvm::orc::InProcessMemoryMapper>(slabSize));
    if (managerOr.isError())
      return managerOr.takeError();

    epc = std::make_unique<llvm::orc::SelfExecutorProcessControl>(
        std::make_shared<llvm::orc::SymbolStringPool>(),
        std::make_unique<llvm::orc::InPlaceTaskDispatcher>(), tt, *pageSize,
        /*MemMgr=*/std::move(*managerOr));
  }
  auto sessionPtr =
      std::make_unique<llvm::orc::ExecutionSession>(std::move(epc));

  // Now we can actually create the ExecutionEngine.
  auto ee = std::unique_ptr<ExecutionEngine>(
      new ExecutionEngine(std::move(sessionPtr), layout));

  // Open the config object so we can use it.
  auto cfgOr = MojoConfig::open();
  if (cfgOr.isError())
    return cfgOr.takeError();
  MojoConfig cfg = std::move(*cfgOr);

  // Get the ORC runtime binary.
  auto orcRTBuf = initializeOrcRT(cfg);
  if (orcRTBuf.isError())
    return orcRTBuf.takeError();
  std::optional<BufferRef> rtBuf = orcRTBuf.takeValue();

  // Windows *requires* the orc runtime.
  if (!rtBuf && tt.isOSBinFormatCOFF())
    return Error("unable to locate orc_rt");

  // Construct the object linking layer.
  ee->objectLayer =
      std::make_unique<llvm::orc::ObjectLinkingLayer>(*ee->executionSession);

  // Construct the platform stdlib - this way we don't have to worry about
  // whether or not we have it later on.
  llvm::orc::JITDylib &platformStdlib =
      ee->executionSession->createBareJITDylib(platformStdlibName.str());

  // If we have the platform support library, use it. This requires the
  // compilation target to be a subset of the host process, so disable it for
  // cross-compilation.
  if (!options.crossCompiling) {
    if (auto err = setupPlatform(rtBuf, ee->dataLayout, platformStdlib,
                                 *ee->executionSession, *ee->objectLayer))
      return err.takeError();
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
      auto err =
          toModularErrorOr(platformStdlib.define(llvm::orc::absoluteSymbols(
              {{session.intern("_llvm_orc_registerJITLoaderGDBWrapper"),
                {llvm::orc::ExecutorAddr::fromPtr(
                     &llvm_orc_registerJITLoaderGDBWrapper),
                 llvm::JITSymbolFlags::Exported |
                     llvm::JITSymbolFlags::Absolute}},
               {session.intern("_llvm_orc_registerJITLoaderGDBAllocAction"),
                {llvm::orc::ExecutorAddr::fromPtr(
                     &llvm_orc_registerJITLoaderGDBAllocAction),
                 llvm::JITSymbolFlags::Exported |
                     llvm::JITSymbolFlags::Absolute}}})));
      if (err)
        return err.takeError();

      // Create and register the JIT DebugInfo plugin.
      auto plugin =
          toModularErrorOr(llvm::orc::GDBJITDebugInfoRegistrationPlugin::Create(
              session, platformStdlib, tt));
      if (plugin.isError())
        return plugin.takeError();

      ee->objectLayer->addPlugin(std::move(*plugin));
    } else if (tt.isOSBinFormatELF()) {
      // Register the DebugObjectManagerPlugin.
      ee->objectLayer->addPlugin(
          std::make_unique<llvm::orc::DebugObjectManagerPlugin>(
              session,
              std::make_unique<llvm::orc::EPCDebugObjectRegistrar>(
                  session, llvm::orc::ExecutorAddr::fromPtr(
                               &llvm_orc_registerJITLoaderGDBWrapper))));
    }
  }

  if (options.registerPerfPlugins) {
    auto debugInfo = llvm::orc::DebugInfoPreservationPlugin::Create();
    if (!debugInfo)
      return toModularError(debugInfo.takeError());
    ee->objectLayer->addPlugin(std::move(debugInfo.get()));
    auto perf = std::make_unique<llvm::orc::PerfSupportPlugin>(
        ee->objectLayer->getExecutionSession().getExecutorProcessControl(),
        llvm::orc::ExecutorAddr::fromPtr(&llvm_orc_registerJITLoaderPerfStart),
        llvm::orc::ExecutorAddr::fromPtr(&llvm_orc_registerJITLoaderPerfEnd),
        llvm::orc::ExecutorAddr::fromPtr(&llvm_orc_registerJITLoaderPerfImpl),
        true, true);
    ee->objectLayer->addPlugin(std::move(perf));
  }

  // Add the platform dylib to the search order.
  if (auto err = ee->addToSearchOrder(platformStdlibName, &platformStdlib))
    return err.takeError();

  // Prepare the CompilerRT dylib.
  if (auto err =
          initializeCompilerRT(*ee->executionSession, cfg, layout, options))
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

  ErrorOr<CompiledFunc> ctorResult = lookup(getGlobalCtorFnName());
  if (failed(ctorResult))
    return ctorResult.takeError();
  ErrorOr<CompiledFunc> dtorResult = lookup(getGlobalDtorFnName());
  if (failed(dtorResult))
    return dtorResult.takeError();

  // Lookup the entry point symbol and directly invoke it rather than going
  // through the runtime.
  ErrorOr<CompiledFunc> mainFn = lookup(entryPoint);
  if (mainFn.isError())
    return mainFn.takeError();
  ctorResult->invoke<void>();
  if (ErrorOrSuccess err = runFn(mainFn->getFunctionPointer()))
    return err.takeError();
  dtorResult->invoke<void>();
  return success();
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
  [[maybe_unused]] auto [_, didInsert] = knownDylibs.insert(name);
  assert(didInsert && "must have uniquely-named dylibs");

  // If this isn't the platform stdlib, setup CompilerRT and mlirc.
  if (name != platformStdlibName) {
    dylib->addToLinkOrder(
        *executionSession->getJITDylibByName(compilerRTlibName));
    dylib->addToLinkOrder(*executionSession->getJITDylibByName(mlirclibName));
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
    return toModularError(sym.takeError());

  // Add the additional context from the layer's error.
  return Error(llvm::toString(sym.takeError()) +
               " (from the layer: " + (*found)->takeError().get() + ")");
}
