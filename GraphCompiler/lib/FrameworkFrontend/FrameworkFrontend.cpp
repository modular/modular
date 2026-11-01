//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/FrameworkFrontend.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/Utils.h"
#include "Cache/CacheTelemetryContext.h"
#include "Cache/CachedTransform.h"
#include "Config/Version.h"
#include "GraphCompiler/Driver/Common/DriverConfig.h"
#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "GraphCompiler/Driver/Common/Utils.h"
#include "GraphCompiler/Driver/Compile/CompileDriver.h"
#include "GraphCompiler/Driver/Compile/CompileToBinary.h"
#include "GraphCompiler/Driver/Execution/DriverCLOptions.h"
#include "GraphCompiler/GraphCompilerCLOptions.h"
#include "GraphCompiler/MEFCacheKey.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/ParametersUtils.h"
#include "GraphCompiler/Passes.h"
#include "GraphCompiler/RMODialect/RMOAttrs.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "GraphCompiler/Version.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Mojo/KGENDialect/KGENCompilationContext.h"
#include "Support/Configuration.h"
#include "Support/DeviceSpecs.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/Paths.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "max/max_license/MOF/Driver/DriverImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"

#include <string>
#include <utility>

using namespace M;
using namespace M::AsyncRT;
using namespace M::MOF;

namespace {

class MOFDriverOptions : public DriverOptions,
                         public MO::GraphCompilerOptions {};

/// Union of configuration gathered from MOFContext::setConfigs().
struct MOFConfig {
  SmallVector<std::string> kernelLibraryPaths;
  std::optional<std::string> defaultDeviceLabel;
  std::optional<size_t> deviceID;
  std::optional<DeviceSpec> deviceSpec;
  bool tileBasedFusion = false;

  void overrideLoweringOptions(MO::GCPipelineOptions &gcPipelineOptions) const {
    gcPipelineOptions.compiledFrameworkLabel =
        CompiledFrameworkLabel::asLabelString(
            CompiledFrameworkLabel::kModularModel);
    gcPipelineOptions.kernelLibraryPaths = kernelLibraryPaths;
    if (defaultDeviceLabel) {
      gcPipelineOptions.defaultDeviceLabel = *defaultDeviceLabel;
      if (deviceID)
        gcPipelineOptions.defaultDeviceId = *deviceID;
    }
    gcPipelineOptions.defaultDeviceTargetTriple =
        deviceSpec ? deviceSpec.value().target.triple.str() : "";
    gcPipelineOptions.defaultDeviceTargetArch =
        deviceSpec ? deviceSpec.value().target.arch : "";
  }
};

} // namespace

struct MOFContext::Impl final {
  ContextRef maxContext;
  std::unique_ptr<MOFDriverOptions> clOptions;
  const DriverOptions *driverOptions{nullptr};
  std::unique_ptr<MO::GraphCompilerCLOptions> graphCompilerParser;
  std::unique_ptr<MOFConfig> mofConfig;
  bool optionsParsed{false};

  const CommonOptions *getCommonCLOptions() const {
    return driverOptions ? static_cast<const CommonOptions *>(driverOptions)
                         : clOptions.get();
  }

  bool getEnableMEFMetadata() const {
    return driverOptions ? driverOptions->enableMEFMetadata
                         : (clOptions ? clOptions->enableMEFMetadata : true);
  }
};

/// Constructor when invoking from command line.
MOFContext::MOFContext(ContextRef maxContext, int &argc, char **&argv)
    : impl(std::make_unique<Impl>()) {
  impl->maxContext = std::move(maxContext);
  impl->clOptions = std::make_unique<MOFDriverOptions>();
  impl->driverOptions = impl->clOptions.get();
  impl->clOptions->programName = argv[0];
  (void)argc;
  impl->graphCompilerParser =
      std::make_unique<MO::GraphCompilerCLOptions>(*impl->clOptions);
}

/// Constructor when invoking via API.
MOFContext::MOFContext(ContextRef maxContext) : impl(std::make_unique<Impl>()) {
  impl->maxContext = std::move(maxContext);
}

MOFContext::~MOFContext() = default;

const ExecutorDriverOptions *MOFContext::getExecutorDriverOptions() const {
  return impl->driverOptions
             ? static_cast<const ExecutorDriverOptions *>(impl->driverOptions)
             : impl->clOptions.get();
}

void MOFContext::setDriverOptions(const DriverOptions &options) {
  impl->driverOptions = &options;
}

ErrorOrSuccess MOFContext::setOptions(ArrayRef<std::string> opts) {
  if (opts.empty())
    return success();

  if (!impl->clOptions)
    return Error("CL Options parser is not initialized");

  if (impl->optionsParsed)
    return success();

  for (StringRef opt : opts) {
    const char *optChars[] = {impl->clOptions->getProgramName().data(),
                              opt.data()};
    if (!llvm::cl::ParseCommandLineOptions(std::size(optChars), optChars))
      return Error("unable to parse argument: " + opt);
  }

  impl->optionsParsed = true;
  return success();
}

ErrorOrSuccess
MOFContext::setConfigs(ArrayRef<std::unique_ptr<DriverConfig>> configs) {
  auto &mofConfig = impl->mofConfig;
  if (!mofConfig)
    mofConfig = std::make_unique<MOFConfig>();

  for (auto &driverConfig : configs) {
    if (auto kernelLibraryConfig =
            dyn_cast<KernelLibraryDriverConfig>(driverConfig.get())) {
      mofConfig->kernelLibraryPaths.push_back(
          kernelLibraryConfig->getPath().string());
    } else if (auto deviceConfig = dyn_cast<DeviceConfig>(driverConfig.get())) {
      mofConfig->defaultDeviceLabel = deviceConfig->getDeviceLabel();
      mofConfig->deviceID = deviceConfig->getDeviceID();
      mofConfig->deviceSpec = deviceConfig->deviceSpec;
    } else if (auto tileBasedFusionConfig =
                   dyn_cast<TileBasedFusionDriverConfig>(driverConfig.get())) {
      mofConfig->tileBasedFusion = tileBasedFusionConfig->getTileBasedFusion();
    }
  }
  return success();
}

void MOFContext::overrideLoweringOptions(
    MO::GCPipelineOptions &gcPipelineOptions) {
  if (impl->driverOptions)
    gcPipelineOptions.minimumCpuAlignment =
        impl->driverOptions->minimumCpuAlignment;
  if (impl->clOptions)
    static_cast<const MO::GraphCompilerOptions &>(*impl->clOptions)
        .overrideLoweringOptions(gcPipelineOptions);
  gcPipelineOptions.compiledFrameworkLabel =
      CompiledFrameworkLabel::asLabelString(
          CompiledFrameworkLabel::kModularModel);
  if (impl->mofConfig)
    impl->mofConfig->overrideLoweringOptions(gcPipelineOptions);

  if (auto configOr = M::Config::open(); !configOr.isError()) {
    if (configOr->getValueAsBool("max-debug.nan-check", false))
      gcPipelineOptions.nanCheck = true;
    StringRef strideStr = configOr->getValue("max-debug.nan-check-stride");
    if (!strideStr.empty()) {
      uint64_t stride = 0;
      if (!strideStr.getAsInteger(/*radix=*/10, stride) && stride > 0)
        gcPipelineOptions.nanCheckStride = static_cast<size_t>(stride);
    }

    if (configOr->getValueAsBool("max-debug.profile-scope-tracing", false))
      gcPipelineOptions.enableProfileScopeTracing = true;
    if (configOr->getValueAsBool(
            "max-debug.profile-scope-tracing-trace-unlabelled", false))
      gcPipelineOptions.profileScopeTracingTraceUnlabelled = true;

    if (configOr->getValueAsBool("max-debug.pre-jit", false))
      gcPipelineOptions.preJit = true;

    // mo-opt's flag doesn't reach the in-process Python compile path.
    StringRef timingOutDir =
        configOr->getValue("max-debug.mlir-timing-out-dir");
    if (!timingOutDir.empty())
      gcPipelineOptions.mlirTimingOutDir = timingOutDir.str();
  }
}

/// Prints the first error diagnostic (like the first Python exception) with a
/// Python stacktrace from the graph api location, else falls back to
/// FormatScopedDiagnosticHandler.
class GraphApiDiagnosticHandler : public mlir::ScopedDiagnosticHandler {
public:
  GraphApiDiagnosticHandler(MLIRContext *ctx)
      : ScopedDiagnosticHandler(ctx, [&](Diagnostic &diag) {
          diagnostics.push_back(std::move(diag));
        }) {}

  std::string formatMessage() const {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    emitDiagnosticToStream(messageStream);
    return message;
  }

private:
  std::vector<mlir::Diagnostic> diagnostics;

  void emitDiagnosticToStream(raw_ostream &os) const {
    auto errorDiagnostics =
        llvm::make_filter_range(diagnostics, [](const Diagnostic &diag) {
          return diag.getSeverity() == mlir::DiagnosticSeverity::Error;
        });
    if (errorDiagnostics.empty()) {
      // Still use fallback diagnostics when there are no errors.
      emitFallbackDiagnosticsToStream(os);
      return;
    }

    for (const Diagnostic &diag : errorDiagnostics) {
      // Python op creation attaches the user call stack to the location
      // (graph.py `_location()` -> `_graph.frame_loc()`).
      if (auto attr = RMO::PythonFrameAttr::fromLocation(diag.getLocation())) {
        os << "\nSource Traceback:\n"
           // Formatted via Python's traceback.format_list.
           << attr.stack_trace() << '\n';
      }
      FormatScopedDiagnosticHandler::emitDiagnosticSeverityToStream(os, diag);

      for (const Diagnostic &note : diag.getNotes()) {
        // Skip notes that would dump the op's IR, judged by the first
        // argument (typically the message).
        if (note.getArguments().empty() ||
            note.getArguments().front().getKind() !=
                mlir::DiagnosticArgument::DiagnosticArgumentKind::String ||
            !note.getArguments().front().getAsString().starts_with(
                "see current operation")) {
          FormatScopedDiagnosticHandler::emitDiagnosticToStream(os, note);
        }
      }
    }
  }

  void emitFallbackDiagnosticsToStream(raw_ostream &os) const {
    int pid = llvm::sys::Process::getProcessId();
    std::string fileName = "max-full-diagnostics-" + Twine(pid).str();

    auto diagFile = TempFile::create(fileName);
    if (diagFile.isError()) {
      os << "Error creating file for full diagnostics (" << fileName
         << "): " << diagFile.takeError() << "\n"
         << "All diagnostics will be emitted to the console.\n";
      for (auto &diag : diagnostics)
        FormatScopedDiagnosticHandler::emitDiagnosticToStream(os, diag);
      return;
    }

    // Minimal console output; everything goes to a file.
    diagFile->keep();
    std::error_code unused;
    llvm::raw_fd_ostream fullOutputStream(diagFile->getPath().c_str(), unused);

    std::string previousDiagnostic;
    for (auto &diag : diagnostics) {
      std::string currentDiagnostic;
      llvm::raw_string_ostream currentDiagnosticStream(currentDiagnostic);
      FormatScopedDiagnosticHandler::emitDiagnosticToStream(
          currentDiagnosticStream, fullOutputStream, diag);

      if (currentDiagnostic == previousDiagnostic)
        continue;
      previousDiagnostic = currentDiagnostic;
      os << currentDiagnostic << "\n";
    }

    os << "================================================\n";
    os << "Elided diagnostics have been emitted to "
       << diagFile->getPath().c_str() << "\n";
    os << "================================================\n";
  }
};

/// Sinks the `tile_based_fusion` compile option into every
/// `mo.device_info_mapping` entry, CPU included, so SIMD can later become a
/// subset of TileTensor.
static void enableTileBasedFusion(ModuleOp moduleOp) {
  MLIRContext *ctx = moduleOp.getContext();
  auto dict =
      moduleOp->getAttrOfType<DictionaryAttr>(MO::DeviceInfoMapping::kAttrName);
  if (!dict)
    return;

  SmallVector<NamedAttribute> entries;
  entries.reserve(dict.size());
  bool changed = false;
  for (NamedAttribute entry : dict) {
    auto info = dyn_cast<DeviceInfoAttr>(entry.getValue());
    if (info && !info.getTileBasedFusion()) {
      entries.emplace_back(entry.getName(),
                           DeviceInfoAttr::get(ctx, info.getLabel(),
                                               info.getApi(), info.getArch(),
                                               info.getModel(),
                                               /*tile_based_fusion=*/true));
      changed = true;
    } else {
      entries.push_back(entry);
    }
  }

  if (changed)
    moduleOp->setAttr(MO::DeviceInfoMapping::kAttrName,
                      DictionaryAttr::get(ctx, entries));
}

/// Lowers a module op to MGP then exports to MEF.
static ErrorOr<BufferRef> convertMOToMEF(ModuleOp moduleOp,
                                         const MO::GCPipelineOptions &options,
                                         bool enableMEFMetadata) {
  bool const dumpStages = !getMaxIRDumpDir().empty();
  GraphApiDiagnosticHandler diagnostics(moduleOp.getContext());
  if (failed(MO::runGCPipeline(moduleOp, options, dumpStages)))
    return Error(Twine("Graph compilation failed:\n") +
                 diagnostics.formatMessage());

  // Nothing to export without the JIT; erroring also keeps the half-lowered
  // result out of the MEF cache. emit-mojo already wrote the Mojo.
  if (options.preJit)
    return Error("compilation stopped after emitting Mojo, as requested by "
                 "pre-jit; the emitted Mojo is in ir-output-dir");

  auto bufOr = MO::convertMGPToMEF(moduleOp, enableMEFMetadata);
  if (bufOr)
    return bufOr.takeError();
  return bufOr.takeValue();
}

M::ErrorOr<M::BufferRef> MOFContext::compileToBinary(void *source,
                                                     FileFormat format) {
  // Cheap: weights are dense_resource attributes owned by the MLIRContext.
  OwningOpRef<ModuleOp> moduleOp =
      ModuleOp::getFromOpaquePointer(source).clone();

  auto &maxContext = impl->maxContext;
  MLIRContext *mlirCtx = moduleOp->getContext();
  registerContext(*mlirCtx, maxContext);

  auto telemetryContext = maxContext->get<GML::MaxEngineTelemetryContext>();

  // Must precede the telemetry diagnostic handler.
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, mlirCtx);

  GML::MaxEngineTelemetryScope<GML::MaxEngineErrorCode::GraphLoweringError>
      loweringDiagScope(mlirCtx, telemetryContext);

  // Gather all the options from all the sources.
  MO::GCPipelineOptions gcPipelineOptions;
  overrideLoweringOptions(gcPipelineOptions);

  // Set iff Python `Analysis::seed_kernel_decls` already seeded kernel decls,
  // so mogg-import-packages can be skipped.
  if ((*moduleOp)->hasAttr(MOGG::kOpaqueTypeMappingAttrName))
    gcPipelineOptions.kernelDeclsPrePopulated = true;

  // Before the cache key, so tile and non-tile compiles never collide.
  if (impl->mofConfig && impl->mofConfig->tileBasedFusion)
    enableTileBasedFusion(*moduleOp);

  bool enableMEFMetadata = impl->getEnableMEFMetadata();

  auto baseCacheDirOr = M::getMaxCacheDir();

  if (!isModelIRCacheEnabled() || !baseCacheDirOr)
    return convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);

  const std::filesystem::path &baseCacheDir = *baseCacheDirOr;
  auto mefBackendOr = Cache::getLocalDefaultBackendChain(
      (baseCacheDir / "mof" / "mef").string(),
      std::string(M::getModularVersionString()));

  if (mefBackendOr.isError()) {
    return convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);
  }

  auto keyHashOr = computeMEFCacheKey(*moduleOp, gcPipelineOptions, maxContext);
  if (keyHashOr.isError())
    return keyHashOr.takeError();
  std::string keyHash = keyHashOr.takeValue();

  auto mefCache = RCRef<Cache::TransformCache>::create(*mefBackendOr);
  auto maybeMefOr = mefCache->findKeyedSync(keyHash);
  if (maybeMefOr.isError())
    return maybeMefOr.takeError();
  std::optional<BufferRef> maybeMef = maybeMefOr.takeValue();
  if (maybeMef) {
    auto onHitLambda = Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
        "mof::M_compileToBinaryFromSource");
    onHitLambda(*moduleOp);

    return std::move(*maybeMef);
  }

  auto onMissLambda = Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
      "mof::M_compileToBinaryFromSource",
      "mof.compiletobinaryfromsource.cache.miss.time",
      {{"pipeline", "mof::M_compileToBinaryFromSource"}});
  onMissLambda(*moduleOp);

  auto outBufOr =
      convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);
  if (outBufOr.isError())
    return outBufOr.takeError();
  BufferRef outputBuffer = outBufOr.takeValue();

  // Insertion failure is ignored.
  (void)mefCache->insertKeyedSync(keyHash, outputBuffer.copy());

  return outputBuffer;
}

ErrorOr<M::BufferRef> MOFContext::compileToBinary(M::BufferRef file,
                                                  FileFormat format) {
  CompileToBinaryCtx compileCtx(impl->maxContext.copy());
  auto telemetryContext =
      impl->maxContext->get<GML::MaxEngineTelemetryContext>();
  // Gather all the options from all the sources.
  MO::GCPipelineOptions gcPipelineOptions;
  overrideLoweringOptions(gcPipelineOptions);

  MOGG::GraphCompilerCacheKey graphCompilerOptions{&compileCtx.mlirCtx};

  auto loweringTransform =
      [&, file = file.copy()]() mutable -> ErrorOr<BufferRef> {
    switch (format) {
    case FileFormat::kMODialect:
      [[fallthrough]];
    case FileFormat::kMGPDialect:
      compileCtx.module = parseMODialect(file.copy(), &compileCtx.mlirCtx);
      break;

    default: {
      auto identityOr = getFileContentIdentity(file.copy(), format);
      if (identityOr.isError())
        return identityOr.takeError();
      return Error("cannot compile input with format " + *identityOr);
    }
    }

    GML::MaxEngineTelemetryScope<GML::MaxEngineErrorCode::GraphLoweringError>
        loweringDiagScope(&compileCtx.mlirCtx, telemetryContext);

    switch (format) {
    case FileFormat::kMODialect: {
      GraphApiDiagnosticHandler diagnostics(&compileCtx.mlirCtx);
      if (failed(runGcPipeline(compileCtx.module.get(), gcPipelineOptions)))
        return Error(Twine("Graph compilation failed") +
                     diagnostics.formatMessage());
      format = FileFormat::kMGPDialect;
      break;
    }
    case FileFormat::kMGPDialect: {
      break;
    }
    default:
      auto identityOr = getFileContentIdentity(file.copy(), format);
      if (identityOr.isError())
        return identityOr.takeError();
      auto identity = std::move(*identityOr);
      return Error("cannot lower to MEF data of format " + identity);
    }

    return MO::convertMGPToMEF(compileCtx.module.get(),
                               impl->getEnableMEFMetadata());
  };

  // Custom kernels would have to be inspected for the key; don't cache.
  if (!gcPipelineOptions.kernelLibraryPaths.empty() ||
      !isModelIRCacheEnabled()) {
    return loweringTransform();
  }

  WriteableBufferRef keyBuffer =
      makeCacheKey(std::move(file), gcPipelineOptions, graphCompilerOptions);

  // Mojo defines (LOGGING_LEVEL, ASSERT_LEVEL, etc.) must invalidate the cache.
  if (auto *compCtx = impl->maxContext->get<KGEN::CompilationContext>())
    compCtx->print(*keyBuffer);

  return M::cacheMEFTransform("mof", loweringTransform, std::move(keyBuffer));
}
