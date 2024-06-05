//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-build.h"
#include "../../common/Telemetry.h"
#include "../Common/Compilation.h"

#include "Cache/CachedTransform.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/CompilerSupport/Context.h"
#include "Support/Config.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"
#include "Support/Init/Init.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Command line argument parsing
//===----------------------------------------------------------------------===//

#define DRIVER_OPTIONS_PATH "Build/BuildOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct BuildOptTable : public llvm::opt::PrecomputedOptTable {
  BuildOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Parses the command line arguments from the given `state` object.
static std::optional<int> parseArgs(State &state, llvm::opt::InputArgList &args,
                                    llvm::SourceMgr &sourceManager,
                                    CompilationOptions &compilationOptions,
                                    MLIRContext &ctx, TargetInfoAttr &target) {
  BuildOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  args = options.ParseArgs(state.arguments, missingIndex, missingCount);

  // If `--help` was specified, print help before checking any other arguments.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Build/BuildOptionsHelpText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format))
    return result;
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  if (!args.hasArg(options::OPT_INPUT))
    return state.reportError("no input file provided");
  if (args.hasMultipleArgs(options::OPT_INPUT)) {
    std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
    return state.reportError(llvm::formatv(
        "too many input files, cannot process both '{0}' and '{1}'", inputs[0],
        inputs[1]));
  }

  // Open the provided input file path, or exit with an error if it's not a
  // valid argument that can be opened.
  auto bufferOrErr =
      openMojoInputFile(args.getLastArgValue(options::OPT_INPUT));
  if (failed(bufferOrErr))
    return state.reportError(bufferOrErr.getError());

  // Initialize the source manager with the input file buffer and an appropriate
  // diagnostic handler.
  sourceManager.setDiagHandler(getDiagHandler(state.diagnosticFormat));
  sourceManager.AddNewSourceBuffer(std::move(*bufferOrErr), llvm::SMLoc());

  // Build the compilation options based on the provided arguments.
  if (ErrorOrSuccess err = parseCompilationOptions(
          state, args, compilationOptions, sourceManager, ctx, options::OPT_I,
          options::OPT_no_optimization, options::OPT_debug_level,
          options::OPT_sanitize, options::OPT_debug_info_language))
    return state.reportError(err.getError());
  if (ErrorOrSuccess err = parseTargetOptions(
          state, args, compilationOptions, sourceManager, ctx, target,
          options::OPT_target_triple, options::OPT_target_cpu,
          options::OPT_target_features, options::OPT_march, options::OPT_mcpu,
          options::OPT_mtune))
    return state.reportError(err.getError());
  return {};
}

//===----------------------------------------------------------------------===//
// Mojo program execution
//===----------------------------------------------------------------------===//

/// Given a module representing a Mojo program, compile the program to a static
/// archive. Returns an unsuccessful exit code if the archive could not be
/// created successfully, and nullopt otherwise.
static std::optional<int>
compileModuleToArchive(const State &state, LLCL::Runtime &runtime,
                       MLIRContext &context, const CompilationOptions &options,
                       ModuleOp moduleOp, TargetInfoAttr target,
                       BufferRef &archive) {
  KGENCompiler compiler(context, options);
  mlir::PassManager &pm = compiler.getPassManager();
  configurePassManager(pm);

  // Compile the moduleOp down to the post-elaboration phase, because before
  // that phase we don't have flat symbols.
  auto objectCompiler =
      ObjectCompiler::create(pm, ".mojo_cache", options, false);

  if (ErrorOrSuccess err = compiler.runKGENPipeline(moduleOp, target))
    return state.reportError(err.getError());

  // Generate a symbol table and an export map for the module post-compile.
  SymbolTable symtab(moduleOp);
  if (!symtab.lookup("main"))
    return state.reportError("module does not contain a 'main' function");

  // Generate an archive for the module.
  auto standaloneOr = objectCompiler->produceStandaloneArchive(
      symtab, getExportedSymbols(moduleOp));

  if (failed(standaloneOr))
    return state.reportError("failed to produce an archive for the module: " +
                             Twine(standaloneOr.getError()));
  archive = std::move(*standaloneOr);
  return std::nullopt;
}

#if defined(__APPLE__)
/// Generate a dSYM bundle for the given binary in the same directory.
static int generateDSYM(const State &state, StringRef binaryOutputPath) {
  // Resolve the xcrun path.
  llvm::ErrorOr<std::string> xcrun = llvm::sys::findProgramByName("xcrun");
  if (!xcrun)
    return state.reportError("unable to find xcrun");

  std::string errorMsg;
  // Note: this .dSYM bundle is tied to the specific executable generated
  // above via an embedded UUID.
  std::string dsymBundle = (binaryOutputPath + ".dSYM").str();
  SmallVector<StringRef> xcrunArgs = {*xcrun, "dsymutil", binaryOutputPath,
                                      "-o", dsymBundle};
  int xcrunExitCode = llvm::sys::ExecuteAndWait(
      *xcrun, xcrunArgs, /*Env=*/std::nullopt, /*Redirects=*/{},
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errorMsg);
  if (xcrunExitCode) {
    if (!errorMsg.empty())
      errorMsg.insert(0, ": ");
    return state.reportError("failed to create dSYM bundle" + errorMsg);
  }
  return EXIT_SUCCESS;
}
#endif

/// Given a static archive generated from a mojo module, link an executable from
/// that archive. Returns a successful exit code if the executable was linked
/// successfully, otherwise returns a failure code.
static int linkExecutable(const State &state,
                          const llvm::opt::InputArgList &args,
                          const CompilationOptions &options,
                          BufferRef &archive) {
  // For now we just use the system C++ compiler as the linker on non-windows,
  // which makes it a tad bit easier to link in the necessary system and runtime
  // dependencies of KGENCompilerRT.
#ifdef _WIN32
  StringRef linkerFilename = "link.exe";
  StringRef binaryExt = ".exe";
  StringRef libExt = ".lib";
#else
  StringRef linkerFilename = "c++";
  StringRef binaryExt = "";
  StringRef libExt = ".a";
#endif
  // Read the mojo configuration.
  ErrorOr<MojoConfig> configOr = MojoConfig::open();
  if (failed(configOr)) {
    return state.reportError(Twine("failed to parse 'modular.cfg': ") +
                             configOr.getError());
  }
  MojoConfig config = std::move(*configOr);

  // Resolve the path to the CompilerRT library.
  std::error_code ec;
  StringRef compilerRTPath = config.getStaticCompilerRTPath();
  if (!std::filesystem::exists(compilerRTPath.str(), ec) || ec)
    return state.reportError("unable to locate Mojo CompilerRT library");

  // Build a default output name based on the input file and the current working
  // directory.
  StringRef inputName = args.getLastArgValue(options::OPT_INPUT);
  std::string defaultOutputName =
      std::filesystem::path((inputName.rsplit('.').first + binaryExt).str())
          .filename();
  std::filesystem::path cwd = std::filesystem::current_path(ec);
  if (!ec)
    defaultOutputName = cwd.append(defaultOutputName);

  // Invoke the system linker to link the archive into an executable. The
  // checked linked depends on the target platform.
  StringRef outputName =
      args.getLastArgValue(options::OPT_o, defaultOutputName);

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  // Check that the parent directory of the output exists.
  auto outputDirPath =
      std::filesystem::absolute(outputName.str(), ec).parent_path();
  if (!std::filesystem::exists(outputDirPath, ec) || ec) {
    return state.reportError(
        llvm::formatv("unable to write file. The path '{0}' does not exist.",
                      outputDirPath.string()));
  }

  // Resolve the linker path.
  llvm::ErrorOr<std::string> linker =
      llvm::sys::findProgramByName(linkerFilename);
  if (!linker) {
    return state.reportError(
        "unable to find suitable c++ compiler for linking");
  }

  // Write the archive to a temporary file.
  auto archiveFileOr =
      writeTempFile("mojo_archive-%%%%%%%" + libExt, archive->getBuffer());
  if (archiveFileOr.isError()) {
    return state.reportError("unable to write temporary files for linking: " +
                             Twine(archiveFileOr.getError()));
  }
  std::string archivePath = archiveFileOr->getPath().string();

  // Invoke the linker command.
  SmallVector<StringRef> linkerArgs = {*linker, archivePath, compilerRTPath};

#ifdef _WIN32
  std::string outputArg = ("/out:" + outputName).str();
  linkerArgs.emplace_back(outputArg);
  linkerArgs.emplace_back("/nologo");
  linkerArgs.emplace_back("/SUBSYSTEM:CONSOLE");

  // Ignore `no object files specified; libraries used` warnings.
  linkerArgs.emplace_back("/IGNORE:4001");

// Add the right VCRT to match the one used when building KGENCompilerRT.
#if _DEBUG
  linkerArgs.emplace_back("msvcrtd.lib");
#else
  linkerArgs.emplace_back("msvcrt.lib");
#endif

  // Mojo only supports X86_64 COFF right now.
  linkerArgs.emplace_back("/machine:X64");
#else
  linkerArgs.emplace_back("-o");
  linkerArgs.emplace_back(outputName);

  // Add the necessary sanitizer flags.
  // First we have to match the sanitizer flags used when building
  // KGENCompilerRT, if any.
#if LLVM_ADDRESS_SANITIZER_BUILD
  linkerArgs.emplace_back("-fsanitize=address");
#elif LLVM_MEMORY_SANITIZER_BUILD
  linkerArgs.emplace_back("-fsanitize=memory");
#elif LLVM_THREAD_SANITIZER_BUILD
  linkerArgs.emplace_back("-fsanitize=thread");
#else
  // Otherwise, base this on the compilation options.
  if (options.sanitizers.has(Sanitizers::kAddress))
    linkerArgs.emplace_back("-fsanitize=address");
  if (options.sanitizers.has(Sanitizers::kThread))
    linkerArgs.emplace_back("-fsanitize=thread");
#endif
#endif

  // Apply options for stripping unused code.
#if !defined(_WIN32) &&                                                        \
    !(LLVM_ADDRESS_SANITIZER_BUILD || LLVM_MEMORY_SANITIZER_BUILD ||           \
      LLVM_THREAD_SANITIZER_BUILD)
  // Avoid stripping in sanitizer or debug builds, to avoid dropping symbols
  // that are actually referenced externally.
  if (options.optimizationLevel != 0 &&
      options.debugLevel != M::KGEN::CompilationOptions::kFullDebugInfo &&
      !options.sanitizers) {
    linkerArgs.emplace_back("-ffunction-sections");
    linkerArgs.emplace_back("-fdata-sections");

#if defined(__APPLE__)
    linkerArgs.emplace_back("-Wl,-dead_strip");
#else
    linkerArgs.emplace_back("-Wl,--gc-sections");
#endif // defined(__APPLE__)
  }
#endif // !defined(_WIN32) && !SANITIZER_BUILD

  // Add any necessary system libraries.
  config.getSystemLibraryLinkArgs(linkerArgs);

  std::string errorMsg;
  int linkExitCode = llvm::sys::ExecuteAndWait(
      *linker, linkerArgs, /*Env=*/std::nullopt, /*Redirects=*/{},
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errorMsg);
  if (linkExitCode) {
    if (!errorMsg.empty())
      errorMsg.insert(0, ": ");
    return state.reportError("failed to link executable" + errorMsg);
  }

#if defined(__APPLE__)
  // On macOS, the debug info needs to be generated at link time using dsymutil.
  if (options.debugLevel != CompilationOptions::kNoDebug) {
    if (int code = generateDSYM(state, outputName))
      return code;
  }
#endif

  return EXIT_SUCCESS;
}

/// Given a path to a Mojo source file, open that file, and compile it to an
/// executable. Returns an integer representing a successful exit code if the
/// source file could be compiled without raising an error, otherwise returns a
/// failure code.
static int build(const State &subcommandState) {
  CompilationOptions options;

  // Parse arguments.
  State state = subcommandState;
  MLIRContext mlirCtx{MLIRContext::Threading::DISABLED};
  TargetInfoAttr target;
  llvm::opt::InputArgList args;
  llvm::SourceMgr sourceMgr;
  if (std::optional<int> exitCode =
          parseArgs(state, args, sourceMgr, options, mlirCtx, target))
    return *exitCode;

  // Create our context (including the runtime).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(LLCL::RuntimeOptions()));
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);
  registerContext(mlirCtx, ctx);

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), args, /*privateArgs=*/
      {options::OPT_D, options::OPT_I, options::OPT_o});

  // Lower the input file to an MLIR module.
  LLCL::Runtime &runtime = *ctx->get<LLCL::Runtime>();
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlirCtx);
  ErrorOr<OwningOpRef<ModuleOp>> moduleOp = invokeMojoParser(
      state, args, options, &mlirCtx, runtime,
      options::OPT_diagnose_missing_doc_strings,
      options::OPT_validate_doc_strings, options::OPT_max_notes, options::OPT_D,
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        return LIT::importMojoFile(runtime, sourceMgr, parserConfig, ts);
      });
  if (failed(moduleOp))
    return state.reportError(moduleOp.getError());

  // Compile the module to a static archive.
  BufferRef archive;
  if (std::optional<int> exitCode = compileModuleToArchive(
          state, runtime, mlirCtx, options, **moduleOp, target, archive))
    return *exitCode;

  // Link an executable from the archive.
  return linkExecutable(state, args, options, archive);
}

void M::registerBuildSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build", build);
}
