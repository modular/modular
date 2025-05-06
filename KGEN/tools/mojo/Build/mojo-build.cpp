//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-build.h"
#include "../../common/Telemetry.h"
#include "../Common/Compilation.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "Cache/CachedTransform.h"
#include "Init/Init.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/Config.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/PlatformLibNames.h"

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
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

#define DEBUG_TYPE "mojo-build"

//===----------------------------------------------------------------------===//
// Command line argument parsing
//===----------------------------------------------------------------------===//

#define DRIVER_OPTIONS_PATH "Build/BuildOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct BuildOptTable : public llvm::opt::PrecomputedOptTable {
  BuildOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
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
          options::OPT_optimization_level, options::OPT_debug_level,
          options::OPT_sanitize, options::OPT_shared_libasan,
          options::OPT_external_libasan, options::OPT_debug_info_language,
          options::OPT_num_threads))
    return state.reportError(err.getError());
  if (ErrorOrSuccess err = parseTargetOptions(
          state, args, compilationOptions, sourceManager, ctx, target,
          options::OPT_target_triple, options::OPT_target_cpu,
          options::OPT_target_features, options::OPT_march, options::OPT_mcpu,
          options::OPT_mtune, options::OPT_target_accelerator,
          options::OPT_mcmodel, options::OPT_large_data_threshold,
          options::OPT_loop_unrolling_warn_threshold))
    return state.reportError(err.getError());
  return {};
}

//===----------------------------------------------------------------------===//
// Mojo program execution
//===----------------------------------------------------------------------===//

// What output file type `mojo build` will generate.
enum class OutputType {
  // Produce an executable file containing machine code, e.g. a `.exe` on
  // Windows, or an extensionless binary on Unix-like operating systems.
  //
  // Produced by default or when `--emit exe` is specified.
  executable,
  // Produce a shared (dynamic) library, with the appropriate file extension
  // for the OS (.dylib, .so, or .dll).
  //
  // Produced when `--emit shared-lib` is specified.
  sharedLibrary,
  // Produce an object file(.o) containing machine code.
  //
  // Produced when `--emit object` is specified.
  object,
  // Also a shared library, but with extra code generated and special file ext
  // and linker options.
  //
  // Produced when `--emit shared-lib` and `--gen-py` are specified.
  pythonExtensionModule,
  // Produce LLVM IR, with the appropriate file extension (.ll).
  //
  // Produced when `--emit llvm` is specified.
  llvm,
  // Produce assembly code, with the appropriate file extension (.s).
  //
  // Produced when `--emit asm` is specified.
  assembly,
};

/// Helper function to create an output file with the given extension
static std::unique_ptr<llvm::ToolOutputFile>
createOutputFile(const State &state, bool hasBinaryOutput,
                 StringRef fileExtension) {
  // Get the input filename
  StringRef inputName;
  for (const char *arg : state.arguments) {
    if (StringRef(arg).starts_with("-"))
      continue;
    inputName = arg;
    break;
  }

  if (inputName.empty()) {
    state.reportError("no input file provided");
    return nullptr;
  }

  // Get the file base name, e.g. `foo` in `foo.mojo`
  StringRef inputBaseName = inputName.rsplit('.').first;

  // Create the output filename
  std::string outputName = (inputBaseName + fileExtension).str();

  // Check if -o was specified
  StringRef outputPath = outputName;
  for (size_t i = 0; i < state.arguments.size(); ++i) {
    if (StringRef(state.arguments[i]) == "-o" &&
        i + 1 < state.arguments.size()) {
      outputPath = state.arguments[i + 1];
      break;
    }
  }

  // Create the output file
  std::error_code ec;
  auto outFile = std::make_unique<llvm::ToolOutputFile>(outputPath, ec,
                                                        llvm::sys::fs::OF_None);

  if (ec) {
    state.reportError("could not open output file: " + ec.message());
    return nullptr;
  }

  return outFile;
}

/// Given a module representing a Mojo program, compile the program to a static
/// archive. Returns an unsuccessful exit code if the archive could not be
/// created successfully, and nullopt otherwise.
static std::optional<int>
compileModuleToArchive(const State &state, AsyncRT::Runtime &runtime,
                       MLIRContext &context, const CompilationOptions &options,
                       OwningOpRef<ModuleOp> module, TargetInfoAttr target,
                       BufferRef &archive, OutputType outputType,
                       const llvm::opt::InputArgList &args) {
  KGENCompiler compiler(context, options);

  // Compile the moduleOp down to the post-elaboration phase, because before
  // that phase we don't have flat symbols.
  ErrorOr<std::unique_ptr<ObjectCompiler>> objectCompilerOr =
      ObjectCompiler::create(".mojo_cache", options, /*isJIT=*/false, context);

  if (objectCompilerOr.isError())
    return state.reportError(objectCompilerOr.getError());

  if (ErrorOrSuccess err = compiler.runKGENPipeline(*module, target))
    return state.reportError(err.getError());

  std::unique_ptr<ObjectCompiler> objectCompiler = objectCompilerOr.takeValue();

  // Generate a symbol table and an export map for the module post-compile.
  SymbolTable symtab(*module);
  switch (outputType) {
  case OutputType::executable:
  case OutputType::object: // NOTE: This isn't a required limitation
    if (!symtab.lookup("main"))
      return state.reportError("module does not contain a 'main' function");
    break;
  case OutputType::sharedLibrary:
  case OutputType::pythonExtensionModule:
    // Python extension modules
    if (symtab.lookup("main"))
      return state.reportError(
          "shared library should not contain a 'main' function");
    break;
  case OutputType::llvm: {
    // Compile Module to LLVM IR
    llvm::LLVMContext llvmCtx;
    ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
        objectCompiler->lowerAllFuncsToLLVM(llvmCtx, *module);
    if (llvmModuleOr.isError())
      return state.reportError(Twine("could not lower funcs to LLVM: ") +
                               llvmModuleOr.getError());

    // Open .ll file
    auto outFile = createOutputFile(state, /*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return state.reportError("could not open .ll output file");

    // Print to .ll file
    std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();
    llvmModule->print(outFile->os(), nullptr);
    outFile->keep();

    // Return with success to avoid the link step
    return EXIT_SUCCESS;
  } break;
  case OutputType::assembly: {
    // Compile Module to Assembly
    auto outFile = createOutputFile(state, /*hasBinaryOutput=*/false, ".s");
    if (!outFile)
      return state.reportError("could not open .s output file");

    if (failed(objectCompiler->emitAssembly(std::move(module), outFile->os())))
      return state.reportError("could not emit assembly");
    outFile->keep();
    return EXIT_SUCCESS;
  } break;
  }

  // Generate an archive for the module.
  auto archiveOr = objectCompiler->emitArchive(std::move(module));
  if (failed(archiveOr))
    return state.reportError("failed to produce an archive for the module: " +
                             Twine(archiveOr.getError()));
  archive = std::move(*archiveOr);
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

/// Given a static archive generated from a mojo module, either
/// 1. Link an executable from that archive.
/// 2. Produce a dynamic library for the Python extension module from that
///    archive.
/// Returns a successful exit code if the executable was linked
/// successfully, otherwise returns a failure code.
static int linkOutput(OutputType outputType, const State &state,
                      const llvm::opt::InputArgList &args,
                      const CompilationOptions &options, BufferRef &archive) {
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

  // Build a default output name based on the input file and the current working
  // directory.
  StringRef inputName = args.getLastArgValue(options::OPT_INPUT);

  // Get the file base name, e.g. `foo` in `foo.mojo`.
  StringRef inputBaseName = inputName.rsplit('.').first;

  std::string defaultOutputName = [outputType, inputBaseName, binaryExt] {
    switch (outputType) {
    case OutputType::executable:
      return (inputBaseName + binaryExt).str();
    case OutputType::sharedLibrary:
      // TODO(MOCO-1772):
      //  Determine this file extension based on the _target_ OS, not the host
      //  that `mojo` itself was compiled for.
      // Returns `foo.(so|dylib|dll)` for a source file called `foo.mojo`.
      return PlatformLibrary::getSharedLibraryName(inputBaseName);
    // Python modules require a .so suffix on all platforms.
    case OutputType::pythonExtensionModule:
      return (inputBaseName + ".so").str();
    case OutputType::llvm:
      return (inputBaseName + ".ll").str();
    case OutputType::object:
      return (inputBaseName + ".o").str();
    case OutputType::assembly:
      return (inputBaseName + ".asm").str();
    }
  }();
  // Validate this is a valid filename using the `path` ctor.
  defaultOutputName = std::filesystem::path(defaultOutputName).filename();

  std::error_code ec;
  std::filesystem::path cwd = std::filesystem::current_path(ec);
  if (!ec)
    defaultOutputName = cwd.append(defaultOutputName);

  // Invoke the system linker to link the archive into an executable or produce
  // a dynamic library using the provided output filename argument. The
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
  llvm::ErrorOr<std::string> linker = config.getLinkerDriver().str();
  if (linker->empty()) {
    linker = llvm::sys::findProgramByName(linkerFilename);
    if (!linker) {
      return state.reportError(
          "unable to find suitable c++ compiler for linking");
    }
  }

  if (outputType == OutputType::object) {
    if (llvm::Error err = llvm::writeToOutput(outputName, [&](raw_ostream &os) {
          os << archive->getBuffer();
          return llvm::Error::success();
        })) {
      return state.reportError("unable to write object file: " +
                               llvm::toString(std::move(err)));
    }

    return EXIT_SUCCESS;
  }

  // Write the archive to a temporary file.
  auto archiveFileOr =
      writeTempFile("mojo_archive-%%%%%%%" + libExt, archive->getBuffer());
  if (archiveFileOr.isError()) {
    return state.reportError("unable to write temporary files for linking: " +
                             Twine(archiveFileOr.getError()));
  }
  std::string archivePath = archiveFileOr->getPath().string();

  // Resolve the path to the CompilerRT library.
  StringRef compilerRTPath = config.getCompilerRTPath();

  if (!std::filesystem::exists(compilerRTPath.str(), ec) || ec)
    return state.reportError("unable to locate Mojo CompilerRT library");

  // Invoke the linker command.
  SmallVector<StringRef> linkerArgs = [&] {
    if (outputType == OutputType::executable)
      return SmallVector<StringRef>{*linker, archivePath, compilerRTPath};

    // Here, we use `--whole-archive` to force every symbol from the `.a` static
    // archive to be included in the resulting library.  In the generated Python
    // bindings case, the exported function symbols otherwise wouldn't appeared
    // "used" by the linker, and so it would get aggressively removed.

    SmallVector<StringRef> linkerInvocation{*linker, "-shared"};

#if defined(__APPLE__)
    linkerInvocation.push_back("-Wl,-force_load");
    linkerInvocation.push_back(archivePath);
#else
    linkerInvocation.push_back("-Wl,--whole-archive");
    linkerInvocation.push_back(archivePath);
    linkerInvocation.push_back("-Wl,--no-whole-archive");
#endif

    linkerInvocation.push_back(compilerRTPath);
    linkerInvocation.push_back("-o");
    linkerInvocation.push_back(outputName);
    return linkerInvocation;
  }();

  // Add other shared libs
  config.getSharedLibraryLinkArgs(linkerArgs);

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
  if (options.sanitizers.has(Sanitizers::kAddress)) {
    if (options.externalLibasan.empty()) {
      linkerArgs.emplace_back("-fsanitize=address");
      if (options.sharedLibasan)
        linkerArgs.emplace_back("-shared-libasan");
    } else {
      linkerArgs.emplace_back(options.externalLibasan);
    }
  }
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

  // Print linker arguments for debugging
  LLVM_DEBUG({
    for (auto arg : linkerArgs) {
      llvm::errs() << arg << " ";
    }
    llvm::errs() << "\n";
  });

  std::string errorMsg;
  int linkExitCode = llvm::sys::ExecuteAndWait(
      *linker, linkerArgs, /*Env=*/std::nullopt, /*Redirects=*/{},
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errorMsg);
  if (linkExitCode) {
    if (!errorMsg.empty())
      errorMsg.insert(0, ": ");
    if (outputType == OutputType::executable)
      return state.reportError("failed to link executable" + errorMsg);
    return state.reportError("failed to produce dynamic library" + errorMsg);
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

  warnBuildingForDebugWithDebugBuiltCompiler(state, options.debugLevel);

  AsyncRT::RuntimeOptions runtimeOptions;
  configureRuntimeOptions(runtimeOptions, options);

  // Create our context (including the runtime).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(runtimeOptions));
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

  StringRef emitFileType =
      args.getLastArgValue(options::OPT_emitted_file_type, "exe");

  OutputType outputType = OutputType::executable;
  if (emitFileType == "exe") {
    // Link an executable from the archive (default).
    outputType = OutputType::executable;
  } else if (emitFileType == "shared-lib") {
    // We have a static archive at this point, go ahead and turn it into a
    // dynamic library.
    outputType = OutputType::sharedLibrary;
  } else if (emitFileType == "llvm") {
    outputType = OutputType::llvm;
  } else if (emitFileType == "object") {
    outputType = OutputType::object;
  } else if (emitFileType == "asm") {
    outputType = OutputType::assembly;
  } else {
    return state.reportError(
        Twine("Unrecognized value for `--emit`. Missing case for: ") +
        emitFileType);
  }

  bool generatePythonBindings =
      args.hasArg(options::OPT_generate_python_extension_module);

  // TODO(MOCO-1375):
  //  Remove this restriction on `--gen-py` when this feature is ready for
  //  public usage.
  if (generatePythonBindings) {
    // This feature is experimental and not intended for general usage yet.
    // To discourage use, gate this behind an additional undocumented
    // environment variable.
    std::optional<std::string> envValue =
        llvm::sys::Process::GetEnv("MODULAR_MOJO_PYBIND");

    if (envValue.value_or("") != "enabled") {
      // Error message is intentionally vague about how to enable support for
      //`--gen-py`.
      return state.reportError("Mojo pybind is not supported yet.");
    }
  }

  // The `--gen-py` flag is only valid when emitting a shared library.
  // If `--gen-py` was validly specified, update `outputType` to track that
  // we're emitting a Python extension module shared library.
  if (generatePythonBindings) {
    if (outputType != OutputType::sharedLibrary) {
      return state.reportError(
          "Mojo Python binding generation is only supported "
          "when emitting a shared library.");
    }
    outputType = OutputType::pythonExtensionModule;
  }

  // Lower the input file to an MLIR module.
  AsyncRT::Runtime &runtime = *ctx->get<AsyncRT::Runtime>();
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlirCtx);
  ErrorOr<OwningOpRef<ModuleOp>> moduleOp = invokeMojoParser(
      state, args, options, &mlirCtx, runtime,
      options::OPT_diagnose_missing_doc_strings,
      options::OPT_validate_doc_strings, options::OPT_max_notes, options::OPT_D,
      options::OPT_strip_file_prefix, options::OPT_disable_builtins,
      options::OPT_mojo_search_paths,
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        return LIT::importMojoFile(runtime, sourceMgr, parserConfig, ts,
                                   nullptr, generatePythonBindings);
      });
  if (failed(moduleOp))
    return state.reportError(moduleOp.getError());

  // Compile the module to a static archive.
  BufferRef archive;
  if (std::optional<int> exitCode = compileModuleToArchive(
          state, runtime, mlirCtx, options, moduleOp.takeValue(), target,
          archive, outputType, args))
    return *exitCode;

  return linkOutput(outputType, state, args, options, archive);
}

void M::registerBuildSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build", build);
}
