//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-build.h"
#include "../Common/Compilation.h"

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
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "Support/Compiler/Diags.h"
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

#ifdef KGEN_ENABLE_PASS_OPTIONS
#include "KGEN/ToolCommon/CLOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#endif // KGEN_ENABLE_PASS_OPTIONS

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

  // First, parse arguments to check for help flags.
  // We need to do this separately because help text is command-specific.
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList allArgs =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  // Check for help before doing any other processing.
  if (allArgs.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Build/BuildOptionsHelpText.inc"
    );
  } else if (allArgs.hasArg(options::OPT_help_hidden)) {
    return state.printHelp(
#include "Build/BuildOptionsHelpHiddenText.inc"
    );
  }

  // Set up common option IDs.
  CommonOptionIDs optionIDs{
      .help = options::OPT_help,
      .helpHidden = options::OPT_help_hidden,
      .diagnosticFormat = options::OPT_diagnostic_format,
      .disableWarnings = options::OPT_disable_warnings,
      .warningsAsErrors = options::OPT_werror,
      .noWarningsAsErrors = options::OPT_wno_error,
      .unknown = options::OPT_UNKNOWN,
      .input = options::OPT_INPUT,
      .includeDirs = options::OPT_I,
      .optimizationLevel = options::OPT_optimization_level,
      .debugLevel = options::OPT_debug_level,
      .sanitize = options::OPT_sanitize,
      .sharedLibasan = options::OPT_shared_libasan,
      .externalLibasan = options::OPT_external_libasan,
      .bitcodeLibs = options::OPT_bitcode_libs,
      .debugInfoLanguage = options::OPT_debug_info_language,
      .numThreads = options::OPT_num_threads,
      .mojoSearchPaths = options::OPT_mojo_search_paths,
      .loopUnrollingWarnThreshold = options::OPT_loop_unrolling_warn_threshold,
      .elaborationErrorLimit = options::OPT_elaboration_error_limit,
      .elaborationErrorIncludePrelude =
          options::OPT_elaboration_error_include_prelude,
      .elaborationErrorVerbose = options::OPT_elaboration_error_verbose,
      .elaborationMaxDepth = options::OPT_elaboration_max_depth,
      .targetTriple = options::OPT_target_triple,
      .targetCpu = options::OPT_target_cpu,
      .targetFeatures = options::OPT_target_features,
      .march = options::OPT_march,
      .mcpu = options::OPT_mcpu,
      .mtune = options::OPT_mtune,
      .targetAccelerator = options::OPT_target_accelerator,
      .mcmodel = options::OPT_mcmodel,
      .largeDataThreshold = options::OPT_large_data_threshold,
      .diagnoseMissingDocStrings = options::OPT_diagnose_missing_doc_strings,
      .maxNotes = options::OPT_max_notes,
      .defines = options::OPT_D,
      .stripFilePrefix = options::OPT_strip_file_prefix,
      .disableBuiltins = options::OPT_disable_builtins,
      .fixit = options::OPT_fixit,
      .exportFixit = options::OPT_export_fixit,
  };

  // Configure parsing for `mojo build` - parse all arguments normally.
  CommonParseConfig config{
      .parseAllArguments = true,
      .requireSingleInput = true,
  };

  // Parse common arguments.
  ErrorOr<CommonParseResult> result = parseCommonMojoArguments(
      state, sourceManager, ctx, options, optionIDs, config);
  if (failed(result))
    return state.reportError(result.getError());

  if (result->exitCode)
    return *result->exitCode;

  // Extract results.
  args = std::move(result->args);
  compilationOptions = std::move(result->compilationOptions);
  target = std::move(result->target);
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
  // Produce LLVM IR, with the appropriate file extension (.ll).
  //
  // Produced when `--emit llvm` is specified.
  llvm,
  // Produce bitcode of LLVM IR, with the appropriate file extension (.bc).
  //
  // Produced when `--emit llvm` is specified.
  llvmBitcode,
  // Produce assembly code, with the appropriate file extension (.s).
  //
  // Produced when `--emit asm` is specified.
  assembly,
};

/// Helper function to create an output file with the given extension
static std::unique_ptr<llvm::ToolOutputFile>
createOutputFile(const State &state, const llvm::opt::InputArgList &args,
                 bool hasBinaryOutput, StringRef fileExtension) {
  // Get the input filename
  StringRef inputName = args.getLastArgValue(options::OPT_INPUT);

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

  // Extract and set bitcode libraries from the module before compilation.
  if (auto arrayAttr =
          module->getOperation()->getAttrOfType<LLVMBitcodeLibArrayAttr>(
              LLVMBitcodeLibArrayAttr::getBitcodeLibsAttrName()))
    arrayAttr.externalize(objectCompiler->getBitcodeLibs());

  // Generate a symbol table and an export map for the module post-compile.
  SymbolTable symtab(*module);
  switch (outputType) {
  case OutputType::object:
    // Objects can be linked as a executable or shared library.
    break;
  case OutputType::executable:
    if (!symtab.lookup("main"))
      return state.reportError("module does not contain a 'main' function");
    break;
  case OutputType::sharedLibrary:
    if (symtab.lookup("main"))
      return state.reportError(
          "shared library should not contain a 'main' function");
    break;
  case OutputType::llvm:
  case OutputType::llvmBitcode: {
    // Compile Module to LLVM IR
    llvm::LLVMContext llvmCtx;
    ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
        objectCompiler->lowerAllFuncsToLLVM(llvmCtx, *module);
    if (llvmModuleOr.isError())
      return state.reportError(Twine("could not lower funcs to LLVM: ") +
                               llvmModuleOr.getError());

    const std::string fileExtension =
        outputType == OutputType::llvm ? ".ll" : ".bc";
    // Open .ll file
    auto outFile =
        createOutputFile(state, args, /*hasBinaryOutput=*/false, fileExtension);
    if (!outFile)
      return state.reportError("could not open .ll output file");

    // Print to .ll file
    std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();
    if (outputType == OutputType::llvmBitcode) {
      if (ErrorOrSuccess err =
              objectCompiler->emitBitcode(*llvmModule, outFile->os()))
        return state.reportError(err.getError());
    } else {
      llvmModule->print(outFile->os(), nullptr);
    }
    outFile->keep();

    // Return with success to avoid the link step
    return EXIT_SUCCESS;
  } break;
  case OutputType::assembly: {
    // Compile Module to Assembly
    auto outFile =
        createOutputFile(state, args, /*hasBinaryOutput=*/false, ".s");
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
  // For now we just use the system C compiler as the linker on non-windows,
  // which makes it a tad bit easier to link in the necessary system and runtime
  // dependencies of KGENCompilerRT.
#ifdef _WIN32
  StringRef linkerFilename = "link.exe";
  StringRef binaryExt = ".exe";
  StringRef libExt = ".lib";
#else
  StringRef linkerFilename = "cc";
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
    case OutputType::llvm:
      return (inputBaseName + ".ll").str();
    case OutputType::llvmBitcode:
      return (inputBaseName + ".bc").str();
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
          "unable to find suitable c compiler for linking");
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
  config.appendSharedLibraryLinkArgs(linkerArgs);

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

  // Apply options for stripping unused code.
#if defined(__APPLE__)
  linkerArgs.emplace_back("-Wl,-dead_strip");
#else
  linkerArgs.emplace_back("-Wl,--gc-sections");
#endif // defined(__APPLE__)

  // Add any necessary system libraries.
  config.appendSystemLibraryLinkArgs(linkerArgs);

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

#ifdef KGEN_ENABLE_PASS_OPTIONS
  const char *cKGENOptions = "KGEN_OPTIONS";
  KGEN::KGENPassCLOptions::registerOptions();
  llvm::cl::ParseCommandLineOptions(0, &cKGENOptions, "", nullptr, nullptr,
                                    cKGENOptions);
#endif // KGEN_ENABLE_PASS_OPTIONS

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
  } else if (emitFileType == "llvm-bitcode") {
    outputType = OutputType::llvmBitcode;
  } else if (emitFileType == "object") {
    outputType = OutputType::object;
  } else if (emitFileType == "asm") {
    outputType = OutputType::assembly;
  } else {
    return state.reportError(
        Twine("Unrecognized value for `--emit`. Missing case for: ") +
        emitFileType);
  }

  // Lower the input file to an MLIR module.
  AsyncRT::Runtime &runtime = *ctx->get<AsyncRT::Runtime>();
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlirCtx);
  ScopedMLIRWarningHandler warningHandler(&mlirCtx, options.disableWarnings,
                                          options.warningsAsErrors);

  ErrorOr<OwningOpRef<ModuleOp>> moduleOp = invokeMojoParser(
      state, args, options, &mlirCtx, runtime,
      options::OPT_diagnose_missing_doc_strings, options::OPT_max_notes,
      options::OPT_D, options::OPT_strip_file_prefix,
      options::OPT_disable_builtins, options::OPT_mojo_search_paths,
      options::OPT_fixit, options::OPT_export_fixit,
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        return LIT::importMojoFile(runtime, sourceMgr, parserConfig, ts,
                                   nullptr);
      });
  if (failed(moduleOp))
    return state.reportError(moduleOp.getError());

  if (!moduleOp.get()->getOperation()) {
    // Only --experimental-fixit returns a null module (after applying fixes).
    // --experimental-export-fixit continues normal execution after writing
    // YAML.
    assert(args.hasArg(options::OPT_fixit));
    return EXIT_SUCCESS;
  }

  // Compile the module to a static archive.
  BufferRef archive;
  if (std::optional<int> exitCode = compileModuleToArchive(
          state, runtime, mlirCtx, options, moduleOp.takeValue(), target,
          archive, outputType, args))
    return *exitCode;

  // Check if any warnings were promoted to errors via -Werror.
  if (warningHandler.wasErrorEmitted())
    return EXIT_FAILURE;

  return linkOutput(outputType, state, args, options, archive);
}

void M::registerBuildSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build", build);
}
