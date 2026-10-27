//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "mojo-build.h"
#include "../Common/Compilation.h"

#include <algorithm>

#include "AsyncRT/CompilerSupport/Context.h"
#include "Cache/CachedTransform.h"
#include "Init/Init.h"
#include "Mojo/Compiler/KGENCompiler.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/Configuration.h"
#include "Mojo/Support/Constants.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
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
#include "Target/TargetTraits.h"

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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#ifdef KGEN_ENABLE_PASS_OPTIONS
#include "Mojo/ToolCommon/CLOptions.h"
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
struct BuildOptTable : public llvm::opt::OptTable {
  BuildOptTable() : llvm::opt::OptTable(optionTables()) {}
};
} // namespace

/// Parses the command line arguments from the given `state` object.
static std::optional<int> parseArgs(State &state, llvm::opt::InputArgList &args,
                                    llvm::SourceMgr &sourceManager,
                                    CompilationOptions &compilationOptions,
                                    FrontendOptions &frontendOptions,
                                    MLIRContext &ctx, TargetInfoAttr &target,
                                    BuildOptTable &options) {

  // First, parse arguments to check for help flags.
  // We need to do this separately because help text is command-specific.
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList allArgs =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  // Check for help before doing any other processing. The emission-kind
  // table is target-dependent, so it is rendered from the registered
  // TargetTraits and spliced into the `--emit` option's help block.
  auto printHelpWithEmissionKinds = [&](std::u8string_view helpText) {
    std::string text(reinterpret_cast<const char *>(helpText.data()),
                     helpText.size());
    std::string table;
    llvm::raw_string_ostream os(table);
    printSupportedEmissionKinds(os, /*indent=*/12);
    size_t emitPos = text.find("--emit <FILE_TYPE>");
    size_t insertPos = emitPos == std::string::npos
                           ? std::string::npos
                           : text.find("\n\n", emitPos);
    if (insertPos == std::string::npos)
      text += "\nEMISSION KINDS (--emit):\n" + table;
    else
      text.insert(insertPos + 1, table);
    return state.printHelp(std::string_view(text));
  };
  if (allArgs.hasArg(options::OPT_help)) {
    return printHelpWithEmissionKinds(
#include "Build/BuildOptionsHelpText.inc"
    );
  } else if (allArgs.hasArg(options::OPT_help_hidden)) {
    return printHelpWithEmissionKinds(
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
      .ignoreIncompatiblePrecompiledFileErrors =
          options::OPT_ignore_incompatible_precompiled_file_errors,
      .unknown = options::OPT_UNKNOWN,
      .input = options::OPT_INPUT,
      .includeDirs = options::OPT_I,
      .optimizationLevel = options::OPT_optimization_level,
      .fpMode = options::OPT_fp_mode,
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
      .targetAbi = options::OPT_target_abi,
      .march = options::OPT_march,
      .mcpu = options::OPT_mcpu,
      .mtune = options::OPT_mtune,
      .targetAccelerator = options::OPT_target_accelerator,
      .mcmodel = options::OPT_mcmodel,
      .largeDataThreshold = options::OPT_large_data_threshold,
      .relocationModel = options::OPT_relocation_model,
      .printEffectiveTarget = options::OPT_print_effective_target,
      .printSupportedTargets = options::OPT_print_supported_targets,
      .printSupportedCpus = options::OPT_print_supported_cpus,
      .printSupportedAccelerators = options::OPT_print_supported_accelerators,
      .diagnoseMissingDocStrings = options::OPT_diagnose_missing_doc_strings,
      .maxNotes = options::OPT_max_notes,
      .defines = options::OPT_D,
      .stripFilePrefix = options::OPT_strip_file_prefix,
      .disableBuiltins = options::OPT_disable_builtins,
      .fixit = options::OPT_fixit,
      .exportFixit = options::OPT_export_fixit,
      .warnOnUnstableAPIs = options::OPT_warn_on_unstable_apis,
      .ignoreDeprecated = options::OPT_ignore_deprecated,
      .lldPath = options::OPT_lld_path,
      .incrParsingCacheFile = options::OPT_incr_parsing_cache_file,
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
  frontendOptions = std::move(result->frontendOptions);
  target = std::move(result->target);
  return {};
}

//===----------------------------------------------------------------------===//
// Mojo program execution
//===----------------------------------------------------------------------===//

/// Return the output file path for a given extension: the value of `-o` if
/// provided, otherwise `<input-stem><fileExtension>`.
static std::string deriveOutputPath(const llvm::opt::InputArgList &args,
                                    StringRef fileExtension) {
  StringRef inputName = args.getLastArgValue(options::OPT_INPUT);
  StringRef inputBaseName = inputName.rsplit('.').first;
  std::string defaultPath = (inputBaseName + fileExtension).str();
  return args.getLastArgValue(options::OPT_o, defaultPath).str();
}

/// Helper function to create an output file with the given extension
static std::unique_ptr<llvm::ToolOutputFile>
createOutputFile(const State &state, const llvm::opt::InputArgList &args,
                 bool hasBinaryOutput, StringRef fileExtension) {
  if (args.getLastArgValue(options::OPT_INPUT).empty()) {
    state.reportError("no input file provided");
    return nullptr;
  }

  std::string outputPath = deriveOutputPath(args, fileExtension);

  std::error_code ec;
  auto outFile = std::make_unique<llvm::ToolOutputFile>(outputPath, ec,
                                                        llvm::sys::fs::OF_None);
  if (ec) {
    state.reportError("could not open output file: " + ec.message());
    return nullptr;
  }

  return outFile;
}

/// The link-step product a kind requests. Every product compiles to
/// EmitAs::OBJECT first, so EmitAs alone cannot distinguish them.
enum class LinkProduct { kNone, kExecutable, kSharedLibrary, kObjectFile };

/// Given a module representing a Mojo program, compile the program to a static
/// archive. Returns an unsuccessful exit code if the archive could not be
/// created successfully, and nullopt otherwise.
static std::optional<int> compileModuleToArchive(
    const State &state, AsyncRT::CPUDevice &cpuDevice, MLIRContext &context,
    const CompilationOptions &options, OwningOpRef<ModuleOp> module,
    TargetInfoAttr target, BufferRef &archive, LinkProduct product,
    EmitAs emitAs, const TargetBackend *deviceBackend,
    const llvm::opt::InputArgList &args, PassManagerConfigOptions pmOptions) {
  // Set offloadOutputPrefix so compileOffloads() writes offload kernel files
  // alongside the host output; offloadOutputKind selects which kind to
  // produce. Must be set before runKGENPipeline().
  CompilationOptions effectiveOptions = options;
  // Device-only kinds use the accelerator's extension.
  llvm::StringRef deviceExt;
  if (deviceBackend)
    deviceExt = deviceBackend->traits()->extensionFor(emitAs);
  if (emitAs == EmitAs::ASM || emitAs == EmitAs::LLVM ||
      emitAs == EmitAs::LLVM_OPT_BITCODE) {
    llvm::StringRef hostExt = deviceBackend            ? deviceExt
                              : emitAs == EmitAs::ASM  ? ".s"
                              : emitAs == EmitAs::LLVM ? ".ll"
                                                       : ".bc";
    std::string outPath = deriveOutputPath(args, hostExt);
    llvm::SmallString<256> prefix(outPath);
    llvm::sys::path::replace_extension(prefix, "");
    effectiveOptions.offloadOutputPrefix = prefix.str().str();
    effectiveOptions.offloadOutputKind = emitAs;
  }

  KGENCompiler compiler(context, effectiveOptions, pmOptions);

  // Compile the moduleOp down to the post-elaboration phase, because before
  // that phase we don't have flat symbols.
  ErrorOr<std::unique_ptr<ObjectCompiler>> objectCompilerOr =
      ObjectCompiler::create(kMojoCacheBaseDirName, effectiveOptions,
                             /*isJIT=*/false, context, pmOptions);

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

  // Device-only kind: the pipeline wrote the kernel files; no host output.
  // TODO: need to relax it if host is a plugin too
  if (deviceBackend)
    return EXIT_SUCCESS;

  // Generate a symbol table and an export map for the module post-compile.
  // "object" needs no check here: objects can be linked as an executable or a
  // shared library.
  SymbolTable symtab(*module);
  if (product == LinkProduct::kExecutable) {
    if (!symtab.lookup("main"))
      return state.reportError("module does not contain a 'main' function");
  } else if (product == LinkProduct::kSharedLibrary) {
    if (symtab.lookup("main"))
      return state.reportError(
          "shared library should not contain a 'main' function");
  } else if (emitAs == EmitAs::LLVM || emitAs == EmitAs::LLVM_BITCODE ||
             emitAs == EmitAs::LLVM_OPT_BITCODE) {
    // Compile Module to LLVM IR
    llvm::LLVMContext llvmCtx;
    ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
        objectCompiler->lowerAllFuncsToLLVM(llvmCtx, *module);
    if (llvmModuleOr.isError())
      return state.reportError(Twine("could not lower funcs to LLVM: ") +
                               llvmModuleOr.getError());

    const std::string fileExtension = emitAs == EmitAs::LLVM ? ".ll" : ".bc";
    // Open .ll file
    auto outFile =
        createOutputFile(state, args, /*hasBinaryOutput=*/false, fileExtension);
    if (!outFile)
      return state.reportError("could not open .ll output file");

    // Print to .ll file
    std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();
    if (emitAs != EmitAs::LLVM) {
      if (ErrorOrSuccess err =
              objectCompiler->emitBitcode(*llvmModule, outFile->os()))
        return state.reportError(err.getError());
    } else {
      llvmModule->print(outFile->os(), nullptr);
    }
    outFile->keep();

    // Return with success to avoid the link step
    return EXIT_SUCCESS;
  } else if (emitAs == EmitAs::ASM) {
    // Compile Module to Assembly
    auto outFile =
        createOutputFile(state, args, /*hasBinaryOutput=*/false, ".s");
    if (!outFile)
      return state.reportError("could not open .s output file");

    if (failed(objectCompiler->emitAssembly(std::move(module), outFile->os())))
      return state.reportError("could not emit assembly");
    outFile->keep();
    return EXIT_SUCCESS;
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
static int linkOutput(LinkProduct product, const State &state,
                      const llvm::opt::InputArgList &args,
                      const CompilationOptions &options, BufferRef &archive) {
  // For now we just use the system C compiler as the linker on non-windows,
  // which makes it a tad bit easier to link in the necessary system and
  // cpuDevice dependencies of KGENCompilerRT.
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

  // TODO(MOCO-1772):
  //  Determine the shared-library extension based on the _target_ OS, not the
  //  host that `mojo` itself was compiled for.
  // Only the link products reach linkOutput; the codegen kinds return early
  // from compileModuleToArchive.
  std::string defaultOutputName;
  if (product == LinkProduct::kSharedLibrary)
    defaultOutputName =
        PlatformLibrary::getSharedLibraryName(inputBaseName); // foo.(so|dll)
  else if (product == LinkProduct::kObjectFile)
    defaultOutputName = (inputBaseName + ".o").str();
  else
    defaultOutputName = (inputBaseName + binaryExt).str(); // kExecutable
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

  std::vector<std::string> extraLinkerArgs =
      args.getAllArgValues(options::OPT_Xlinker);

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

  if (product == LinkProduct::kObjectFile) {
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
    if (product == LinkProduct::kExecutable)
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

  // The Mojo standard library calls libm entry points such as `hypot` and
  // `expm1`. A C compiler driver doesn't link libm implicitly the way a C++
  // driver does, so request it here. Apple platforms and Windows keep those
  // entry points in libSystem and the CRT, which are already linked.
  if (llvm::Triple triple(options.targetTriple);
      !triple.isOSDarwin() && !triple.isOSWindows())
    linkerArgs.emplace_back("-lm");

  // Add any necessary system libraries.
  config.appendSystemLibraryLinkArgs(linkerArgs);

  // Propagate any user-supplied linker flags. Add these last so they take
  // precedence.
  for (const auto &extraArg : extraLinkerArgs) {
    linkerArgs.emplace_back("-Xlinker");
    linkerArgs.emplace_back(extraArg.c_str());
  }

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
    if (product == LinkProduct::kExecutable)
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
  FrontendOptions frontendOptions;
  BuildOptTable optTable;

  // Parse arguments.
  State state = subcommandState;
  MLIRContext mlirCtx{MLIRContext::Threading::DISABLED};
  TargetInfoAttr target;
  llvm::opt::InputArgList args;
  llvm::SourceMgr sourceMgr;
  if (std::optional<int> exitCode =
          parseArgs(state, args, sourceMgr, options, frontendOptions, mlirCtx,
                    target, optTable))
    return *exitCode;

#ifdef KGEN_ENABLE_PASS_OPTIONS
  const char *cKGENOptions = "KGEN_OPTIONS";
  KGEN::KGENPassCLOptions::registerOptions();
  llvm::cl::ParseCommandLineOptions(0, &cKGENOptions, "", nullptr, nullptr,
                                    cKGENOptions);
#endif // KGEN_ENABLE_PASS_OPTIONS

  warnBuildingForDebugWithDebugBuiltCompiler(state, options.debugLevel);

  // Declared first so it outlives both timing objects, which write their
  // reports into it from their destructors.
  TimingReportSink timingReport;
  if (ErrorOrSuccess err = timingReport.configure(
          args, options::OPT_timing_json, options::OPT_timing_file))
    return state.reportError(err.getError());

  // Before the CPU device, because the option pins the thread count to one;
  // before the MLIR timing, so the MLIR report comes out first.
  LLVMPassTiming llvmTiming;
  llvmTiming.configure(args, options::OPT_llvm_timing, options, timingReport);

  AsyncRT::CPUDeviceOptions cpuDeviceOptions;
  configureCPUDeviceOptions(cpuDeviceOptions, options);

  // Create our context (including the cpuDevice).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withCPUDeviceOptions(cpuDeviceOptions), "build");
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);
  registerContext(mlirCtx, ctx);

  StringRef emitFileType =
      args.getLastArgValue(options::OPT_emitted_file_type, "exe");

  // The only place the link products are named.
  LinkProduct product = llvm::StringSwitch<LinkProduct>(emitFileType)
                            .Case("exe", LinkProduct::kExecutable)
                            .Case("shared-lib", LinkProduct::kSharedLibrary)
                            .Case("object", LinkProduct::kObjectFile)
                            .Default(LinkProduct::kNone);

  // Resolve what the kind compiles to from the traits that declare it: the
  // host's for the kinds every target supports, otherwise the accelerator's,
  // since target-declared kinds are emitted for the accelerator.
  EmitAs emitAs;
  const TargetBackend *deviceBackend = nullptr;
  {
    ErrorOr<const TargetTraits *> hostTraitsOr =
        TargetTraitsRegistry::get().lookup(
            llvm::Triple(llvm::sys::getDefaultTargetTriple()));
    const TargetTraits *hostTraits =
        hostTraitsOr.isError() ? nullptr : *hostTraitsOr;
    if (hostTraits && hostTraits->supportsEmissionKind(emitFileType)) {
      emitAs = *hostTraits->emitAsForKind(emitFileType);
    } else if (isKnownEmissionKind(emitFileType)) {
      StringRef accel = args.getLastArgValue(options::OPT_target_accelerator);
      const TargetTraits *traits = traitsForAcceleratorArch(accel);
      if (!traits && !accel.empty())
        return state.reportError(Twine("unknown accelerator architecture '") +
                                 accel +
                                 "' (see --print-supported-accelerators)");
      if (!traits)
        traits = hostTraits;
      if (!traits)
        return state.reportError("no target traits registered for the host");
      ErrorOr<EmitAs> emitAsOr = traits->emitAsForKind(emitFileType);
      if (emitAsOr.isError())
        return state.reportError(
            Twine(emitAsOr.getError()) +
            "; pass a `--target-accelerator` that supports it "
            "(see --print-supported-accelerators)");
      emitAs = *emitAsOr;
      // Backends hold their own traits instance.
      if (traits != hostTraits) {
        for (const std::unique_ptr<TargetBackend> &backend :
             TargetBackendRegistry::get().backends()) {
          if (backend->ownsOffloadLowering() && backend->traits() &&
              backend->traits()->name() == traits->name())
            deviceBackend = backend.get();
        }
      }
    } else {
      return state.reportError(
          Twine("unknown emission kind '") + emitFileType +
          "' for `--emit`; see the `--emit` section of "
          "`mojo build --help` for the kinds each target accepts");
    }
    // An OBJECT kind that is no link product of this tool is rejected, as is
    // any EmitAs the tool never writes.
    if (emitAs == EmitAs::LLVM_OPT ||
        (emitAs == EmitAs::OBJECT && product == LinkProduct::kNone))
      return state.reportError(
          Twine("`--emit ") + emitFileType +
          "` is not supported by this tool; see the `--emit` section of "
          "`mojo build --help` for the kinds each target accepts");
  }

  // Lower the input file to an MLIR module.
  AsyncRT::CPUDevice &cpuDevice = *ctx->get<AsyncRT::CPUDevice>();
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlirCtx);
  ScopedMLIRWarningHandler warningHandler(&mlirCtx,
                                          frontendOptions.disableWarnings,
                                          frontendOptions.warningsAsErrors);

  // The timing shows the parse, the passes, and the code generation. The
  // manager prints the report at the end of this function.
  MLIRPassTiming timing;
  if (ErrorOrSuccess err =
          timing.configure(args, options::OPT_mlir_timing,
                           options::OPT_mlir_timing_display, timingReport))
    return state.reportError(err.getError());

  ErrorOr<OwningOpRef<ModuleOp>> moduleOp = invokeMojoParser(
      state, args, options, frontendOptions, &mlirCtx, cpuDevice,
      options::OPT_diagnose_missing_doc_strings, options::OPT_max_notes,
      options::OPT_D, options::OPT_strip_file_prefix,
      options::OPT_disable_builtins, options::OPT_mojo_search_paths,
      options::OPT_fixit, options::OPT_export_fixit, &timing.rootScope(),
      [&](const FrontendOptions &parserOptions, mlir::TimingScope &ts) {
        return LIT::importMojoFile(ctx, sourceMgr, &mlirCtx, options,
                                   parserOptions, ts, nullptr);
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
          state, cpuDevice, mlirCtx, options, moduleOp.takeValue(), target,
          archive, product, emitAs, deviceBackend, args,
          timing.passManagerOptions()))
    return *exitCode;

  // Check if any warnings were promoted to errors via -Werror.
  if (warningHandler.wasErrorEmitted())
    return EXIT_FAILURE;

  return linkOutput(product, state, args, options, archive);
}

void M::registerBuildSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("build", build);
}
