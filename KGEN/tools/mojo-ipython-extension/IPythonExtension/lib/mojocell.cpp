//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "Init/Init.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/Support/ForceLinkMLIRC.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/CPython/PythonGIL.h"
#include "Support/CPython/PythonObject.h"
#include "Support/CPython/Util.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include <Python.h>
#include <optional>

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace M::CPython;

#define DEBUG_TYPE "mojo-cell"

enum class OutputType {
  executable,
  pythonExtensionModule,
};

static int build(llvm::StringRef code, llvm::StringRef inputName);

extern "C" __attribute__((visibility("default"))) PyObject *
iPythonMagicMojoCellExecute(PyObject *opts) {
  KGEN::forceLinkMLIRC();

  bool doBuild = false;
  std::string cell;
  std::string inputName;
  std::vector<llvm::StringRef> errors;

  {
    PythonGIL gil;
    if (auto v = getDictBool(opts, "build"))
      doBuild = *v;

    if (auto v = getDictValueAs<std::string>(opts, "cell"))
      cell = std::move(*v);
    else
      errors.push_back("Expected key 'cell' not provided.");

    if (auto v = getDictValueAs<std::string>(opts, "modulename"))
      inputName = std::move(*v);
    else
      errors.push_back("Expected key 'modulename' not provided.");
  }

  // Don't build if there are errors
  doBuild = doBuild && errors.empty();

  [[maybe_unused]] int resultCode = 0;
  if (doBuild) {
    resultCode = build(cell, inputName);
  }

  // Populate the return dict w/ build info
  PythonGIL lock;
  PyObject *result = PyDict_New();

  // This is probably not necessary as it's the same data passed in
  setDictKeyValueString(result, "cell", cell);

  setDictKeyValueBool(result, "built", doBuild);
  setDictKeyValueLong(result, "result_code", resultCode);

  if (!errors.empty())
    setDictKeyValueString(result, "error_msg",
                          llvm::join(errors.begin(), errors.end(), "\n"));

  return result;
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
                      const CompilationOptions &options, BufferRef &archive,
                      StringRef inputName) {
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

  // Only used for Python modules which requires a .so suffix on all platforms.
  StringRef dynamicLibraryExtension = ".so";

  // Read the mojo configuration.
  ErrorOr<MojoConfig> configOr = MojoConfig::open();
  if (failed(configOr)) {
    return state.reportError(Twine("failed to parse 'modular.cfg': ") +
                             configOr.getError());
  }
  MojoConfig config = std::move(*configOr);

  // Resolve the path to the CompilerRT library.
  std::error_code ec;
  StringRef compilerRTPath = config.getCompilerRTPath();

  if (!std::filesystem::exists(compilerRTPath.str(), ec) || ec)
    return state.reportError("unable to locate Mojo CompilerRT library");

  // Build a default output name based on the input file and the current working
  // directory.
  //   StringRef inputName =
  //       "mojocellbindings.mojo"; // args.getLastArgValue(options::OPT_INPUT);

  StringRef outputSuffix = outputType == OutputType::executable
                               ? binaryExt
                               : dynamicLibraryExtension;
  std::string defaultOutputName =
      std::filesystem::path((inputName.rsplit('.').first + outputSuffix).str())
          .filename();
  std::filesystem::path cwd = std::filesystem::current_path(ec);
  if (!ec)
    defaultOutputName = cwd.append(defaultOutputName);

  // Invoke the system linker to link the archive into an executable or produce
  // a dynamic library using the provided output filename argument. The
  // checked linked depends on the target platform.
  StringRef outputName = defaultOutputName;

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  // Check that the parent directory of the output exists.
  auto outputDirPath =
      std::filesystem::absolute(outputName.str(), ec).parent_path();
  if (!std::filesystem::exists(outputDirPath, ec) || ec) {
    // return state.reportError(
    //     llvm::formatv("unable to write file. The path '{0}' does not exist.",
    //                   outputDirPath.string()));
  }
#if 1

  // Resolve the linker path.
  llvm::ErrorOr<std::string> linker = config.getLinkerDriver().str();
  if (linker->empty()) {
    linker = llvm::sys::findProgramByName(linkerFilename);
    if (!linker) {
      return state.reportError(
          "unable to find suitable c++ compiler for linking");
    }
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

  // Print linker arguments for debugging
  /*
  LLVM_DEBUG({
    for (auto arg : linkerArgs) {
      llvm::errs() << arg << " ";
    }
    llvm::errs() << "\n";
  });
  */

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
#endif
  return EXIT_SUCCESS;
}

static int build(llvm::StringRef code, llvm::StringRef inputName) {
  CompilationOptions options;

  // State state{"mojo", {"cell"}};
  State state{{}, {}};

  llvm::opt::InputArgList args;
  llvm::SourceMgr sourceMgr;

  // Initialize the MLIR context.
  MLIRContext mlirCtx{MLIRContext::Threading::DISABLED};
  {
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    registerKGENToLLVMTranslation(registry);
    mlirCtx.appendDialectRegistry(registry);
    mlirCtx.loadDialect<MDialect>();
  }

  /*
  if (std::optional<int> exitCode =
          parseArgs(state, args, sourceMgr, options, mlirCtx, target))
    return *exitCode;
    */

  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(code, inputName);

  sourceMgr.setDiagHandler(getDiagHandler(state.diagnosticFormat));
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  // warnBuildingForDebugWithDebugBuiltCompiler(state, options.debugLevel);

  // Create our context (including the runtime).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(AsyncRT::RuntimeOptions()));
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctxref = std::move(*ctxOr);
  registerContext(mlirCtx, ctxref);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  auto targetOr = getTargetInfoFor(&mlirCtx, options.targetTriple,
                                   options.targetCpu, options.targetFeatures);
  if (targetOr.isError())
    exit(101);
  TargetInfoAttr target = targetOr.takeValue();

  // Lower the input file to an MLIR module.
  AsyncRT::Runtime &runtime = *ctxref->get<AsyncRT::Runtime>();
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &mlirCtx);

  MLIRContext *ctx = &mlirCtx;

  // We don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timing = timingManager.getRootScope();

  DialectRegistry registry;
  registerAllKGENDialects(registry);
  ctx->appendDialectRegistry(registry);

  // Parse the input Mojo file into an MLIR module.
  LIT::ParserConfig parseConfig(ctx, options);

#if 0
  auto docDiagnoseMissingId = options::OPT_diagnose_missing_doc_strings;
  auto docErrorOnInvalidDocId = options::OPT_validate_doc_strings;
  parseConfig.diagnoseMissingDocStrings = args.hasArg(docDiagnoseMissingId);
  parseConfig.errorOnInvalidDocStrings = args.hasArg(docErrorOnInvalidDocId);

  int maxNotes = 0;
  if (!args.getLastArgValue(maxNotesId).getAsInteger(10, maxNotes))
    parseConfig.maxNotesPerDiagnostic = maxNotes;
#endif

  mlir::TimingScope mojoScope = timing.nest("Import Mojo");
  OwningOpRef<ModuleOp> module =
      LIT::importMojoFile(runtime, sourceMgr, parseConfig, timing, nullptr);

  if (!module)
    exit(2);
  // return Error("failed to parse the provided Mojo source module");

  // Tag the module with the environment, which includes any definitions the
  // user may have specified on the command line.
  ctx->loadDialect<KGENDialect>();

#if 0
  if (definesId.isValid()) {
    ErrorOr<EnvAttr> envOrErr = compilationOptions.parseDefinesWithDefaults(
        ctx, args.getAllArgValues(definesId));
    if (failed(envOrErr)) {
      return Error(
          llvm::formatv("an internal error occurred when initializing the Mojo "
                        "MLIR module: {0}",
                        envOrErr.getError()));
    }
    (*module)->setAttr(EnvAttr::getEnvAttrName(), *envOrErr);
  }
#endif
  /*
    if (failed(moduleOp))
      return state.reportError(moduleOp.getError());
      */

  // Compile the module to a static archive.

  KGENCompiler compiler(mlirCtx, options);

  // Compile the moduleOp down to the post-elaboration phase, because before
  // that phase we don't have flat symbols.
  ErrorOr<std::unique_ptr<ObjectCompiler>> objectCompilerOr =
      ObjectCompiler::create(".mojo_cache", options, false, mlirCtx);

  if (objectCompilerOr.isError())
    return state.reportError(objectCompilerOr.getError());

  if (ErrorOrSuccess err = compiler.runKGENPipeline(*module, target))
    return state.reportError(err.getError());

  // Generate a symbol table and an export map for the module post-compile.
  SymbolTable symtab(*module);

  if (symtab.lookup("main"))
    return state.reportError(
        "python extension module should not contain a 'main' function");

  std::unique_ptr<ObjectCompiler> objectCompiler = objectCompilerOr.takeValue();
  // Generate an archive for the module.
  auto archiveOr = objectCompiler->emitArchive(std::move(module));
  if (failed(archiveOr))
    return state.reportError("failed to produce an archive for the module: " +
                             Twine(archiveOr.getError()));

  BufferRef archive = std::move(*archiveOr);

  int result = 0;
  result = linkOutput(OutputType::pythonExtensionModule, state, args, options,
                      archive, inputName);
  return result;
}
