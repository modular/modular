//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-package.h"
#include "../../common/Telemetry.h"
#include "../Common/Compilation.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Init/Init.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Cache/CachedTransform.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Config.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"

#include "Support/Filesystem/Paths.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>
#include <stack>
#include <tuple>
#include <utility>

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Options includes + OptTable
//===----------------------------------------------------------------------===//

#define DRIVER_OPTIONS_PATH "Package/PackageOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct PackageOptTable : public llvm::opt::PrecomputedOptTable {
  PackageOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// This function takes a parsed `lit.package` op and creates  a new
/// `lit.package` op. This new `lit.package` op is one that can be serialized
/// into MLIR bytecode and written to disk, as a `.mojopkg` file. The generated
/// package only contains stubs of the original package's contents, and is
/// suitable for importing into other Mojo programs.
static std::pair<OwningOpRef<ModuleOp>, LIT::PackageOp>
buildPackageModule(LIT::PackageOp parsedPackageOp) {
  OwningOpRef<ModuleOp> packageModule =
      ModuleOp::create(parsedPackageOp->getLoc());
  OpBuilder b(packageModule->getBody(), packageModule->getBody()->begin());

  // Clone the relevant operations into the package.
  std::stack<SmartVariant<Operation *, OpBuilder::InsertPoint>> worklist;

  // Clone an op without its regions, and ensure that once that op is finished
  // processing, we reset the OpBuilder's insert point to where it was before we
  // walked the ops inside `op`.
  auto cloneWithoutRegions = [&](auto op) {
    // Save the insert point on the worklist stack first.
    worklist.push(b.saveInsertionPoint());

    auto clonedOp = b.cloneWithoutRegions(op);
    clonedOp.getBodyRegion().push_back(new Block);
    b.setInsertionPointToStart(clonedOp.getBody());
    return clonedOp;
  };

  // Push the ops in `opList` onto the worklist, while preserving their original
  // order.
  auto pushOpsOntoWorklist = [&](auto opList) {
    // Push onto a temporary stack so we can reverse it when we put it onto the
    // worklist - this ensures the ops keep their original order.
    std::stack<Operation *> tmp;
    for (Operation &op : opList)
      tmp.push(&op);

    while (!tmp.empty()) {
      worklist.emplace(tmp.top());
      tmp.pop();
    }
  };

  // Clone the parsed package operation and push its ops onto the worklist.
  LIT::PackageOp thePackage = cloneWithoutRegions(parsedPackageOp);
  pushOpsOntoWorklist(parsedPackageOp.getOps());

  while (!worklist.empty()) {
    auto listFront = worklist.top();
    worklist.pop();
    if (isa<OpBuilder::InsertPoint>(listFront)) {
      b.restoreInsertionPoint(cast<OpBuilder::InsertPoint>(listFront));
      continue;
    }
    TypeSwitch<Operation *>(cast<Operation *>(listFront))
        // Always clone and recurse in the case of a package, module, or struct.
        .Case<LIT::PackageOp, LIT::FileModuleOp, LIT::StructDeclOp>(
            [&](auto op) {
              cloneWithoutRegions(op);
              pushOpsOntoWorklist(op.getOps());
            })
        .Case([&](LIT::UnresolvedImportOp op) {
          // Drop unresolved imports within packages that were used to lazily
          // pull in nested modules. These aren't needed during packaging
          // because everything is recursively resolved.
          if (isa<LIT::PackageOp>(op->getParentOp()))
            return;
          b.clone(*op);
        })
        // None of the cases matched? Just clone the op directly.
        .Default([&](auto op) { b.clone(*op); });
  }

  // Process the package to strip out various information from the package.
  thePackage.walk([&](LIT::ASTDeclInterface astDeclOp) {
    // Strip out locations from doc strings, these aren't needed anymore.
    auto docAttr = astDeclOp.getDocStringAttr();
    if (docAttr && docAttr.getLocation())
      astDeclOp.setDocStringAttr(LIT::DocStringAttr::get(docAttr.getString()));
  });

  return std::make_pair(std::move(packageModule), thePackage);
}

//===----------------------------------------------------------------------===//
// parsePackageArgs
//===----------------------------------------------------------------------===//

namespace {
/// This struct provides an in-memory representation of the arguments passed to
/// the `package` subcommand for structured access.
struct PackageArgs {
  /// The name of the package being output.
  std::string name;
  /// The path to the Mojo package source directory to parse and output as a
  /// package.
  std::string inputPath;
  /// The path to which to output a `.mojopkg` file.
  std::string outputPath;
  /// Compilation options common to all Mojo builds.
  CompilationOptions compileOptions;
  /// The MLIR context used for compilation.
  mlir::MLIRContext ctx{MLIRContext::Threading::DISABLED};
};
} // namespace

/// Returns whether the given path is an existing directory, or an error if one
/// prevents us from determining anything about the given path.
static ErrorOr<bool> isExistingDirectory(const std::filesystem::path &path) {
  std::error_code ec;
  bool exists = std::filesystem::exists(path, ec);
  if (ec)
    return Error(
        llvm::formatv("could not determine if output path '{0}' exists: {1}",
                      path, ec.message()));
  if (!exists)
    return false;

  bool isDirectory = std::filesystem::is_directory(path, ec);
  if (ec)
    return Error(llvm::formatv(
        "could not determine if output path '{0}' is a directory: {1}", path,
        ec.message()));
  return isDirectory;
}

/// Parse the `package` subcommand arguments into a struct.
static ErrorOrSuccess parsePackageArgs(const State &state,
                                       const llvm::opt::InputArgList &args,
                                       llvm::SourceMgr &sourceMgr,
                                       PackageArgs &pkgArgs) {
  if (!args.hasArg(options::OPT_INPUT))
    return Error("no input directory provided");
  if (args.hasMultipleArgs(options::OPT_INPUT))
    return Error("too many inputs, expected exactly one");

  // Reject input files that do not appear to be mojo package directories (this
  // includes stdin "-").
  pkgArgs.inputPath = args.getLastArgValue(options::OPT_INPUT).str();
  if (!Filesystem::isMojoSourcePackagePath(pkgArgs.inputPath)) {
    return Error("'" + pkgArgs.inputPath +
                 "' does not correspond to a Mojo package");
  }

  // Use the output path the user specified, or if none was specified, output
  // "input-directory-name.mojopkg".
  std::string inputDirName =
      std::filesystem::path(pkgArgs.inputPath).filename().string();
  if (args.hasArg(options::OPT_o)) {
    pkgArgs.outputPath = args.getLastArgValue(options::OPT_o);
    if (pkgArgs.outputPath == "-") {
      // If we're outputting to stdout, use the input directory name as the
      // package name.
      pkgArgs.name = inputDirName;
    } else {
      // Otherwise, validate the output path and infer the package name from it.
      std::filesystem::path outputPath(pkgArgs.outputPath);

      // If the user has specified a directory, output an
      // "input-directory-name.mojopkg" within that directory.
      ErrorOr<bool> isDirectoryOr = isExistingDirectory(outputPath);
      if (isDirectoryOr.isError())
        return isDirectoryOr.takeError();
      if (*isDirectoryOr) {
        outputPath = outputPath / (inputDirName + ".mojopkg");
        pkgArgs.outputPath = outputPath;
      }

      if (outputPath.extension() != ".mojopkg" &&
          outputPath.extension() != ".📦")
        return Error("output path must have a '.mojopkg' or '.📦' extension");

      pkgArgs.name = outputPath.stem().string();
    }
  } else {
    pkgArgs.outputPath = inputDirName + ".mojopkg";
    pkgArgs.name = inputDirName;
  }

  // Set up the compilation options now, so we can use them as a single source
  // of truth.
  if (auto err =
          parseCompilationOptions(state, args, pkgArgs.compileOptions,
                                  sourceMgr, pkgArgs.ctx, options::OPT_I))
    return err.takeError();

  // Packages are built with the intention of being agnostic, so use
  // conservative compilation settings as a default.
  pkgArgs.compileOptions.debugLevel = CompilationOptions::kFullDebugInfo;
  pkgArgs.compileOptions.optimizationLevel = 0;

  return success();
}

//===----------------------------------------------------------------------===//
// buildPackage
//===----------------------------------------------------------------------===//

/// Given parsed module and package ops, builds a new module and package op. The
/// newly build package op is suitable for serialization as MLIR bytecode; it
/// may be written to a `.mojopkg` file that can be deserialized and imported
/// into Mojo programs.
static ErrorOr<std::pair<OwningOpRef<ModuleOp>, LIT::PackageOp>>
buildPackage(const PackageArgs &packageArgs, ModuleOp theModule,
             LIT::PackageOp parsedPackageOp, AsyncRT::Runtime &runtime) {
  // Add the dependencies of the package to the package itself, and strip out
  // any post parser metadata for other package.
  SmallVector<FlatSymbolRefAttr> dependencies;
  for (LIT::PackageOp package : theModule.getOps<LIT::PackageOp>()) {
    if (package == parsedPackageOp || !package.getPostParseModuleAttr())
      continue;
    dependencies.push_back(FlatSymbolRefAttr::get(package.getSymNameAttr()));
    package.removePostParseModuleAttr();
  }
  if (!dependencies.empty()) {
    parsedPackageOp.setDependenciesAttr(
        LinkDependencyArrayAttr::get(theModule.getContext(), dependencies));
  }

  auto [packageModule, thePackage] = buildPackageModule(parsedPackageOp);

  // Attach the post-parse module to the package.
  auto postParseModuleAttr = writeModuleToBytecodeAttr(theModule);
  if (!postParseModuleAttr) {
    return Error(
        "compilation failed: unable to write bytecode for package module");
  }
  thePackage.setPostParseModuleAttr(postParseModuleAttr);

  // Run various check passes now to propagate warnings and errors up to the
  // user.
  KGENCompiler compiler(*theModule.getContext(), packageArgs.compileOptions);
  if (failed(compiler.runCheckLITPipeline(theModule)))
    return Error("errors occurred during compilation");

  return std::make_pair(std::move(packageModule), thePackage);
}

//===----------------------------------------------------------------------===//
// package
//===----------------------------------------------------------------------===//

/// Given the path to a mojo directory, compiles it into a precompiled mojo
/// package op by generating an archive and attaching those bytes to a new
/// top-level `lit.package`, suitable for consumption by other mojo
/// programs.
static int package(const State &subcommandState) {
  //===--------------------------------------------------------------------===//
  // Options Parsing
  //===--------------------------------------------------------------------===//

  // Parse command line arguments.
  State state = subcommandState;
  PackageOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Package/PackageOptionsHelpText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format))
    return result;
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  llvm::SourceMgr sourceMgr;
  sourceMgr.setDiagHandler(getDiagHandler(state.diagnosticFormat));
  PackageArgs packageArgs;
  if (auto err = parsePackageArgs(state, args, sourceMgr, packageArgs))
    return state.reportError(err.getError());

  // Create our context (including the runtime).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(AsyncRT::RuntimeOptions()));
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);
  registerContext(packageArgs.ctx, ctx);

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), args,
      /*privateArgs=*/{options::OPT_I, options::OPT_o});

  //===--------------------------------------------------------------------===//
  // Build the package
  //===--------------------------------------------------------------------===//

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out =
      mlir::openOutputFile(packageArgs.outputPath, &outputError);
  if (!out)
    return state.reportError(outputError);

  // Parse the input directory as a Mojo package. This returns a module op that
  // wraps the `lit.package` op, which represents the package contents.
  AsyncRT::Runtime &runtime = *ctx->get<AsyncRT::Runtime>();
  LIT::PackageOp packageOp;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
                                                    &packageArgs.ctx);
  ErrorOr<OwningOpRef<ModuleOp>> module = invokeMojoParser(
      state, args, packageArgs.compileOptions, &packageArgs.ctx, runtime,
      options::OPT_diagnose_missing_doc_strings,
      options::OPT_validate_doc_strings, options::OPT_max_notes,
      /*definesId=*/llvm::opt::OptSpecifier(),
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        OwningOpRef<ModuleOp> moduleOp;
        std::tie(moduleOp, packageOp) = LIT::importMojoPackage(
            runtime, packageArgs.inputPath, packageArgs.name, sourceMgr,
            parserConfig, ts);
        return moduleOp;
      });
  if (failed(module))
    return state.reportError(module.getError());

  // Build a new package op based off of the parsed package op. This new op is
  // suitable for serialization as MLIR bytecode.
  auto builtOrErr = buildPackage(packageArgs, **module, packageOp, runtime);
  if (failed(builtOrErr))
    return state.reportError(builtOrErr.getError());
  auto [builtPackageModule, builtPackage] = builtOrErr.takeValue();

  // Write the new package op as serialized bytecode to the output file.
  if (failed(mlir::writeBytecodeToFile(builtPackage, out->os())))
    return state.reportError("failed to write package bytecode to a file");

  out->keep();

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  return EXIT_SUCCESS;
}

void M::registerPackageSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("package", package);
}
