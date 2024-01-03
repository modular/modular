//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-package.h"
#include "../Common/Compilation.h"
#include "../Common/Telemetry.h"

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/Compiler/ExecutionEngine.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Package/Package.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Driver/DriverSupport.h"

#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

#include <filesystem>
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

//===----------------------------------------------------------------------===//
// PackageBuilder Declaration
//===----------------------------------------------------------------------===//

namespace {
/// This class provides a context for building a package alongside the normal
/// compilation pipeline.
class PackageBuilder {
public:
  /// Construct the PackageBuilder from the parsed package op.
  PackageBuilder(LIT::PackageOp parsedPackageOp, TargetInfoAttr target,
                 EnvAttr env, const CompilationOptions &options);

  /// Given a pre-elaboration module, attach the bytecode for the pre-elaborated
  /// versions of each non-parametric function to the high level lit.func in
  /// the new package.
  ErrorOrSuccess attachPreElaboratorBytecode(ModuleOp moduleOp);

  /// Given an elaborated module, returns an attribute storing its bytecode, or
  /// an error if one could not be created.
  ///
  /// This also sets the new package name and the appropriate linkage on each
  /// non-parametric `lit.func` op in the given symbol table.
  ErrorOr<DenseResourceElementsAttr>
  createPostElaborationModuleAttr(const SymbolTable &symtab,
                                  const ExportMap &exportedSymbols);

  /// Sets the given archive on the package op that's being built.
  void attachArchive(PackageArchiveAttr archive);

  /// Returns an owning reference to the module that contains the newly created
  /// package, as well as the package itself. This releases the builer's owning
  /// reference to the module, and thus invalidates the builder.
  std::pair<OwningOpRef<ModuleOp>, LIT::PackageOp> build() {
    return {packageModule.release(), thePackage};
  }

  /// Get the MLIRContext.
  mlir::MLIRContext *getContext() { return packageModule->getContext(); }

private:
  /// This is the module that contains the new package we're generating.
  OwningOpRef<ModuleOp> packageModule;
  /// This is a reference to the new package op we've created.
  LIT::PackageOp thePackage;
  /// This maps from a flattened name to the LIT::FuncOp in the package
  /// module.
  DenseMap<StringAttr, std::pair<LIT::FuncOp, StringAttr>> flattenedNameToFunc;
};
} // namespace

//===----------------------------------------------------------------------===//
// PackageBuilder Implementation
//===----------------------------------------------------------------------===//

/// Returns true if the given function can be externalized.
static bool canExternalize(LIT::FuncOp func) {
  // If the function is marked as always inline, we can't externalize it.
  if (func.getInlineLevel() == InlineLevel::Always ||
      func.getInlineLevel() == InlineLevel::AlwaysNoDebug ||
      func.getIsAdaptive())
    return false;

  // Check for parameters.
  SignatureType signature = func.getSignature();
  if (!signature.getInputParamTypes().empty() ||
      !signature.getResultParamTypes().empty())
    return false;
  // Check if a parent has parameters.
  LIT::StructDeclOp parentStruct = func->getParentOfType<LIT::StructDeclOp>();
  while (parentStruct) {
    if (!parentStruct.getInputParams().empty())
      return false;
    parentStruct = parentStruct->getParentOfType<LIT::StructDeclOp>();
  }
  return true;
}

PackageBuilder::PackageBuilder(LIT::PackageOp parsedPackageOp,
                               TargetInfoAttr target, EnvAttr env,
                               const CompilationOptions &options) {
  packageModule = ModuleOp::create(parsedPackageOp->getLoc());
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
  thePackage = cloneWithoutRegions(parsedPackageOp);
  thePackage.setCompiledEnvAttr(env);
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
        // It's a func? OK - non-parametric funcs get elided, parametric funcs
        // are cloned as-is.
        .Case([&](LIT::FuncOp func) {
          // If the function is non-parametric, drop its body.
          LIT::FuncOp clonedFunc;
          if (canExternalize(func)) {
            // This will reset the insertion point to where it was before we
            // entered the function.
            OpBuilder::InsertionGuard guard(b);

            Block *bodyBlock = new Block();
            for (BlockArgument arg : func.getArguments())
              bodyBlock->addArgument(arg.getType(), arg.getLoc());

            // Add a block that only contains a lit.extern_func in it.
            clonedFunc = b.cloneWithoutRegions(func);
            clonedFunc.getBodyRegion().push_back(bodyBlock);
            b.setInsertionPointToStart(clonedFunc.getBody());
            b.create<LIT::ExternFuncOp>(clonedFunc.getLoc());
          } else {
            clonedFunc = cast<LIT::FuncOp>(b.clone(*func));
          }

          // Map the function to the alias it will have. Otherwise, use the
          // mangled version of the original func, because that's what its name
          // will be post-elaboration.
          StringAttr preElaborationName = func.getLinkageNameAttr();
          if (!preElaborationName)
            preElaborationName = LIT::MangledSymbol::mangle(func).mangled;
          StringAttr postElaborationName = preElaborationName;
          // If we are sanitizing symbols during elaboration, the
          // post-elaboration name will be different than the pre-elaboration
          // name.
          flattenedNameToFunc.insert(
              {postElaborationName, {clonedFunc, preElaborationName}});
        })
        // Drop export ops unconditionally.
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
}

ErrorOrSuccess PackageBuilder::attachPreElaboratorBytecode(ModuleOp moduleOp) {
  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  WriteableBufferRef str = WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(moduleOp, *str)))
    return Error("could not write bytecode for package module");

  // Hash the bytecode itself - this will give us a unique'd attr name that
  // shouldn't clash even when a large number of packages get imported - and
  // if they do clash, they're guaranteed to be exactly the same.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef<uint8_t>((const uint8_t *)str->getBufferStart(),
                        (const uint8_t *)str->getBufferEnd()));
  thePackage.setPreElaborationModuleAttr(
      createResourceAttr(moduleOp.getContext(), str->getBuffer(),
                         "bytecode_" + llvm::toHex(hash, /*LowerCase=*/true)));
  return success();
}

ErrorOr<DenseResourceElementsAttr>
PackageBuilder::createPostElaborationModuleAttr(
    const SymbolTable &symtab, const ExportMap &exportedSymbols) {
  auto packageName = FlatSymbolRefAttr::get(thePackage.getSymNameAttr());
  auto bytecodeResourceOr = createElaboratedBytecodeAttr(symtab, packageName);
  if (bytecodeResourceOr.isError())
    return bytecodeResourceOr.takeError();
  DenseResourceElementsAttr bytecodeResource = bytecodeResourceOr.takeValue();

  for (auto [symName, exportSym] : exportedSymbols) {
    auto [hlFunc, preElaborationName] = flattenedNameToFunc.lookup(symName);

    // We only care about functions in the package.
    if (!hlFunc)
      continue;

    // If the thing is parametric, then we don't care about it.
    if (!isa_and_nonnull<LIT::ExternFuncOp>(hlFunc.getBody()->getTerminator()))
      continue;
    // Make sure we actually compiled this function.
    if (!symtab.lookup<KGEN::FuncOp>(symName))
      return Error("could not find kgen.func with name " + symName.getValue());

    hlFunc.setPreCompiledModuleRefAttr(packageName);
    hlFunc.setPreElaborationNameAttr(preElaborationName);
    hlFunc.setLinkageName(symName);
  }

  return bytecodeResource;
}

void PackageBuilder::attachArchive(PackageArchiveAttr archive) {
  thePackage.setArchives(archive);
}

//===----------------------------------------------------------------------===//
// parsePackageArgs
//===----------------------------------------------------------------------===//

namespace {
/// This struct provides an in-memory representation of the arguments passed to
/// the `package` subcommand for structured access.
struct PackageArgs {
  std::string name;
  std::string inputPath;
  std::string outputPath;
  CompilationOptions compileOptions;
  mlir::MLIRContext ctx;
  TargetInfoAttr target;
  EnvAttr env;
};
} // namespace

/// Parse the `package` subcommand arguments into a struct.
static ErrorOrSuccess parsePackageArgs(const State &state,
                                       const llvm::opt::InputArgList &args,
                                       llvm::SourceMgr &sourceMgr,
                                       PackageArgs &pkgArgs) {
  if (!args.hasArg(options::OPT_INPUT))
    return Error("no input directory provided");
  if (args.hasMultipleArgs(options::OPT_INPUT))
    return Error("too many inputs, expected exactly one");

  pkgArgs.ctx.loadDialect<KGENDialect>();
  ErrorOr<EnvAttr> envOrErr =
      EnvAttr::parseDefines(&pkgArgs.ctx, args.getAllArgValues(options::OPT_D));
  if (failed(envOrErr))
    return envOrErr.takeError();
  pkgArgs.env = envOrErr.takeValue();

  // Reject input files that do not appear to be mojo package directories (this
  // includes stdin "-").
  pkgArgs.inputPath = args.getLastArgValue(options::OPT_INPUT).str();
  if (!LIT::isMojoSourcePackagePath(pkgArgs.inputPath)) {
    return Error("'" + pkgArgs.inputPath +
                 "' does not correspond to a Mojo package");
  }

  // Use the output path the user specified, or if none was specified, output
  // "directory-name.mojopkg".
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
  return parseCompilationOptions(
      state, args, pkgArgs.compileOptions, sourceMgr, pkgArgs.ctx,
      pkgArgs.target, options::OPT_I, options::OPT_target_triple,
      options::OPT_target_cpu, options::OPT_target_features, options::OPT_march,
      options::OPT_mcpu, options::OPT_mtune, options::OPT_no_optimization,
      options::OPT_debug_level, options::OPT_sanitize,
      options::OPT_debug_info_language);
}

//===----------------------------------------------------------------------===//
// buildPackage
//===----------------------------------------------------------------------===//

/// Elaborate the given module, attaching the generated IR along the way. On
/// success, returns the symbol table and export map after elaboration has run.
static ErrorOr<std::tuple<DenseResourceElementsAttr, SymbolTable, ExportMap>>
elaboratePackage(ModuleOp theModule, PackageBuilder &packageBuilder,
                 const CompilationOptions &options, LLCL::Runtime &runtime,
                 TargetInfoAttr target, EnvAttr env) {
  // Build the backends used for caching compilation.
  auto cacheBackends = getMojoCacheBackends(runtime);
  if (cacheBackends.isError())
    return cacheBackends.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(cacheBackends->first));
  auto regionCache =
      RCRef<Cache::RegionCache>::create(std::move(cacheBackends->second));

  auto fileLine = theModule.getLoc()->findInstanceOf<mlir::FileLineColLoc>();

  llvm::StringMap<Telemetry::MetricAttributeValue> attrs = {
      {"filename", fileLine.getFilename().str()}};

  // Time the compilation.
  [[maybe_unused]] auto timeScope =
      runtime.emplaceContextIfMissing<M::Telemetry::TelemetryContext>()
          .createUInt64Timer<std::chrono::milliseconds>(
              "mojo.kgen.compile.time", M::Telemetry::Level::L2, attrs);

  auto runPipeline = [&](mlir::PassManager &pm) -> ErrorOrSuccess {
    LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
        theModule, regionCache.copy(), transformCache.copy(),
        runtime.getReadyChain().copy(), pm, /*deflateTarget=*/false);
    LLCL::await(ready);
    if (ready.isError())
      return ready.takeDiagnostic().getMessage().copy();
    return success();
  };

  // Lower the module up to the elaborator, and be sure to include the
  // environment attribute in the pre-elaborated bytecode.
  theModule->setAttr(EnvAttr::getEnvAttrName(), env);
  mlir::PassManager preElaboratePM(theModule.getContext());
  buildGenerateLibraryPipeline(preElaboratePM, runtime, options);
  if (auto err = runPipeline(preElaboratePM))
    return err.takeError();
  if (auto err = packageBuilder.attachPreElaboratorBytecode(theModule))
    return err.takeError();

  // Elaborate the module for the given target.
  setTargetInfo(theModule, target);

  mlir::PassManager elaboratePM(theModule.getContext());
  populateElaborateModulePasses(elaboratePM, runtime, target, options);
  if (auto err = runPipeline(elaboratePM))
    return err.takeError();

  // Construct the symbol table and the export map.
  SymbolTable symtab(theModule);
  ExportMap exportedSymbols = getExportedSymbols(theModule);

  // Create the elaborated bytecode attribute, and update the functions in the
  // symbol table.
  auto attrOr =
      packageBuilder.createPostElaborationModuleAttr(symtab, exportedSymbols);
  if (attrOr.isError())
    return attrOr.takeError();
  return std::make_tuple(attrOr.takeValue(), std::move(symtab),
                         std::move(exportedSymbols));
}

/// Given parsed module and package ops, returns either a module and package op
/// "built" for the given target, or an error.
///
/// Here, "building" a package means:
/// 1. Running the package through both the pre-elaboration and elaboration
///    phases of the KGEN compiler, and setting the resulting MLIR bytecode of
///    each of these as attributes on the generated package op.
/// 2. Generating a standalone archive that can be included in a final Mojo
///    program, and setting those bytes as an attribute on the generated package
///    op.
static ErrorOr<std::pair<OwningOpRef<ModuleOp>, LIT::PackageOp>>
buildPackage(const PackageArgs &packageArgs, ModuleOp theModule,
             LIT::PackageOp parsedPackageOp, LLCL::Runtime &runtime) {
  // Set up the package builder.
  PackageBuilder packageBuilder(parsedPackageOp, packageArgs.target,
                                packageArgs.env, packageArgs.compileOptions);
  const CompilationOptions &compilationOptions = packageArgs.compileOptions;

  // For now we implicilty export everything in the package, so add exports to
  // the main module for the contents of the module.
  parsedPackageOp.walk<mlir::WalkOrder::PreOrder>([&](LIT::FuncOp func) {
    if (canExternalize(func))
      func.setExported();
    return WalkResult::skip();
  });

  // Elaborate the package, attaching the generated IR along the way.
  auto elaboratedOr =
      elaboratePackage(theModule, packageBuilder, compilationOptions, runtime,
                       packageArgs.target, packageArgs.env);
  if (failed(elaboratedOr)) {
    return Error(
        llvm::formatv("compilation failed: {0}", elaboratedOr.getError()));
  }
  auto [elaboratedBytecode, symtab, exportMap] = std::move(*elaboratedOr);

  auto archiveOr =
      createPackageArchive(symtab, exportMap, packageArgs.target,
                           elaboratedBytecode, compilationOptions, runtime);
  if (archiveOr.isError())
    return archiveOr.takeError();
  packageBuilder.attachArchive(archiveOr.takeValue());
  return packageBuilder.build();
}

//===----------------------------------------------------------------------===//
// package
//===----------------------------------------------------------------------===//

/// Given the path to a mojo directory, compiles it into a precompiled mojo
/// package op by generating an archive and attaching those bytes to a new
/// top-level `lit.package`, suitable for consumption by other mojo
/// programs.
static int package(const State &state) {
  //===--------------------------------------------------------------------===//
  // Options Parsing
  //===--------------------------------------------------------------------===//

  // Parse command line arguments.
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

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  llvm::SourceMgr sourceMgr;
  PackageArgs packageArgs;
  if (auto err = parsePackageArgs(state, args, sourceMgr, packageArgs))
    return state.reportError(err.getError());

  std::unique_ptr<LLCL::Runtime> runtime = LLCL::createRuntime();
  auto &telemetryCtx =
      runtime->emplaceContext<M::Telemetry::TelemetryContext>();

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  initializeTelemetry(telemetryCtx, state, args, /*privateArgs=*/
                      {options::OPT_D, options::OPT_I, options::OPT_o});

  //===--------------------------------------------------------------------===//
  // Build the package
  //===--------------------------------------------------------------------===//

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out =
      mlir::openOutputFile(packageArgs.outputPath, &outputError);
  if (!out)
    return state.reportError(outputError);

  // Parse the package.
  LIT::PackageOp packageOp;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
                                                    &packageArgs.ctx);
  ErrorOr<OwningOpRef<ModuleOp>> module = invokeMojoParser(
      state, args, packageArgs.compileOptions, &packageArgs.ctx, *runtime,
      options::OPT_warn_missing_dog_strings, options::OPT_max_notes,
      options::OPT_D, options::OPT_parsing_stdlib,
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        OwningOpRef<ModuleOp> moduleOp;
        std::tie(moduleOp, packageOp) =
            LIT::importMojoPackage(packageArgs.inputPath, packageArgs.name,
                                   sourceMgr, parserConfig, ts);
        return moduleOp;
      });
  if (failed(module))
    return state.reportError(module.getError());

  // Build the package from the inputs we just parsed, and write the output to
  // `out`.
  auto builtOrErr = buildPackage(packageArgs, **module, packageOp, *runtime);
  if (failed(builtOrErr))
    return state.reportError(builtOrErr.getError());
  auto [builtModule, builtPackage] = builtOrErr.takeValue();

  if (failed(mlir::writeBytecodeToFile(builtPackage, out->os())))
    return state.reportError("failed to write package bytecode to a file");

  out->keep();
  return EXIT_SUCCESS;
}

void M::registerPackageSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("package", package);
}
