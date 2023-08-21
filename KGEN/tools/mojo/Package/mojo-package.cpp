//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-package.h"
#include "../Common/Compilation.h"
#include "../Common/Telemetry.h"

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
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
  PackageBuilder(LIT::PackageOp parsedPackageOp, TargetInfoAttr target);

  /// Given a pre-elaboration module, attach the bytecode for the pre-elaborated
  /// versions of each non-parametric function to the high level lit.func in
  /// the new package.
  ErrorOrSuccess attachPreElaboratorBytecode(ModuleOp moduleOp);

  /// Given an elaborated module, attach the bytecode for the elaborated
  /// versions of each non-parametric function to the high level lit.func in
  /// the new package.
  ErrorOrSuccess attachElaboratedBytecode(const SymbolTable &symtab,
                                          const ExportMap &exportedSymbols);

  /// Given the module and the static archive bytes corresponding to that
  /// module, generate a resource attribute and attach it to the package we're
  /// building.
  ErrorOrSuccess
  attachCompiledArchiveBytes(ModuleOp theModule, Cache::BufferRef archive,
                             const CompilationOptions &compilationOptions);

  /// Write the package - this takes the ToolOutputFile because if we're
  /// printing to the stdout we want to print the full module (so the dialect
  /// resource is printed), but if we're printing to a file, we simply print
  /// the package bytecode, which will include the resources.
  ErrorOrSuccess writePackage(llvm::ToolOutputFile &out) {
    if (out.getFilename() == "-")
      packageModule->print(out.os());
    else if (failed(mlir::writeBytecodeToFile(thePackage, out.os())))
      return Error("failed to write package bytecode to a file");

    return success();
  }

  /// Get the MLIRContext.
  mlir::MLIRContext *getContext() { return packageModule->getContext(); }

private:
  /// This takes a BufferRef `bytes` and a name, and generates a
  /// DenseResourceElementsAttr. It's a small helper, but it is somewhat fiddly
  /// so it's useful to only write this code once.
  DenseResourceElementsAttr createResourceAttr(Cache::BufferRef bytes,
                                               Twine name);

  /// This is the module that contains the new package we're generating.
  OwningOpRef<ModuleOp> packageModule;
  /// This is a reference to the new package op we've created.
  LIT::PackageOp thePackage;
  /// This maps from a flattened name to the LIT::FuncOp in the package
  /// module.
  DenseMap<StringAttr, LIT::FuncOp> flattenedNameToFunc;
};
} // namespace

//===----------------------------------------------------------------------===//
// PackageBuilder Implementation
//===----------------------------------------------------------------------===//

/// Returns true if the given function can be externalized.
static bool canExternalize(LIT::FuncOp func) {
  // If the function is marked as always inline, we can't externalize it.
  if (func.getInlineLevel() == InlineLevel::Always ||
      func.getInlineLevel() == InlineLevel::AlwaysNoDebug)
    return false;

  // Check for parameters.
  SignatureType signature = func.getSignature();
  if (!signature.getInputParamTypes().empty() ||
      !signature.getResultParamTypes().empty())
    return false;
  // Check if a parent has parameters.
  LIT::StructDeclOp parentStruct = func->getParentOfType<LIT::StructDeclOp>();
  while (parentStruct) {
    if (!parentStruct.getInputParams().empty() ||
        parentStruct.getParamVarargs())
      return false;
    parentStruct = parentStruct->getParentOfType<LIT::StructDeclOp>();
  }
  return true;
}

PackageBuilder::PackageBuilder(LIT::PackageOp parsedPackageOp,
                               TargetInfoAttr target) {
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
  thePackage.setCompiledForAttr(target);
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
          StringAttr postElaborationName = func.getLinkageNameAttr();
          if (!postElaborationName)
            postElaborationName = LIT::MangledSymbol::mangle(func).mangled;
          flattenedNameToFunc.try_emplace(postElaborationName, clonedFunc);
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
  // Prepare the operations within the module for use when importing the
  // package.
  SmallVector<std::pair<ExportInterface, ExportKind>> exportKinds;
  for (ExportInterface op : moduleOp.getOps<ExportInterface>()) {
    auto exportKind = op.getExportKind();
    if (exportKind != ExportKind::NotExported) {
      exportKinds.push_back({op, exportKind});
      op.setNotExported();
    }
  }

  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  Cache::WriteableBufferRef str = Cache::WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(moduleOp, *str)))
    return Error("could not write bytecode for package module");

  // Reset the ops now that we've written to bytecode.
  for (auto [op, exportKind] : exportKinds)
    op.setExportKind(exportKind);

  // Hash the bytecode itself - this will give us a unique'd attr name that
  // shouldn't clash even when a large number of packages get imported - and
  // if they do clash, they're guaranteed to be exactly the same.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef<uint8_t>((const uint8_t *)str->getBufferStart(),
                        (const uint8_t *)str->getBufferEnd()));
  thePackage.setPreElaborationModuleAttr(createResourceAttr(
      std::move(str), "bytecode_" + llvm::toHex(hash, /*LowerCase=*/true)));
  return success();
}

/// Attach the elaborated bytecode to the high-level lit.func ops.
ErrorOrSuccess
PackageBuilder::attachElaboratedBytecode(const SymbolTable &symtab,
                                         const ExportMap &exportedSymbols) {
  ModuleOp theModule = cast<ModuleOp>(symtab.getOp());

  // Prepare the functions within the module for use when importing the package.
  auto packageName = FlatSymbolRefAttr::get(thePackage.getSymNameAttr());
  for (KGEN::FuncOp func : theModule.getOps<KGEN::FuncOp>()) {
    // Attach a reference to the precompiled body to the KGEN::FuncOp.
    func.setPrecompiledBodyRefAttr(packageName);
    func.setExported();
  }

  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  Cache::WriteableBufferRef str = Cache::WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(symtab.getOp(), *str)))
    return Error("could not write bytecode for package module");

  // Reset the precompiled references now that we've written to bytecode.
  for (KGEN::FuncOp func : theModule.getOps<KGEN::FuncOp>())
    func.removePrecompiledBodyRefAttr();

  // Hash the bytecode itself - this will give us a unique'd attr name that
  // shouldn't clash even when a large number of packages get imported - and
  // if they do clash, they're guaranteed to be exactly the same.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef<uint8_t>((const uint8_t *)str->getBufferStart(),
                        (const uint8_t *)str->getBufferEnd()));
  DenseResourceElementsAttr bytecodeResource = createResourceAttr(
      std::move(str), "bytecode_" + llvm::toHex(hash, /*LowerCase=*/true));

  for (auto [symName, exportSym] : exportedSymbols) {
    LIT::FuncOp hlFunc = flattenedNameToFunc.lookup(symName);

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
    hlFunc.setLinkageName(symName);
  }

  thePackage.setPostElaborationModuleAttr(bytecodeResource);
  return success();
}

/// Attach the compiled archive bytes to the new lit.package op.
ErrorOrSuccess PackageBuilder::attachCompiledArchiveBytes(
    ModuleOp theModule, Cache::BufferRef archive,
    const CompilationOptions &compilationOptions) {
  // Get the standalone archive key to use as the archive name.
  Cache::WriteableBufferRef produceStandaloneArchiveKey =
      Cache::WriteableBuffer::get();
  compilationOptions.print(*produceStandaloneArchiveKey
                           << "produceStandaloneArchive(");
  *produceStandaloneArchiveKey << ")";
  if (failed(
          mlir::writeBytecodeToFile(theModule, *produceStandaloneArchiveKey)))
    return Error("failed to write bytecode file");
  // Hash it so the name isn't enormous.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)produceStandaloneArchiveKey->getBufferStart(),
               produceStandaloneArchiveKey->getBufferSize()));

  auto attr = createResourceAttr(
      std::move(archive), "archive_" + llvm::toHex(hash, /*LowerCase=*/true));
  thePackage.setArchiveBytesAttr(attr);
  return success();
}

/// Generate a DenseResourceElementsAttr from `bytes` with the given `name`.
DenseResourceElementsAttr
PackageBuilder::createResourceAttr(Cache::BufferRef bytes, Twine name) {
  mlir::MLIRContext *ctx = packageModule->getContext();

  auto resourceManager =
      mlir::DenseResourceElementsHandle::getManagerInterface(ctx);

  // Pretend this is a "tensor" of data.
  auto attrType =
      RankedTensorType::get({(int64_t)bytes->getBufferSize()},
                            IntegerType::get(ctx, 8, IntegerType::Unsigned));
  auto blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
      ArrayRef<char>(bytes->getBufferStart(), bytes->getBufferSize()),
      /*align=*/8,
      [bytes = bytes.copy()](void *data, size_t size, size_t align) {
        // Drop the ref to the BufferRef to deallocate the bytes.
      });

  // Some convenience typedefs to simplify this code a little bit.
  using HandleTy = mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>;
  auto *dialect = cast<mlir::BuiltinDialect>(resourceManager.getDialect());
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.getBlobManager().insert<HandleTy>(
                    dialect, name.str(), std::move(blob)));
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

  if (!args.hasArg(options::OPT_name))
    return Error("must provide a package name");
  if (args.hasMultipleArgs(options::OPT_name))
    return Error("too many package names, expected exactly one");

  pkgArgs.ctx.loadDialect<KGENDialect>();
  ErrorOr<EnvAttr> envOrErr =
      EnvAttr::parseDefines(&pkgArgs.ctx, args.getAllArgValues(options::OPT_D));
  if (failed(envOrErr))
    return envOrErr.takeError();
  pkgArgs.env = envOrErr.takeValue();

  pkgArgs.name = args.getLastArgValue(options::OPT_name);

  // Reject input files that do not appear to be mojo package directories (this
  // includes stdin "-").
  pkgArgs.inputPath = args.getLastArgValue(options::OPT_INPUT).str();
  if (!isMojoSourcePackagePath(pkgArgs.inputPath)) {
    return Error("'" + pkgArgs.inputPath +
                 "' does not correspond to a Mojo package");
  }

  pkgArgs.outputPath = args.getLastArgValue(options::OPT_o, "-");

  // Set up the compilation options now, so we can use them as a single source
  // of truth.
  return parseCompilationOptions(
      state, args, pkgArgs.compileOptions, sourceMgr, pkgArgs.ctx,
      pkgArgs.target, options::OPT_I, options::OPT_target_triple,
      options::OPT_target_cpu, options::OPT_target_features, options::OPT_march,
      options::OPT_mcpu, options::OPT_mtune, options::OPT_no_optimization,
      options::OPT_debug_level, options::OPT_sanitize);
}

//===----------------------------------------------------------------------===//
// buildPackage
//===----------------------------------------------------------------------===//

/// Elaborate the given module, attaching the generated IR along the way. On
/// success, returns the symbol table and export map after elaboration has run.
static ErrorOr<std::pair<SymbolTable, ExportMap>>
elaboratePackage(ModuleOp theModule, PackageBuilder &packageBuilder,
                 const CompilationOptions &options, LLCL::Runtime &runtime,
                 BuildInfoAttr build, TargetInfoAttr target, EnvAttr env) {
  // Set the target and build info now, so it's included in the cache key.
  theModule->setAttr(EnvAttr::getEnvAttrName(), env);
  setTargetInfo(theModule, target);
  setBuildInfo(theModule, build);

  // Build the backends used for caching compilation.
  auto transformCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".mojo_cache") / "transform").string(),
      KGEN_VERSION_STRING);
  if (transformCacheBackend.isError())
    return transformCacheBackend.takeError();
  auto regionCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".mojo_cache") / "region").string(),
      KGEN_VERSION_STRING);
  if (regionCacheBackend.isError())
    return regionCacheBackend.takeError();
  auto transformCache = LLCL::RCRef<Cache::TransformCache>::create(
      std::move(*transformCacheBackend));
  auto regionCache =
      LLCL::RCRef<Cache::RegionCache>::create(std::move(*regionCacheBackend));

  // Time the compilation.
  [[maybe_unused]] auto timeScope =
      runtime.emplaceContextIfMissing<M::Telemetry::TelemetryContext>()
          .createUInt64Timer<std::chrono::milliseconds>(
              "mojo.kgen.compile.time", M::Telemetry::Level::L2);

  auto runPipeline = [&](mlir::PassManager &pm) -> ErrorOrSuccess {
    LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
        theModule, regionCache.copy(), transformCache.copy(),
        runtime.getReadyChain().copy(), pm, /*deflateTarget=*/false);
    LLCL::await(ready);
    if (ready.isError())
      return ready.takeDiagnostic().getMessage().copy();
    return success();
  };

  // Lower the module up to the elaborator.
  mlir::PassManager preElaboratePM(theModule.getContext());
  populateGenerateLibraryFilePasses(preElaboratePM, runtime, options);
  if (auto err = runPipeline(preElaboratePM))
    return err.takeError();
  if (auto err = packageBuilder.attachPreElaboratorBytecode(theModule))
    return err.takeError();

  // Elaborate the module.
  mlir::PassManager elaboratePM(theModule.getContext());
  populateElaborateModulePasses(elaboratePM, runtime, target, build, options);
  if (auto err = runPipeline(elaboratePM))
    return err.takeError();

  // Construct the symbol table and the export map.
  SymbolTable symtab(theModule);
  ExportMap exportedSymbols = getExportedSymbols(theModule);

  // Attach the elaborated bytecode to the individual functions.
  if (auto err =
          packageBuilder.attachElaboratedBytecode(symtab, exportedSymbols))
    return err.takeError();
  return std::make_pair(std::move(symtab), std::move(exportedSymbols));
}

/// We have all the arguments and all the state we need, we can now start
/// building the package itself.
static ErrorOrSuccess buildPackage(const PackageArgs &packageArgs,
                                   ModuleOp theModule,
                                   LIT::PackageOp parsedPackageOp,
                                   llvm::ToolOutputFile &out,
                                   LLCL::Runtime &runtime) {
  // Set up the package builder.
  PackageBuilder packageBuilder(parsedPackageOp, packageArgs.target);
  mlir::MLIRContext *ctx = packageBuilder.getContext();
  const CompilationOptions &compilationOptions = packageArgs.compileOptions;

  // For now we implicilty export everything in the package, so add exports to
  // the main module for the contents of the module.
  parsedPackageOp.walk<mlir::WalkOrder::PreOrder>([&](LIT::FuncOp func) {
    if (canExternalize(func))
      func.setExported();
    return WalkResult::skip();
  });

  // Elaborate the package, attaching the generated IR along the way.
  auto symTabAndExportedSymbolsOr =
      elaboratePackage(theModule, packageBuilder, compilationOptions, runtime,
                       BuildInfoAttr::getForCurrentBuild(ctx),
                       packageArgs.target, packageArgs.env);
  if (failed(symTabAndExportedSymbolsOr)) {
    return Error(llvm::formatv("compilation failed: {0}",
                               symTabAndExportedSymbolsOr.getError()));
  }
  auto [symtab, exportMap] = std::move(*symTabAndExportedSymbolsOr);

  // Now we can start to generate the archive.
  mlir::PassManager archivePM(ctx);
  auto objectCompiler = ObjectCompiler::create(
      runtime, archivePM, ".mojo_cache", compilationOptions, /*isJIT=*/false);
  if (failed(objectCompiler))
    return objectCompiler.takeError();
  ErrorOr<Cache::BufferRef> archiveOr =
      objectCompiler->produceStandaloneArchive(symtab, exportMap);
  if (failed(archiveOr))
    return Error(archiveOr.getError());
  Cache::BufferRef archive = std::move(*archiveOr);

  // Compile the module, and attach the archive to the package op.
  if (auto err = packageBuilder.attachCompiledArchiveBytes(
          theModule, std::move(archive), compilationOptions))
    return err.takeError();

  // Write the module to the output. We write the whole module because we have
  // to get the resources as well.
  if (auto err = packageBuilder.writePackage(out))
    return err.takeError();

  return success();
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

  if (args.hasArg(options::OPT_help, options::OPT_help_text)) {
    return state.printHelp(/*plainText=*/args.hasArg(options::OPT_help_text),
#include "Package/PackageOptionsHelpText.inc"
    );
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  llvm::SourceMgr sourceMgr;
  PackageArgs packageArgs;
  if (auto err = parsePackageArgs(state, args, sourceMgr, packageArgs))
    return state.reportError(err.getError());

  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

  auto &telemetryCtx = runtime.emplaceContext<M::Telemetry::TelemetryContext>();

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  initializeTelemetry(
      telemetryCtx, state, args, /*privateArgs=*/
      {options::OPT_D, options::OPT_I, options::OPT_name, options::OPT_o});

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
      state, args, packageArgs.compileOptions, &packageArgs.ctx, runtime,
      options::OPT_warn_missing_dog_strings, options::OPT_max_notes,
      options::OPT_D, options::OPT_parsing_stdlib,
      [&](MojoParserConfig &parserConfig, mlir::TimingScope &ts) {
        // TODO: We allow naming the package but parser caching doesn't
        // currently take this into account.
        parserConfig.moduleCachingLevel = MojoParserConfig::kCacheNone;

        OwningOpRef<ModuleOp> moduleOp;
        std::tie(moduleOp, packageOp) =
            M::importMojoPackage(packageArgs.inputPath, packageArgs.name,
                                 sourceMgr, parserConfig, ts);
        return moduleOp;
      });
  if (failed(module))
    return state.reportError(module.getError());

  // Build the package from the inputs we just parsed, and write the output to
  // `out`.
  if (auto err = buildPackage(packageArgs, **module, packageOp, *out, runtime))
    return state.reportError(err.getError());

  out->keep();
  return EXIT_SUCCESS;
}

void M::registerPackageSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("package", package);
}
