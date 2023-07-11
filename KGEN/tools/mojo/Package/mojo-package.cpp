//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-package.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
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
  PackageBuilder(LIT::PackageOp parsedPackageOp);

  /// Set the target on the new package.
  void setTarget(TargetInfoAttr target) {
    thePackage.setCompiledForAttr(target);
  }

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

  /// Verify the package module.
  LogicalResult verify() { return mlir::verify(*packageModule); }

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
  if (func.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
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

PackageBuilder::PackageBuilder(LIT::PackageOp parsedPackageOp) {
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
          StringAttr name = func.getPostElaborationNameAttr();
          if (!name)
            name = LIT::MangledSymbol::mangle(func).mangled;
          flattenedNameToFunc.try_emplace(name, clonedFunc);
        })
        // Drop export ops unconditionally.
        .Case([&](ExportOp op) { /* do nothing */ })
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
    if (!hlFunc)
      return Error("could not find lit.func with name " + symName.getValue());

    // If the thing is parametric, then we don't care about it.
    if (!isa_and_nonnull<LIT::ExternFuncOp>(hlFunc.getBody()->getTerminator()))
      continue;
    // Make sure we actually compiled this function.
    if (!symtab.lookup<KGEN::FuncOp>(symName))
      return Error("could not find kgen.func with name " + symName.getValue());

    hlFunc.setPostElaborationModuleRefAttr(packageName);
    hlFunc.setPostElaborationNameAttr(symName);
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
// setupExecutionEngine
//===----------------------------------------------------------------------===//

/// This function sets up the ExecutionEngine. Its remit is initializing the
/// LLVM MC targets, the target machine, the cache backends, and the execution
/// engine itself. This function does not provide an ExecutionEngine suitable
/// for JIT'ing - its purpose is solely to generate binaries for AOT
/// consumption.
static ErrorOr<std::unique_ptr<ExecutionEngine>>
setupExecutionEngine(LLCL::Runtime &runtime, mlir::PassManager &pm,
                     TargetInfoAttr target,
                     const CompilationOptions &compilationOptions) {
  // Now create the execution engine so we can JIT.
  auto tmOr = KGEN::createTargetMachine(compilationOptions,
                                        /*isJIT=*/false);
  if (tmOr.isError())
    return tmOr.takeError();

  auto engineOr = KGEN::ExecutionEngine::create({}, **tmOr);
  if (failed(engineOr))
    return engineOr.takeError();
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // Add the object compiler layer.
  auto compiler =
      ObjectCompiler::create(runtime, pm, ".kgen_cache", compilationOptions);
  if (failed(compiler))
    return compiler.takeError();

  auto &objLayer = engine->addLayer<ObjectCompilerLayer>(
      std::move(*compiler), engine->getLinkingLayer());

  // Notify the object layer that anything we build is not for immediate
  // execution.
  objLayer.notForImmediateExecution();

  // Add the KGEN compiler layer.
  // First though, get the backend chains to pass into the compile layer.
  auto transformCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".kgen_cache") / "transform").string(),
      KGEN_VERSION_STRING);
  if (transformCacheBackend.isError())
    return transformCacheBackend.takeError();

  auto regionCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".kgen_cache") / "region").string(),
      KGEN_VERSION_STRING);
  if (regionCacheBackend.isError())
    return regionCacheBackend.takeError();

  // Get the build info from the current build.
  BuildInfoAttr build = BuildInfoAttr::getForCurrentBuild(pm.getContext());

  engine->addLayer<KGENCompilerLayer>(
      pm, runtime, target, build, compilationOptions, objLayer,
      std::move(*transformCacheBackend), std::move(*regionCacheBackend));
  return std::move(engine);
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
};
} // namespace

/// Parse the `package` subcommand arguments into a struct.
static ErrorOrSuccess parsePackageArgs(const State &state,
                                       const llvm::opt::InputArgList &args,
                                       PackageArgs &pkgArgs) {
  if (!args.hasArg(options::OPT_INPUT))
    return Error("no input directory provided");
  if (args.hasMultipleArgs(options::OPT_INPUT))
    return Error("too many inputs, expected exactly one");

  if (!args.hasArg(options::OPT_name))
    return Error("must provide a package name");
  if (args.hasMultipleArgs(options::OPT_name))
    return Error("too many package names, expected exactly one");

  pkgArgs.name = args.getLastArgValue(options::OPT_name);

  // Reject input files that do not appear to be mojo package directories (this
  // includes stdin "-").
  pkgArgs.inputPath = args.getLastArgValue(options::OPT_INPUT).str();
  if (!isMojoSourcePackagePath(pkgArgs.inputPath)) {
    return Error("'" + pkgArgs.inputPath +
                 "' does not correspond to a Mojo package");
  }

  pkgArgs.outputPath = args.getLastArgValue(options::OPT_o, "-");

  StringRef triple = args.getLastArgValue(options::OPT_triple);
  if (args.hasMultipleArgs(options::OPT_triple))
    return Error("too many specified target triples, expected exactly one");

  StringRef cpu = args.getLastArgValue(options::OPT_cpu);
  if (args.hasMultipleArgs(options::OPT_cpu))
    return Error("too many specified target CPUs, expected exactly one");

  StringRef features = args.getLastArgValue(options::OPT_features);
  if (args.hasMultipleArgs(options::OPT_features))
    return Error("too many specified target features, expected exactly one");

  // Set up the compilation options now, so we can use them as a single source
  // of truth.
  CompilationOptions &compilationOptions = pkgArgs.compileOptions;
  // If the user specified the triple, the target CPU, or the target feature
  // set, use those to override the defaults.
  if (!triple.empty())
    compilationOptions.targetTriple = triple.str();
  if (!cpu.empty())
    compilationOptions.targetCpu = cpu.str();
  if (!features.empty())
    compilationOptions.targetFeatures = features.str();

  return success();
}

//===----------------------------------------------------------------------===//
// setupMLIRContext
//===----------------------------------------------------------------------===//

/// Set up the MLIR context with all the dialects we need.
static void setupMLIRContext(mlir::MLIRContext &ctx) {
  // Register the various dialects we need.
  DialectRegistry dialectRegistry;
  registerAllKGENDialects(dialectRegistry);

  mlir::registerBuiltinDialectTranslation(dialectRegistry);
  mlir::registerLLVMDialectTranslation(dialectRegistry);

  // Set up the dialects in the context.
  ctx.appendDialectRegistry(dialectRegistry);
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx.allowUnregisteredDialects();
}

//===----------------------------------------------------------------------===//
// buildPackage
//===----------------------------------------------------------------------===//

/// We have all the arguments and all the state we need, we can now start
/// building the package itself.
static ErrorOrSuccess buildPackage(const PackageArgs &packageArgs,
                                   ModuleOp theModule,
                                   LIT::PackageOp parsedPackageOp,
                                   llvm::ToolOutputFile &out,
                                   LLCL::Runtime &runtime) {
  // Set up the package builder.
  PackageBuilder packageBuilder(parsedPackageOp);

  // For now we implicilty export everything in the package, so add exports to
  // the main module for the contents of the module.
  OpBuilder exportBuilder = OpBuilder::atBlockEnd(theModule.getBody());
  parsedPackageOp.walk<mlir::WalkOrder::PreOrder>([&](LIT::FuncOp func) {
    if (!canExternalize(func))
      return WalkResult::skip();
    SymbolRefAttr fullName = LIT::getFullyResolvedSymbolRef(func);
    exportBuilder.create<ExportOp>(func.getLoc(), fullName);
    return WalkResult::skip();
  });

  mlir::MLIRContext *ctx = packageBuilder.getContext();
  // Initialize targets first - we rely on this for getTargetInfo as well as for
  // the ExecutionEngine.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Pull out the compilation options.
  const CompilationOptions &compilationOptions = packageArgs.compileOptions;

  // Construct a target specification using the command line options.
  ErrorOr<TargetInfoAttr> targetOr = getTargetInfoFor(
      ctx, compilationOptions.targetTriple, compilationOptions.targetCpu,
      compilationOptions.targetFeatures);
  if (targetOr.isError())
    return targetOr.takeError();
  TargetInfoAttr target = targetOr.takeValue();
  packageBuilder.setTarget(target);

  // Set up the ExecutionEngine with all the requisite layers.
  mlir::PassManager pm(ctx);
  ErrorOr<std::unique_ptr<ExecutionEngine>> execEngineOr =
      setupExecutionEngine(runtime, pm, target, compilationOptions);
  if (failed(execEngineOr))
    return execEngineOr.takeError();
  std::unique_ptr<ExecutionEngine> engine = std::move(*execEngineOr);
  // Pull out references to the layers we're going to use later.
  auto &compileLayer = engine->getLayer<KGENCompilerLayer>();

  // This currently compiles the module, so we don't need to try to look
  // anything up just yet.
  if (auto err = compileLayer.add("package", theModule))
    return Error("compilation failed");

  // Construct the symbol table and the export map.
  SymbolTable symtab(theModule);
  ExportMap exportedSymbols = getExportedSymbols(theModule);

  // Attach the elaborated bytecode to the individual functions.
  if (auto err =
          packageBuilder.attachElaboratedBytecode(symtab, exportedSymbols))
    return err.takeError();

  // Look up the first item in the exported symbols to trigger archive
  // generation.
  ErrorOr<CompiledFunc> funcOr = engine->lookup(exportedSymbols.front().first);
  if (funcOr.isError())
    return funcOr.takeError();
  // And lookup the archive.
  std::optional<Cache::BufferRef> archiveOr =
      engine->getLayer<ObjectCompilerLayer>().lookupArchive(theModule);
  assert(archiveOr.has_value());
  Cache::BufferRef archive = std::move(*archiveOr);

  // Compile the module, and attach the archive to the package op.
  if (auto err = packageBuilder.attachCompiledArchiveBytes(
          theModule, std::move(archive), compilationOptions))
    return err.takeError();

  // Verify the cloned module to ensure nothing has gone egregiously wrong.
  if (failed(packageBuilder.verify()))
    return Error("new package failed to verify");

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

  if (args.hasArg(options::OPT_UNKNOWN)) {
    int result = 1;
    for (llvm::opt::Arg *arg : args.filtered(options::OPT_UNKNOWN))
      result = state.reportError("unrecognized argument '" +
                                 arg->getSpelling() + "'\n");
    return result;
  }

  PackageArgs packageArgs;
  if (auto err = parsePackageArgs(state, args, packageArgs))
    return state.reportError(err.getError());

  //===--------------------------------------------------------------------===//
  // MLIRContext/LLCL::Runtime setup
  //===--------------------------------------------------------------------===//

  mlir::MLIRContext ctx;
  setupMLIRContext(ctx);

  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

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
  mlir::TimingScope parseTimeScope;
  MojoParserConfig parserConfig(&ctx, runtime, packageArgs.compileOptions);
  llvm::SourceMgr sourceMgr;
  auto [ownedModuleOp, packageOp] =
      M::importMojoPackage(packageArgs.inputPath, packageArgs.name, sourceMgr,
                           parserConfig, parseTimeScope);
  if (!ownedModuleOp)
    return state.reportError("could not parse the provided package");

  // Build the package from the inputs we just parsed, and write the output to
  // `out`.
  if (auto err =
          buildPackage(packageArgs, *ownedModuleOp, packageOp, *out, runtime))
    return state.reportError(err.getError());

  out->keep();
  return EXIT_SUCCESS;
}

void M::registerPackageSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("package", package);
}
