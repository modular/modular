//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-package.h"
#include "../mojo-driver.h"

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

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
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

#define MOJO_DRIVER_OPTIONS_PATH "Package/PackageOptions.inc"
#include "../OptTable.inc"

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
  /// Construct the PackageBuilder. This requires knowing the name of the
  /// package, and the module from which we are constructing said package.
  PackageBuilder(StringRef packageName, ModuleOp theModule);

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

/// Construct the PackageBuilder.
PackageBuilder::PackageBuilder(StringRef packageName, ModuleOp theModule) {
  packageModule = ModuleOp::create(theModule->getLoc());
  OpBuilder b(packageModule->getBody(), packageModule->getBody()->begin());
  thePackage = b.create<LIT::PackageOp>(packageModule->getLoc(),
                                        b.getStringAttr(packageName));
  b.setInsertionPointToStart(thePackage.getBody());

  // Clone the relevant operations into the package.
  std::stack<SmartVariant<Operation *, OpBuilder::InsertPoint>> worklist;
  for (Operation &op : theModule.getOps())
    worklist.emplace(&op);

  // Clone an op without its regions, and ensure that once that op is finished
  // processing, we reset the OpBuilder's insert point to where it was before we
  // walked the ops inside `op`.
  auto cloneWithoutRegions = [&](auto op) {
    // Save the insert point on the worklist stack first.
    worklist.push(b.saveInsertionPoint());

    auto clonedOp = b.cloneWithoutRegions(op);
    clonedOp.getBodyRegion().push_back(new Block);
    b.setInsertionPointToStart(clonedOp.getBody());
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

  while (!worklist.empty()) {
    auto listFront = worklist.top();
    worklist.pop();
    if (isa<OpBuilder::InsertPoint>(listFront)) {
      b.restoreInsertionPoint(cast<OpBuilder::InsertPoint>(listFront));
      continue;
    }
    Operation *front = cast<Operation *>(listFront);

    // If it's a package, we might want to clone it, but we definitely want to
    // put the ops onto the worklist.
    if (auto package = dyn_cast<LIT::PackageOp>(front)) {
      if (package->getParentOp() != theModule)
        cloneWithoutRegions(package);

      pushOpsOntoWorklist(package.getOps());
      continue;
    }

    // It's a file? Clone it without its regions and push the ops onto the
    // worklist.
    if (auto file = dyn_cast<LIT::FileModuleOp>(front)) {
      cloneWithoutRegions(file);
      pushOpsOntoWorklist(file.getOps());
      continue;
    }

    // It's a struct? Same as a file.
    if (auto structDecl = dyn_cast<LIT::StructDeclOp>(front)) {
      cloneWithoutRegions(structDecl);
      pushOpsOntoWorklist(structDecl.getOps());
      continue;
    }

    // It's a func? OK - non-parametric funcs get elided, parametric funcs are
    // cloned as-is.
    if (auto func = dyn_cast<LIT::FuncOp>(front)) {
      SignatureType sig = func.getSignature();
      // If the function is non-parametric, drop its body.
      LIT::FuncOp clonedFunc;
      if (sig.getInputParamTypes().empty() &&
          sig.getResultParamTypes().empty()) {
        // This will reset the insertion point to where it was before we entered
        // the function.
        OpBuilder::InsertionGuard guard(b);

        // Add a block that only contains a lit.end_func in it.
        clonedFunc = b.cloneWithoutRegions(func);
        clonedFunc.getBodyRegion().push_back(new Block);
        b.setInsertionPointToStart(clonedFunc.getBody());
        b.create<LIT::ExternFuncOp>(clonedFunc.getLoc());
      } else {
        clonedFunc = cast<LIT::FuncOp>(b.clone(*func));
      }

      // Use the mangled version of the original func, because that's what
      // its name will be post-elaboration.
      auto mangled = LIT::MangledSymbol::mangle(func);
      flattenedNameToFunc.try_emplace(mangled.mangled, clonedFunc);
      continue;
    }

    // Drop export ops unconditionally.
    if (auto exportOp = dyn_cast<KGEN::ExportOp>(front))
      continue;

    // None of the cases matched? Just clone the op directly.
    b.clone(*front);
  }
}

/// Attach the elaborated bytecode to the high-level lit.func ops.
ErrorOrSuccess
PackageBuilder::attachElaboratedBytecode(const SymbolTable &symtab,
                                         const ExportMap &exportedSymbols) {
  // This lambda takes a LIT::FuncOp and the thing it turns into
  // post-elaboration (a KGEN::FuncOp) and attaches the bytecode for the
  // lowered func to the high-level func as a resource attribute.
  auto attachElaboratedBytecodeToFunc =
      [this](LIT::FuncOp hlFunc, KGEN::FuncOp llFunc) -> ErrorOrSuccess {
    Cache::WriteableBufferRef str = Cache::WriteableBuffer::get();
    if (failed(mlir::writeBytecodeToFile(llFunc, *str)))
      return Error("could not write bytecode for kgen.func");

    LIT::MangledSymbol mangled = LIT::MangledSymbol::mangle(hlFunc);

    hlFunc.setPostElaborationBodyRefAttr(createResourceAttr(
        std::move(str), mangled.mangled.getValue() + "_bytecode"));
    return success();
  };

  for (auto [symName, _] : exportedSymbols) {
    LIT::FuncOp hlFunc = flattenedNameToFunc.lookup(symName);
    if (!hlFunc)
      return Error("could not find lit.func with name " + symName.getValue());

    // If the thing is parametric, then we don't care about it.
    if (!isa_and_nonnull<LIT::ExternFuncOp>(hlFunc.getBody()->getTerminator()))
      continue;

    auto func = symtab.lookup<KGEN::FuncOp>(symName);
    if (!func)
      return Error("could not find kgen.func with name " + symName.getValue());

    if (auto err = attachElaboratedBytecodeToFunc(hlFunc, func))
      return Error(err.getError());
  }

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
  // TODO: Once the frontend is set up, this should be a directory, not a single
  //       file.
  if (!args.hasArg(options::OPT_INPUT))
    return Error("no input file provided");
  if (args.hasMultipleArgs(options::OPT_INPUT))
    return Error("too many input files, expected exactly one");

  if (!args.hasArg(options::OPT_name))
    return Error("must provide a package name");
  if (args.hasMultipleArgs(options::OPT_name))
    return Error("too many package names, expected exactly one");

  pkgArgs.name = args.getLastArgValue(options::OPT_name);

  // Reject input files that do not appear to be mlir files (this includes stdin
  // "-").
  StringRef inputPath = args.getLastArgValue(options::OPT_INPUT);
  if (!inputPath.ends_with(".mlir")) {
    return Error(
        "cannot open '" + inputPath +
        "', this command temporarily only supports manually-formed packages.");
  }
  pkgArgs.inputPath = inputPath.str();

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
  dialectRegistry.insert<DebugInfo::DebugInfoDialect, mlir::index::IndexDialect,
                         mlir::LLVM::LLVMDialect>();

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
static ErrorOrSuccess buildPackage(PackageArgs packageArgs, ModuleOp theModule,
                                   llvm::ToolOutputFile &out) {
  // Set up the LLCL runtime here, it's the first place we need it.
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

  // Set up the package builder.
  PackageBuilder packageBuilder(packageArgs.name, theModule);

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
  ErrorOr<CompiledFunc> funcOr =
      engine->lookup(exportedSymbols.front().second.alias);
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

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
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

  //===--------------------------------------------------------------------===//
  // Build the package
  //===--------------------------------------------------------------------===//

  // Open the input file, or exit with an error.
  std::string inputError;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      mlir::openInputFile(packageArgs.inputPath, &inputError);
  if (!buffer)
    return state.reportError(inputError);

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out =
      mlir::openOutputFile(packageArgs.outputPath, &outputError);
  if (!out)
    return state.reportError(outputError);

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  // Parse the mlir file.
  OwningOpRef<ModuleOp> theModule =
      mlir::parseSourceFile<ModuleOp>(sourceManager, &ctx);
  if (!theModule)
    return state.reportError("could not parse the provided source file");

  // Build the package from the inputs we just parsed, and write the output to
  // `out`.
  if (auto err = buildPackage(packageArgs, *theModule, *out))
    return state.reportError(err.getError());

  out->keep();
  return EXIT_SUCCESS;
}

void M::registerPackageSubCommand(SubcommandRegistry &registry) {
  registry.addCallback("package", package);
}
