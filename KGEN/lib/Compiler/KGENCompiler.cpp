//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/JITSupport.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "kgen-compiler"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// evaluateSpecializations
//===----------------------------------------------------------------------===//

/// A default specialization evaluator that JITs and invokes the specialized
/// functions with the provided evaluator.
static ErrorOr<ElaboratorSearchFn>
evaluateSpecializations(FuncOp evaluator, const SymbolTable &symtab,
                        LLCL::Runtime &runtime, TargetInfoAttr target,
                        const CompilationOptions &options,
                        ArrayRef<FuncOp> specializations) {
  // TODO(#2717): Cross-compilation and execution for search!
  if (target.getArch() != llvm::sys::getHostCPUName())
    return Error("cross-compilation execution in search is not yet supported");

  mlir::PassManager mgr(target.getContext());
  ExecutionEngineOptions eeOptions;
  if (options.debugLevel != CompilationOptions::kNoDebug)
    eeOptions.registerDebugPlugins = true;
  auto engineOr =
      initializeExecutionEngine(runtime, mgr, options, std::move(eeOptions),
                                /*isJIT=*/true, target, /*isSearch=*/true);
  if (engineOr.isError())
    return engineOr.takeError();
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // We only want the funcs passed-in and the evaluator to be code-generated.
  SmallVector<FuncOp> funcsToCompile(specializations);
  funcsToCompile.push_back(evaluator);

  // Create the set of symbols to export.
  ExportMap exportedSymbols;
  for (FuncOp func : funcsToCompile) {
    exportedSymbols.insert(
        {func.getSymNameAttr(), ExportedSymbol(ExportKind::Exported)});
  }

  // Add the exported symbols to the ObjectCompilerLayer. This will not actually
  // compile anything - that happens at lookup time.
  if (auto err = engine->add<ObjectCompilerLayer>("evaluateSpecializations",
                                                  symtab, exportedSymbols))
    return err.takeError();

  SmallVector<void *> candidatePtrs;
  {
    CompilerTimeTraceScope traceScope("compile-specializations");
    // Get pointers to all the candidates.
    for (FuncOp candidate : specializations) {
      auto funcOr = engine->lookup(candidate.getNameAttr());
      if (funcOr.isError())
        return funcOr.takeError();
      candidatePtrs.push_back(funcOr->getFunctionPointer());
    }
  }

  // Lookup the evaluator function
  auto evaluatorFuncOr = engine->lookup(evaluator.getNameAttr());
  if (evaluatorFuncOr.isError())
    return evaluatorFuncOr.takeError();
  auto evaluatorFunc = std::move(*evaluatorFuncOr);

  return
      [engine = std::move(engine), evaluatorFunc = std::move(evaluatorFunc),
       candidatePtrs = std::move(candidatePtrs)]() mutable -> ErrorOr<ssize_t> {
        CompilerTimeTraceScope traceScope("execute-specializations");
        return evaluatorFunc.invoke<ssize_t, void **, ssize_t>(
            candidatePtrs.data(), candidatePtrs.size());
      };
}

//===----------------------------------------------------------------------===//
// compileElaboratorAsm
//===----------------------------------------------------------------------===//

/// Generate a stub function that calls into the sliced function with input
/// parameters, then rename it to match the expected symbol name and export it
/// This is how compilation is rooted at instantiations of parametric functions.
static void generateInstantiateStub(GeneratorOp func, SymbolConstantAttr symbol,
                                    StringAttr name, IRMapping &mapping) {
  GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(sliced));
  StringAttr stubName = b.getStringAttr(name.getValue() + "_asm_stub");

  // Build debuginfo for the stub if requested.
  if (auto scope = func.getSubprogramScope()) {
    scope = scope.cloneWith(
        DebugInfo::SourceNameAttr::get("asm_stub", scope.getName()), stubName);
    DebugInfo::DIAttrTypeReplacer replacer;
    replacer.addReplacement(
        [scope](DebugInfo::DISubprogramAttr) { return scope; });
    b.setLoc(cast<mlir::LocationAttr>(replacer.replace(b.getLoc())));
  }

  sliced.setNotExported();
  sliced.setInlineLevel(InlineLevel::Always);
  sliced.setSymNameAttr(stubName);
  SignatureType sig = symbol.getType();
  auto wrapper = b.create<GeneratorOp>(name, sig);
  wrapper.setExported();
  wrapper.setLLVMMetadataAttr(sliced.getLLVMMetadataAttr());
  Block *entry =
      b.createBlock(&wrapper.getBodyRegion(), {}, sig.getValueInputs(),
                    llvm::map_to_vector(sliced.getArguments(),
                                        [](Value v) { return v.getLoc(); }));

  // Re-declare the captured parameter values.
  for (auto [decl, value] :
       llvm::zip(sliced.getInputParams(), symbol.getParamValues()))
    b.create<ParamDeclareOp>(decl, value);

  auto call =
      b.create<CallOp>(sig.getValueResults(),
                       SymbolConstantAttr::get(FlatSymbolRefAttr::get(stubName),
                                               symbol.getParamValues(), sig),
                       std::nullopt, entry->getArguments());
  b.create<ReturnOp>(call.getResults());
}

/// HACK HACK HACK https://github.com/modularml/modular/issues/22959
/// HACK: Read out the magic attribute used to propagate captures across device
/// boundaries, generate the capture function, and write them into the buffer.
static LogicalResult writeCaptureArgs(ModuleOp module, StringAttr name,
                                      WriteableBufferRef buf) {
  // First, go find the elaborated instance of the function.
  FuncOp sliced;
  for (auto func : module.getOps<FuncOp>()) {
    if (func.getSymNameAttr() == name) {
      sliced = func;
      break;
    }
  }
  // This is held together with duct tape, so check the invariant.
  assert(sliced && sliced.isExported() && "expected a sliced function");
  ArrayRef<StringAttr> captures;
  if (auto capturesAttr =
          sliced->getAttrOfType<StringArrayAttr>("kgen.cross_device_captures"))
    captures = capturesAttr;

  // The location to use for generated code. Remove all debuginfo from it.
  Location loc = sliced.getLoc();
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](mlir::FusedLocWith<DebugInfo::DIAttr> loc) {
    return FusedLoc::get(loc.getContext(), loc.getLocations());
  });
  loc = cast<mlir::LocationAttr>(replacer.replace(loc));

  // Generate a function on the host side that opaquely populates a piece of
  // memory with the capture values.
  ImplicitLocOpBuilder b(loc, OpBuilder(name.getContext()));

  // The expected signature is `fn(Pointer[None]) capturing -> None`.
  auto noneType = b.getType<KGEN::NoneType>();
  auto nonePtr = PointerType::get(noneType);
  auto sig = SignatureType::get(
      b.getFunctionType(PointerType::get(nonePtr), noneType),
      ValueInputConvention::BorrowedInReg, FnEffects().setCapturing());
  OwningOpRef<FuncOp> func =
      b.create<FuncOp>(b.getStringAttr(name.getValue() + "_populate_captures"),
                       sig, InlineLevel::Always);

  // Populate the body. Generate a local variable for each capture argument
  // and store the addresses to the pointer. The function is marked as
  // `always_inline`, so this is okay.
  // FIXME: This does not account for copy constructors, obviously.
  Block *body = b.createBlock(&func->getBodyRegion());
  Value argPtrs = body->addArgument(sig.getValueInputs().front(), b.getLoc());
  for (auto [i, type, capture] : llvm::enumerate(
           sliced.getArgumentTypes().take_front(captures.size()), captures)) {
    // ```
    // %value = pop.compiler.global_load "var" : T
    // %ptr = pop.stack_allocation 1 x T
    // pop.store %value, %ptr
    // %gep = pop.offset %argPtrs[%i]
    // %opaque = pop.pointer.bitcast %pt : pointer<T> to pointer<none>
    // pop.store %opaque, %gep
    // ```
    Value value = b.create<POP::CompilerGlobalLoadOp>(type, capture);
    Value ptr = b.create<POP::StackAllocationOp>(PointerType::get(type), 1);
    b.create<POP::StoreOp>(value, ptr);
    Value gep = b.create<POP::OffsetOp>(
        argPtrs, b.create<ParamConstantOp>(b.getIndexAttr(i)));
    Value opaque = b.create<POP::PointerBitcastOp>(nonePtr, ptr);
    b.create<POP::StoreOp>(opaque, gep);
  }
  b.create<ReturnOp>(
      b.create<ParamConstantOp>(b.getAttr<NoneAttr>()).getResult());

  // Now write this into the buffer as bytecode. Add space for a header that
  // contains the size of the bytecode first.
  uint64_t size = 0;
  buf->write((char *)&size, sizeof(size));
  // Then write the bytecode.
  if (failed(mlir::writeBytecodeToFile(*func, *buf)))
    return failure();
  // Now write the size of the bytecode into the allocate header space. Be
  // mindful of endianness here.
  size = buf->tell() - sizeof(size);
  size = llvm::support::endian::byte_swap(size, llvm::endianness::little);
  buf->pwrite((char *)&size, sizeof(size), /*Offset=*/0);

  // Write the number of captures.
  uint64_t numCaptures = llvm::support::endian::byte_swap(
      captures.size(), llvm::endianness::little);
  buf->write((char *)&numCaptures, sizeof(uint64_t));
  return success();
}

/// HACK: Read out the capture function and generated code.
static ErrorOr<CrossDeviceFunction> readCaptureArgs(MLIRContext *ctx,
                                                    BufferRef buf) {
  // First, read the bytecode header size.
  const char *it = buf->getBufferStart();
  uint64_t size = llvm::support::endian::read64le(it);
  it += sizeof(uint64_t);

  // Then read the bytecode for the capture population function.
  std::unique_ptr<llvm::MemoryBuffer> bytecode =
      llvm::MemoryBuffer::getMemBuffer(StringRef(it, size), /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  Block container;
  if (failed(mlir::readBytecodeFile(
          *bytecode, &container,
          mlir::ParserConfig(ctx, /*verifyAfterParse=*/false))))
    return Error("failed to read capture function bytecode");
  assert(container.getOperations().size() == 1 && "expected a single function");
  it += size;

  // Take ownership of the function.
  Operation *captureFunc = &container.front();
  captureFunc->remove();
  OwningOpRef<Operation *> func = captureFunc;

  // Read the number of captures.
  uint64_t numCaptures = llvm::support::endian::read64le(it);
  it += sizeof(uint64_t);

  // Read out the rest of the data as the payload.
  auto contents =
      StringAttr::get(StringRef(it, std::distance(it, buf->getBufferEnd())),
                      StringType::get(ctx));

  return CrossDeviceFunction{contents, (unsigned)numCaptures, std::move(func)};
}

/// Given the pre-elaboration function `func` belonging to a module with the
/// symbol table `symtab`, slice out a standalone module rooted at `func` and
/// elaborate it and compile to assembly for the provided `target.
static ErrorOr<CrossDeviceFunction>
compileElaboratorAsm(GeneratorOp func, SymbolConstantAttr symbol,
                     StringAttr name, const SymbolTable &symtab,
                     LLCL::Runtime &runtime, TargetInfoAttr target,
                     EmissionKind emissionKind, CompilationOptions options) {
  // Configure the compilation options given the new target.
  options.targetTriple = target.getTripleStr();
  options.targetCpu = target.getArch();
  options.targetFeatures = target.getFeatures();
  options.relocModel = target.getRelocationModel();

  // Initialize the object compiler.
  mlir::PassManager compilerPm(target.getContext());
  ErrorOr<ObjectCompiler> compilerOr = ObjectCompiler::create(
      runtime, compilerPm, ".mojo_cache", options, /*isJIT=*/false);
  if (compilerOr.isError())
    return compilerOr.takeError();
  ObjectCompiler compiler = compilerOr.takeValue();

  // Initialize the target machine.
  auto tmOr = createTargetMachine(options, /*isJIT=*/false);
  if (tmOr.isError())
    return tmOr.takeError();
  std::unique_ptr<llvm::TargetMachine> tm = tmOr.takeValue();

  // Slice out a pre-elaboration module for the new target to compile for.
  ExportMap exportedSymbols;
  exportedSymbols.insert({func.getSymNameAttr(), ExportKind::Exported});
  // Make sure to slice out anything referenced in the input parameters. When
  // generator references are instantiated in the standalone module, they are
  // instantiated with the new target.
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](SymbolConstantAttr ref) {
    exportedSymbols.insert(
        {ref.getSymbol().getRootReference(), ExportKind::NotExported});
  });
  for (TypedAttr attr : symbol.getParamValues())
    walker.walk(attr);

  IRMapping mapping;
  OwningOpRef<ModuleOp> module =
      compiler.produceStandaloneModule(symtab, exportedSymbols, mapping);
  // Override the target.
  eraseTargetInfo(*module);
  setTargetInfo(*module, target);

  // If there are input parameters, we have to go generate a stub to root
  // instantiation of the generator. Go find the cloned generator.
  if (!symbol.getParamValues().empty())
    generateInstantiateStub(func, symbol, name, mapping);

  // Run elaboration through to the end of the optimization pipeline.
  ElaborateGeneratorsOptions elaboratorOptions;
  elaboratorOptions.enableSearch = options.enableSearch;
  elaboratorOptions.elaborateDebugInfo =
      options.debugLevel == CompilationOptions::kLineTablesOnly ||
      options.debugLevel == CompilationOptions::kFullDebugInfo;
  mlir::PassManager pm(target.getContext());
  pm.addPass(createElaborateGenerators(
      runtime, target, BuildInfoAttr::getForCurrentBuild(target.getContext()),
      elaboratorOptions,
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      [=, &runtime](GeneratorOp func, SymbolConstantAttr symbol,
                    StringAttr name, const SymbolTable &symtab,
                    TargetInfoAttr target, EmissionKind emissionKind) {
        // Recursion...!
        return compileElaboratorAsm(func, symbol, name, symtab, runtime, target,
                                    emissionKind, options);
      }));
  buildPostElaborationPipeline(pm, runtime, options);

  // This functor runs the desired transformation to cache.
  auto compileToAsm =
      [&pm, &compiler, &options, &runtime, tm = std::move(tm), name,
       emissionKind](Operation *op,
                     WriteableBufferRef buffer) mutable -> ErrorOrSuccess {
    if (failed(pm.run(op)))
      return Error("failed to run the pass manager");
    if (failed(writeCaptureArgs(cast<ModuleOp>(op), name, buffer.copy())))
      return Error("failed to generate capture stub");
    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> llvmModule =
        compiler.lowerAllFuncsToLLVM(llvmCtx, cast<ModuleOp>(op));

    if (emissionKind == EmissionKind::LLVM) {
      llvmModule->print(*buffer, nullptr);
      return success();
    }

    if (failed(compileLLVMToObject(*llvmModule, *tm, *buffer, options, runtime,
                                   /*emitAssembly=*/true)))
      return Error("failed to emit assembly");
    return success();
  };

  // Cache the compilation down to assembly as a single step. Finer-grain
  // caching here would not create any re-use with the rest of the stack.
  WriteableBufferRef key = WriteableBuffer::get();
  pm.printAsTextualPipeline(*key);
  options.print(*key);
  // Encode the cache key to disambiguiate between different emission kinds.
  *key << (int)emissionKind;
  // Functor to adapt the transform functor to the required API.
  auto runTransformation =
      [func = std::move(compileToAsm)](Operation *op, WriteableBufferRef buf,
                                       AnyAsyncValueRef chain) mutable {
        auto output = AsyncValueRef<BufferRef>::allocate(chain.getRuntime());
        std::move(chain).andThenSync(
            [op, func = std::move(func), output = output.copy(),
             buf = std::move(buf)](AnyAsyncValueRef &&chain) mutable {
              if (chain.isError())
                return std::move(output).setToError(chain.takeDiagnostic());
              if (ErrorOrSuccess err = func(op, buf.copy()); err.isError())
                return std::move(output).setToError(
                    LLCL::getMLIRDiagnostic(err.takeError(), op->getLoc()));
              return std::move(output).emplace(std::move(buf));
            });
        return output;
      };
  // On cache hit, just return the assembly buffer.
  auto onCacheHit = [](Operation *op, BufferRef buf) { return buf.copy(); };
  AnyAsyncValueRef result = Cache::cachedTransform(
      *module, compiler.getTransformCache(),
      AsyncValueRef<Chain>::createReady(runtime), std::move(key),
      std::move(runTransformation), onCacheHit);
  await(result);
  if (result.isError())
    return std::move(result.takeDiagnostic().getMessage());

  BufferRef buf = std::move(result.get<BufferRef>());
  return readCaptureArgs(func.getContext(), buf.copy());
}

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

void KGEN::populateElaborateModulePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    BuildInfoAttr build, const CompilationOptions &options,
    EvaluatorExecutorFn evaluatorExecutorFn,
    PackageLinkHandlerFn packageLinkHandlerFn) {
  buildElaborateModulePipeline(
      pm, runtime, target, build, options, std::move(evaluatorExecutorFn),
      /*compileAsmFn=*/
      [=, &runtime](GeneratorOp func, SymbolConstantAttr symbol,
                    StringAttr name, const SymbolTable &symtab,
                    TargetInfoAttr target, EmissionKind emissionKind) {
        return compileElaboratorAsm(func, symbol, name, symtab, runtime, target,
                                    emissionKind, options);
      },
      std::move(packageLinkHandlerFn));
  buildPostElaborationPipeline(pm, runtime, options);
}

void KGEN::populateElaborateModulePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    BuildInfoAttr build, const CompilationOptions &options,
    PackageLinkHandlerFn packageLinkHandlerFn) {
  populateElaborateModulePasses(
      pm, runtime, target, build, options,
      /*evaluatorExecutorFn=*/
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      std::move(packageLinkHandlerFn));
}

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

ErrorOr<
    std::pair<RCRef<Cache::BlobCacheBackend>, RCRef<Cache::BlobCacheBackend>>>
KGEN::getMojoCacheBackends(LLCL::Runtime &runtime) {
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

  return std::make_pair(transformCacheBackend.takeValue(),
                        regionCacheBackend.takeValue());
}

//===----------------------------------------------------------------------===//
// KGENCompilerMaterializationUnit
//===----------------------------------------------------------------------===//

/// Produce an ExportMap with every symbol in the module.
static ExportMap getAllSymbols(ModuleOp theModule) {
  ExportMap exports;
  for (auto sym : theModule.getOps<mlir::SymbolOpInterface>())
    exports.insert({sym.getNameAttr(), {ExportKind::Exported}});
  return exports;
}

class KGENCompilerLayer::KGENCompilerMaterializationUnit
    : public llvm::orc::MaterializationUnit {
public:
  KGENCompilerMaterializationUnit(KGENCompilerLayer &layer, SymbolTable s,
                                  ExportMap e)
      : MaterializationUnit(layer.getInterface(e)), genLayer(layer),
        symtab(std::move(s)), exports(std::move(e)) {}

  /// Provide a name for this MU that will show up in ORC debug logs.
  StringRef getName() const override {
    return "KGEN::KGENCompilerMaterializationUnit";
  }

  /// Given a MaterializationResponsibility, materialize the code for those
  /// symbols and forward them to the next layer.
  void materialize(
      std::unique_ptr<llvm::orc::MaterializationResponsibility> mr) override {
    genLayer.emit(std::move(mr), symtab, exports);
  }

  /// Notify that the symbol `name` has been overridden and this MU should
  /// remove it from the source. This removes the symbol from the module.
  void discard(const llvm::orc::JITDylib &jd,
               const llvm::orc::SymbolStringPtr &name) override {
    // If the operation exists, erase it. Otherwise, do nothing.
    if (auto sym = symtab.lookup<mlir::SymbolOpInterface>(*name))
      symtab.erase(sym);
  }

private:
  KGENCompilerLayer &genLayer;
  SymbolTable symtab;
  ExportMap exports;
};

//===----------------------------------------------------------------------===//
// KGENCompilerLayer
//===----------------------------------------------------------------------===//

char KGENCompilerLayer::ID;

KGENCompilerLayer::KGENCompilerLayer(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    BuildInfoAttr build, const CompilationOptions &options,
    ObjectCompilerLayer &base,
    RCRef<Cache::BlobCacheBackend> transformCacheBackend,
    RCRef<Cache::BlobCacheBackend> regionCacheBackend,
    llvm::orc::ExecutionSession &sess, const llvm::DataLayout &dl,
    MaterializationLayer::AddToSearchOrderFn add)
    : llvm::RTTIExtends<KGENCompilerLayer, MaterializationLayer>(
          sess, dl, std::move(add)),
      pm(pm), runtime(runtime), target(target), build(build), options(options),
      baseLayer(base) {
  // Construct the caches.
  transformCache =
      RCRef<Cache::TransformCache>::create(std::move(transformCacheBackend));
  regionCache =
      RCRef<Cache::RegionCache>::create(std::move(regionCacheBackend));
}

ErrorOrSuccess
KGENCompilerLayer::add(StringRef libName, ModuleOp theModule,
                       PackageLinkHandlerFn packageLinkHandlerFn) {
  CompilerTimeTraceScope traceScope("KGENCompilerLayer::add(" + libName.str() +
                                    ")");
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();

  llvm::orc::JITDylib *dylib = *dylibOr;
  llvm::orc::ResourceTrackerSP resourceTracker =
      dylib->getDefaultResourceTracker();

  // Set the target and build info now, so it's included in the cache key.
  setTargetInfo(theModule, target);
  setBuildInfo(theModule, build);
  // Populate the passes.
  buildGenerateLibraryPipeline(pm, runtime, options);
  populateElaborateModulePasses(pm, runtime, target, build, options,
                                packageLinkHandlerFn);

  // Run the passes as a cached transform. Don't deflate the op as part of this
  // - we don't want that cost right now.
  {
    [[maybe_unused]] auto timeScope =
        runtime.emplaceContextIfMissing<M::Telemetry::TelemetryContext>()
            .createUInt64Timer<std::chrono::milliseconds>(
                "mojo.kgen.compile.time", M::Telemetry::Level::L2);

    LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
        theModule, regionCache.copy(), transformCache.copy(),
        runtime.getReadyChain().copy(), pm, /*deflateTarget=*/false);
    LLCL::await(ready);
    if (ready.isError())
      return ready.takeDiagnostic().getMessage().copy();
  }

  // Add the materialization unit by computing the exports and the symbol
  // table, and passing those off.
  SymbolTable st(theModule);
  ExportMap ex = getExportedSymbols(theModule);
  if (ex.empty())
    ex = getAllSymbols(theModule);

  return toModularErrorOr(
      dylib->define(std::make_unique<KGENCompilerMaterializationUnit>(
                        *this, std::move(st), std::move(ex)),
                    resourceTracker));
}

void KGENCompilerLayer::emit(
    std::unique_ptr<llvm::orc::MaterializationResponsibility> mr,
    SymbolTable &symtab, const ExportMap &exports) {
  // Delegate all requested symbols to the base layer.
  baseLayer.emit(std::move(mr), symtab, exports);
}

llvm::orc::MaterializationUnit::Interface
KGENCompilerLayer::getInterface(const ExportMap &exports) {
  llvm::orc::MangleAndInterner mangler(session, dataLayout);
  llvm::orc::SymbolFlagsMap symbols;

  for (auto &[name, symbol] : exports)
    symbols[mangler(name)] = getFlagsForExportedSymbol(symbol);

  if (baseLayer.getRawCompiler().getIsJIT()) {
    symbols[mangler(ExecutionEngine::getGlobalCtorFnName())] =
        getGlobalFnSymbolFlags();
    symbols[mangler(ExecutionEngine::getGlobalDtorFnName())] =
        getGlobalFnSymbolFlags();
  }

  return {std::move(symbols), /*InitSymbol=*/nullptr};
}

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
KGEN::createElaborateGeneratorsWithDefaultJIT(LLCL::Runtime &runtime) {
  CompilationOptions options;
  return createElaborateGenerators(
      runtime, /*target=*/{}, /*build=*/{}, /*options=*/{},
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      [=, &runtime](GeneratorOp func, SymbolConstantAttr symbol,
                    StringAttr name, const SymbolTable &symtab,
                    TargetInfoAttr target, EmissionKind emissionKind) {
        return compileElaboratorAsm(func, symbol, name, symtab, runtime, target,
                                    emissionKind, options);
      });
}

ErrorOr<std::unique_ptr<ExecutionEngine>>
KGEN::initializeExecutionEngine(LLCL::Runtime &runtime, mlir::PassManager &pm,
                                const CompilationOptions &compilationOptions,
                                ExecutionEngineOptions executionEngineOptions,
                                bool isJIT, TargetInfoAttr target,
                                bool isSearch) {
  MLIRContext *ctx = pm.getContext();

  // Now create the execution engine so we can JIT.
  auto tmOr = createTargetMachine(compilationOptions, isJIT);
  if (tmOr.isError())
    return tmOr.takeError();

  auto engineOr = ExecutionEngine::createWithStandardLayers(
      std::move(executionEngineOptions), **tmOr);
  if (failed(engineOr))
    return engineOr.takeError();
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // Add the object compiler layer.
  auto compiler = ObjectCompiler::create(runtime, pm, ".mojo_cache",
                                         compilationOptions, isJIT, isSearch);
  if (failed(compiler))
    return compiler.takeError();

  auto &objLayer = engine->addLayer<ObjectCompilerLayer>(
      std::move(*compiler), engine->getLinkingLayer());

  // Add the KGEN compiler layer. First though, get the backend chains to pass
  // into the compile layer.
  auto cacheBackends = getMojoCacheBackends(runtime);
  if (cacheBackends.isError())
    return cacheBackends.takeError();

  // Get the build info from the current build.
  BuildInfoAttr build = BuildInfoAttr::getForCurrentBuild(ctx);

  engine->addLayer<KGENCompilerLayer>(
      pm, runtime, target, build, compilationOptions, objLayer,
      std::move(cacheBackends->first), std::move(cacheBackends->second));
  return std::move(engine);
}
