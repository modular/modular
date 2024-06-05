//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/KGENCompiler.h"
#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ExecutionEngine/JIT/ObjectCompilerLayer.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "ObjectCompiler/KGENToLLVMPipeline.h"
#include "Pipeline/Pipeline.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Config.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "kgen-compiler"

using namespace M;
using namespace KGEN;

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
    b.setLoc(cast<LocationAttr>(replacer.replace(b.getLoc())));
  }

  sliced.setNotExported();
  sliced.setInlineLevel(InlineLevel::Always);
  sliced.setSymNameAttr(stubName);
  SignatureType sig = symbol.getType();
  auto wrapper = b.create<GeneratorOp>(name, sig);
  wrapper.setExported();
  wrapper.setLLVMMetadataAttr(sliced.getLLVMMetadataAttr());
  Block *entry =
      b.createBlock(&wrapper.getBodyRegion(), {}, sig.getArguments(),
                    llvm::map_to_vector(sliced.getArguments(),
                                        [](Value v) { return v.getLoc(); }));

  // Re-declare the captured parameter values.
  for (auto [decl, value] :
       llvm::zip(sliced.getInputParams(), symbol.getParamValues()))
    b.create<ParamDeclareOp>(decl, value);

  auto call =
      b.create<CallOp>(sig.getResults(),
                       SymbolConstantAttr::get(FlatSymbolRefAttr::get(stubName),
                                               symbol.getParamValues(), sig),
                       entry->getArguments());
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
  loc = cast<LocationAttr>(replacer.replace(loc));

  // Generate a function on the host side that opaquely populates a piece of
  // memory with the capture values.
  ImplicitLocOpBuilder b(loc, OpBuilder(name.getContext()));

  // The expected signature is `fn(Pointer[None]) capturing -> None`.
  auto noneType = b.getType<KGEN::NoneType>();
  auto nonePtr = PointerType::get(noneType);
  auto sig = SignatureType::get(b.getFunctionType(nonePtr, noneType),
                                ArgConvention::BorrowedInReg,
                                FnEffects().setCapturing());
  OwningOpRef<FuncOp> func =
      b.create<FuncOp>(b.getStringAttr(name.getValue() + "_populate_captures"),
                       sig, InlineLevel::Always);

  // Populate the body. Generate a local variable for each capture argument
  // and store the addresses to the pointer. The function is marked as
  // `always_inline`, so this is okay.
  // FIXME: This does not account for copy constructors, obviously.
  Block *body = b.createBlock(&func->getBodyRegion());
  Value argPtrs = body->addArgument(sig.getArguments().front(), b.getLoc());
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
    Value argPtrPtrs =
        b.create<POP::PointerBitcastOp>(PointerType::get(nonePtr), argPtrs);
    Value gep = b.create<POP::OffsetOp>(
        argPtrPtrs, b.create<ParamConstantOp>(b.getIndexAttr(i)));
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
  OwningOpRef<Operation *> func = readOpFromBytecodeFile(
      *bytecode, mlir::ParserConfig(ctx, /*verifyAfterParse=*/false));
  if (!func)
    return Error("failed to read capture function bytecode");
  it += size;

  // Read the number of captures.
  uint64_t numCaptures = llvm::support::endian::read64le(it);
  it += sizeof(uint64_t);

  // Read out the rest of the data as the payload.
  auto contents =
      StringAttr::get(StringRef(it, std::distance(it, buf->getBufferEnd())),
                      StringType::get(ctx));

  return CrossDeviceFunction{contents, (unsigned)numCaptures, std::move(func)};
}

//===----------------------------------------------------------------------===//
// compileElaboratorAsm
//===----------------------------------------------------------------------===//

/// Given the pre-elaboration function `func` belonging to a module with the
/// symbol table `symtab`, slice out a standalone module rooted at `func` and
/// elaborate it and compile to assembly for the provided `target.
static ErrorOr<CrossDeviceFunction>
compileElaboratorAsm(GeneratorOp func, SymbolConstantAttr symbol,
                     StringAttr name, const SymbolTable &symtab,
                     TargetInfoAttr target, EmissionKind emissionKind,
                     CompilationOptions options) {
  // Configure the compilation options given the new target.
  options.targetTriple = target.getTripleStr();
  options.targetCpu = target.getArch();
  options.targetFeatures = target.getFeatures();
  options.relocModel = target.getRelocationModel();

  // Initialize the object compiler.
  ErrorOr<std::unique_ptr<ObjectCompiler>> compilerOr = ObjectCompiler::create(
      ".mojo_cache", options, /*isJIT=*/false, *target.getContext());

  if (compilerOr.isError())
    return compilerOr.takeError();

  std::unique_ptr<ObjectCompiler> compiler = compilerOr.takeValue();

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
      compiler->produceStandaloneModule(symtab, exportedSymbols, mapping);
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
  configurePassManager(pm);
  pm.addPass(createElaborateGenerators(
      target, elaboratorOptions,
      [=](GeneratorOp func, SymbolConstantAttr symbol, StringAttr name,
          const SymbolTable &symtab, TargetInfoAttr target,
          EmissionKind emissionKind) {
        // Recursion...!
        return compileElaboratorAsm(func, symbol, name, symtab, target,
                                    emissionKind, options);
      }));
  buildPostElaborationPipeline(pm, options);

  // This functor runs the desired transformation to cache.
  auto compileToAsm =
      [&pm, &compiler, &options, tm = std::move(tm), name, emissionKind](
          Operation *op, WriteableBufferRef buffer) mutable -> ErrorOrSuccess {
    if (failed(pm.run(op)))
      return Error("failed to run the pass manager");
    if (failed(writeCaptureArgs(cast<ModuleOp>(op), name, buffer.copy())))
      return Error("failed to generate capture stub");
    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> llvmModule =
        compiler->lowerAllFuncsToLLVM(llvmCtx, cast<ModuleOp>(op));
    if (!llvmModule)
      return Error("failed to lower to LLVM");

    if (emissionKind == EmissionKind::LLVM) {
      llvmModule->print(*buffer, nullptr);
      return success();
    }

    LLCL::Runtime &runtime =
        *loadContext(pm.getContext())->get<LLCL::Runtime>();
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
  auto runTransformation = [func = std::move(compileToAsm)](
                               Operation *op, WriteableBufferRef buf,
                               AnyAsyncValueRef chain) mutable {
#ifdef MODULAR_ENABLE_TELEMETRY
    Cache::CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheMiss("KGEN::compileElaboratorAsm");
#endif

    auto output = AsyncValueRef<BufferRef>::allocate(chain.getRuntime());
    std::move(chain).andThenSync(
        [op, func = std::move(func), output = output.copy(),
         buf = std::move(buf)](AnyAsyncValueRef &&chain) mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
          [[maybe_unused]] auto timeScope =
              loadContext(op->getContext())
                  ->get<M::Telemetry::TelemetryContext>()
                  ->createUInt64Timer<std::chrono::milliseconds>(
                      "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
                      {{"pipeline", "KGEN::compileElaboratorAsm"}});
#endif
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
  auto onCacheHit = [](Operation *op, BufferRef buf) {
#ifdef MODULAR_ENABLE_TELEMETRY
    Cache::CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheHit("KGEN::compileElaboratorAsm");
#endif

    return buf.copy();
  };
  LLCL::Runtime &runtime =
      *loadContext(target.getContext())->get<LLCL::Runtime>();
  AnyAsyncValueRef result = Cache::cachedTransform(
      *module, compiler->getTransformCache(),
      AsyncValueRef<Chain>::createReady(runtime), std::move(key),
      std::move(runTransformation), onCacheHit);
  await(result);
  if (result.isError())
    return std::move(result.takeDiagnostic().getMessage());

  BufferRef buf = std::move(result.get<BufferRef>());
  return readCaptureArgs(func.getContext(), buf.copy());
}

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

/// Returns Mojo transform backend, or an error if the backend could not be
/// created.
static ErrorOr<RCRef<Cache::BlobCacheBackend>> getMojoCacheBackend() {
  return Cache::getLocalDefaultBackendChain(
      std::filesystem::path(".mojo_cache") / "transform", KGEN_VERSION_STRING);
}

static ErrorOr<BufferRef> specializePackageLinkForPreElaborationLinking(
    PackageLinkOp packageLink, const KGEN::CompilationOptions &compileOptions) {
  auto cacheBackend = getMojoCacheBackend();
  if (cacheBackend.isError())
    return cacheBackend.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(*cacheBackend));
  DenseResourceElementsAttr bytecodeAttr = packageLink.getPostParseModuleAttr();
  if (!bytecodeAttr || !bytecodeAttr.getRawHandle().getBlob())
    return Error("package link does not contain post parser bytecode data");
  ArrayRef<char> bytecodeData =
      bytecodeAttr.getRawHandle().getBlob()->getData();

  auto runTransform = [&](WriteableBufferRef buf,
                          LLCL::AnyAsyncValueRef chain) mutable {
    auto output = AsyncValueRef<BufferRef>::allocate(chain.getRuntime());
    std::move(chain).andThenSync([&, output = output.copy(),
                                  buf = std::move(buf)](
                                     AnyAsyncValueRef &&chain) mutable {
      if (chain.isError())
        return std::move(output).setToError(chain.takeDiagnostic());
      auto setError = [&](Error err) {
        return std::move(output).setToError(
            LLCL::getMLIRDiagnostic(std::move(err), packageLink->getLoc()));
      };

      // Read in the bytecode and run the library generation pipeline.
      OwningOpRef<ModuleOp> packageModuleOr =
          readOpFromBytecodeFile<ModuleOp>(bytecodeAttr);
      if (!packageModuleOr)
        return setError("unable to load parsed module bytecode");

      KGENCompiler compiler(*(packageModuleOr->getContext()), compileOptions);
      if (auto err = compiler.runGenerateLibraryPipeline(*packageModuleOr))
        return setError(err.takeError());

      // Write the bytecode back out to the buffer.
      if (failed(mlir::writeBytecodeToFile(*packageModuleOr, *buf)))
        return setError("failed to write bytecode for package module");
      return std::move(output).emplace(std::move(buf));
    });
    return output;
  };
  auto onCacheHit = [](BufferRef buf) { return buf; };

  // Build the cache key for the specialization.
  auto bytecodeHash = llvm::BLAKE3::hash(ArrayRef<uint8_t>(
      (const uint8_t *)bytecodeData.data(), bytecodeData.size()));
  WriteableBufferRef key = WriteableBuffer::get();
  *key << "specializePackageForPreElaboration";
  compileOptions.print(*key);
  key->write_impl((const char *)bytecodeHash.data(), bytecodeHash.size());

  LLCL::Runtime &runtime =
      *loadContext(packageLink.getContext())->get<LLCL::Runtime>();
  LLCL::AnyAsyncValueRef ready = cachedTransform(
      LLCL::MLIRLocationDecoder::getEncodedLocation(packageLink->getLoc()),
      transformCache.copy(), AsyncValueRef<Chain>::createReady(runtime),
      std::move(key), runTransform, onCacheHit);
  await(ready);
  if (ready.isError())
    return std::move(ready.takeDiagnostic().getMessage());

  return std::move(ready.get<BufferRef>());
}

//===----------------------------------------------------------------------===//
// createElaborateGeneratorsWithDefaultJIT
//===----------------------------------------------------------------------===//

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass> KGEN::createElaborateGeneratorsWithDefaultJIT() {
  CompilationOptions options;
  return createElaborateGenerators(
      /*target=*/{}, /*options=*/{},
      [=](GeneratorOp func, SymbolConstantAttr symbol, StringAttr name,
          const SymbolTable &symtab, TargetInfoAttr target,
          EmissionKind emissionKind) {
        return compileElaboratorAsm(func, symbol, name, symtab, target,
                                    emissionKind, options);
      });
}

//===----------------------------------------------------------------------===//
// createMaterializePackagesWithDefaultGen
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> KGEN::createMaterializePackagesWithDefaultGen(
    const CompilationOptions &options) {
  return createMaterializePackages([&](KGEN::PackageLinkOp packageLink) {
    return specializePackageLinkForPreElaborationLinking(packageLink, options);
  });
}

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

void KGEN::populateElaborateModulePasses(mlir::PassManager &pm,
                                         TargetInfoAttr target,
                                         const CompilationOptions &options) {
  buildElaborateModulePipeline(
      pm, target, options, /*compileAsmFn=*/
      [=](GeneratorOp func, SymbolConstantAttr symbol, StringAttr name,
          const SymbolTable &symtab, TargetInfoAttr target,
          EmissionKind emissionKind) {
        return compileElaboratorAsm(func, symbol, name, symtab, target,
                                    emissionKind, options);
      },
      [=](PackageLinkOp packageLink) {
        return specializePackageLinkForPreElaborationLinking(packageLink,
                                                             options);
      });
  buildPostElaborationPipeline(pm, options);
}

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<ExecutionEngine>> KGEN::initializeExecutionEngine(
    mlir::MLIRContext &context, const CompilationOptions &compilationOptions,
    ExecutionEngineOptions executionEngineOptions, bool isJIT,
    PassManagerConfigOptions pmOptions, bool isSearch) {

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
  auto compiler =
      ObjectCompiler::create(".mojo_cache", compilationOptions, isJIT, context,
                             std::move(pmOptions), isSearch);
  if (failed(compiler))
    return compiler.takeError();

  engine->addLayer<ObjectCompilerLayer>(std::move(*compiler),
                                        engine->getLinkingLayer());

  return std::move(engine);
}

//===----------------------------------------------------------------------===//
// KGENCompiler
//===----------------------------------------------------------------------===//

KGENCompiler::KGENCompiler(MLIRContext &context, CompilationOptions options,
                           PassManagerConfigOptions pmConfigOptions)
    : options(std::move(options)), pmConfigOptions(std::move(pmConfigOptions)),
      context(context) {}

ErrorOrSuccess KGENCompiler::runKGENPipeline(ModuleOp theModule,
                                             TargetInfoAttr target) {
  llvm::StringMap<Telemetry::MetricAttributeValue> attrs;
  auto fileLine = theModule.getLoc()->findInstanceOf<mlir::FileLineColLoc>();
  if (fileLine)
    attrs["filename"] = fileLine.getFilename().str();

  auto cacheBackend = getMojoCacheBackend();
  if (cacheBackend.isError())
    return cacheBackend.takeError();

  // Run the passes as a cached transform.
  {
    ContextRef ctx = loadContext(target.getContext());
    [[maybe_unused]] auto timeScope =
        ctx->get<M::Telemetry::TelemetryContext>()
            ->createUInt64Timer<std::chrono::milliseconds>(
                "mojo.kgen.compile.time", M::Telemetry::Level::L2, attrs);

    auto transformCache =
        RCRef<Cache::TransformCache>::create(std::move(*cacheBackend));

    LLCL::AnyAsyncValueRef ready = runKGENPipeline(
        theModule, target, transformCache,
        ctx->get<LLCL::Runtime>()->getReadyChain().copy(),
        Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
            "KGENCompiler::runKGENPipeline", "mojo.compiler.cache.miss.time",
            {{"pipeline", "KGEN"}}),
        Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
            "KGENCompiler::runKGENPipeline"));

    if (ready.isError())
      return ready.takeDiagnostic().getMessage().copy();
  }
  return {};
}

static mlir::PassManager
createPassManager(const std::optional<std::string> &operationName,
                  mlir::MLIRContext *context) {
  if (operationName)
    return {context, *operationName};
  return {context};
}

AnyAsyncValueRef
KGENCompiler::runKGENPipeline(ModuleOp theModule, TargetInfoAttr target,
                              RCRef<Cache::TransformCache> transformCache,
                              AnyAsyncValueRef chain,
                              std::function<void(Operation *)> moreOnMiss,
                              std::function<void(Operation *)> moreOnHit) {

  // Set the target now, so it's included in the cache key.
  setTargetInfo(theModule, target);
  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  if (failed(pmConfigOptions.configurePassManager(pm))) {
    LLCL::Runtime &runtime = *loadContext(&context)->get<LLCL::Runtime>();
    auto output = LLCL::AsyncValueRef<std::string>::allocate(runtime);
    std::move(output).setToError(LLCL::getMLIRDiagnostic(
        "configure PassManager in KGENCompiler::runKGENPipeline failed",
        theModule->getLoc()));
    return output;
  }

  // Populate the passes.
  buildGenerateLibraryPipeline(pm, options);
  populateElaborateModulePasses(pm, target, options);

  // Run the passes as a cached transform.

  LLCL::AnyAsyncValueRef ready =
      Cache::cachedTransform(theModule, transformCache.copy(), std::move(chain),
                             pm, std::move(moreOnMiss), std::move(moreOnHit));

  // This await here is important since pm is local in this function.
  LLCL::await(ready);
  return ready;
}

ErrorOrSuccess
KGENCompiler::runGenerateLibraryPipeline(ModuleOp module,
                                         bool materializeDependencies) {
  auto cacheBackend = getMojoCacheBackend();
  if (cacheBackend.isError())
    return cacheBackend.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(*cacheBackend));

  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  if (failed(pmConfigOptions.configurePassManager(pm))) {
    return Error("configure PassManager in "
                 "KGENCompiler::runGenerateLibraryPipeline failed");
  }

  buildGenerateLibraryPipeline(pm, options);
  if (materializeDependencies)
    pm.addPass(createMaterializePackagesWithDefaultGen(options));

  LLCL::Runtime &runtime =
      *loadContext(module.getContext())->get<LLCL::Runtime>();
  LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
      module, transformCache.copy(), AsyncValueRef<Chain>::createReady(runtime),
      pm,
      Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
          "KGEN::runGenerateLibraryPipeline", "mojo.compiler.cache.miss.time",
          {{"pipeline", "KGEN"}}),
      Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
          "KGEN::runGenerateLibraryPipeline"));

  // This await here is important since pm is local in this function.
  LLCL::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();

  // Strip the implicit package exports, we don't need these because we're going
  // to link the package into an existing module as-is.
  for (ExportInterface op : module.getOps<ExportInterface>())
    if (op.isPackageExported())
      op.setNotExported();
  return success();
}

LogicalResult KGENCompiler::runCheckLITPipeline(ModuleOp module) {
  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  if (failed(pmConfigOptions.configurePassManager(pm))) {
    return Error(
        "configure PassManager in KGENCompiler::runCheckLITPipeline failed");
  }

  buildCheckLITPipeline(pm, options);
  return pm.run(module);
}

/// Run the compilation pipeline till the end of elaboration to produce a fully
/// concrete KGEN module. This allows the transform to be cached.
/// Note that this function also awaits the AsyncValue because it uses
/// a local PassManager.
/// Returns the same AnyAsyncValueRef for error handling in the caller
/// if needed.
AnyAsyncValueRef KGENCompiler::runElaborationPipeline(
    ModuleOp module, TargetInfoAttr target, LLCL::Runtime &runtime,
    std::optional<AnyAsyncValueRef> chain,
    std::function<void(Operation *)> moreOnMiss,
    std::function<void(Operation *)> moreOnHit) {
  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  if (failed(pmConfigOptions.configurePassManager(pm))) {
    auto output = LLCL::AsyncValueRef<std::string>::allocate(runtime);
    std::move(output).setToError(LLCL::getMLIRDiagnostic(
        "configure PassManager in KGENCompiler::runKGENPipeline failed",
        module->getLoc()));
    return output;
  }

  populateElaborateModulePasses(pm, target, options);
  auto cacheBackend = getMojoCacheBackend();

  if (cacheBackend.isError() || !chain) {
    auto output = LLCL::AsyncValueRef<std::string>::allocate(runtime);
    if (failed(pm.run(module))) {
      std::move(output).setToError(LLCL::getMLIRDiagnostic(
          "KGENCompiler::runElaborationPipeline failed", module->getLoc()));
    } else {
      std::move(output).emplace(
          "KGENCompiler::runElaborationPipeline success without caching.");
    }
    return std::move(output);
  }

  AnyAsyncValueRef ready = Cache::cachedTransform(
      module, RCRef<Cache::TransformCache>::create(std::move(*cacheBackend)),
      std::move(*chain), pm, std::move(moreOnMiss), std::move(moreOnHit));

  // This await here is important since pm is local in this function.
  LLCL::await(ready);

  return ready;
}
