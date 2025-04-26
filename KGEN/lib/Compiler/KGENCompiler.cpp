//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/KGENCompiler.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/CompilerSupport/MLIRLocationDecoder.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/Support/BuildInfo.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/NameMangling.h"
#include "KGEN/ToolCommon/Debug.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/SlicingUtils.h"
#include "ObjectCompiler/KGENToLLVMPipeline.h"
#include "Pipeline/Pipeline.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/Config.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "kgen-compiler"
#define KGEN_DEBUG_TYPE "kgen-compiler"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// compileElaboratorAsm
//===----------------------------------------------------------------------===//

/// Generate a stub function that calls into the sliced function with input
/// parameters, then rename it to match the expected symbol name and export it
/// This is how compilation is rooted at instantiations of parametric functions.
static void
generateInstantiateStub(GeneratorOp func, SymbolConstantAttr symbol,
                        StringAttr name, IRMapping &mapping,
                        SymbolTable *symtab = nullptr,
                        std::optional<uint64_t> kernelId = std::nullopt) {

  GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(sliced));
  StringAttr stubName = b.getStringAttr(name.getValue() + "_asm_stub");
  FuncTypeGeneratorType sigGen = symbol.getType();
  FuncType sigBase = sigGen.getBody();

  // Build debuginfo for the stub if requested.
  if (auto sp = func.getSubprogramScope()) {
    // The original DISubroutineType for the subprogram may contain parameter
    // references that are no longer in scope in the stub. Re-create a
    // DISubroutineType from the concretized signature of the stub (this is ok
    // since the stub is a compiler-synthesized function).
    auto stubSourceName =
        DebugInfo::SourceNameAttr::get("asm_stub", sp.getSourceName());
    FunctionType stubFuncType = sigBase.getValues();
    DebugInfo::DIUnresolvedMLIRType (*mapToDIUnresolvedType)(Type) =
        &DebugInfo::DIUnresolvedMLIRType::get;
    auto stubSp = DebugInfo::DISubprogramAttr::get(
        sp.getCompileUnit(), sp.getScope(), stubSourceName, stubName,
        sp.getFile(), sp.getLine(), sp.getScopeLine(), sp.getSubprogramFlags(),
        DebugInfo::DISubroutineType::get(
            func.getContext(),
            SmallVector<DebugInfo::DIType>(
                map_range(stubFuncType.getInputs(), mapToDIUnresolvedType)),
            SmallVector<DebugInfo::DIType>(
                map_range(stubFuncType.getResults(), mapToDIUnresolvedType))));
    DebugInfo::DIAttrTypeReplacer replacer;
    replacer.addReplacement(
        [stubSp](DebugInfo::DISubprogramAttr) { return stubSp; });
    b.setLoc(cast<LocationAttr>(replacer.replace(b.getLoc())));
  }

  sliced.setNotExported();
  sliced.setInlineLevel(InlineLevel::Always);
  if (symtab) {
    sliced = sliced.clone();
    sliced.setSymNameAttr(stubName);
    symtab->insert(sliced);
  } else {
    sliced.setSymNameAttr(stubName);
  }

  auto wrapper = b.create<GeneratorOp>(name, sigGen);
  wrapper.setExported();

  SmallVector<Attribute> metadataArray =
      llvm::to_vector(sliced.getLLVMMetadataArrayAttr().getValue());
  if (kernelId) {
    metadataArray.push_back(
        StringAttr::get(sliced->getContext(), "kgen.offload.kernelid"));
    metadataArray.push_back(b.getIndexAttr(*kernelId));
  }
  wrapper.setLLVMMetadataArrayAttr(
      ArrayAttr::get(sliced.getContext(), metadataArray));
  wrapper.setLLVMArgMetadataArrayAttr(sliced.getLLVMArgMetadataArrayAttr());
  Block *entry =
      b.createBlock(&wrapper.getBodyRegion(), {}, sigBase.getArguments(),
                    llvm::map_to_vector(sliced.getArguments(),
                                        [](Value v) { return v.getLoc(); }));

  // Re-declare the captured parameter values.
  for (auto [decl, value] :
       llvm::zip(sliced.getInputParams(), symbol.getParamValues()))
    b.create<ParamDeclareOp>(decl, value);

  auto call = b.create<CallOp>(
      SymbolConstantAttr::get(stubName, sigGen, symbol.getParamValues()),
      entry->getArguments());
  b.create<ReturnOp>(call.getResults());
}

/// HACK HACK HACK https://github.com/modularml/modular/issues/22959
/// HACK: Read out the magic attribute used to propagate captures across device
/// boundaries, generate the capture function, and write them into the buffer.
static std::pair<OwningOpRef<FuncOp>, unsigned>
writeCaptureArgs(ModuleOp module, StringAttr name) {
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
  ArrayRef<StringAttr> captures = sliced.getCrossDeviceCaptures();

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
  auto sig = FuncType::get(b.getFunctionType(nonePtr, noneType),
                           ArgConvention::ReadReg, FnEffects().setCapturing());
  OwningOpRef<FuncOp> func =
      b.create<FuncOp>(b.getStringAttr(name.getValue() + "_populate_captures"),
                       sig, InlineLevel::Always);

  // Populate the body. Generate a local variable for each capture argument
  // and store the addresses to the pointer.
  // The function has to be `always_inline`, so that the stack allocated ptr
  // will not go out of scope before use.
  // FIXME: This does not account for copy constructors, obviously.
  Block *body = b.createBlock(&func->getBodyRegion());
  Value argPtrs = body->addArgument(sig.getArguments().front(), b.getLoc());
  for (auto [i, capture] : llvm::enumerate(captures)) {
    // ```
    // %value = pop.compiler.global_load "var" : T
    // %ptr = pop.stack_allocation 1 x T
    // pop.store %value, %ptr
    // %gep = pop.offset %argPtrs[%i]
    // %opaque = pop.pointer.bitcast %ptr : pointer<T> to pointer<none>
    // pop.store %opaque, %gep
    // ```
    Type type = capture.getType();
    Value value = b.create<POP::CompilerGlobalLoadOp>(
        // Make sure to strip off the type of the StringAttr.
        type, b.getStringAttr(capture.getValue()));
    Value ptr = b.create<POP::StackAllocationOp>(PointerType::get(type));
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

  return {std::move(func), captures.size()};
}

static ElaboratorCompileOffloadRetType compileOffloads(
    ModuleOp theModule,
    llvm::MapVector<TargetInfoAttr, OffloadInfo> &targetOffloadInfos,
    const SymbolTable &symtab, CompilationOptions compilationOptions,
    ElaborateGeneratorsOptions elabOptions, mlir::DiagnosticEngine::HandlerID);

/// Given the pre-elaboration function `func` belonging to a module with the
/// symbol table `symtab`, slice out a standalone module rooted at `func` and
/// elaborate it and compile to assembly for the provided `target.
static ErrorOr<CrossDeviceFunction> compileElaboratorAsm(
    GeneratorOp func, SymbolConstantAttr symbol, StringAttr name,
    const SymbolTable &symtab, TargetInfoAttr target, EmitAs emissionKind,
    EmissionOptions emissionOptions, CompilationOptions compilationOptions,
    ElaborateGeneratorsOptions elaboratorOptions,
    mlir::DiagnosticEngine::HandlerID diagHandlerID) {
  // Configure the compilation options given the new target.
  compilationOptions.targetTriple = target.getTripleStr();
  compilationOptions.targetCpu = target.getArch();
  compilationOptions.targetFeatures = target.getFeatures();
  if (compilationOptions.targetAccelerator.empty()) {
    compilationOptions.targetAccelerator =
        AsyncRT::Device::getAcceleratorArchOrEmpty();
  }
  compilationOptions.relocModel = target.getRelocationModel();
  StringRef targetDataLayout = target.getDataLayout().toString();
  if (!targetDataLayout.empty())
    compilationOptions.targetDataLayout = targetDataLayout;

  // Initialize the object compiler.
  ErrorOr<std::unique_ptr<ObjectCompiler>> compilerOr =
      ObjectCompiler::create(".mojo_cache", compilationOptions, /*isJIT=*/
                             false, *target.getContext());

  if (compilerOr.isError())
    return compilerOr.takeError();

  std::unique_ptr<ObjectCompiler> compiler = compilerOr.takeValue();

  // Initialize the target machine.
  auto tmOr = createTargetMachine(compilationOptions, /*isJIT=*/false);
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
  OwningOpRef<ModuleOp> module = produceStandaloneModule(
      symtab, exportedSymbols, mapping,
      /*overrideExported*/ isGPUBackend(compilationOptions));
  // Override the target.
  eraseTargetInfo(*module);
  setTargetInfo(*module, target);

  // If there are input parameters, we have to go generate a stub to root
  // instantiation of the generator. Go find the cloned generator.
  if (!symbol.getParamValues().empty())
    generateInstantiateStub(func, symbol, name, mapping);

  // Run elaboration through to the end of the optimization pipeline.
  mlir::PassManager pm(target.getContext());
  if constexpr (KGEN::kIsTracingEnabled)
    pm.enableTiming(std::make_unique<TimeProfilerTimingManager>());
  configurePassManager(pm);

  pm.addPass(createElaborateGenerators(target, elaboratorOptions,
                                       compilationOptions, compileElaboratorAsm,
                                       compileOffloads));
  buildPostElaborationPipeline(pm, compilationOptions);

  if (failed(pm.run(*module)))
    return Error("failed to run the pass manager");
  auto [capturesFunc, numCaptures] = writeCaptureArgs(*module, name);

  // Handle the emission options.
  ErrorOrSuccess parseResult = parseEmissionOptions(emissionOptions);
  if (parseResult.isError()) {
    return parseResult.takeError();
  }

  // Prepare a buffer to write string output to.
  SmallVector<char> buf;
  buf.reserve(256 * 128); // 32 KB
  llvm::raw_svector_ostream os(buf);

  // Emit the module in the requested form.
  switch (emissionKind) {
  case EmitAs::ASM:
    if (ErrorOrSuccess err = compiler->emitAssembly(std::move(module), os))
      return err.takeError();
    break;

  case EmitAs::LLVM: {
    LLVMModuleAndContext llvmModule;
    if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
          return compiler->lowerAllFuncsToLLVM(ctx, *module);
        }))
      return err.takeError();
    llvmModule->print(os, nullptr);
    break;
  }

  case EmitAs::LLVM_OPT:
    if (ErrorOrSuccess err = compiler->emitLLVMIR(*module, os))
      return err.takeError();
    break;

  case EmitAs::OBJECT:
    if (ErrorOrSuccess err = compiler->emitSharedObject(std::move(module), os))
      return err.takeError();
    break;
  }

  return CrossDeviceFunction{
      StringAttr::get(buf, StringType::get(func.getContext())), numCaptures,
      std::move(capturesFunc)};
}

//===----------------------------------------------------------------------===//
// compileOffloads
//===----------------------------------------------------------------------===//

static ElaboratorCompileOffloadRetType compileOffloads(
    ModuleOp theModule,
    llvm::MapVector<TargetInfoAttr, OffloadInfo> &targetOffloadInfos,
    const SymbolTable &symtab, CompilationOptions compilationOptions,
    ElaborateGeneratorsOptions elabOptions, mlir::DiagnosticEngine::HandlerID) {

  DenseMap<TargetInfoAttr,
           DenseMap<StringRef, DenseMap<uint64_t, OffloadCompilationResult>>>
      result;

  // Compiling offload for different targets.
  // This loop cannot be parallelized since different targets may need
  // different llvm options that are global states.
  for (auto [target, info] : targetOffloadInfos) {
    DenseMap<StringRef, DenseMap<uint64_t, OffloadCompilationResult>>
        &targetEmissionResult = result[target];

    for (auto [emissionOptionsStr, offloadInfo] : info.groups) {

      DenseMap<uint64_t, OffloadCompilationResult> &targetResult =
          targetEmissionResult[emissionOptionsStr];

      IRMapping mapping;

      // Configure the compilation options given the new target.
      compilationOptions.targetTriple = target.getTripleStr();
      compilationOptions.targetCpu = target.getArch();
      compilationOptions.targetFeatures = target.getFeatures();
      if (compilationOptions.targetAccelerator.empty()) {
        compilationOptions.targetAccelerator =
            AsyncRT::Device::getAcceleratorArchOrEmpty();
      }
      compilationOptions.relocModel = target.getRelocationModel();
      StringRef targetDataLayout = target.getDataLayout().toString();
      if (!targetDataLayout.empty())
        compilationOptions.targetDataLayout = targetDataLayout;
      compilationOptions.emissionOptions = emissionOptionsStr;

      OwningOpRef<ModuleOp> module = produceStandaloneModule(
          symtab, offloadInfo.exportedSymbols, mapping,
          /*overrideExported*/ isGPUTriple(target.getTriple()));

      // Override the target.
      eraseTargetInfo(*module);
      setTargetInfo(*module, target);
      SymbolTable slicedSymtab(*module);

      // Collect SymbolConstantAttr names to rename.
      DenseMap<SymbolRefAttr, StringAttr> symToRename;

      for (auto [op, symbolInfo] : offloadInfo.symbols) {
        // If there are input parameters, we have to go generate a stub to root
        // instantiation of the generator. Go find the cloned generator.
        auto func = cast<GeneratorOp>(op);
        StringAttr newCalleeName = StringAttr::get(
            func->getContext(),
            FlatSymbolRefAttr::get(func).getAttr().str() + "_callee");

        std::optional<StringAttr> newName;
        for (auto [symbol, kernelInfo] : symbolInfo) {
          if (!symbol.getParamValues().empty()) {
            // Add "_callee" postfix to the generator that is both a callee of a
            // kernel and a kernel entry function itself. So that we don't end
            // up with two functions with the same name one for the kernel entry
            // wrapper for instantiated stub, and one for the callee in another
            // kernel (they have different function bodies).
            module->walk(
                [&newCalleeName, &newName, &func, &symToRename](CallOp call) {
                  if (call.getCalleeSymbol() == FlatSymbolRefAttr::get(func)) {
                    newName = newCalleeName;
                    symToRename.insert(
                        {call.getCallee().getSymbol(), newCalleeName});
                  }
                });

            generateInstantiateStub(func, symbol, kernelInfo.name, mapping,
                                    &slicedSymtab, kernelInfo.kernelId);
          } else {
            // Set kernelId
            GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
            ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(sliced));
            SmallVector<Attribute> metadataArray =
                llvm::to_vector(sliced.getLLVMMetadataArrayAttr().getValue());
            metadataArray.push_back(
                StringAttr::get(sliced->getContext(), "kgen.offload.kernelid"));
            metadataArray.push_back(b.getIndexAttr(kernelInfo.kernelId));
            sliced.setLLVMMetadataArrayAttr(
                ArrayAttr::get(sliced.getContext(), metadataArray));
          }
        }
        if (newName) {
          // Rename the generator since it is used both as kernel entry function
          // and callee for another kernel.
          GeneratorOp sliced = cast<GeneratorOp>(mapping.lookup(func));
          sliced.setSymNameAttr(*newName);
        }
      }

      // Replace the SymbolConstantAttr names for the renamed generator
      // references.
      if (!symToRename.empty()) {
        mlir::AttrTypeReplacer replacer;
        replacer.addReplacement([&symToRename](SymbolConstantAttr attr) {
          auto iter = symToRename.find(attr.getSymbol());
          if (iter != symToRename.end()) {
            return SymbolConstantAttr::get(iter->second, attr.getType(),
                                           attr.getParamValues());
          }
          return attr;
        });

        replacer.recursivelyReplaceElementsIn(*module, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/true,
                                              /*replaceTypes=*/true);
      }

      // Initialize the object compiler.
      ErrorOr<std::unique_ptr<ObjectCompiler>> compilerOr =
          ObjectCompiler::create(".mojo_cache", compilationOptions, /*isJIT=*/
                                 false, *target.getContext());

      if (compilerOr.isError())
        return compilerOr.takeError();

      std::unique_ptr<ObjectCompiler> compiler = compilerOr.takeValue();

      // Initialize the target machine.
      auto tmOr = createTargetMachine(compilationOptions, /*isJIT=*/false);
      if (tmOr.isError())
        return tmOr.takeError();
      std::unique_ptr<llvm::TargetMachine> tm = tmOr.takeValue();

      // Run elaboration through to the end of the optimization pipeline.
      mlir::PassManager pm(target.getContext());
      if constexpr (KGEN::kIsTracingEnabled)
        pm.enableTiming(std::make_unique<TimeProfilerTimingManager>());
      configurePassManager(pm);

      pm.addPass(
          createElaborateGenerators(target, elabOptions, compilationOptions,
                                    compileElaboratorAsm, compileOffloads));

      buildPostElaborationPipeline(pm, compilationOptions);

      if (failed(pm.run(*module)))
        return Error("failed to run the pass manager for offload functions");

      llvm::MapVector<uint64_t, std::pair<OwningOpRef<FuncOp>, unsigned>>
          captures;

      llvm::DenseMap<uint64_t, llvm::SmallSet<EmitAs, 4>> kernelEmissionKinds;

      for (auto [op, symbols] : offloadInfo.symbols) {
        for (auto [symbol, kernel] : symbols) {

          auto [capturesFunc, numCaptures] =
              writeCaptureArgs(*module, kernel.name);

          captures.insert(
              {kernel.kernelId,
               std::make_pair(std::move(capturesFunc), numCaptures)});
          kernelEmissionKinds.insert({kernel.kernelId, kernel.emissionKinds});
        }
      }

      // Handle the emission options.
      // These are the global llvm options needed to compile this target.
      // These options can't be kernel specific since they are global llvm
      // states, they are shared for parallel kernel compilation. However,
      // offloads for different targets won't have to share since we compile
      // them in order and we can reset these options for each targets.
      SmallVector<StringRef> emissionOptions;
      emissionOptionsStr.split(emissionOptions, /*Separator=*/",",
                               /*MaxSplit=*/-1, /*KeepEmpty=*/false);

      KGEN_DEBUG(0, {
        llvm::dbgs() << "Emit offloads with options: " << emissionOptionsStr
                     << "\n";
      });
      ErrorOrSuccess parseResult = parseEmissionOptions(emissionOptions);
      if (parseResult.isError()) {
        return parseResult.takeError();
      }

      ErrorOr<DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>>>
          compiledKernelsOr =
              compiler->emitGPUKernels(std::move(module), kernelEmissionKinds);

      if (compiledKernelsOr.isError())
        return compiledKernelsOr.takeError();

      OpBuilder b(theModule);
      for (auto idAndKernels : *compiledKernelsOr) {
        uint64_t kernelID = idAndKernels.first;
        DenseMap<EmitAs, BufferRef> &bufs = idAndKernels.second;

        auto iter = captures.find(kernelID);
        if (iter == captures.end())
          return Error("Can't find offload capture.");

        OwningOpRef<FuncOp> func = std::move(iter->second.first);
        unsigned numCaptures = iter->second.second;

        auto populate = cast<FuncOp>(func.get());
        auto populateFnRef = SymbolConstantAttr::get(populate);
        DenseMap<EmitAs, StringAttr> contents;

        for (auto kindAndContent : bufs) {
          EmitAs kind = kindAndContent.first;
          contents.insert(
              {kind,
               StringAttr::get(kindAndContent.second->getBuffer(),
                               StringType::get(theModule->getContext()))});
        }

        targetResult.insert(
            {kernelID, OffloadCompilationResult{{std::move(func)},
                                                b.getIndexAttr(numCaptures),
                                                populateFnRef,
                                                std::move(contents)}});
      }

      // Reset the global llvm options once compiling this target is done.
      ErrorOrSuccess resetResult = resetEmissionOptions(emissionOptions);
      if (resetResult.isError()) {
        return resetResult.takeError();
      }
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

/// Returns Mojo transform backend, or an error if the backend could not be
/// created.
static ErrorOr<RCRef<Cache::BlobCacheBackend>> getMojoCacheBackend() {
  return Cache::getLocalDefaultBackendChain(
      std::filesystem::path(".mojo_cache") / "transform", getVersionString());
}

//===----------------------------------------------------------------------===//
// createElaborateGeneratorsWithDefaultJIT
//===----------------------------------------------------------------------===//

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass> KGEN::createElaborateGeneratorsWithDefaultJIT() {
  return createElaborateGenerators(TargetInfoAttr(), /*elabOpts=*/{},
                                   /*options=*/{}, compileElaboratorAsm,
                                   compileOffloads);
}

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

void KGEN::populateElaborateModulePasses(mlir::PassManager &pm,
                                         TargetInfoAttr target,
                                         const CompilationOptions &options) {
  buildElaborateModulePipeline(pm, target, options, compileElaboratorAsm,
                               compileOffloads);
  buildPostElaborationPipeline(pm, options);
}

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<ExecutionEngine>> KGEN::initializeExecutionEngine(
    MLIRContext &context, const CompilationOptions &compilationOptions,
    ExecutionEngineOptions executionEngineOptions, bool isJIT,
    PassManagerConfigOptions pmOptions) {

  // Now create the execution engine so we can JIT.
  auto tmOr = createTargetMachine(compilationOptions, isJIT);
  if (tmOr.isError())
    return tmOr.takeError();

  return ExecutionEngine::createWithStandardLayers(
      std::move(executionEngineOptions), **tmOr);
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
  ContextRef ctx = loadContext(target.getContext());
  [[maybe_unused]] auto timeScope =
      ctx->get<M::Telemetry::TelemetryContext>()
          ->createUInt64Timer<std::chrono::milliseconds>(
              "mojo.kgen.compile.time", M::Telemetry::Level::L2, attrs);

  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(*cacheBackend));

  return runKGENPipeline(theModule, target, transformCache,
                         ctx->get<AsyncRT::Runtime>()->getReadyChain().copy(),
                         Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
                             "KGENCompiler::runKGENPipeline",
                             "mojo.compiler.cache.miss.time",
                             {{"pipeline", "KGEN"}}),
                         Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
                             "KGENCompiler::runKGENPipeline"));
}

static mlir::PassManager
createPassManager(const std::optional<std::string> &operationName,
                  MLIRContext *context) {
  if (operationName)
    return {context, *operationName};
  return {context};
}

ErrorOrSuccess
KGENCompiler::runKGENPipeline(ModuleOp theModule, TargetInfoAttr target,
                              RCRef<Cache::TransformCache> transformCache,
                              AnyAsyncValueRef chain,
                              std::function<void(Operation *)> moreOnMiss,
                              std::function<void(Operation *)> moreOnHit) {
  // Set the target now, so it's included in the cache key.
  if (!getTargetInfo(theModule))
    setTargetInfo(theModule, target);

  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM)
    return configPM.takeError();

  // Populate the passes.
  buildGenerateLibraryPipeline(pm, options);
  populateElaborateModulePasses(pm, target, options);

  // Run the passes as a cached transform.
  AsyncRT::AnyAsyncValueRef ready =
      Cache::cachedTransform(theModule, transformCache.copy(), std::move(chain),
                             pm, std::move(moreOnMiss), std::move(moreOnHit));

  // This await here is important since pm is local in this function.
  AsyncRT::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();
  return success();
}

ErrorOrSuccess KGENCompiler::runGenerateLibraryPipeline(ModuleOp module) {
  auto cacheBackend = getMojoCacheBackend();
  if (cacheBackend.isError())
    return cacheBackend.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(*cacheBackend));

  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM) {
    return Error(
        std::string("configure PassManager in "
                    "KGENCompiler::runGenerateLibraryPipeline failed, ") +
        configPM.getError());
  }

  buildGenerateLibraryPipeline(pm, options);

  AsyncRT::Runtime &runtime =
      *loadContext(module.getContext())->get<AsyncRT::Runtime>();
  AsyncRT::AnyAsyncValueRef ready = Cache::cachedTransform(
      module, transformCache.copy(), AsyncValueRef<Chain>::createReady(runtime),
      pm,
      Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
          "KGEN::runGenerateLibraryPipeline", "mojo.compiler.cache.miss.time",
          {{"pipeline", "KGEN"}}),
      Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
          "KGEN::runGenerateLibraryPipeline"));

  // This await here is important since pm is local in this function.
  AsyncRT::await(ready);
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

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM) {
    return Error(std::string("configure PassManager in "
                             "KGENCompiler::runCheckLITPipeline failed, ") +
                 configPM.getError());
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
ErrorOrSuccess KGENCompiler::runElaborationPipeline(
    ModuleOp module, TargetInfoAttr target, AsyncRT::Runtime &runtime,
    std::optional<AnyAsyncValueRef> chain,
    std::function<void(Operation *)> moreOnMiss,
    std::function<void(Operation *)> moreOnHit) {
  mlir::PassManager pm =
      createPassManager(pmConfigOptions.operationName, &context);

  ErrorOrSuccess configPM = pmConfigOptions.configurePassManager(pm);
  if (configPM)
    return configPM.takeError();

  populateElaborateModulePasses(pm, target, options);
  auto cacheBackend = getMojoCacheBackend();

  if (cacheBackend.isError() || !chain) {
    if (failed(pm.run(module)))
      return Error("KGENCompiler::runElaborationPipeline failed");
    return success();
  }

  AnyAsyncValueRef ready = Cache::cachedTransform(
      module, RCRef<Cache::TransformCache>::create(std::move(*cacheBackend)),
      std::move(*chain), pm, std::move(moreOnMiss), std::move(moreOnHit));

  // This await here is important since pm is local in this function.
  AsyncRT::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();
  return success();
}

static ErrorOrSuccess
setEmissionOptions(llvm::StringMap<llvm::cl::Option *> &options,
                   StringRef emissionOpt, bool reset) {
  if (!emissionOpt.contains("=")) {
    return Error("emission option must be of the form `option=value`");
  }
  auto [key, value] = emissionOpt.split("=");
  if (value.equals_insensitive("true") || value.equals_insensitive("false")) {
    auto *boolVal = static_cast<llvm::cl::opt<bool> *>(options[key]);
    if (!boolVal)
      return Error("emission option \"" + Twine(key) + "\" is not found");
    if (reset) {
      boolVal->reset();
    } else {
      boolVal->addOccurrence(0, key,
                             std::to_string(value.equals_insensitive("true")));
    }
  } else if (llvm::all_of(value, llvm::isDigit)) {
    auto *intVal = static_cast<llvm::cl::opt<int> *>(options[key]);
    if (!intVal)
      return Error("emission option \"" + Twine(key) + "\" is not found");
    if (reset)
      intVal->reset();
    else
      intVal->addOccurrence(0, key, value);
  } else {
    return Error("invalid emission option \"" + emissionOpt +
                 "\" (only boolean and integer values "
                 "are currently supported)");
  }
  return success();
}

ErrorOrSuccess KGEN::parseEmissionOptions(EmissionOptions emissionOptions) {
  // Handle the emission options.
  // Parse the emission options from a comma separated list of values.
  llvm::StringMap<llvm::cl::Option *> options =
      llvm::cl::getRegisteredOptions();

  for (StringRef elem : emissionOptions) {
    ErrorOrSuccess setOr = setEmissionOptions(options, elem, false);
    if (setOr.isError())
      return setOr.takeError();
  }
  return success();
}

ErrorOrSuccess KGEN::resetEmissionOptions(EmissionOptions emissionOptions) {
  // Handle the emission options.
  // Parse the emission options from a comma separated list of values.
  llvm::StringMap<llvm::cl::Option *> options =
      llvm::cl::getRegisteredOptions();

  for (StringRef elem : emissionOptions) {
    ErrorOrSuccess setOr = setEmissionOptions(options, elem, true);
    if (setOr.isError())
      return setOr.takeError();
  }
  return success();
}
