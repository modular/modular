//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/CustomDialect/CustomUtils.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/TransformUtils/SlicingUtils.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace KGEN;
using namespace Custom;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCUSTOMOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class LowerCustomOpsPass : public impl::LowerCustomOpsBase<LowerCustomOpsPass> {
public:
  LowerCustomOpsPass(const LibraryOptConfig &lib = {}) : lib(lib) {};

  void runOnOperation() override;

private:
  LibraryOptConfig lib;
};

using CompiledPattern = std::function<mlir::LogicalResult(
    mlir::Operation *, mlir::PatternRewriter &)>;

/// A canonicalization pattern for an op in the `custom` dialect.
struct CustomOpPattern : RewritePattern {
  CustomOpPattern(StringAttr opName, CompiledPattern canonicalizationFn)
      : RewritePattern(opName.strref(), /*benefit=*/9, opName.getContext()),
        canonicalizationFn(canonicalizationFn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &b) const override {
    return canonicalizationFn(op, b);
  }

private:
  CompiledPattern canonicalizationFn;
};
} // namespace

static std::pair<OwningOpRef<ModuleOp>, SymbolTable>
loadBytecodeForInstantiation(DenseResourceElementsAttr bytecode) {
  OwningOpRef<ModuleOp> module = readOpFromBytecodeFile<ModuleOp>(bytecode);
  SymbolTable symtab(*module);
  for (GeneratorOp gen : module->getOps<GeneratorOp>())
    gen.setNotExported();
  return {std::move(module), std::move(symtab)};
}

static ParameterExprArrayAttr unwrapPreservedParams(Attribute attr) {
  return cast<ParameterExprArrayAttr>(cast<PreservedAttr>(attr).getValue());
}

static StringAttr generateInstantiateStub(SymbolTable &symtab, StringAttr name,
                                          ParameterExprArrayAttr params,
                                          StringRef prefix, unsigned &counter) {
  auto callee = symtab.lookup<GeneratorOp>(name);
  assert(callee && "could not find the expected pattern");

  ParameterEvaluator evaluator(callee.getInputParams(), params);
  auto funcType =
      cast<FunctionType>(evaluator.getReboundType(callee.getFunctionType()));
  ImplicitLocOpBuilder b{callee.getLoc(), OpBuilder(name.getContext())};
  StringAttr instName = b.getStringAttr(prefix + Twine(counter++));
  auto inst = b.create<GeneratorOp>(instName, SignatureType::get(funcType));
  inst.setExported();
  symtab.insert(inst);

  Block *body = b.createBlock(&inst.getBodyRegion());
  SmallVector<Value> args;
  for (Type type : funcType.getInputs())
    args.push_back(body->addArgument(type, b.getLoc()));
  SignatureType calleeSig =
      callee.getSignature().getSpecializedSignature(params);
  auto call =
      b.create<CallOp>(SymbolConstantAttr::get(name, calleeSig, params), args);
  b.create<ReturnOp>(call.getResults());

  return instName;
}

template <typename AttrOrType>
static void doSlicing(AttrOrType value, DenseSet<const void *> &visited,
                      SmallVectorImpl<StringAttr> &worklist) {
  if (!visited.insert(value.getAsOpaquePointer()).second)
    return;
  if constexpr (std::is_same_v<AttrOrType, Attribute>)
    if (auto ref = dyn_cast<FlatSymbolRefAttr>(value))
      worklist.push_back(ref.getAttr());

  value.walkImmediateSubElements(
      [&](Attribute attr) { doSlicing(attr, visited, worklist); },
      [&](Type type) { doSlicing(type, visited, worklist); });
}

static void sliceInto(SymbolTable &dst, SymbolTable &src, StringAttr name) {
  SmallVector<StringAttr> worklist = {name};
  DenseSet<const void *> visited;
  auto visit = [&](auto value) { doSlicing(value, visited, worklist); };

  while (!worklist.empty()) {
    StringAttr attr = worklist.pop_back_val();
    if (dst.lookup(attr))
      continue;
    Operation *op = src.lookup(attr);
    assert(op && "op was already moved?");
    src.remove(op);
    op->remove();
    dst.insert(op);
    if (auto func = dyn_cast<FuncOp>(op))
      func.setNotExported();

    op->walk([&](Operation *op) {
      visit(op->getAttrDictionary());
      for (Type type : op->getResultTypes())
        visit(type);
      for (Region &region : op->getRegions())
        for (Type type : region.getArgumentTypes())
          visit(type);
    });
  }
}

void LowerCustomOpsPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto templates =
      module->getAttrOfType<DenseResourceElementsAttr>(kCustomOpImplModuleAttr);
  if (!templates)
    return;
  MLIRContext *ctx = &getContext();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  auto [patternModule, patternSymtab] = loadBytecodeForInstantiation(templates);

  // First step, we JIT the patterns. Collect the pattern functions per
  // prototype function name.
  SmallVector<StringAttr> names;
  SmallVector<SmallVector<StringAttr>> patterns;
  DenseMap<StringAttr, CustomOpImplAttr> remap;
  unsigned counter = 0;
  for (auto func : module.getOps<FuncOp>()) {
    CustomOpImplAttr impl = func.getPatternsAttr();
    if (!impl)
      continue;
    // Build the reverse map.
    remap.try_emplace(func.getSymNameAttr(), impl);
    if (impl.getPatterns().empty())
      continue;

    // Build an instantiate stub for every pattern.
    SmallVector<StringAttr> funcs;
    for (auto ref : impl.getPatterns().getAsRange<ArrayAttr>()) {
      auto name = cast<StringAttr>(ref[0]);
      ParameterExprArrayAttr params = unwrapPreservedParams(ref[1]);
      funcs.push_back(generateInstantiateStub(patternSymtab, name, params,
                                              "__pattern_inst_", counter));
    }
    names.push_back(impl.getProto());
    patterns.push_back(std::move(funcs));
  }

  // If no patterns were found, exit early.
  if (patterns.empty())
    return;

  // Compile the patterns.
  auto compiledPatterns = lib.compilePatterns(*patternModule, patterns);
  if (compiledPatterns.isError()) {
    mlir::emitError(module.getLoc(), "failed to compile patterns: ")
        << compiledPatterns.getError();
    return signalPassFailure();
  }

  // Raise all calls.
  DenseMap<std::pair<StringAttr, ArrayAttr>, StringAttr> instances;
  getOperation().walk([&](CallOp call) {
    StringAttr concreteName = call.getCallee().getSymbol().getLeafReference();
    CustomOpImplAttr impl = remap.lookup(concreteName);
    if (!impl)
      return;
    ImplicitLocOpBuilder b{call.getLoc(), OpBuilder(call)};
    OperationState state(call.getLoc(),
                         ("custom." + impl.getProto().getValue()).str(),
                         call.getOperands(), call.getResultTypes());
    SmallVector<Attribute> attrs;
    llvm::append_range(attrs, unwrapPreservedParams(impl.getParams()));
    auto params = ArrayAttr::get(ctx, attrs);
    instances[{impl.getProto(), params}] = concreteName;
    state.addAttribute("params", params);
    Operation *op = b.create(state);
    call->replaceAllUsesWith(op->getResults());
    call.erase();
  });

  // Now run the canonicalizer.
  mlir::RewritePatternSet rewritePatterns(ctx);
  for (auto [name, patternFuncs] : llvm::zip(names, *compiledPatterns)) {
    for (CAPICanonicalizationFn &fn : patternFuncs) {
      CompiledPattern pattern = [&](Operation *op, PatternRewriter &b) {
        MlirOperation opWrapped = wrap(op);
        MlirRewriterBase bWrapped = wrap(&b);
        return mlir::success(fn(&opWrapped, &bWrapped));
      };
      rewritePatterns.add<CustomOpPattern>(
          StringAttr::get(ctx, "custom." + name.getValue()),
          std::move(pattern));
    }
  }

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(rewritePatterns));
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  (void)applyPatternsAndFoldGreedily(getOperation(), frozenPatterns, config);

  // Now collect all required instantiations.
  struct OpRewrite {
    Operation *op;
    StringAttr name;
    ArrayAttr params;
  };
  SmallVector<OpRewrite> rewrites;
  getOperation().walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "custom")
      return;
    auto attrs = op->getAttrOfType<ArrayAttr>("params");
    if (!attrs)
      attrs = ArrayAttr::get(ctx, {});
    auto name = StringAttr::get(ctx, op->getName().stripDialect());
    rewrites.push_back({op, name, attrs});
    instances[{name, attrs}];
  });

  // Instantiate required ops.
  auto [instanceModule, instanceSymtab] =
      loadBytecodeForInstantiation(templates);
  counter = 0;
  for (auto &[instance, concrete] : instances) {
    if (concrete)
      continue;
    auto [name, attrs] = instance;
    SmallVector<TypedAttr> params;
    for (Attribute attr : attrs)
      params.push_back(cast<TypedAttr>(attr));
    concrete = generateInstantiateStub(instanceSymtab, name,
                                       ParameterExprArrayAttr::get(ctx, params),
                                       "__op_inst_", counter);
  }

  // Instantiate ops if required.
  if (counter) {
    mlir::PassManager mgr(ctx);
    lib.buildElaboratePipeline(mgr, lib);
    if (failed(mgr.run(*instanceModule))) {
      mlir::emitError(instanceModule->getLoc(), "op instantiation failed");
      return signalPassFailure();
    }
    instanceSymtab = SymbolTable(*instanceModule);
  }

  // Now go rewrite all the ops, lazily pulling in instantiated ops.
  for (const OpRewrite &rewrite : rewrites) {
    auto [op, name, params] = rewrite;
    StringAttr concrete = instances[{name, params}];
    assert(concrete && "should have been instantiated");

    sliceInto(symtab, instanceSymtab, concrete);
    auto func = symtab.lookup<FuncOp>(concrete);
    assert(func && "should have found an impl");

    ImplicitLocOpBuilder b{op->getLoc(), OpBuilder(op)};
    auto call =
        b.create<CallOp>(SymbolConstantAttr::get(func), op->getOperands());
    op->replaceAllUsesWith(call.getResults());
    op->erase();
  }

  // Reset all metadata. Remove `no_inline` from pattern functions.
  for (auto func : module.getOps<FuncOp>()) {
    if (func.getPatternsAttr()) {
      func.removePatternsAttr();
      func.setInlineLevel(InlineLevel::Automatic);
    }
  }
  module->removeAttr(kCustomOpImplModuleAttr);
}

std::unique_ptr<Pass> KGEN::createLowerCustomOps(const LibraryOptConfig &lib) {
  return std::make_unique<LowerCustomOpsPass>(lib);
}
