//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/CustomDialect/CustomUtils.h"
#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/TransformUtils/SCCUtils.h"
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

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

using namespace M;
using namespace KGEN;
using namespace Custom;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCUSTOMOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Pass Declaration
//===----------------------------------------------------------------------===//

namespace {
class LowerCustomOpsPass : public impl::LowerCustomOpsBase<LowerCustomOpsPass> {
public:
  LowerCustomOpsPass(const LibraryOptConfig &lib = {}) : lib(lib) {}

  void runOnOperation() override;

private:
  LibraryOptConfig lib;
};

//===----------------------------------------------------------------------===//
// CustomOpPattern
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Bytecode Instantiation Helpers
//===----------------------------------------------------------------------===//

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
  SignatureType calleeSig = callee.getSignature().getSpecializedSignature(
      params, [&] { return mlir::emitError(callee.getLoc()); });
  auto inst = b.create<GeneratorOp>(
      instName,
      SignatureGeneratorType::get({}, funcType, calleeSig.getArgConventions(),
                                  calleeSig.getFnEffects()));
  inst.setExported();
  symtab.insert(inst);

  Block *body = b.createBlock(&inst.getBodyRegion());
  SmallVector<Value> args;
  for (Type type : funcType.getInputs())
    args.push_back(body->addArgument(type, b.getLoc()));
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
    Operation *op = src.lookup(attr);
    if (!op)
      continue;
    src.remove(op);
    op->remove();
    if (Operation *existing = dst.lookup(attr))
      dst.erase(existing);
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

//===----------------------------------------------------------------------===//
// MemorySSA Helpers
//===----------------------------------------------------------------------===//

namespace {
struct CallGraphNode : public SCCNode<CallGraphNode, FuncOp, CallOp> {
  CallGraphNode(FuncOp func)
      : SCCNode(func), doesCapture(func ? func.getNumArguments() : 0, false) {}

  llvm::BitVector doesCapture;
};

/// Interprocedural pointer argument capture analysis.
struct CallGraph : public SCCGraph<CallGraph, CallGraphNode> {
  CallGraph(const SymbolTable &symtab) : symtab(symtab) {}

  bool shouldAddToGraph(CallOp call, CallGraphNode *node) { return true; }

  CallGraphNode *getCalleeNode(TypedAttr symbol) {
    auto callee = symtab.lookup<FuncOp>(
        cast<SymbolConstantAttr>(symbol).getSymbol().getRootReference());
    assert(callee);
    return &nodes.find(callee)->second;
  }

  bool doAnalysis(CallGraphNode *node);
  void doRewrite(const CallGraphNode *node) {}

  LLVM_DUMP_METHOD void dump() {
    for (auto &[func, node] : nodes) {
      if (!func)
        continue;
      llvm::errs() << '@' << func.getSymName() << '[';
      for (unsigned i = 0, e = func.getNumArguments(); i != e; ++i)
        llvm::errs() << node.doesCapture[i];
      llvm::errs() << ']' << "\n";
    }
  }

  const SymbolTable &symtab;
};

struct ModRefResult {
  DenseSet<Operation *> prevModRef;
  DenseSet<Operation *> nextModRef;

  DenseSet<Value> varReads;
  DenseSet<Value> varWrites;
  bool refEscaped = false;
};

struct MemoryUseNode {
  llvm::SetVector<Operation *> prevModRef;
  llvm::SetVector<Operation *> nextModRef;
};

struct ModRefAnalysis : public mlir::RewriterBase::Listener {
  ModRefAnalysis(AsyncRT::Runtime &runtime, SymbolTable &symtab)
      : symtab(symtab), runtime(runtime), cg(symtab) {}

  void computeCaptures(ModuleOp module);
  void processRegion(
      Region &region, const HLCF::CFGAnalysis &cfg,
      llvm::SetVector<Operation *> &prev,
      llvm::MapVector<Operation *, ModRefResult> &results,
      DenseMap<HLCF::CFGNode, llvm::SetVector<Operation *>> &virtualPhiNodes,
      bool dryRun);
  void compute(ModuleOp module);

  void rauw(ArrayRef<Operation *> from, ArrayRef<Operation *> to);
  void notifyOperationErased(Operation *op) override {
    if (auto it = allResults.find(op); it != allResults.end()) {
      if (it->second->nextModRef.empty() && it->second->prevModRef.empty())
        return;
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "erasing operation that still has memory effect observers, please "
            "use rauw on the dependency analysis: "
         << *op;
      llvm::report_fatal_error(StringRef(msg));
    }
  }

  SymbolTable &symtab;
  AsyncRT::Runtime &runtime;
  CallGraph cg;
  DenseMap<Operation *, std::unique_ptr<MemoryUseNode>> allResults;
};

struct CMRAnalysis {
  ModRefAnalysis &mr;
};
} // namespace

extern "C" {
MLIR_CAPI_EXPORTED void mlirCMRAnalysisRAUW(void *ptr,
                                            ArrayRef<MlirOperation> from,
                                            ArrayRef<MlirOperation> to) {
  SmallVector<Operation *> fromOps, toOps;
  for (MlirOperation op : from)
    fromOps.push_back(unwrap(op));
  for (MlirOperation op : to)
    toOps.push_back(unwrap(op));
  ((CMRAnalysis *)ptr)->mr.rauw(fromOps, toOps);
}

MLIR_CAPI_EXPORTED size_t mlirCMRAnalysisGetNextModRefCount(void *ptr,
                                                            MlirOperation op) {
  return ((CMRAnalysis *)ptr)->mr.allResults.at(unwrap(op))->nextModRef.size();
}

MLIR_CAPI_EXPORTED size_t mlirCMRAnalysisGetPrevModRefCount(void *ptr,
                                                            MlirOperation op) {
  return ((CMRAnalysis *)ptr)->mr.allResults.at(unwrap(op))->prevModRef.size();
}

MLIR_CAPI_EXPORTED void
mlirCMRAnalysisGetNextModRefValues(void *ptr, MlirOperation op,
                                   MlirOperation *results) {
  for (Operation *op : ((CMRAnalysis *)ptr)
                           ->mr.allResults.at(unwrap(op))
                           ->nextModRef.getArrayRef()) {
    (*results) = wrap(op);
    ++results;
  }
}

MLIR_CAPI_EXPORTED void
mlirCMRAnalysisGetPrevModRefValues(void *ptr, MlirOperation op,
                                   MlirOperation *results) {
  for (Operation *op : ((CMRAnalysis *)ptr)
                           ->mr.allResults.at(unwrap(op))
                           ->prevModRef.getArrayRef()) {
    (*results) = wrap(op);
    ++results;
  }
}

MLIR_CAPI_EXPORTED MlirSymbolTable mlirCMRAnalysisGetSymbolTable(void *ptr) {
  return wrap(&((CMRAnalysis *)ptr)->mr.symtab);
}

MLIR_CAPI_EXPORTED MlirAttribute
mlirSymbolConstantGetSymbolRef(MlirAttribute attr) {
  return wrap(cast<SymbolConstantAttr>(unwrap(attr)).getSymbol());
}
}

static bool isEscapedPointer(Value arg, CallGraph &cg) {
  if (!isa<PointerType>(arg.getType()))
    return false;

  // Build a worklist of all SSA values that trivially alias the block
  // argument.
  SmallVector<Value> worklist;
  worklist.push_back(arg);

  // Iterate while the worklist isn't empty and if the analysis for the value
  // is not at fixed point.
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      // Aliasing ops.
      Operation *user = use.getOwner();
      if (isa<POP::ArrayGEPOp, StructGEPOp, POP::OffsetOp,
              POP::PointerBitcastOp>(user)) {
        assert(user->getNumResults() == 1);
        worklist.push_back(user->getResult(0));
        continue;
      }

      // Loads are terminals.
      if (isa<POP::LoadOp, POP::StackAllocLifetimeStartOp,
              POP::StackAllocLifetimeEndOp>(use.getOwner()))
        continue;

      // Stores that don't capture the value are terminals.
      if (auto store = dyn_cast<POP::StoreOp>(user)) {
        if (store.getArg() == value)
          return true;
        continue;
      }

      // Handle calls specially.
      auto callee = user->getAttrOfType<SymbolConstantAttr>("callee");
      if (!callee)
        return true;

      CallGraphNode *calleeNode = cg.getCalleeNode(callee);
      if (calleeNode->doesCapture.test(use.getOperandNumber()))
        return true;
    }
  }
  return false;
}

bool CallGraph::doAnalysis(CallGraphNode *node) {
  bool changed = false;

  // Check every block argument.
  for (BlockArgument arg : node->func.getArguments()) {
    if (!isa<PointerType>(arg.getType()) ||
        node->doesCapture.test(arg.getArgNumber()))
      continue;

    if (isEscapedPointer(arg, *this)) {
      changed = true;
      node->doesCapture.set(arg.getArgNumber());
    }
  }

  return changed;
}

void ModRefAnalysis::computeCaptures(ModuleOp module) {
  cg.build(module, symtab);
  cg.run(runtime);
}

void ModRefAnalysis::processRegion(
    Region &region, const HLCF::CFGAnalysis &cfg,
    llvm::SetVector<Operation *> &prev,
    llvm::MapVector<Operation *, ModRefResult> &results,
    DenseMap<HLCF::CFGNode, llvm::SetVector<Operation *>> &virtualPhiNodes,
    bool dryRun) {
  assert(llvm::hasSingleElement(region));
  for (Operation &op : region.getOps()) {
    ModRefResult &result = results[&op];

    // For each terminator, propagate the prev set to the virtual phi node set.
    if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(op)) {
      if (auto it = cfg.successors.find(term); it != cfg.successors.end())
        for (HLCF::CFGNode node : it->second)
          virtualPhiNodes[node].insert(prev.begin(), prev.end());
      continue;
    }

    // Skip side-effect free ops and lifetime markers.
    if (mlir::isMemoryEffectFree(&op) ||
        isa<POP::StackAllocLifetimeStartOp, POP::StackAllocLifetimeEndOp>(op)) {
      // For unknown region operations, treat the region as isolated.
      for (Region &region : op.getRegions()) {
        llvm::SetVector<Operation *> nestedPrev;
        processRegion(region, cfg, nestedPrev, results, virtualPhiNodes,
                      /*dryRun=*/false);
      }
      continue;
    }

    auto node = dyn_cast<HLCF::ControlFlowNode>(op);
    if (!node) {
      // For all operations, chain the prev and next operations.
      for (Operation *prevOp : prev) {
        result.prevModRef.insert(prevOp);
        results[prevOp].nextModRef.insert(&op);
      }
      prev.clear();
      if (dryRun)
        return;
      prev.insert(&op);

      // For unknown region operations, treat the region as isolated.
      for (Region &region : op.getRegions()) {
        llvm::SetVector<Operation *> nestedPrev;
        processRegion(region, cfg, nestedPrev, results, virtualPhiNodes,
                      /*dryRun=*/false);
      }
      continue;
    }

    for (Region &region : op.getRegions()) {
      llvm::SetVector<Operation *> &phiEntry =
          virtualPhiNodes[{node, region.getRegionNumber()}];
      // Initialize the prev set with the phi node set of the region.
      phiEntry.insert(prev.begin(), prev.end());
      prev.clear();

      size_t curPhiSize = phiEntry.size();

      llvm::SetVector<Operation *> nestedPrev = phiEntry;
      processRegion(region, cfg, nestedPrev, results, virtualPhiNodes, dryRun);

      // Re-query because the map might reallocate.
      const llvm::SetVector<Operation *> &nextPhiEntry =
          virtualPhiNodes.at({node, region.getRegionNumber()});
      if (nextPhiEntry.size() != curPhiSize) {
        llvm::SetVector<Operation *> nestedPrev = nextPhiEntry;
        processRegion(region, cfg, nestedPrev, results, virtualPhiNodes,
                      /*dryRun=*/true);
      }
    }
    if (dryRun)
      return;

    // Take the phi node set of the parent op and set it as the current prev
    // set.
    prev = virtualPhiNodes[{node, std::nullopt}];
  }
}

static Value getIdentifiedVariable(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (isa_and_nonnull<POP::StackAllocationOp>(defOp))
    return value;
  if (isa_and_nonnull<POP::ArrayGEPOp, StructGEPOp, POP::OffsetOp,
                      POP::PointerBitcastOp>(defOp)) {
    return getIdentifiedVariable(defOp->getOperand(0));
  }
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return {};
  auto func = dyn_cast<FuncOp>(arg.getOwner()->getParentOp());
  if (!func)
    return {};
  ArgConvention conv = func.getSignatureGenerator().getBody().getArgConvention(
      arg.getArgNumber());
  if (SignatureType::hasAddress(conv))
    return arg;
  return {};
}

void ModRefAnalysis::compute(ModuleOp module) {
  for (FuncOp func : module.getOps<FuncOp>()) {
    HLCF::CFGAnalysis cfg(func);
    llvm::SetVector<Operation *> prev;
    llvm::MapVector<Operation *, ModRefResult> results;
    DenseMap<HLCF::CFGNode, llvm::SetVector<Operation *>> virtualPhiNodes;
    processRegion(func.getBodyRegion(), cfg, prev, results, virtualPhiNodes,
                  /*dryRun=*/false);

    // Now consider aliasing between nonescaping identified variables. The user
    // only cares about modref relationships between operations on these
    // variables.
    for (auto &[op, result] : results) {
      // The user doesn't care about anything other than function calls, because
      // those ops cannot be optimized.
      auto callee = op->getAttrOfType<SymbolConstantAttr>("callee");
      if (!callee) {
        if (auto alloc = dyn_cast<POP::StackAllocationOp>(op)) {
          result.varWrites.insert(alloc.getResult());
          continue;
        }
        if (auto load = dyn_cast<POP::LoadOp>(op)) {
          Value obj = getIdentifiedVariable(load.getPtr());
          if (!obj)
            continue;
          if (isEscapedPointer(obj, cg)) {
            result.refEscaped = true;
            continue;
          }
          result.varReads.insert(obj);
          continue;
        }
        if (auto store = dyn_cast<POP::StoreOp>(op)) {
          Value obj = getIdentifiedVariable(store.getPtr());
          if (!obj)
            continue;
          if (isEscapedPointer(obj, cg)) {
            result.refEscaped = true;
            continue;
          }
          result.varWrites.insert(obj);
          continue;
        }
        continue;
      }

      // Find nonescaping identified variables that are known written to and
      // read from by the function call.
      SignatureType sig = callee.getType();
      for (auto [i, arg] : llvm::enumerate(op->getOperands())) {
        Value obj = getIdentifiedVariable(arg);
        if (!obj)
          continue;
        if (isEscapedPointer(obj, cg)) {
          result.refEscaped = true;
          continue;
        }
        if (llvm::is_contained({ArgConvention::ReadMem, ArgConvention::Ref},
                               sig.getArgConvention(i)))
          result.varReads.insert(arg);
        else
          result.varWrites.insert(arg);
      }
    }
    for (auto &[op, result] : results) {
      allResults.try_emplace(op,
                             std::make_unique<MemoryUseNode>(MemoryUseNode{}));
    }

    for (auto &[op, result] : results) {
      MemoryUseNode &allResult = *allResults.at(op);

      if (result.refEscaped) {
        for (Operation *prev : result.prevModRef) {
          allResult.prevModRef.insert(prev);
        }
        for (Operation *next : result.nextModRef) {
          allResult.nextModRef.insert(next);
        }
      }

      if (result.varWrites.empty() && result.varReads.empty())
        continue;

      SmallVector<Operation *> worklist;
      DenseSet<Operation *> visited;
      llvm::append_range(worklist, result.prevModRef);
      while (!worklist.empty()) {
        Operation *prev = worklist.pop_back_val();
        if (!visited.insert(prev).second)
          continue;
        const ModRefResult &prevResult = results[prev];
        bool keep = false;
        for (Value v :
             llvm::concat<const Value>(result.varWrites, result.varReads)) {
          if (prevResult.varWrites.contains(v)) {
            keep = true;
            break;
          }
        }
        if (keep) {
          allResult.prevModRef.insert(prev);
          continue;
        }
        llvm::append_range(worklist, prevResult.prevModRef);
      }

      visited.clear();
      llvm::append_range(worklist, result.nextModRef);
      while (!worklist.empty()) {
        Operation *next = worklist.pop_back_val();
        if (!visited.insert(next).second)
          continue;
        const ModRefResult &nextResult = results[next];
        bool keep = false;
        bool foundDef = false;
        for (Value v : result.varWrites) {
          if (nextResult.varReads.contains(v)) {
            keep = true;
          }
          if (nextResult.varWrites.contains(v)) {
            keep = true;
            foundDef = true;
          }
        }
        if (keep) {
          allResult.nextModRef.insert(next);
          if (foundDef)
            continue;
        }
        llvm::append_range(worklist, nextResult.nextModRef);
      }
    }

    /*
    for (auto &[op, result] : allResults) {
      llvm::errs() << "\n\n" << *op << "\n";
      llvm::errs() << "prev:\n";
      for (Operation *op : result->prevModRef)
        llvm::errs() << "  " << *op << "\n";
      llvm::errs() << "next:\n";
      for (Operation *op : result->nextModRef)
        llvm::errs() << "  " << *op << "\n";
    }
    */
  }
}

void ModRefAnalysis::rauw(ArrayRef<Operation *> from,
                          ArrayRef<Operation *> to) {
  if (from.empty()) {
    llvm::report_fatal_error("source dependency chain is empty");
  }

  SmallVector<std::pair<Operation *, MemoryUseNode *>> srcChain;
  for (auto [i, op] : llvm::enumerate(from)) {
    MemoryUseNode *ch = allResults.at(op).get();
    if (!srcChain.empty()) {
      auto [prevOp, prev] = srcChain.back();
      if (prev->nextModRef.size() != 1 || !prev->nextModRef.contains(op) ||
          ch->prevModRef.size() != 1 || !ch->prevModRef.contains(prevOp)) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "index " << i << " source chain from " << *prevOp
           << " does not uniquely connect to " << *op;
        llvm::report_fatal_error(StringRef(msg));
      }
    }

    srcChain.emplace_back(op, ch);
  }

  SmallVector<std::pair<Operation *, MemoryUseNode *>> dstChain;
  for (Operation *op : to) {
    allResults.try_emplace(op,
                           std::make_unique<MemoryUseNode>(MemoryUseNode{}));
    MemoryUseNode *ch = allResults.at(op).get();
    if (!dstChain.empty()) {
      auto [prevOp, prev] = dstChain.back();
      prev->nextModRef.insert(op);
      ch->prevModRef.insert(prevOp);
    }
    dstChain.emplace_back(op, ch);
  }

  assert(!srcChain.empty());
  llvm::SetVector<Operation *> &head = srcChain.front().second->prevModRef;
  llvm::SetVector<Operation *> &tail = srcChain.back().second->nextModRef;
  for (Operation *prevOp : head) {
    MemoryUseNode *prev = allResults.at(prevOp).get();
    prev->nextModRef.remove(srcChain.front().first);
    if (dstChain.empty()) {
      prev->nextModRef.insert(tail.begin(), tail.end());
    } else {
      prev->nextModRef.insert(dstChain.front().first);
    }
  }
  for (Operation *nextOp : tail) {
    MemoryUseNode *next = allResults.at(nextOp).get();
    next->prevModRef.remove(srcChain.back().first);
    if (dstChain.empty()) {
      next->prevModRef.insert(head.begin(), head.end());
    } else {
      next->prevModRef.insert(dstChain.back().first);
    }
  }

  for (auto [op, result] : srcChain) {
    allResults.erase(op);
  }
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

void LowerCustomOpsPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto templates =
      module->getAttrOfType<DenseResourceElementsAttr>(kCustomOpImplModuleAttr);
  if (!templates)
    return;
  module->removeAttr(kCustomOpImplModuleAttr);
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
    auto patternsAttr = cast<ArrayAttr>(impl.getPatterns().getValue());
    if (patternsAttr.empty())
      continue;

    // Build an instantiate stub for every pattern.
    SmallVector<StringAttr> funcs;
    ParameterEvaluator evaluator;
    for (auto ref : patternsAttr.getAsRange<ArrayAttr>()) {
      auto name = cast<StringAttr>(ref[0]);
      auto params = cast<ParameterExprArrayAttr>(ref[1]);
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
  for (auto func : patternModule->getOps<GeneratorOp>()) {
    func.removePatternsAttr();
  }
  auto compiledPatterns =
      lib.compilePatterns(std::move(patternModule), patterns);
  if (compiledPatterns.isError()) {
    mlir::emitError(module.getLoc(), "failed to compile patterns: ")
        << compiledPatterns.getError();
    return signalPassFailure();
  }

  // Compute modref before raising.
  AsyncRT::Runtime &runtime =
      *loadContext(&getContext())->get<AsyncRT::Runtime>();
  ModRefAnalysis mr(runtime, symtab);
  mr.computeCaptures(getOperation());
  llvm::nulls() << (void *)mlirCMRAnalysisRAUW
                << (void *)mlirCMRAnalysisGetNextModRefCount
                << (void *)mlirCMRAnalysisGetPrevModRefCount
                << (void *)mlirCMRAnalysisGetNextModRefValues
                << (void *)mlirCMRAnalysisGetPrevModRefValues
                << (void *)mlirCMRAnalysisGetSymbolTable
                << (void *)mlirSymbolConstantGetSymbolRef;

  // Raise all calls.
  llvm::MapVector<std::pair<StringAttr, ArrayAttr>, StringAttr> instances;
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
    state.addAttribute("callee", call.getCallee());
    Operation *op = b.create(state);
    call->replaceAllUsesWith(op->getResults());
    call.erase();
  });

  // Now run the canonicalizer.
  mr.compute(getOperation());
  mlir::RewritePatternSet rewritePatterns(ctx);
  for (auto [name, patternFuncs] : llvm::zip(names, *compiledPatterns)) {
    for (CAPICanonicalizationFn &fn : patternFuncs) {
      CompiledPattern pattern = [&](Operation *op, PatternRewriter &b) {
        MlirOperation opWrapped = wrap(op);
        MlirRewriterBase bWrapped = wrap(&b);
        CMRAnalysis cmr{mr};
        return mlir::success(fn(&opWrapped, &bWrapped, (void *)&cmr));
      };
      rewritePatterns.add<CustomOpPattern>(
          StringAttr::get(ctx, "custom." + name.getValue()),
          std::move(pattern));
    }
  }

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(rewritePatterns));
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  config.listener = &mr;
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
    mlir::AttrTypeWalker symFinder;
    symFinder.addWalk([&, symtabRef = &instanceSymtab](FlatSymbolRefAttr ref) {
      if (symtabRef->lookup(ref.getAttr()))
        return;
      Operation *op = symtab.lookup(ref.getAttr());
      if (!op)
        return;
      symtabRef->insert(op->clone());
      return;
    });
    SmallVector<TypedAttr> params;
    for (Attribute attr : attrs) {
      symFinder.walk(attr);
      params.push_back(cast<TypedAttr>(attr));
    }
    concrete = generateInstantiateStub(instanceSymtab, name,
                                       ParameterExprArrayAttr::get(ctx, params),
                                       "__op_inst_", counter);
  }

  // Instantiate ops if required.
  mlir::PassManager mgr(ctx);
  lib.buildElaboratePipeline(mgr, lib);
  if (failed(mgr.run(*instanceModule))) {
    mlir::emitError(instanceModule->getLoc(), "op instantiation failed");
    return signalPassFailure();
  }
  instanceSymtab = SymbolTable(*instanceModule);

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
}

std::unique_ptr<Pass> KGEN::createLowerCustomOps(const LibraryOptConfig &lib) {
  return std::make_unique<LowerCustomOpsPass>(lib);
}
