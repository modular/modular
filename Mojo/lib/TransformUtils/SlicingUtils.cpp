//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/TransformUtils/SlicingUtils.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "slicing-utils"

/// Dig into an attribute or type to find references to other symbols. If we see
/// a new one, copy it over.
template <typename AttrOrType>
static void sliceDependenciesFrom(AttrOrType value, SymbolTable &sliceSymtab,
                                  const SymbolTable &symtab,
                                  DenseSet<const void *> &visited,
                                  IRMapping &reusedMapping,
                                  std::vector<Operation *> &worklist) {
  // If we've already visited this value, we know we've extracted all
  // dependencies from it and its subtree.
  if (!visited.insert(value.getAsOpaquePointer()).second)
    return;

  // Check if this is a symbol reference.
  if constexpr (std::is_same_v<AttrOrType, Attribute>) {
    // Decorators are currently hacky and won't appear in the symbol table, so
    // we skip them.
    if (isa<KGEN::DecoratorsAttr>(value))
      return;

    if (auto ref = dyn_cast<FlatSymbolRefAttr>(value)) {
      // We know this is a new symbol because this is the first time we've
      // visited this attribute.
      StringAttr name = ref.getAttr();
      Operation *symbol = symtab.lookup(name);
      // If the symbol reference attribute doesn't reference a symbol, it must
      // be a reference that is not relative to `symtab`. Do not clone it.
      if (!symbol)
        return;

      // Clone the symbol into the new symbol table. Reuse an IRMapping to save
      // memory pressure.
      reusedMapping.clear();
      Operation *copy = symbol->clone(reusedMapping);
      sliceSymtab.insert(copy);

      // We need to recurse on this newly cloned operation.
      worklist.push_back(copy);

      // There are no further subelements. We can exit.
      return;
    }
  }

  // Recurse into subelements.
  value.walkImmediateSubElements(
      [&](Attribute attr) {
        sliceDependenciesFrom(attr, sliceSymtab, symtab, visited, reusedMapping,
                              worklist);
      },
      [&](Type type) {
        sliceDependenciesFrom(type, sliceSymtab, symtab, visited, reusedMapping,
                              worklist);
      });
}

/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
void M::sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                          const SymbolTable &symtab, IRMapping &reusedMapping,
                          DenseSet<const void *> &visited) {
  std::vector<Operation *> worklist;
  auto visit = [&](auto value) {
    sliceDependenciesFrom(value, sliceSymtab, symtab, visited, reusedMapping,
                          worklist);
  };
  auto extractDependencies = [&](Operation *op) {
    // Extract references to type declarations.
    visit(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      visit(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        visit(type);
  };

  worklist.push_back(op);
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();
    op->walk(extractDependencies);
  }
}

OwningOpRef<ModuleOp>
M::produceStandaloneModule(const SymbolTable &symtab,
                           const ExportMap &exportedSymbols,
                           bool overrideExported) {
  IRMapping unused;
  return produceStandaloneModule(symtab, exportedSymbols, unused,
                                 overrideExported);
}

OwningOpRef<ModuleOp>
M::produceStandaloneModule(const SymbolTable &symtab,
                           const ExportMap &exportedSymbols, IRMapping &mapping,
                           bool overrideExported) {

  DenseSet<const void *> visited;
  CompilerTimeTraceScope traceScope("produceStandaloneModule");
  auto module = cast<ModuleOp>(symtab.getOp());
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  OwningOpRef<ModuleOp> singleModule = ModuleOp::create(module->getLoc());
  singleModule.get()->setAttrs(module->getAttrDictionary());

  // Create a new symbol table for the sliced module.
  SymbolTable sliceSymtab(*singleModule);

  IRMapping reusedMapping;
  DenseSet<ExportInterface> exported;

  for (auto [sym, exportValKind] : exportedSymbols) {
    auto func = symtab.lookup<ExportInterface>(sym);
    // Absence is reachable rather than an invariant violation: the offload path
    // exports names taken from a kernel's parameter values, and a kernel
    // selected at run time never binds to a symbol the elaborator emitted.
    if (!func) {
      mlir::emitError(module->getLoc())
          << "cannot slice out exported symbol '" << sym.getValue()
          << "': it is not defined in this module";
      return {};
    }

    // Traverse the call graph and clone all the callees into this module.
    sliceDependencies(func, sliceSymtab, symtab, reusedMapping, visited);

    // Clone the func into this new module. We don't want to remove it from
    // the current module. Make sure the function is also exported in the slice.
    auto sliceFn = sliceSymtab.lookup<ExportInterface>(sym);
    if (!sliceFn) {
      sliceFn = cast<ExportInterface>(func->clone(mapping));
      sliceSymtab.insert(sliceFn);
    } else {
      mapping.map(func.getOperation(), sliceFn.getOperation());
    }
    ExportKind kind = func.getExportKind();
    sliceFn.setExportKind(kind == ExportKind::NotExported ? exportValKind
                                                          : kind);
    exported.insert(sliceFn);
  }

  if (overrideExported) {
    // Override exported info here for some cases.
    // For example:
    // GPU kernel should only have kernel entry function as exported.
    // Mark anyone else to not be exported here.
    // Some of they may be marked as exported because graph compiler needs
    // them to be or are needed as CPU functions .
    singleModule->walk([&](ExportInterface func) {
      if (!exported.contains(func) &&
          func.getExportKind() == ExportKind::Exported)
        func.setExportKind(ExportKind::NotExported);
    });
  }

  return singleModule;
}

LogicalResult LLVMDialectModuleSplitter::collectSymbols() {
  for (Operation &op : *module->getBody()) {
    if (isa<mlir::LLVM::LLVMFuncOp, mlir::LLVM::GlobalOp>(op)) {
      symbolIndex.insert({cast<mlir::SymbolOpInterface>(op).getNameAttr(),
                          static_cast<unsigned>(symbols.size())});
      symbols.push_back(&op);
      continue;
    }

    // Aliases, ifuncs and comdats are symbols with their own grouping rules,
    // which the partition does not implement.
    if (isa<mlir::SymbolOpInterface>(op))
      return op.emitOpError(
          "is not supported when splitting a module per exported function");

    // A module-level op that names a symbol, such as a list of constructors,
    // belongs with that symbol and not in every slice.
    bool hasSymbolUse = false;
    op.getAttrDictionary().walk([&](SymbolRefAttr) { hasSymbolUse = true; });
    if (hasSymbolUse)
      return op.emitOpError("names a symbol, which is not supported when "
                            "splitting a module per exported function");
    moduleOps.push_back(&op);
  }
  return success();
}

/// Whether `op` translates to an LLVM IR instruction for any operands.
/// Constants are not instructions, and debug records are not since LLVM 19.
static bool isInstruction(Operation *op) {
  return !isa<mlir::LLVM::ConstantOp, mlir::LLVM::ZeroOp, mlir::LLVM::UndefOp,
              mlir::LLVM::PoisonOp, mlir::LLVM::AddressOfOp,
              mlir::LLVM::DSOLocalEquivalentOp, mlir::LLVM::BlockAddressOp,
              mlir::LLVM::BlockTagOp, mlir::LLVM::NoneTokenOp,
              mlir::LLVM::MetadataAsValueOp, mlir::LLVM::DbgDeclareOp,
              mlir::LLVM::DbgValueOp, mlir::LLVM::DbgLabelOp>(op);
}

/// Whether translation folds `op` into a constant when its operands are
/// constants. The `IRBuilder` it uses folds these instructions.
static bool isFoldable(Operation *op) {
  namespace LLVM = mlir::LLVM;
  return isa<
      LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::UDivOp, LLVM::SDivOp,
      LLVM::URemOp, LLVM::SRemOp, LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp,
      LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp, LLVM::ICmpOp, LLVM::FCmpOp,
      LLVM::FAddOp, LLVM::FSubOp, LLVM::FMulOp, LLVM::FDivOp, LLVM::FRemOp,
      LLVM::FNegOp, LLVM::GEPOp, LLVM::BitcastOp, LLVM::AddrSpaceCastOp,
      LLVM::IntToPtrOp, LLVM::PtrToIntOp, LLVM::PtrToAddrOp, LLVM::SExtOp,
      LLVM::ZExtOp, LLVM::TruncOp, LLVM::SIToFPOp, LLVM::UIToFPOp,
      LLVM::FPToSIOp, LLVM::FPToUIOp, LLVM::FPExtOp, LLVM::FPTruncOp,
      LLVM::ExtractElementOp, LLVM::ExtractValueOp, LLVM::InsertElementOp,
      LLVM::InsertValueOp, LLVM::ShuffleVectorOp, LLVM::SelectOp>(op);
}

/// Counts what `llvm::Function::getInstructionCount` will count after
/// translation. The arguments of a non-entry block become phis.
static uint64_t countInstructions(mlir::LLVM::LLVMFuncOp func) {
  // Translation visits blocks in dominance order, which can differ from
  // region order. Iterate until no new constant appears.
  DenseSet<Value> constants;
  auto isConstant = [&](Value value) { return constants.contains(value); };
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](Operation *op) {
      if (op == func || op->getNumResults() == 0 ||
          isConstant(op->getResult(0)))
        return;
      if (isInstruction(op) &&
          !(isFoldable(op) && llvm::all_of(op->getOperands(), isConstant)))
        return;
      constants.insert(op->result_begin(), op->result_end());
      changed = true;
    });
  }

  uint64_t count = 0;
  for (Block &block : func.getBody()) {
    if (!block.isEntryBlock())
      count += block.getNumArguments();
    for (Operation &op : block)
      count += isInstruction(&op) &&
               (op.getNumResults() == 0 || !isConstant(op.getResult(0)));
  }
  return count;
}

LogicalResult
LLVMDialectModuleSplitter::collectDependencies(bool countInstructions) {
  MLIRContext *ctx = module->getContext();
  Dialect *llvmDialect = ctx->getLoadedDialect<mlir::LLVM::LLVMDialect>();
  splitSymbols.resize(symbols.size());

  // The first LLVM dialect op in each symbol that names an undefined symbol.
  // Diagnostics are emitted after the parallel walk, in module order.
  std::vector<std::pair<Operation *, StringAttr>> undefined(symbols.size());

  mlir::parallelFor(ctx, 0, symbols.size(), [&](size_t idx) {
    SmallVector<unsigned, 16> found;
    auto addSymbol = [&](StringAttr name) {
      auto it = symbolIndex.find(name);
      if (it == symbolIndex.end())
        return false;
      found.push_back(it->second);
      return true;
    };
    auto addRequiredSymbol = [&](Operation *user, FlatSymbolRefAttr ref) {
      if (ref && !addSymbol(ref.getAttr()) && !undefined[idx].first)
        undefined[idx] = {user, ref.getAttr()};
    };
    auto addSymbolsIn = [&](Attribute attr) {
      attr.walk([&](SymbolRefAttr ref) { addSymbol(ref.getRootReference()); });
    };

    Operation *symbol = symbols[idx];
    addSymbolsIn(symbol->getAttrDictionary());
    if (auto func = dyn_cast<mlir::LLVM::LLVMFuncOp>(symbol)) {
      addRequiredSymbol(func, func.getPersonalityAttr());
      if (countInstructions)
        splitSymbols[idx].instructionCount = ::countInstructions(func);
    } else if (auto global = dyn_cast<mlir::LLVM::GlobalOp>(symbol)) {
      addRequiredSymbol(global, global.getAssociatedAttr());
    }

    // `getAttrDictionary` builds a dictionary from the op properties, which is
    // too slow to do for every op of a function. The LLVM dialect ops below
    // are the only ones that hold a symbol.
    for (Region &region : symbol->getRegions()) {
      region.walk([&](Operation *op) {
        if (op->getDialect() != llvmDialect) {
          addSymbolsIn(op->getAttrDictionary());
        } else if (auto call = dyn_cast<mlir::LLVM::CallOp>(op)) {
          addRequiredSymbol(op, call.getCalleeAttr());
        } else if (auto invoke = dyn_cast<mlir::LLVM::InvokeOp>(op)) {
          addRequiredSymbol(op, invoke.getCalleeAttr());
        } else if (auto addressOf = dyn_cast<mlir::LLVM::AddressOfOp>(op)) {
          addRequiredSymbol(op, addressOf.getGlobalNameAttr());
        } else if (auto equivalent =
                       dyn_cast<mlir::LLVM::DSOLocalEquivalentOp>(op)) {
          addRequiredSymbol(op, equivalent.getFunctionNameAttr());
        } else if (auto blockAddress =
                       dyn_cast<mlir::LLVM::BlockAddressOp>(op)) {
          addRequiredSymbol(op, blockAddress.getBlockAddr().getFunction());
        } else if (auto constant = dyn_cast<mlir::LLVM::ConstantOp>(op)) {
          if (!isa<IntegerAttr, FloatAttr>(constant.getValue()))
            addSymbolsIn(constant.getValue());
        }
      });
    }

    llvm::sort(found);
    found.erase(llvm::unique(found), found.end());
    splitSymbols[idx].dependencies.assign(found.begin(), found.end());
  });

  for (auto [user, name] : undefined)
    if (user)
      return user->emitOpError() << "names '" << name.getValue()
                                 << "', which is not defined in the module";
  return success();
}

void LLVMDialectModuleSplitter::classifySymbols(
    function_ref<bool(mlir::LLVM::GlobalOp)> isSliceLocalGlobal) {
  for (auto [symbol, splitSymbol] : llvm::zip_equal(symbols, splitSymbols)) {
    splitSymbol.name = cast<mlir::SymbolOpInterface>(symbol).getName();
    if (auto func = dyn_cast<mlir::LLVM::LLVMFuncOp>(symbol)) {
      mlir::LLVM::Linkage linkage = func.getLinkage();
      splitSymbol.isFunction = true;
      splitSymbol.isDeclaration = func.isExternal();
      splitSymbol.isAnchor =
          !func.isExternal() &&
          llvm::is_contained(
              {mlir::LLVM::Linkage::External, mlir::LLVM::Linkage::Weak,
               mlir::LLVM::Linkage::WeakODR, mlir::LLVM::Linkage::ExternWeak},
              linkage);

      bool alwaysInline = func.isAlwaysInline();
      if (ArrayAttr passthrough = func.getPassthroughAttr())
        alwaysInline |= llvm::any_of(passthrough, [](Attribute attr) {
          auto name = dyn_cast<StringAttr>(attr);
          return name && name.getValue() == "alwaysinline";
        });
      splitSymbol.canOutline = !func.isExternal() &&
                               linkage == mlir::LLVM::Linkage::Internal &&
                               !alwaysInline;
      continue;
    }

    auto global = cast<mlir::LLVM::GlobalOp>(symbol);
    splitSymbol.isDeclaration =
        !global.getValueOrNull() && !global.getInitializerBlock();
    bool isLocal = llvm::is_contained(
        {mlir::LLVM::Linkage::Internal, mlir::LLVM::Linkage::Private},
        global.getLinkage());
    if (isLocal && global.getConstant())
      continue;
    bool isSliceLocal = isSliceLocalGlobal(global);
    splitSymbol.isAnchor = !isLocal && !isSliceLocal;
    splitSymbol.tiesUsers = !global.getConstant() && !isSliceLocal;
  }
}

/// Builds the declaration that a slice holds of a symbol that another slice
/// defines. `KGEN::splitPerExported` leaves the same in its partitions.
static Operation *createDeclaration(Operation *symbol) {
  Operation *declaration = symbol->cloneWithoutRegions();
  if (auto func = dyn_cast<mlir::LLVM::LLVMFuncOp>(declaration)) {
    func.setLinkage(mlir::LLVM::Linkage::External);
    func.setDsoLocal(false);
    func.removeComdatAttr();
    func.removePersonalityAttr();
    // The LLVM verifier rejects a declaration that carries the subprogram of
    // a definition.
    if (auto loc = dyn_cast<mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>>(
            func.getLoc()))
      func->setLoc(mlir::FusedLoc::get(func.getContext(), loc.getLocations()));
    return declaration;
  }
  auto global = cast<mlir::LLVM::GlobalOp>(declaration);
  global.removeValueAttr();
  global.setLinkage(mlir::LLVM::Linkage::External);
  global.setDsoLocal(false);
  global.removeComdatAttr();
  return declaration;
}

LogicalResult LLVMDialectModuleSplitter::plan(
    function_ref<bool(mlir::LLVM::GlobalOp)> isSliceLocalGlobal,
    unsigned outlineThreads) {
  CompilerTimeTraceScope traceScope("splitLLVMDialectModule");
  if (failed(collectSymbols()) ||
      failed(collectDependencies(/*countInstructions=*/outlineThreads != 0)))
    return failure();
  classifySymbols(isSliceLocalGlobal);
  SplitPlan splitPlan = planSplitPerExported(splitSymbols, outlineThreads);
  std::vector<std::vector<unsigned>> &sets = splitPlan.slices;

  // With one kernel, or kernels that must stay together, the module is the
  // slice.
  if (sets.size() <= 1) {
    slices.emplace_back().module = std::move(module);
    return success();
  }

  // An outlined function needs a definition that the other slices resolve
  // to. weak_odr, unlike weak, still lets its own slice inline it. It stays
  // dso_local, as an internal function is.
  for (unsigned idx : splitPlan.outlineRoots) {
    auto func = cast<mlir::LLVM::LLVMFuncOp>(symbols[idx]);
    func.setLinkage(mlir::LLVM::Linkage::WeakODR);
    func.setDsoLocal(true);
    outlinedFunctions.push_back(func.getSymNameAttr());
  }

  numSlicesUsing.assign(symbols.size(), 0);
  for (const std::vector<unsigned> &set : sets)
    for (unsigned member : set)
      ++numSlicesUsing[member];

  constexpr unsigned kNone = ~0u;
  std::vector<unsigned> heldBy(symbols.size(), kNone);
  std::vector<unsigned> declaredBy(symbols.size(), kNone);
  slices.reserve(sets.size());
  for (std::vector<unsigned> &set : sets) {
    unsigned sliceIdx = slices.size();
    Slice &slice = slices.emplace_back();
    for (unsigned member : set)
      heldBy[member] = sliceIdx;
    for (unsigned member : set) {
      for (unsigned dep : splitSymbols[member].dependencies) {
        if (heldBy[dep] == sliceIdx || declaredBy[dep] == sliceIdx)
          continue;
        declaredBy[dep] = sliceIdx;
        slice.declared.push_back(dep);
        if (declarationOf.count(dep))
          continue;
        if (!declarations)
          declarations = ModuleOp::create(module.get()->getLoc());
        Operation *declaration = createDeclaration(symbols[dep]);
        declarations->getBody()->push_back(declaration);
        declarationOf[dep] = declaration;
      }
    }
    llvm::sort(slice.declared);

    slice.module = ModuleOp::create(module.get()->getLoc());
    slice.module.get()->setAttrs(module.get()->getAttrDictionary());
    Block *body = slice.module->getBody();
    for (unsigned member : set) {
      if (numSlicesUsing[member] != 1)
        continue;
      symbols[member]->remove();
      body->push_back(symbols[member]);
    }
    slice.members = std::move(set);
  }

  LLVM_DEBUG(llvm::dbgs() << "split a module of " << symbols.size()
                          << " symbols into " << slices.size() << " slices\n");
  return success();
}

OwningOpRef<ModuleOp> LLVMDialectModuleSplitter::takeSlice(size_t idx) {
  Slice &slice = slices[idx];
  if (!module)
    return std::move(slice.module);

  // The symbols of this slice alone are already in its module. Clone the
  // shared ones and the declarations in between, to keep module order.
  Block *body = slice.module->getBody();
  Block::iterator next = body->begin();
  for (Operation *op : moduleOps)
    body->getOperations().insert(next, op->clone());
  auto declared = slice.declared.begin();
  auto declareBefore = [&](unsigned limit) {
    for (; declared != slice.declared.end() && *declared < limit; ++declared)
      body->getOperations().insert(
          next, declarationOf.find(*declared)->second->clone());
  };
  for (unsigned member : slice.members) {
    declareBefore(member);
    if (numSlicesUsing[member] == 1) {
      assert(&*next == symbols[member] && "slice is not in module order");
      ++next;
    } else {
      body->getOperations().insert(next, symbols[member]->clone());
    }
  }
  declareBefore(~0u);
  return std::move(slice.module);
}
