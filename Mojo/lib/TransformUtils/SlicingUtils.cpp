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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
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
  dependencies.resize(symbols.size());
  if (countInstructions)
    instructionCounts.assign(symbols.size(), 0);

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
        instructionCounts[idx] = ::countInstructions(func);
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
    dependencies[idx].assign(found.begin(), found.end());
  });

  for (auto [user, name] : undefined)
    if (user)
      return user->emitOpError() << "names '" << name.getValue()
                                 << "', which is not defined in the module";
  return success();
}

void LLVMDialectModuleSplitter::classifySymbols(
    function_ref<bool(mlir::LLVM::GlobalOp)> isSliceLocalGlobal) {
  isAnchor.assign(symbols.size(), false);
  isMutableGlobal.assign(symbols.size(), false);

  for (auto [idx, symbol] : llvm::enumerate(symbols)) {
    if (auto func = dyn_cast<mlir::LLVM::LLVMFuncOp>(symbol)) {
      mlir::LLVM::Linkage linkage = func.getLinkage();
      isAnchor[idx] =
          !func.isExternal() &&
          llvm::is_contained(
              {mlir::LLVM::Linkage::External, mlir::LLVM::Linkage::Weak,
               mlir::LLVM::Linkage::WeakODR, mlir::LLVM::Linkage::ExternWeak},
              linkage);
      continue;
    }

    auto global = cast<mlir::LLVM::GlobalOp>(symbol);
    bool isLocal = llvm::is_contained(
        {mlir::LLVM::Linkage::Internal, mlir::LLVM::Linkage::Private},
        global.getLinkage());
    if (isLocal && global.getConstant())
      continue;
    bool isSliceLocal = isSliceLocalGlobal(global);
    isAnchor[idx] = !isLocal && !isSliceLocal;
    isMutableGlobal[idx] = !global.getConstant() && !isSliceLocal;
  }
}

SmallVector<unsigned>
LLVMDialectModuleSplitter::collectReachable(unsigned root,
                                            bool stopAtOutlineRoots) const {
  SmallVector<unsigned> members;
  DenseSet<unsigned> seen;
  SmallVector<unsigned> worklist{root};
  while (!worklist.empty()) {
    unsigned idx = worklist.pop_back_val();
    if (idx != root && stopAtOutlineRoots && isOutlineRoot[idx])
      continue;
    if (!seen.insert(idx).second)
      continue;
    members.push_back(idx);
    llvm::append_range(worklist, dependencies[idx]);
  }
  return members;
}

/// Longest-processing-time makespan, as `KGEN::splitPerExported` estimates it.
static uint64_t estimateMakespan(ArrayRef<uint64_t> sizes, unsigned threads) {
  SmallVector<uint64_t> load(threads, 0);
  SmallVector<uint64_t> sorted(sizes);
  llvm::sort(sorted, std::greater<uint64_t>());
  for (uint64_t size : sorted)
    *llvm::min_element(load) += size;
  return *llvm::max_element(load);
}

/// Mirrors `pickOutlineRoots` in `KGEN::splitPerExported`, which documents
/// the rule. Matching it keeps the partition and the LLVM work per slice the
/// same in either IR.
void LLVMDialectModuleSplitter::pickOutlineRoots(unsigned threads) {
  constexpr unsigned kNone = ~0u;
  size_t numSymbols = symbols.size();
  std::vector<SmallVector<unsigned, 4>> users(numSymbols);
  for (unsigned idx = 0; idx != numSymbols; ++idx)
    for (unsigned dep : dependencies[idx])
      users[dep].push_back(idx);

  // Which anchors reach each symbol, as positions in `partitionSize`, and how
  // big each anchor's closure is.
  std::vector<SmallVector<unsigned, 2>> reachedBy(numSymbols);
  SmallVector<uint64_t> partitionSize;
  std::vector<unsigned> seenBy(numSymbols, kNone);
  SmallVector<unsigned> worklist;
  for (unsigned anchor = 0; anchor != numSymbols; ++anchor) {
    if (!isAnchor[anchor])
      continue;
    unsigned position = partitionSize.size();
    uint64_t size = 0;
    worklist.assign(1, anchor);
    while (!worklist.empty()) {
      unsigned idx = worklist.pop_back_val();
      if (seenBy[idx] == position)
        continue;
      seenBy[idx] = position;
      size += instructionCounts[idx];
      if (idx != anchor)
        reachedBy[idx].push_back(position);
      llvm::append_range(worklist, dependencies[idx]);
    }
    partitionSize.push_back(size);
  }

  struct Candidate {
    unsigned root;
    uint64_t size;
    uint64_t exclusive;
    uint64_t saving;
  };
  SmallVector<Candidate> candidates;
  std::vector<bool> dominated(numSymbols, false);
  for (unsigned root = 0; root != numSymbols; ++root) {
    if (reachedBy[root].size() < 2)
      continue;
    auto func = dyn_cast<mlir::LLVM::LLVMFuncOp>(symbols[root]);
    if (!func || func.isExternal() ||
        func.getLinkage() != mlir::LLVM::Linkage::Internal)
      continue;
    bool alwaysInline = func.isAlwaysInline();
    if (ArrayAttr passthrough = func.getPassthroughAttr())
      alwaysInline |= llvm::any_of(passthrough, [](Attribute attr) {
        auto name = dyn_cast<StringAttr>(attr);
        return name && name.getValue() == "alwaysinline";
      });
    if (alwaysInline)
      continue;

    // The part of the subtree that only the root reaches leaves the anchors'
    // slices. The rest stays there and gains a copy in the root's slice.
    SmallVector<unsigned> members = collectReachable(root, false);
    dominated[root] = true;
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned member : members) {
        if (dominated[member] || users[member].empty())
          continue;
        if (llvm::all_of(users[member],
                         [&](unsigned user) { return dominated[user]; })) {
          dominated[member] = true;
          changed = true;
        }
      }
    }

    uint64_t size = 0, exclusive = 0;
    for (unsigned member : members) {
      size += instructionCounts[member];
      if (dominated[member])
        exclusive += instructionCounts[member];
      dominated[member] = false;
    }
    uint64_t shared = size - exclusive;
    uint64_t gained = exclusive * (reachedBy[root].size() - 1);
    if (gained <= shared)
      continue;
    candidates.push_back({root, size, exclusive, gained - shared});
  }

  auto nameOf = [&](unsigned idx) {
    return cast<mlir::SymbolOpInterface>(symbols[idx]).getName();
  };
  llvm::sort(candidates, [&](const Candidate &lhs, const Candidate &rhs) {
    if (lhs.saving != rhs.saving)
      return lhs.saving > rhs.saving;
    return nameOf(lhs.root) < nameOf(rhs.root);
  });

  uint64_t makespan = estimateMakespan(partitionSize, threads);
  std::vector<bool> covered(numSymbols, false);
  SmallVector<uint64_t> trial;
  for (const Candidate &candidate : candidates) {
    if (covered[candidate.root])
      continue;

    trial.assign(partitionSize.begin(), partitionSize.end());
    for (unsigned position : reachedBy[candidate.root])
      trial[position] -= std::min(trial[position], candidate.exclusive);
    trial.push_back(candidate.size);
    uint64_t trialMakespan = estimateMakespan(trial, threads);
    if (trialMakespan > makespan)
      continue;

    isOutlineRoot[candidate.root] = true;
    makespan = trialMakespan;
    std::swap(partitionSize, trial);
    for (unsigned member : collectReachable(candidate.root, false))
      if (member != candidate.root)
        covered[member] = true;
  }
}

std::vector<std::vector<unsigned>> LLVMDialectModuleSplitter::partition() {
  constexpr unsigned kNone = ~0u;

  // The closure of one anchor, and the group that it merges into when it
  // shares an anchor or a mutable global with another closure.
  struct Closure {
    unsigned root;
    SmallVector<unsigned> members;
    std::optional<unsigned> group;
  };
  std::vector<Closure> closures;
  std::vector<unsigned> closureOf(symbols.size(), kNone);
  for (unsigned idx = 0, e = symbols.size(); idx != e; ++idx) {
    if (!isAnchor[idx])
      continue;
    closureOf[idx] = static_cast<unsigned>(closures.size());
    closures.push_back({idx, {}, std::nullopt});
  }

  // The closures that must share a slice, keyed by the symbol that ties them.
  llvm::MapVector<unsigned, llvm::SetVector<unsigned>> tiedClosures;
  std::vector<unsigned> visitedBy(symbols.size(), kNone);
  SmallVector<unsigned> worklist;
  for (unsigned closureIdx = 0, e = closures.size(); closureIdx != e;
       ++closureIdx) {
    Closure &closure = closures[closureIdx];
    worklist.assign(1, closure.root);
    while (!worklist.empty()) {
      unsigned idx = worklist.pop_back_val();
      // An outline root has a slice of its own. It does not tie the closures
      // that reach it, which would undo the parallelism the split creates.
      if (isOutlineRoot[idx] || visitedBy[idx] == closureIdx)
        continue;
      visitedBy[idx] = closureIdx;
      closure.members.push_back(idx);

      // Another anchor is not cloned because its symbol must stay unique.
      // Its own closure covers what it references.
      if (idx != closure.root && closureOf[idx] != kNone) {
        llvm::SetVector<unsigned> &tied = tiedClosures[idx];
        tied.insert(closureIdx);
        tied.insert(closureOf[idx]);
        continue;
      }

      if (isMutableGlobal[idx])
        tiedClosures[idx].insert(closureIdx);
      llvm::append_range(worklist, dependencies[idx]);
    }
  }

  std::vector<SmallVector<unsigned>> groups(tiedClosures.size());
  unsigned groupIdx = 0;
  for (auto it = tiedClosures.begin(); it != tiedClosures.end();
       ++it, ++groupIdx) {
    for (unsigned closureIdx : it->second) {
      std::optional<unsigned> previous = closures[closureIdx].group;
      if (!previous || *previous == groupIdx) {
        groups[groupIdx].push_back(closureIdx);
        closures[closureIdx].group = groupIdx;
        continue;
      }
      for (unsigned moved : groups[*previous]) {
        groups[groupIdx].push_back(moved);
        closures[moved].group = groupIdx;
      }
      groups[*previous].clear();
    }
  }

  std::vector<std::vector<unsigned>> sets;
  std::fill(visitedBy.begin(), visitedBy.end(), kNone);
  for (SmallVector<unsigned> &group : groups) {
    if (group.empty())
      continue;
    unsigned setIdx = static_cast<unsigned>(sets.size());
    std::vector<unsigned> &set = sets.emplace_back();
    for (unsigned closureIdx : group) {
      for (unsigned member : closures[closureIdx].members) {
        if (visitedBy[member] == setIdx)
          continue;
        visitedBy[member] = setIdx;
        set.push_back(member);
      }
    }
  }
  for (Closure &closure : closures)
    if (!closure.group)
      sets.emplace_back(closure.members.begin(), closure.members.end());
  for (unsigned idx = 0, e = symbols.size(); idx != e; ++idx) {
    if (!isOutlineRoot[idx])
      continue;
    SmallVector<unsigned> members =
        collectReachable(idx, /*stopAtOutlineRoots=*/true);
    sets.emplace_back(members.begin(), members.end());
  }

  // Module order within a slice, and the larger slices first so that they
  // start first. Both orders are fixed, which keeps the output reproducible.
  for (std::vector<unsigned> &set : sets)
    llvm::sort(set);
  llvm::sort(sets, [](const auto &lhs, const auto &rhs) {
    if (lhs.size() != rhs.size())
      return lhs.size() > rhs.size();
    return lhs < rhs;
  });
  return sets;
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
  isOutlineRoot.assign(symbols.size(), false);
  if (outlineThreads != 0 && llvm::count(isAnchor, true) > 1)
    pickOutlineRoots(std::max(2u, outlineThreads));

  std::vector<std::vector<unsigned>> sets = partition();

  // With one kernel, or kernels that must stay together, the module is the
  // slice.
  if (sets.size() <= 1) {
    slices.emplace_back().module = std::move(module);
    return success();
  }

  // An outlined function needs a definition that the other slices resolve
  // to. weak_odr, unlike weak, still lets its own slice inline it. It stays
  // dso_local, as an internal function is.
  for (auto [idx, symbol] : llvm::enumerate(symbols)) {
    if (!isOutlineRoot[idx])
      continue;
    auto func = cast<mlir::LLVM::LLVMFuncOp>(symbol);
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
      for (unsigned dep : dependencies[member]) {
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
