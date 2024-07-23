//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/SlicingUtils.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

using namespace M;
using namespace KGEN;

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
    if (auto ref = dyn_cast<FlatSymbolRefAttr>(value)) {
      // We know this is a new symbol because this is the first time we've
      // visited this attribute.
      StringAttr name = ref.getAttr();
      Operation *symbol = symtab.lookup(name);
      // If the symbol reference attribute doesn't reference a symbol, somehow
      // invalid IR made it to the ObjectCompiler.
      assert(symbol && "missing symbol, invalid IR?");

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
static void sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                              const SymbolTable &symtab,
                              IRMapping &reusedMapping,
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
                           const ExportMap &exportedSymbols) {
  IRMapping unused;
  return produceStandaloneModule(symtab, exportedSymbols, unused);
}

OwningOpRef<ModuleOp>
M::produceStandaloneModule(const SymbolTable &symtab,
                           const ExportMap &exportedSymbols,
                           IRMapping &mapping) {
  CompilerTimeTraceScope traceScope("produceStandaloneModule");
  auto module = cast<ModuleOp>(symtab.getOp());
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  OwningOpRef<ModuleOp> singleModule = ModuleOp::create(module->getLoc());
  singleModule.get()->setAttrs(module->getAttrDictionary());

  // Create a new symbol table for the sliced module.
  SymbolTable sliceSymtab(*singleModule);

  IRMapping reusedMapping;
  DenseSet<const void *> visited;
  for (auto [sym, exportVal] : exportedSymbols) {
    auto func = symtab.lookup<ExportInterface>(sym);
    assert(func && "Unknown exported symbol");

    // Traverse the call graph and clone all the callees into this module.
    sliceDependencies(func, sliceSymtab, symtab, reusedMapping, visited);

    // Clone the func into this new module. We don't want to remove it from
    // the current module. Make sure the function is also exported in the slice.
    auto sliceFn = sliceSymtab.lookup<ExportInterface>(sym);
    if (!sliceFn) {
      sliceFn = cast<ExportInterface>(func->clone(mapping));
      sliceSymtab.insert(sliceFn);
    }
    ExportKind kind = func.getExportKind();
    sliceFn.setExportKind(kind == ExportKind::NotExported ? exportVal.kind
                                                          : kind);
  }

  return singleModule;
}
