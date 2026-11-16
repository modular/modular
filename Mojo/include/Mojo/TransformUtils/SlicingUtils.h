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

#ifndef KGEN_TRANSFORMUTILS_SLICINGUTILS_H
#define KGEN_TRANSFORMUTILS_SLICINGUTILS_H

#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <vector>

namespace mlir::LLVM {
class GlobalOp;
} // namespace mlir::LLVM

namespace M {
/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
void sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                       const SymbolTable &symtab, IRMapping &reusedMapping,
                       DenseSet<const void *> &visited);

/// Produce a standalone MLIR module by slicing out the dependencies of the
/// provided exported ops.
OwningOpRef<ModuleOp>
produceStandaloneModule(const SymbolTable &symtab,
                        const KGEN::ExportMap &exportedSymbols,
                        bool overrideExported = false);

/// Produce a standalone MLIR module by slicing out the dependencies of the
/// provided exported ops. An `IRMapping` can be provided to be able to map
/// into the sliced module.
OwningOpRef<ModuleOp>
produceStandaloneModule(const SymbolTable &symtab,
                        const KGEN::ExportMap &exportedSymbols,
                        IRMapping &mapping, bool overrideExported = false);

/// Splits a module in the LLVM dialect into one module per exported function,
/// so that each slice translates to LLVM IR in its own LLVM context and on its
/// own thread.
///
/// The partition matches `KGEN::splitPerExported` on LLVM IR. A slice is the
/// transitive closure of one exported symbol, and the users of a mutable global
/// or of another exported symbol share a slice.
///
/// `produceStandaloneModule` clones one closure for each call. This class
/// walks the module once for all slices, and clones only the symbols that
/// more than one slice holds.
class LLVMDialectModuleSplitter {
public:
  explicit LLVMDialectModuleSplitter(OwningOpRef<ModuleOp> module)
      : module(std::move(module)) {}

  /// Computes the partition. Fails with a diagnostic when the module holds an
  /// alias, an ifunc, a comdat or a module-level op that names a symbol. It
  /// also fails when an LLVM dialect op names a symbol that the module does not
  /// define. A failure leaves the module unchanged. A symbol reference in
  /// a discardable attribute or in an op of another dialect may name something
  /// outside the module, so the partition ignores it when it does not resolve.
  ///
  /// `isSliceLocalGlobal` names the globals that each slice can hold its own
  /// copy of, such as one in the shared memory of a GPU block. Such a global
  /// neither roots a slice nor ties its users together.
  LogicalResult
  plan(function_ref<bool(mlir::LLVM::GlobalOp)> isSliceLocalGlobal);

  size_t getNumSlices() const { return slices.size(); }
  size_t getNumSymbols(size_t idx) const { return slices[idx].members.size(); }

  /// Returns slice `idx`. Call once per slice. Calls for different slices may
  /// run on different threads.
  OwningOpRef<ModuleOp> takeSlice(size_t idx);

private:
  struct Slice {
    OwningOpRef<ModuleOp> module;
    /// Indices into `symbols`, in module order.
    std::vector<unsigned> members;
  };

  LogicalResult collectSymbols();
  LogicalResult collectDependencies();
  void
  classifySymbols(function_ref<bool(mlir::LLVM::GlobalOp)> isSliceLocalGlobal);
  std::vector<std::vector<unsigned>> partition();

  OwningOpRef<ModuleOp> module;

  /// The functions and globals of the module, in module order.
  std::vector<Operation *> symbols;
  DenseMap<StringAttr, unsigned> symbolIndex;
  /// The symbols that each symbol references, sorted.
  std::vector<SmallVector<unsigned, 4>> dependencies;
  /// Module-level ops that are not symbols. Every slice gets a copy.
  SmallVector<Operation *> moduleOps;

  /// A symbol that roots a slice.
  std::vector<bool> isAnchor;
  std::vector<bool> isMutableGlobal;

  /// The number of slices that hold each symbol. A symbol in one slice moves
  /// there; a symbol in several stays in the module and each slice clones it.
  std::vector<unsigned> numSlicesUsing;
  std::vector<Slice> slices;
};
} // namespace M

#endif // KGEN_TRANSFORMUTILS_SLICINGUTILS_H
