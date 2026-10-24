//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MODULEHASHER_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MODULEHASHER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace M::KGEN {
class GeneratorOp;
}

namespace M::MOGG {

void hashModuleContents(mlir::ModuleOp module);
void hashModuleContents(mlir::ModuleOp module, const mlir::SymbolTable &symtab);

/// Get the semantic hash attached to the generator by `hashModuleContents`.
std::string getSemanticHash(M::KGEN::GeneratorOp generator);

/// Computes the semantic hash for each operation in the given module and then
/// renames the `symbolsToRename` operations based on their semantic hash.
llvm::DenseMap<mlir::StringAttr, KGEN::GeneratorOp> applySemanticNaming(
    mlir::ModuleOp module, llvm::ArrayRef<mlir::StringAttr> symbolsToRename,
    const mlir::SymbolTable &symtab, llvm::StringRef namePrefix = "");

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MODULEHASHER_H
