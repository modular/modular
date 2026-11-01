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
//
// Helper for manipulating / renaming parameters for MO / MOGG ops
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_PARAMETERUTILS_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_PARAMETERUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/DenseMap.h"

namespace M::MOGG {

/// Update parameter names of all types and attributes nested in `op` according
/// to the name mapping.
void remapTypesAndAttrs(
    mlir::Operation *op,
    const llvm::DenseMap<StringAttr, TypedAttr> &oldNameToNew,
    bool skipRootOp = false);

/// Normalize all names in the given module. This includes:
/// 1. Symbol names for the top-level symbols in the module.
/// 2. Parameter names used within each graph/function.
void normalizeAllNames(ModuleOp module);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_PARAMETERUTILS_H
