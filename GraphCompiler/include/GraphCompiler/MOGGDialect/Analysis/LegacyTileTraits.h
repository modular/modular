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
// TODO(GEX-4340): The graph-compiler side of the interim legacy tile traits
// (`extensibility.legacy_tile_traits`). Delete this file with them when the
// tile epilogue interface of GEX-4467 lands. The removal checklist is
// GraphCompiler/docs/LegacyTileTraitsRemoval.md.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_LEGACYTILETRAITS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_LEGACYTILETRAITS_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace M::MOGG {

class PlatformPredicate;

/// The tile epilogue lane a legacy tile trait contract selects.
enum class LegacyTileLane { store, compute };

/// The lane `contract` selects, or nullopt if it binds no legacy tile trait.
std::optional<LegacyTileLane> legacyTileLane(const TensorArgContract &contract);

/// Whether `arg` is a fused output bound by a legacy tile trait.
bool isLegacyTileFusedOutput(const KernelDeclOperandAdaptor &arg);

/// Rejects a kernel whose functions bind a legacy tile trait unless its
/// registration is on the fixed list in LegacyTileTraits.cpp. `functions` may
/// hold null attributes for functions the kernel does not define.
mlir::LogicalResult
verifyLegacyTileRegistration(mlir::Location loc, llvm::StringRef registration,
                             const PlatformPredicate &platform,
                             llvm::ArrayRef<KernelFunctionAttr> functions);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_LEGACYTILETRAITS_H
