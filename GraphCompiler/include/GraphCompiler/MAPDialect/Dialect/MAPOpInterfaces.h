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

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_H

#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace M::MAP {
/// In `IndexTransformOpInterface::mapLoopDims`, a coordinate that is 0 for
/// every element.
inline constexpr int64_t kPinnedCoordinate = -1;
} // namespace M::MAP

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h.inc"

namespace M::MAP {
/// The loop dimension each coordinate of `index` is (see
/// `IndexTransformOpInterface::mapLoopDims`), following the index transforms
/// from `loopIdx`. Nullopt when one of them doesn't map loop dimensions.
std::optional<llvm::SmallVector<int64_t>> traceLoopDims(mlir::Value index,
                                                        mlir::Value loopIdx);
} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_H
