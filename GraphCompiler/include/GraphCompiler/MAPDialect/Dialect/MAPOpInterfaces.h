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

#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
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

/// Whether `attr` is a SIMD width limit: a constant dim, a parameter dim, or a
/// `GCDAttr` of limits.
bool isWidthLimit(::mlir::Attribute attr);

/// `limit` with nested GCDs flattened, constants folded, duplicate dims
/// dropped and a 1 absorbing the rest; a GCD of one element is that element.
::mlir::TypedAttr simplifyWidthLimit(::mlir::TypedAttr limit);

/// The constant and parameter dims `limit` is the GCD of, nested GCDs
/// flattened.
void collectWidthLimitDims(::mlir::TypedAttr limit,
                           ::llvm::SmallVectorImpl<::mlir::TypedAttr> &dims);

/// Parses a limit: a dim (`2`, `N0`) or a `#map.gcd<[...]>`.
::mlir::ParseResult parseWidthLimit(::mlir::AsmParser &parser,
                                    ::mlir::TypedAttr &limit);

/// Prints a limit as `parseWidthLimit` reads it.
void printWidthLimit(::mlir::AsmPrinter &printer, ::mlir::TypedAttr limit);

/// A number every SIMD width of the enclosing kernel must divide, for the ops
/// in `region` (see `MaxSimdWidthInterface`); null for no limit. Not
/// simplified.
::mlir::TypedAttr getMaxSimdWidth(::mlir::Region &region);

/// Whether `op`, which doesn't implement `MaxSimdWidthInterface`, produces an
/// index or hands one to its regions: index computation the width inference
/// can't follow.
bool computesIndex(::mlir::Operation *op);
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
