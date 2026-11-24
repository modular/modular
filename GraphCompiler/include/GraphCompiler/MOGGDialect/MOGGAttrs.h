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
// This file declares the attribute classes for the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_H

#include "GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallSet.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h.inc"

namespace M::MOGG {

/// Prints ` fusionInfo = #...`. The attribute prints in full, so the dialect's
/// alias for it stands in for the tree.
void printFusionInfo(mlir::AsmPrinter &p, mlir::Operation *op,
                     FusionInfoAttr fusionInfo);

mlir::ParseResult parseFusionInfo(mlir::AsmParser &p,
                                  FusionInfoAttr &fusionInfo);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_H
