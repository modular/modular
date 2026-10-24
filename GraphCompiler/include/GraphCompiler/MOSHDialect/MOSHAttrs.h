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
// This file declares attributes for the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_H
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/MDialect/MAttrs.h"

namespace M::MOSH {
class ChainType;
class ShapeType;
class ShapeAttr;

/// Return none if parser encountered something other than ShapeAttr or
/// UnknownAttr.
OptionalParseResult parseShapeAttr(AsmParser &p, TypedAttr &shapeAttr);

/// Print an attribute that has MOSH::ShapeType.
void printShapeTypedAttr(AsmPrinter &p, TypedAttr shapeAttr);

/// Convert an attribute that has MOSH::ShapeType to string.
std::string shapeTypedAttrToStr(TypedAttr attr);

/// Returns the index of `decl` in `shape`, or -1 if absent.
int64_t findDeclIdx(ShapeAttr shape, KGEN::ParamDeclAttr decl);

/// Parse the parametric dimensions of a MOSH::ShapeAttr.
ParseResult parseShapeValue(AsmParser &p, SmallVectorImpl<TypedAttr> &values);
ParseResult parseShapeValue(AsmParser &p, SmallVectorImpl<TypedAttr> &values,
                            MOSH::ShapeType type);

/// Print the parametric dimensions of a MOSH::ShapeAttr.
void printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values,
                     MOSH::ShapeType type);
void printShapeValue(AsmPrinter &p, ArrayRef<TypedAttr> values);

/// Print a single parametric dimensions from a MOSH::ShapeAttr.
void printDimensionValue(AsmPrinter &p, TypedAttr value);

} // namespace M::MOSH

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_H
