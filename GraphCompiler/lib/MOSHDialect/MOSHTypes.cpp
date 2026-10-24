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
// This file implements the MOSH dialect types.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//

LogicalResult ShapeType::printValue(AsmPrinter &p, TypedAttr value) const {
  if (!llvm::isa<KGEN::ParamDeclRefAttr, ShapeAttr>(value))
    return failure();
  printShapeTypedAttr(p, value);
  return success();
}

OptionalParseResult ShapeType::parseValue(AsmParser &p,
                                          TypedAttr &value) const {
  return parseShapeAttr(p, value);
}

//===----------------------------------------------------------------------===//
// MOSHDialect type support
//===----------------------------------------------------------------------===//

void MOSHDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MOSHDialect/MOSHTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MOSHDialect/MOSHTypes.cpp.inc"
