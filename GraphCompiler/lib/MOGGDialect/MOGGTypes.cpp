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

#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOGG;

//===----------------------------------------------------------------------===//
// MOGGDialect type support
//===----------------------------------------------------------------------===//

void MOGGDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MOGGDialect/MOGGTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// !mogg.dim custom directive
//===----------------------------------------------------------------------===//

namespace M::MOGG {
// Prints/parses the optional `!mogg.dim<name>` parameter as a bare identifier
// (like MOSH shapes render dynamic dims), reusing KGEN's param-name round-trip.
// A dimension parameter always has the shape dim type (si64).
static mlir::ParseResult parseDimName(mlir::AsmParser &parser,
                                      KGEN::ParamDeclRefAttr &name) {
  mlir::StringAttr rawName;
  if (KGEN::parseParamName(parser, rawName))
    return mlir::failure();
  name = KGEN::ParamDeclRefAttr::get(
      rawName, MOSH::ShapeAttr::getDimType(parser.getContext()));
  return mlir::success();
}
static void printDimName(mlir::AsmPrinter &printer,
                         KGEN::ParamDeclRefAttr name) {
  KGEN::printParamName(printer, name.getName());
}
} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGTypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MOGGDialect/MOGGTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// SimdType and TileType
//===----------------------------------------------------------------------===//

static LogicalResult
verifyScalarElementType(function_ref<InFlightDiagnostic()> emitError,
                        Type elementType) {
  // `fromMLIRType` maps `none` to the unknown dtype, so it is excluded first.
  if (!isa<IntegerType, FloatType>(elementType) ||
      MO::fromMLIRType(elementType).isError())
    return emitError() << "expected a scalar dtype element type, got "
                       << elementType;
  return success();
}

LogicalResult
MOGG::SimdType::verify(function_ref<InFlightDiagnostic()> emitError,
                       Type elementType) {
  return verifyScalarElementType(emitError, elementType);
}

LogicalResult
MOGG::TileType::verify(function_ref<InFlightDiagnostic()> emitError,
                       Type elementType) {
  return verifyScalarElementType(emitError, elementType);
}

//===----------------------------------------------------------------------===//
// MOKernelOperandTypeInterface
//===----------------------------------------------------------------------===//

MO::KernelOperandKind MOGG::TensorType::getKernelOperandKind() const {
  return MO::KernelOperandKind::Tensor;
}
MO::KernelOperandKind MOGG::BufferType::getKernelOperandKind() const {
  return MO::KernelOperandKind::Buffer;
}
MO::KernelOperandKind MOGG::ScalarType::getKernelOperandKind() const {
  return MO::KernelOperandKind::Scalar;
}
