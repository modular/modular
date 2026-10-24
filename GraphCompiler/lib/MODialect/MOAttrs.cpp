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
// This file implements the MO dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MO;

#include "GraphCompiler/MODialect/MOEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// MODialect attribute support
//===----------------------------------------------------------------------===//

void MODialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/MODialect/MOAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// LayoutAttr
//===----------------------------------------------------------------------===//

LayoutAttr LayoutAttr::get(MLIRContext *ctx, StringRef formatStr) {
  return get(ctx, StringAttr::get(ctx, formatStr));
}

LayoutAttr LayoutAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *ctx, StringRef formatStr) {
  auto format = StringAttr::get(ctx, formatStr);
  if (failed(verify(emitError, format)))
    return {};
  return get(ctx, format);
}

Attribute LayoutAttr::parse(AsmParser &p, Type t) {
  if (p.parseLess())
    return {};

  StringRef layoutFormat;
  if (failed(p.parseKeyword(&layoutFormat)))
    return {};

  if (p.parseGreater())
    return {};

  MLIRContext *ctx = p.getContext();
  auto formatAttr = StringAttr::get(ctx, layoutFormat);
  return p.getChecked<LayoutAttr>(p.getCurrentLocation(), ctx, formatAttr);
}

void LayoutAttr::print(AsmPrinter &p) const {
  p << '<' << getFormat().strref() << '>';
}

LogicalResult
LayoutAttr::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                   mlir::StringAttr format) {
  // TODO(#18622) make this more extensible
  static StringSet<> kSupportedLayoutStrs = {
      // conv and convTranspose inputs
      // 2d
      "NHWC",
      "NCHW",
      // 3d
      "NDHWC",

      // conv filter
      // 2d
      "RSCF",
      "FCRS",
      // 3d
      "QRSCF",
      "FCQRS",

      // convTranspose filter
      // 2d
      "RSFC",
      "CFRS",
      "QRSFC",

      // packed conv and convTranspose filter
      // 2d
      "FRSCf",
      // 3d
      "FQRSCf",

      // matmul weights
      "KN",

      // transposed matmul weights
      "NK",

      // packed matmul weights
      "KNkni",
  };

  if (!kSupportedLayoutStrs.contains(format.strref()))
    return emitError() << "invalid layout format: " << format;
  return success();
}

//===----------------------------------------------------------------------===//
// ChainAttr
//===----------------------------------------------------------------------===//

bool ChainAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MODialect/MOAttrs.cpp.inc"
