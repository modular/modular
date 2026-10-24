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
// This file implements the RMO dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/RMODialect/RMOAttrs.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

using namespace M;
using namespace RMO;

//===----------------------------------------------------------------------===//
// RMODialect attribute support
//===----------------------------------------------------------------------===//

void RMODialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/RMODialect/RMOAttrs.cpp.inc"
      >();
  addAttributes<PythonFrameAttr>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/RMODialect/RMOAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Python frame location materialization
//===----------------------------------------------------------------------===//

void RMO::materializePythonFrameLocations(mlir::Operation *root) {
  mlir::MLIRContext *ctx = root->getContext();
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [ctx](mlir::OpaqueLoc opaqueLoc) -> mlir::LocationAttr {
        mlir::Location fallback = opaqueLoc.getFallbackLocation();
        if (auto frame = PythonFrameAttr::fromOpaque(opaqueLoc)) {
          llvm::StringRef stackTrace = frame.stack_trace();
          if (!stackTrace.empty())
            return mlir::FusedLoc::get(ctx, {fallback},
                                       mlir::StringAttr::get(ctx, stackTrace));
        }
        return fallback;
      });
  root->walk([&replacer](mlir::Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/false,
                               /*replaceLocs=*/true, /*replaceTypes=*/false);
  });
}
