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

#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"
#include "Support/MDialect/MDialect.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MOGG;

//===----------------------------------------------------------------------===//
// MOGGDialect
//===----------------------------------------------------------------------===//

namespace {
struct MOGGOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  // A fusion tree can be hundreds of characters long, so ops print a short
  // alias named after the fused ops and the tree is spelled out once at the
  // top of the module.
  AliasResult getAlias(mlir::Attribute attr,
                       llvm::raw_ostream &os) const override {
    if (auto fusionInfo = dyn_cast<FusionInfoAttr>(attr)) {
      os << joinNameParts(fusionInfo.getLeafNames(),
                          TruncationMarker::DroppedCount);
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void MOGGDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
  addInterfaces<MOGGOpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGDialect.cpp.inc"
