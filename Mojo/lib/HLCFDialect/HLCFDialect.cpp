//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "mlir/Interfaces/FoldInterfaces.h"

using namespace M;
using namespace HLCF;

namespace {

//===----------------------------------------------------------------------===//
// HLCFDialectFoldInterface
//===----------------------------------------------------------------------===//

struct HLCFDialectFoldInterface : public mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Never hoist a constant out of a declaration scope. We could scan the
  /// parameters declarations to find the highest scope a constant could be
  /// hoisted into, but that is expensive to do. We also do not hoist constants
  /// out of ops that define a subprogram location scope, since the hoisted
  /// constant would carry incorrect scope information into their new scope.
  bool shouldMaterializeInto(Region *region) const override {
    if (DebugInfo::shouldMaterializeConstantsInto(*region))
      return true;
    return isa<KGEN::DeclInterface>(region->getParentOp());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HLCFDialect
//===----------------------------------------------------------------------===//

void M::HLCF::HLCFDialect::initialize() {
  registerAttributes();

  addInterfaces<HLCFDialectFoldInterface>();

  addOperations<
#define GET_OP_LIST
#include "Mojo/HLCFDialect/HLCF.cpp.inc"
      >();
}

Operation *HLCFDialect::materializeConstant(OpBuilder &b, Attribute value,
                                            Type type, Location loc) {
  return KGEN::ParamConstantOp::create(b, loc, cast<TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// Generated Definitions
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFDialect.cpp.inc"
